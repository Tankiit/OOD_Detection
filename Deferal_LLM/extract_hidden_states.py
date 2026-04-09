"""
extract_hidden_states.py
──────────────────────────────
Extracts hidden states and generation log-probs from an LLM on TruthfulQA.
Uses HuggingFace transformers with output_hidden_states=True.

── Which layers? ─────────────────────────────────────────────────────────────
Factual knowledge peaks at 60-80% of network depth (Orgad et al. ICLR 2025,
Li et al. NeurIPS 2023). We extract the last `--layers` transformer layers
(default 8), which for a 32-layer model covers layers 24-32 — the right range.

A layer sweep is run automatically on 50 questions to find the single best
layer. The main extraction uses mean-of-last-N. The sweep result is saved
separately for ablation.

── Which tokens? ─────────────────────────────────────────────────────────────
Factual signal concentrates at the first entity/claim token of the answer
(Orgad et al. 2025). We extract three representations per (question, answer):
  h_first  : hidden state at first answer token only        ← strongest signal
  h_mean   : mean-pool over all answer tokens               ← robust baseline
  h_early  : mean of first min(3, ans_len) answer tokens    ← compromise

The probe in train_probe uses h_first by default (--token-mode first).

── OVA expert labels ─────────────────────────────────────────────────────────
TruthfulQA questions span 38 categories. Human accuracy varies by category
(Lin et al. 2022, Table 2): Science ~92%, Health ~83%, Conspiracies ~52%.
y_expert = 1 when a reliable human would answer correctly (human_acc >= threshold).
This is the f_defer training target — distinct from y_model (did LLM get it right?).

Output: outputs/hidden_states.pt

Usage:
    python extract_hidden_states.py
    python extract_hidden_states.py --max_questions 50   # quick test
    python extract_hidden_states.py --layers 12          # deeper
    python extract_hidden_states.py --token-mode mean    # all tokens
    python extract_hidden_states.py --resume             # after interrupt
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "google/gemma-2-2b-it"
OUTPUT_DIR      = Path("outputs")
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
OUTPUT_FILE     = OUTPUT_DIR / "hidden_states.pt"
SAVE_EVERY      = 50

# Human accuracy by TruthfulQA category (Lin et al. 2022, Table 2 estimates).
# y_expert = 1 if human_acc >= EXPERT_THRESHOLD → expert is reliable → defer there
# y_expert = 0 if human_acc < threshold → expert also wrong → don't defer there
EXPERT_THRESHOLD = 0.80   # tune: 0.70 = more expert-reliable, 0.90 = stricter
HUMAN_ACC_BY_CATEGORY = {
    "Misconceptions":              0.65,
    "Conspiracies":                0.52,
    "Myths and Fairytales":        0.67,
    "Paranormal":                  0.60,
    "Superstitions":               0.63,
    "Fiction":                     0.75,
    "Advertising":                 0.72,
    "Psychology":                  0.78,
    "Sociology":                   0.76,
    "Economics":                   0.79,
    "History":                     0.88,
    "Politics":                    0.81,
    "Law":                         0.84,
    "Health":                      0.83,
    "Science":                     0.92,
    "Nutrition":                   0.80,
    "Statistics":                  0.85,
    "Weather":                     0.90,
    "Geography":                   0.91,
    "Religion":                    0.77,
    "Language":                    0.82,
    "Logical Falsehoods":          0.88,
    "Distraction":                 0.85,
    # Default for any missing category
    "__default__":                 0.80,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--max_questions",  type=int, default=None)
    p.add_argument("--layers",         type=int, default=None,
                   help="Number of last transformer layers. Defaults to principled scaling (e.g. 8 for 32 layers, 12 for 80).")
    p.add_argument("--token-mode",     default="first",
                   choices=["first", "mean", "early"],
                   help="How to pool over answer tokens. "
                        "'first'=first token (best per Orgad et al.), "
                        "'mean'=all tokens, 'early'=first 3 tokens")
    p.add_argument("--output",         default=str(OUTPUT_FILE))
    p.add_argument("--resume",         action="store_true")
    p.add_argument("--skip-sweep",     action="store_true",
                   help="Skip the layer sweep (saves ~5 min)")
    p.add_argument("--expert-threshold", type=float, default=EXPERT_THRESHOLD)
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        print("  Using MPS (Apple Silicon)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("  Using CUDA")
        return torch.device("cuda")
    print("  Using CPU")
    return torch.device("cpu")


def load_model(model_name, device):
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
    if device.type == "mps":
        # Gemma-2 and some other models need eager attention on MPS
        # (SDPA uses ops not yet supported in MPS backend)
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to(device).eval()

    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.0f}M params | {n_layers} layers | d_model={d_model}")
    return tokenizer, model, n_layers, d_model


def load_truthfulqa(expert_threshold):
    """
    Load TruthfulQA (multiple_choice split).
    Constructs y_expert from category-level human accuracy.
    """
    print("\nLoading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    records = []
    for item in ds:
        choices = item["mc1_targets"]["choices"]
        labels  = item["mc1_targets"]["labels"]
        wrong   = [c for c, l in zip(choices, labels) if l == 0]
        cat     = item.get("category", "Unknown")

        # Human accuracy for this category → expert reliability label
        human_acc    = HUMAN_ACC_BY_CATEGORY.get(cat,
                       HUMAN_ACC_BY_CATEGORY["__default__"])
        expert_label = int(human_acc >= expert_threshold)

        records.append({
            "question":         item["question"],
            "best_answer":      item["best_answer"],
            "wrong_answers":    wrong,
            "category":         cat,
            "human_acc":        human_acc,
            # OVA labels:
            # y_expert_correct: expert reliable for the correct-answer representation
            # y_expert_wrong:   expert reliable for the wrong-answer representation
            # (same value — expert reliability is question-level, not answer-level)
            "y_expert_correct": expert_label,
            "y_expert_wrong":   expert_label,
        })

    cats = set(r["category"] for r in records)
    n_reliable = sum(1 for r in records if r["y_expert_correct"] == 1)
    print(f"  {len(records)} questions | {len(cats)} categories")
    print(f"  Expert reliable (human_acc ≥ {expert_threshold}): "
          f"{n_reliable}/{len(records)} ({n_reliable/len(records)*100:.0f}%)")
    return records


# ── Core extraction ───────────────────────────────────────────────────────────

def pool_hidden_state(hs_layers, ans_start, ans_end, token_mode):
    """
    Pool hidden states over answer token positions and across layers.

    hs_layers : list of tensors, each (1, seq_len, d_model)
    Returns   : float32 tensor (d_model,)

    token_mode options:
      'first' — first answer token only (Orgad et al. recommendation)
      'early' — mean of first min(3, ans_len) tokens
      'mean'  — mean of all answer tokens
    """
    ans_len = ans_end - ans_start

    layer_vecs = []
    for hs in hs_layers:
        h = hs[0, ans_start:ans_end, :]   # (ans_len, d)

        if token_mode == "first" or ans_len == 1:
            vec = h[0]                     # (d,)
        elif token_mode == "early":
            n   = min(3, ans_len)
            vec = h[:n].mean(dim=0)        # (d,)
        else:  # mean
            vec = h.mean(dim=0)            # (d,)

        layer_vecs.append(vec)

    # Mean across layers
    return torch.stack(layer_vecs).mean(dim=0).cpu().float()   # (d,)


def extract_one(model, tokenizer, device, question, answer, n_layers, token_mode):
    """
    Single forward pass → (h_pooled, gen_logprob) or (None, None).

    h_pooled   : float32 tensor (d_model,)
    gen_logprob: float — mean log-prob of answer tokens (length-normalised)
    """
    q_prefix  = f"Question: {question}\nAnswer: "
    full_text = q_prefix + answer

    full_ids   = tokenizer(full_text,  return_tensors="pt").input_ids
    prefix_ids = tokenizer(q_prefix,   return_tensors="pt").input_ids

    ans_start = prefix_ids.shape[1]
    ans_end   = full_ids.shape[1]

    if ans_end <= ans_start:
        return None, None

    full_ids = full_ids.to(device)

    with torch.no_grad():
        out = model(full_ids, output_hidden_states=True)

    # ── hidden states ─────────────────────────────────────────────────────────
    # out.hidden_states: tuple of (n_transformer_layers + 1) tensors
    # Index 0 = embedding layer output (skip — no attention, no factual encoding)
    # We take the last n_layers transformer layers
    hs_last_n = out.hidden_states[-n_layers:]   # each: (1, seq_len, d)

    h = pool_hidden_state(list(hs_last_n), ans_start, ans_end, token_mode)

    # ── generation log-probability ────────────────────────────────────────────
    # logits[t] predicts token t+1, so for answer token at position t,
    # its log-prob comes from logits[t-1]
    logits    = out.logits[0]                                    # (seq, vocab)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    ans_ids    = full_ids[0, ans_start:ans_end]                  # (ans_len,)
    pred_lp    = log_probs[ans_start-1 : ans_end-1]              # (ans_len, vocab)
    token_lps  = pred_lp[torch.arange(len(ans_ids)), ans_ids]    # (ans_len,)

    gen_logprob = token_lps.mean().item()   # length-normalised

    return h, gen_logprob


# ── Layer sweep ───────────────────────────────────────────────────────────────

def run_layer_sweep(model, tokenizer, device, records, n_total_layers,
                    token_mode, n_sweep=50):
    """
    Sweep all transformer layers to find which single layer has the highest
    linear probe AUROC for predicting model correctness.

    This identifies the "peak knowledge layer" — used for ablation in paper.
    Runtime: ~5 min on Mac for 50 questions × n_total_layers forward passes.

    Shortcut: we do ONE forward pass per question and save ALL layer outputs,
    then train a probe per layer — not one pass per layer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    print(f"\nRunning layer sweep on {n_sweep} questions "
          f"({n_total_layers} layers)...")
    records_sub = records[:n_sweep]

    all_hs    = {i: [] for i in range(n_total_layers + 1)}  # layer → list of vecs
    y_correct = []  # 1 = correct rep, 0 = wrong rep

    for rec in tqdm(records_sub, desc="Layer sweep"):
        q     = rec["question"]
        c_ans = rec["best_answer"]
        w_ans = rec["wrong_answers"][0] if rec["wrong_answers"] else None
        if not w_ans:
            continue

        for ans, label in [(c_ans, 1), (w_ans, 0)]:
            q_prefix  = f"Question: {q}\nAnswer: "
            full_text = q_prefix + ans
            full_ids  = tokenizer(full_text,  return_tensors="pt").input_ids
            pfx_ids   = tokenizer(q_prefix,   return_tensors="pt").input_ids
            a_start   = pfx_ids.shape[1]
            a_end     = full_ids.shape[1]
            if a_end <= a_start:
                continue

            with torch.no_grad():
                out = model(full_ids.to(device), output_hidden_states=True)

            # Store each layer's hidden state
            for layer_i, hs in enumerate(out.hidden_states):
                # hs: (1, seq_len, d)
                h = pool_hidden_state([hs], a_start, a_end, token_mode)
                all_hs[layer_i].append(h.numpy())
            y_correct.append(label)

    y_arr = np.array(y_correct)

    # Train a logistic probe per layer, evaluate AUROC via 5-fold CV
    layer_aurocs = {}
    for layer_i in tqdm(range(n_total_layers + 1), desc="Probing layers"):
        if len(all_hs[layer_i]) < 10:
            continue
        X = np.stack(all_hs[layer_i])
        if X.shape[0] != len(y_arr):
            continue
        sc  = StandardScaler()
        X_s = sc.fit_transform(X)
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        scores = cross_val_score(clf, X_s, y_arr, cv=5, scoring="roc_auc")
        layer_aurocs[layer_i] = float(scores.mean())

    best_layer = max(layer_aurocs, key=layer_aurocs.get)
    best_auroc = layer_aurocs[best_layer]
    frac       = best_layer / n_total_layers

    print(f"\n  Layer sweep results:")
    print(f"  Best layer: {best_layer}/{n_total_layers} "
          f"({frac*100:.0f}% depth) — AUROC = {best_auroc:.4f}")
    print(f"  (Bottom 5 AUROC layers: "
          f"{sorted(layer_aurocs, key=layer_aurocs.get)[:5]})")

    return layer_aurocs, best_layer


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(results, idx, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, checkpoint_dir / f"checkpoint_{idx:04d}.pt")


def load_latest_checkpoint(checkpoint_dir):
    cps = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not cps:
        return None, 0
    data = torch.load(cps[-1], map_location="cpu", weights_only=False)
    idx  = int(cps[-1].stem.split("_")[1])
    print(f"  Resuming from checkpoint at question {idx}")
    return data, idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    OUTPUT_DIR.mkdir(exist_ok=True)

    tokenizer, model, n_total_layers, d_model = load_model(args.model, device)
    
    if args.layers is None:
        if n_total_layers >= 80:
            extract_layers = 12
        elif n_total_layers >= 42:
            extract_layers = 10
        else:
            extract_layers = 8
    else:
        extract_layers = args.layers

    print(f"\n{'='*58}")
    print(f"  Generation-Discrimination Gap — Step 1")
    print(f"  Model:       {args.model}")
    print(f"  Layers:      last {extract_layers} (of {n_total_layers}) transformer layers")
    print(f"  Token mode:  {args.token_mode}")
    print(f"  Expert thr:  {args.expert_threshold}")
    print(f"{'='*58}")

    records = load_truthfulqa(args.expert_threshold)

    if args.max_questions:
        records = records[:args.max_questions]
        print(f"  Capped at {len(records)} questions")

    # ── Layer sweep ───────────────────────────────────────────────────────────
    sweep_results = {}
    best_layer    = None

    if not args.skip_sweep:
        layer_aurocs, best_layer = run_layer_sweep(
            model, tokenizer, device, records,
            n_total_layers, args.token_mode,
            n_sweep=min(50, len(records))
        )
        sweep_results = {
            "layer_aurocs": layer_aurocs,
            "best_layer":   best_layer,
            "n_total_layers": n_total_layers,
        }
        # Save sweep separately
        torch.save(sweep_results, OUTPUT_DIR / "layer_sweep.pt")
        print(f"\n  Sweep saved → outputs/layer_sweep.pt")

    # ── Resume or fresh start ─────────────────────────────────────────────────
    empty_results = lambda: {k: [] for k in [
        "h_correct", "h_wrong",
        "lp_correct", "lp_wrong",
        "questions", "correct_ans", "wrong_ans",
        "categories", "human_accs",
        "y_expert_correct", "y_expert_wrong",
    ]}

    if args.resume:
        results, start_idx = load_latest_checkpoint(CHECKPOINT_DIR)
        if results is None:
            print("  No checkpoint found — starting fresh")
            start_idx = 0
            results   = empty_results()
    else:
        start_idx = 0
        results   = empty_results()

    records = records[start_idx:]

    # ── Extraction loop ───────────────────────────────────────────────────────
    print(f"\nExtracting hidden states "
          f"({len(records)} questions remaining)...")
    print(f"  Token mode: '{args.token_mode}' "
          f"(first answer token is most factually informative)")

    skipped       = 0
    processed     = start_idx

    for i, rec in enumerate(tqdm(records, desc="Questions")):
        q     = rec["question"]
        c_ans = rec["best_answer"]
        w_ans = rec["wrong_answers"][0] if rec["wrong_answers"] else None

        if not w_ans:
            skipped += 1
            continue

        h_c, lp_c = extract_one(
            model, tokenizer, device, q, c_ans,
            extract_layers, args.token_mode
        )
        h_w, lp_w = extract_one(
            model, tokenizer, device, q, w_ans,
            extract_layers, args.token_mode
        )

        if h_c is None or h_w is None:
            skipped += 1
            continue

        results["h_correct"].append(h_c)
        results["h_wrong"].append(h_w)
        results["lp_correct"].append(lp_c)
        results["lp_wrong"].append(lp_w)
        results["questions"].append(q)
        results["correct_ans"].append(c_ans)
        results["wrong_ans"].append(w_ans)
        results["categories"].append(rec["category"])
        results["human_accs"].append(rec["human_acc"])
        results["y_expert_correct"].append(rec["y_expert_correct"])
        results["y_expert_wrong"].append(rec["y_expert_wrong"])
        processed += 1

        if (i + 1) % SAVE_EVERY == 0:
            save_checkpoint(results, processed, CHECKPOINT_DIR)
            tqdm.write(f"  Checkpoint at {processed} questions")

    N = len(results["h_correct"])
    print(f"\n  Done — processed {N}, skipped {skipped}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dict = {
        "h_correct":        torch.stack(results["h_correct"]),      # (N, d)
        "h_wrong":          torch.stack(results["h_wrong"]),        # (N, d)
        "lp_correct":       torch.tensor(results["lp_correct"]),    # (N,)
        "lp_wrong":         torch.tensor(results["lp_wrong"]),      # (N,)
        "questions":        results["questions"],
        "correct_ans":      results["correct_ans"],
        "wrong_ans":        results["wrong_ans"],
        "categories":       results["categories"],
        "human_accs":       results["human_accs"],
        "y_expert_correct": torch.tensor(
            results["y_expert_correct"], dtype=torch.long),         # (N,)
        "y_expert_wrong":   torch.tensor(
            results["y_expert_wrong"],   dtype=torch.long),         # (N,)
        "model":            args.model,
        "n_layers":         extract_layers,
        "token_mode":       args.token_mode,
        "n_questions":      N,
        "d_model":          d_model,
        "n_total_layers":   n_total_layers,
        "sweep":            sweep_results,
        "expert_threshold": args.expert_threshold,
    }
    torch.save(save_dict, args.output)

    # Sanity stats
    lp_c = save_dict["lp_correct"]
    lp_w = save_dict["lp_wrong"]
    gap  = lp_c - lp_w
    n_exp = int(save_dict["y_expert_correct"].sum())

    print(f"\nSaved → {args.output}")
    print(f"  Shape:             {save_dict['h_correct'].shape}")
    print(f"  lp_correct > lp_wrong:  "
          f"{(lp_c > lp_w).float().mean()*100:.1f}% of questions")
    print(f"  Gap mean ± std:    "
          f"{gap.mean():.3f} ± {gap.std():.3f}")
    print(f"  Expert-reliable:   {n_exp}/{N} questions "
          f"(human_acc ≥ {args.expert_threshold})")
    if sweep_results:
        bl = sweep_results["best_layer"]
        ba = sweep_results["layer_aurocs"][bl]
        print(f"  Best probe layer:  {bl}/{n_total_layers} "
              f"(AUROC={ba:.4f})")
    print("\n✓  Step 1 complete — run train_probe.py next")


if __name__ == "__main__":
    main()
