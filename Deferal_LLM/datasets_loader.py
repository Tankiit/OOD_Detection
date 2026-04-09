"""
datasets_loader.py
──────────────────
Unified dataset loading for the generation-discrimination gap paper.
Imported by modal_extract.py and run_multi_model_eval.py.

── Dataset narrative arc ─────────────────────────────────────────────────────

  TruthfulQA  → "Does the gap capture misconceptions across 38 topic categories?"
                 38 categories, human-verified MC wrong answers, 817 questions.
                 y_expert varies by category (expert unreliable on Conspiracies).

  HaluEval    → "Does the gap distinguish internal knowledge from hallucination?"
                 TRUE OVA test: y_model ≠ y_expert on wrong reps.
                 GPT-generated hallucinations paired with human-verified answers.

  TriviaQA    → "Does the gap generalise to open-domain factoid QA?"
                 17k trivia questions, cross-question negatives (principled).

  PopQA       → "Does the gap track entity popularity?"
                 14k Wikidata-derived QA pairs with Wikipedia page-view counts.
                 Rare entities → smaller gap → wider interval (EMNLP result).

  BioASQ      → "Does the gap survive domain shift to medical QA?"
                 Biomedical expert-authored questions. Models have less pretraining
                 data here → gap should shrink → more deferral warranted.

── Wrong answer construction ──────────────────────────────────────────────────
  TruthfulQA  — human-verified misconceptions from dataset         ← gold
  HaluEval    — GPT-generated hallucinations from dataset          ← gold
  TriviaQA    — correct answer from adjacent question (shift-by-1) ← principled
  PopQA       — correct answer from adjacent question (shift-by-1) ← principled
  BioASQ      — opposite yes/no answer; or adjacent for factoid    ← principled

── OVA expert labels ──────────────────────────────────────────────────────────
  y_expert_correct / y_expert_wrong: is a human expert reliable here?
  - TruthfulQA: varies by category (Lin et al. 2022 human accuracy)
  - HaluEval:   y_expert=1 for BOTH reps (human always knows the right answer)
                KEY: y_model_wrong=0 but y_expert_wrong=1 → true OVA divergence
  - TriviaQA:   y_expert=1 (well-known facts, expert reliable)
  - PopQA:      y_expert = f(page_views) — rare entities → expert less reliable
  - BioASQ:     y_expert=1 (biomedical experts annotated the dataset)

── HuggingFace dataset IDs ────────────────────────────────────────────────────
  TruthfulQA  truthful_qa / multiple_choice  (validation, 817 rows)
  HaluEval    pminervini/HaluEval / qa       (data, 10k rows; use 2k)
  TriviaQA    trivia_qa / rc                 (validation, 17k rows; use 2k)
  PopQA       akariasai/PopQA               (test, 14k rows; use 2k)
  BioASQ      aps/bioasq_task_b             (test, factoid + yesno; use 1k)
"""

import random
from typing import List, Dict, Any

# ── TruthfulQA human accuracy by category (Lin et al. 2022) ──────────────────
EXPERT_THRESHOLD = 0.80
HUMAN_ACC_BY_CATEGORY = {
    "Misconceptions":        0.65,
    "Conspiracies":          0.52,
    "Myths and Fairytales":  0.67,
    "Paranormal":            0.60,
    "Superstitions":         0.63,
    "Fiction":               0.75,
    "Advertising":           0.72,
    "Psychology":            0.78,
    "Sociology":             0.76,
    "Economics":             0.79,
    "History":               0.88,
    "Politics":              0.81,
    "Law":                   0.84,
    "Health":                0.83,
    "Science":               0.92,
    "Nutrition":             0.80,
    "Statistics":            0.85,
    "Weather":               0.90,
    "Geography":             0.91,
    "Religion":              0.77,
    "Language":              0.82,
    "Logical Falsehoods":    0.88,
    "Distraction":           0.85,
    "__default__":           0.80,
}

# PopQA page_view → expert reliability (log10 scale)
# < 1k views/month → rare → expert less likely to know
POPQA_EXPERT_THRESHOLD_VIEWS = 1000   # monthly Wikipedia page views


def _make_record(
    question: str,
    best_answer: str,
    wrong_answer: str,
    category: str,
    human_acc: float,
    y_model_correct: int,
    y_model_wrong: int,
    y_expert_correct: int,
    y_expert_wrong: int,
    **extra,
) -> Dict[str, Any]:
    """Canonical record format used across all datasets."""
    return {
        "question":         question,
        "best_answer":      best_answer,
        "wrong_answer":     wrong_answer,
        "category":         category,
        "human_acc":        human_acc,
        # OVA labels
        "y_model_correct":  y_model_correct,   # model correct on best_answer?
        "y_model_wrong":    y_model_wrong,     # model correct on wrong_answer?
        "y_expert_correct": y_expert_correct,  # expert reliable for best_answer?
        "y_expert_wrong":   y_expert_wrong,    # expert reliable for wrong_answer?
        **extra,
    }


# ── TruthfulQA ────────────────────────────────────────────────────────────────

def load_truthfulqa(ds) -> List[Dict]:
    """
    HF: truthful_qa / multiple_choice / validation (817 rows)

    Wrong answers: human-verified misconceptions from mc1_targets.
    y_expert: from category-level human accuracy (Lin et al. 2022).

    OVA note: y_model ≈ y_expert here (both identify same correct answer),
    BUT y_expert varies by category. On Conspiracies (human_acc=0.52),
    y_expert=0 → expert is unreliable → f_defer should NOT fire.
    On Science (human_acc=0.92), y_expert=1 → f_defer CAN fire.
    This gives f_defer a semantically meaningful training signal even here.
    """
    records = []
    for item in ds:
        choices = item["mc1_targets"]["choices"]
        labels  = item["mc1_targets"]["labels"]
        correct = [c for c, l in zip(choices, labels) if l == 1]
        wrong   = [c for c, l in zip(choices, labels) if l == 0]
        if not correct or not wrong:
            continue

        cat       = item.get("category", "__default__")
        human_acc = HUMAN_ACC_BY_CATEGORY.get(cat, HUMAN_ACC_BY_CATEGORY["__default__"])
        y_exp     = int(human_acc >= EXPERT_THRESHOLD)

        records.append(_make_record(
            question        = item["question"],
            best_answer     = correct[0],
            wrong_answer    = wrong[0],
            category        = cat,
            human_acc       = human_acc,
            y_model_correct = 1,
            y_model_wrong   = 0,
            y_expert_correct= y_exp,
            y_expert_wrong  = y_exp,
        ))
    return records


# ── HaluEval QA ───────────────────────────────────────────────────────────────

def load_halueval(ds) -> List[Dict]:
    """
    HF: pminervini/HaluEval / qa / data (10k rows)
    Fields: question, right_answer, hallucinated_answer

    This is the TRUE OVA test dataset.

    OVA divergence:
      Correct rep:   y_model=1, y_expert=1  (both agree: answer is right)
      Wrong rep:     y_model=0, y_expert=1  (model hallucinated, expert knows right)
                     ↑ THIS IS THE KEY DIVERGENCE ↑
                     Gap should be strongly negative on wrong reps:
                       f_pred(h_wrong) → low (model unsure of hallucination)
                       f_defer(h_wrong)→ high (expert would answer this correctly)
                       Δ = f_pred - f_defer → strongly negative → DEFER

    The hallucinated_answer is GPT-generated: plausible but factually wrong.
    This is a much harder test than random wrong answers.
    """
    records = []
    for item in ds:
        q     = item.get("question", "").strip()
        right = item.get("right_answer", "").strip()
        wrong = item.get("hallucinated_answer", "").strip()
        if not q or not right or not wrong:
            continue

        records.append(_make_record(
            question        = q,
            best_answer     = right,
            wrong_answer    = wrong,
            category        = "HaluEval-QA",
            human_acc       = 1.0,   # human always knows the right answer
            y_model_correct = 1,
            y_model_wrong   = 0,
            # KEY: expert reliable for BOTH (correct answer is known)
            y_expert_correct= 1,
            y_expert_wrong  = 1,     # ← diverges from y_model_wrong=0
        ))
    return records


# ── TriviaQA ──────────────────────────────────────────────────────────────────

def load_triviaqa(ds) -> List[Dict]:
    """
    HF: trivia_qa / rc / validation (~17k rows, use 2k)
    Fields: question, answer.aliases (list of correct phrasings)

    Wrong answer construction: shift-by-1 across questions.
    wrong[i] = correct[i+1 mod N]

    Why NOT "I don't know":
      That trains the probe on fluency (hedged = wrong, confident = correct).
      The gap Δ(x) would then measure fluency differences, not
      the internal external knowledge discrepancy we care about.

    Why shift-by-1:
      - It's a real fact (not gibberish)
      - It's wrong for THIS question (factually incorrect here)
      - Matched in style/length to a real answer
      - Forces probe to distinguish "correct for this Q" from "correct generally"

    y_expert = 1 (trivia facts are well-known; expert reliable)
    """
    all_correct = []
    raw         = []
    for item in ds:
        q       = item.get("question", "").strip()
        aliases = item.get("answer", {}).get("aliases", [])
        if not q or not aliases:
            continue
        correct = aliases[0]
        all_correct.append(correct)
        raw.append({"question": q, "correct": correct})

    if not raw:
        return []

    # Deterministic shift
    shifted = all_correct[1:] + [all_correct[0]]
    records = []
    for rec, wrong in zip(raw, shifted):
        records.append(_make_record(
            question        = rec["question"],
            best_answer     = rec["correct"],
            wrong_answer    = wrong,
            category        = "TriviaQA",
            human_acc       = 0.95,
            y_model_correct = 1,
            y_model_wrong   = 0,
            y_expert_correct= 1,
            y_expert_wrong  = 1,
        ))
    return records


# ── PopQA ─────────────────────────────────────────────────────────────────────

def load_popqa(ds) -> List[Dict]:
    """
    HF: akariasai/PopQA / test (14k rows, use 2k)
    Fields: question, possible_answers (list), prop, subj, obj,
            s_pop (subject page views), o_pop (object page views)

    PopQA is unique because it has POPULARITY METADATA.
    Wikipedia monthly page views → proxy for how well-known the entity is.

    Key result for paper:
      Rare entities (low page_views) → model has less training signal
      → gap Δ(x) should be SMALLER (model uncertain internally AND externally)
      → interval width [p_low, p_high] should be WIDER
      → more deferral warranted for rare entities

    This is the quantitative backing for the EMNLP interval-width result:
    plot gap_separation vs log10(page_views) → should be monotonically increasing.

    y_expert: 1 if entity is popular (page_views ≥ 1000), else 0.
    Rare-entity questions → expert also less likely to know → y_expert=0.

    Wrong answer: shift-by-1 (same rationale as TriviaQA).
    """
    all_correct  = []
    all_pop      = []   # (s_pop, o_pop) tuple per record
    raw          = []

    for item in ds:
        q        = item.get("question", "").strip()
        answers  = item.get("possible_answers", [])
        if isinstance(answers, str):
            # Sometimes stored as JSON string
            import json
            try:
                answers = json.loads(answers)
            except Exception:
                answers = [answers]
        if not q or not answers:
            continue

        correct = answers[0] if answers else ""
        if not correct:
            continue

        s_pop = item.get("s_pop", 0) or 0   # subject entity page views
        o_pop = item.get("o_pop", 0) or 0   # object entity page views
        pop   = max(int(s_pop), int(o_pop))

        all_correct.append(correct)
        all_pop.append(pop)
        raw.append({
            "question": q,
            "correct":  correct,
            "pop":      pop,
            "prop":     item.get("prop", ""),
            "subj":     item.get("subj", ""),
        })

    if not raw:
        return []

    shifted = all_correct[1:] + [all_correct[0]]
    records = []
    for rec, wrong in zip(raw, shifted):
        pop      = rec["pop"]
        human_acc = min(0.95, 0.50 + 0.45 * min(pop, 100000) / 100000)
        # ↑ smooth function: rare entity (pop→0) → human_acc→0.50
        #                    popular entity (pop≥100k) → human_acc→0.95
        y_exp    = int(pop >= POPQA_EXPERT_THRESHOLD_VIEWS)

        records.append(_make_record(
            question        = rec["question"],
            best_answer     = rec["correct"],
            wrong_answer    = wrong,
            category        = f"PopQA-{rec['prop']}",  # e.g. PopQA-occupation
            human_acc       = round(human_acc, 3),
            y_model_correct = 1,
            y_model_wrong   = 0,
            y_expert_correct= y_exp,
            y_expert_wrong  = y_exp,
            # Extra fields for popularity analysis
            page_views      = pop,
            prop            = rec["prop"],
            subj            = rec["subj"],
        ))
    return records


# ── BioASQ ────────────────────────────────────────────────────────────────────

def load_bioasq(ds) -> List[Dict]:
    """
    HF: aps/bioasq_task_b / test
    Fields: question, type (yesno/factoid/list/summary),
            exact_answer (list), ideal_answer (list)

    We use factoid and yes/no questions only (clear correct/wrong pairs).

    Yes/No questions:
      correct = "yes" or "no" (from exact_answer)
      wrong   = "no"  or "yes" (opposite)
      These are clean binary pairs with no construction needed.

    Factoid questions:
      correct = first exact_answer entity
      wrong   = shift-by-1 from other factoid answers
      (same principled construction as TriviaQA/PopQA)

    BioASQ is the DOMAIN SHIFT test:
      Models have less biomedical pretraining data than general knowledge.
      → Gap should be smaller on biomedical questions
      → More deferral warranted
      → y_expert=1 (biomedical experts authored and verified all answers)

    Note: BioASQ requires registration to access training data.
    The aps/bioasq_task_b HF dataset contains the publicly released test sets.
    Use bigbio/bioasq_task_b as fallback — same format.
    """
    yesno_records  = []
    factoid_raw    = []
    factoid_correct= []

    for item in ds:
        q_type = item.get("type", "").lower()
        q      = item.get("question", "").strip()

        # exact_answer is a list of lists or list of strings depending on version
        exact = item.get("exact_answer", [])
        if isinstance(exact, list) and exact:
            # Flatten: [[answer]] → answer, or [answer] → answer
            ans = exact[0]
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            correct = str(ans).strip()
        else:
            correct = ""

        if not q or not correct:
            continue

        if q_type == "yesno":
            # Opposite is the wrong answer — clean, no construction needed
            wrong = "no" if correct.lower() == "yes" else "yes"
            yesno_records.append(_make_record(
                question        = q,
                best_answer     = correct,
                wrong_answer    = wrong,
                category        = "BioASQ-YesNo",
                human_acc       = 1.0,
                y_model_correct = 1,
                y_model_wrong   = 0,
                y_expert_correct= 1,
                y_expert_wrong  = 1,
            ))

        elif q_type == "factoid":
            factoid_correct.append(correct)
            factoid_raw.append({"question": q, "correct": correct})

    # Factoid: shift-by-1
    factoid_records = []
    if factoid_raw:
        shifted = factoid_correct[1:] + [factoid_correct[0]]
        for rec, wrong in zip(factoid_raw, shifted):
            factoid_records.append(_make_record(
                question        = rec["question"],
                best_answer     = rec["correct"],
                wrong_answer    = wrong,
                category        = "BioASQ-Factoid",
                human_acc       = 1.0,
                y_model_correct = 1,
                y_model_wrong   = 0,
                y_expert_correct= 1,
                y_expert_wrong  = 1,
            ))

    # Combine and shuffle for balanced sampling
    all_records = yesno_records + factoid_records
    random.seed(42)
    random.shuffle(all_records)
    return all_records


# ── Router ────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "truthfulqa":  {
        "hf_id":     "truthful_qa",
        "hf_config": "multiple_choice",
        "split":     "validation",
        "max_rows":  None,      # all 817
        "loader":    load_truthfulqa,
        "description": "TruthfulQA MC — 817q, 38 categories, human-verified wrongs",
    },
    "halueval_qa": {
        "hf_id":     "pminervini/HaluEval",
        "hf_config": "qa",
        "split":     "data",
        "max_rows":  2000,
        "loader":    load_halueval,
        "description": "HaluEval QA — 2k subset, GPT hallucinations, TRUE OVA test",
    },
    "triviaqa": {
        "hf_id":     "trivia_qa",
        "hf_config": "rc",
        "split":     "validation",
        "max_rows":  2000,
        "loader":    load_triviaqa,
        "description": "TriviaQA RC — 2k subset, cross-question negatives",
    },
    "popqa": {
        "hf_id":     "akariasai/PopQA",
        "hf_config": None,
        "split":     "test",
        "max_rows":  2000,
        "loader":    load_popqa,
        "description": "PopQA — 2k subset, entity popularity axis (page views)",
    },
    "bioasq": {
        "hf_id":     "aps/bioasq_task_b",
        "hf_config": None,
        "split":     "test",
        "max_rows":  1000,
        "loader":    load_bioasq,
        "description": "BioASQ — 1k subset, biomedical domain shift test",
    },
}


def load_dataset_records(dataset_name: str) -> List[Dict]:
    """
    Main entry point. Loads and normalises any registered dataset.

    Returns list of canonical records (see _make_record schema).
    Prints OVA divergence statistics for diagnostics.
    """
    from datasets import load_dataset

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Options: {list(DATASET_REGISTRY.keys())}"
        )

    cfg = DATASET_REGISTRY[dataset_name]
    print(f"\n  Loading {dataset_name} ({cfg['description']})...")

    kwargs = {"split": cfg["split"]}
    if cfg["hf_config"]:
        ds = load_dataset(cfg["hf_id"], cfg["hf_config"], **kwargs)
    else:
        ds = load_dataset(cfg["hf_id"], **kwargs)

    if cfg["max_rows"]:
        n = min(cfg["max_rows"], len(ds))
        ds = ds.select(range(n))

    records = cfg["loader"](ds)

    if not records:
        print(f"  ✗ No records loaded from {dataset_name}")
        return []

    # ── Diagnostics ───────────────────────────────────────────────────────────
    n = len(records)
    n_div = sum(
        1 for r in records
        if r["y_model_wrong"] != r["y_expert_wrong"]
    )
    n_exp_reliable = sum(1 for r in records if r["y_expert_correct"] == 1)

    print(f"  ✓  {n} records")
    print(f"     Expert reliable (y_expert_correct=1): "
          f"{n_exp_reliable}/{n} ({n_exp_reliable/n*100:.0f}%)")
    if n_div > 0:
        print(f"     OVA divergence (y_model_wrong ≠ y_expert_wrong): "
              f"{n_div}/{n} ({n_div/n*100:.0f}%) ← true OVA test cases")
    else:
        print(f"     OVA divergence: 0 (y_model == y_expert; "
              f"gap valid but not testing full OVA claim)")

    # Extra diagnostic for PopQA
    if dataset_name == "popqa":
        pops = [r.get("page_views", 0) for r in records]
        import math
        n_rare    = sum(1 for p in pops if p < POPQA_EXPERT_THRESHOLD_VIEWS)
        n_popular = n - n_rare
        print(f"     Rare entities (<{POPQA_EXPERT_THRESHOLD_VIEWS} views/mo): "
              f"{n_rare}/{n}")
        print(f"     Popular entities: {n_popular}/{n}")

    return records


def dataset_summary() -> None:
    """Print a table of all datasets for --dry-run."""
    print(f"\n{'Dataset':<14}  {'HF ID':<32}  {'Size':>6}  Description")
    print("  " + "─" * 75)
    for name, cfg in DATASET_REGISTRY.items():
        size = cfg["max_rows"] or 817
        print(f"  {name:<14}  {cfg['hf_id']:<32}  {size:>6}  "
              f"{cfg['description'][:50]}")


if __name__ == "__main__":
    # Quick verification: try loading each dataset header
    import sys
    dataset_summary()
    print("\nTo test a loader:")
    print("  python -c \"from datasets_loader import load_dataset_records; "
          "r = load_dataset_records('truthfulqa'); print(r[0])\"")
