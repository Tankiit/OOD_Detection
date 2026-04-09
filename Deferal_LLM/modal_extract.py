"""
Run hidden-state extraction on Modal, one GPU container per dataset, in parallel.

This mirrors the pattern in Modal's hp_sweep_gpt example (parallel ``.starmap``):
https://modal.com/docs/examples/hp_sweep_gpt

Artifacts are written to a Modal Volume under ``/vol/extract/<dataset>/hidden_states.pt``.
Download with::

    modal volume get deferral-llm-extract-v1 /extract/<dataset>/hidden_states.pt ./outputs/

Setup::

    pip install modal
    modal setup

Remote jobs do **not** use your conda env; they use the Docker image in this file
(a Linux GPU box with PyTorch from the CUDA 12.1 wheel index).

**Do not put HF tokens in source code.**

Create a Modal secret named ``huggingface`` with ``HF_TOKEN`` (from
https://huggingface.co/settings/tokens ). The function uses
``secrets=[modal.Secret.from_name("huggingface")]``; if you truly need no token,
comment that line out (only for checkpoints that download without auth).

**Hugging Face errors**

- **401** — Hub does not see a valid login: fix ``HF_TOKEN`` in the Modal secret (key name must be ``HF_TOKEN``, not ``HF_TOKENS``).
- **403** — “not in the authorized list”: your token works, but this **account** is not yet allowed to use that **exact** repo. Open the model URL in a browser while logged into HF, click **Agree / Request access** (Meta Llama often needs approval; can take time). Until then, use a non‑Meta preset e.g. ``qwen-1.5b``, ``mistral-7b``, or ``phi-3-mini``.

If the secret already exists: update ``HF_TOKEN`` in the Modal dashboard (or ``modal secret delete huggingface`` then create again).

**Model presets** (pass as ``--model <alias>`` or a full HF model id):

| alias            | Hugging Face id |
|------------------|-----------------|
| ``qwen-1.5b``    | Qwen/Qwen2.5-1.5B-Instruct |
| ``mistral-7b``   | mistralai/Mistral-7B-Instruct-v0.3 |
| ``phi-3-mini``   | microsoft/Phi-3-mini-4k-instruct |
| ``gemma-2-9b``   | google/gemma-2-9b-it |
| ``llama-3-8b``   | meta-llama/Llama-3.1-8B-Instruct |
| ``llama-3-70b``  | meta-llama/Llama-3.1-70B-Instruct |
| ``llama-3-8b-v3``| meta-llama/Meta-Llama-3-8B-Instruct (older 3.0 id if you only have that repo) |

``llama-3-*`` and ``gemma-2-9b`` require HF license access; **403** until your account is on the allow-list.

``llama-3-70b`` needs a much larger GPU than the default A10G; edit ``gpu=`` on
``extract_dataset_job`` (e.g. ``A100-80GB``) before running.

Artifacts are stored per model under ``/vol/extract/<model_slug>/<dataset>/``.

Run all listed datasets in parallel::

    modal run modal_extract.py --datasets truthfulqa,halueval_qa,triviaqa,popqa,bioasq

After extraction, train probes locally::

    python train_probe.py --input_path ./outputs/hidden_states.pt
"""

from __future__ import annotations

from pathlib import Path, PosixPath

import modal

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App("deferral-llm-hidden-states")

volume = modal.Volume.from_name("deferral-llm-extract-v1", create_if_missing=True)
VOL_ROOT = PosixPath("/vol")
EXTRACT_ROOT = VOL_ROOT / "extract"
HF_HOME = VOL_ROOT / "hf"

_REPO = Path(__file__).resolve().parent

# Short names → full HF repo ids (``--model`` accepts either).
MODEL_ALIASES: dict[str, str] = {
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    # Llama 3.1 ids (common HF naming); use llama-3-8b-v3 for Meta-Llama-3-8B-Instruct if needed
    "llama-3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3-8b-v3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70b-v3": "meta-llama/Meta-Llama-3-70B-Instruct",
}


def resolve_model_id(name: str) -> str:
    key = name.strip().lower()
    return MODEL_ALIASES.get(key, name.strip())


def hf_model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


# Substrings in HF repo ids that are almost always license-gated → need HF_TOKEN in the container.
_GATED_HF_MARKERS = ("meta-llama/", "meta-llama-", "google/gemma-")


def looks_gated_on_hf(model_id: str) -> bool:
    mid = model_id.lower()
    return any(s.lower() in mid for s in _GATED_HF_MARKERS)


# GPU PyTorch from PyTorch's CUDA wheel index (not plain PyPI).
# - transformers>=4.44 refuses torch<2.4 → "Disabling PyTorch… PyTorch >= 2.4 required"
# - Unpinned numpy pulls 2.x; torch wheels pre-2.4 expect numpy 1.x → import/_ARRAY_API failures
_PYTORCH_CUDA = "https://download.pytorch.org/whl/cu121"

extract_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        index_url=_PYTORCH_CUDA,
    )
    .pip_install(
        "numpy<2",
        "transformers>=4.44.0,<5",
        "datasets>=2.19.0",
        "accelerate>=0.30.0",
        "scikit-learn>=1.4.0",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "PYTHONPATH": "/root/proj",
            "HF_HOME": str(HF_HOME),
            "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
            "HF_DATASETS_CACHE": str(HF_HOME / "datasets"),
        }
    )
    .add_local_dir(
        _REPO,
        remote_path="/root/proj",
        ignore=[
            "*.pt",
            "*.pth",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "outputs",
        ],
    )
)


@app.function(
    image=extract_image,
    gpu="A10G",
    volumes={str(VOL_ROOT): volume},
    timeout=8 * HOURS,
    secrets=[modal.Secret.from_name("huggingface")],
)
def extract_dataset_job(
    dataset_name: str,
    model: str,
    max_questions: int | None,
    layers_arg: int | None,
    token_mode: str,
    expert_threshold: float,
    skip_sweep: bool,
) -> str:
    import argparse
    import os

    raw_tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKENS")
    )
    tok = (raw_tok or "").strip() or None

    if looks_gated_on_hf(model) and not tok:
        raise RuntimeError(
            "Gated Hugging Face model requires HF_TOKEN inside the container.\n"
            "1) Create a Modal secret named exactly `huggingface` with env key HF_TOKEN, e.g.\n"
            "     modal secret create huggingface HF_TOKEN=hf_...\n"
            "2) Accept the model license on huggingface.co while logged in with that token.\n"
            "3) Keep secrets=[modal.Secret.from_name('huggingface')] on extract_dataset_job."
        )

    if tok:
        from huggingface_hub import login

        login(token=tok, add_to_git_credential=False)

    HF_HOME.mkdir(parents=True, exist_ok=True)

    from datasets_loader import load_dataset_records
    from extract_hidden_states import (
        adapt_unified_records,
        default_extract_layers,
        get_device,
        load_model,
        run_extraction,
    )

    slug = hf_model_slug(model)
    out_dir = EXTRACT_ROOT / slug / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_pt = out_dir / "hidden_states.pt"

    args = argparse.Namespace(
        model=model,
        max_questions=max_questions,
        layers=layers_arg,
        token_mode=token_mode,
        output=str(output_pt),
        resume=False,
        skip_sweep=skip_sweep,
        expert_threshold=expert_threshold,
        dataset=dataset_name,
    )

    device = get_device()
    tokenizer, model_obj, n_total_layers, d_model = load_model(args.model, device)
    extract_layers = default_extract_layers(n_total_layers, args.layers)

    raw = load_dataset_records(dataset_name)
    records = adapt_unified_records(raw)

    run_extraction(
        args,
        records,
        device=device,
        tokenizer=tokenizer,
        model=model_obj,
        n_total_layers=n_total_layers,
        d_model=d_model,
        extract_layers=extract_layers,
    )

    volume.commit()
    return str(output_pt)


@app.local_entrypoint()
def main(
    datasets: str = "truthfulqa,halueval_qa",
    model: str = "qwen-1.5b",
    max_questions: int = 0,
    layers: int = -1,
    token_mode: str = "first",
    expert_threshold: float = 0.80,
    skip_sweep: bool = False,
):
    """
    Launch one GPU container per dataset name (comma-separated), in parallel.

    ``max_questions`` / ``layers``: use 0 or negative to mean “use script defaults”
    (all questions per registry cap, or heuristic layer count).

    ``model`` may be a key from ``MODEL_ALIASES`` (e.g. ``mistral-7b``) or any HF id.
    """
    names = [d.strip() for d in datasets.split(",") if d.strip()]
    if not names:
        raise SystemExit("Pass at least one dataset in --datasets")

    model_id = resolve_model_id(model)
    mq = None if max_questions <= 0 else max_questions
    ly = None if layers < 0 else layers

    payload = [
        (
            ds,
            model_id,
            mq,
            ly,
            token_mode,
            expert_threshold,
            skip_sweep,
        )
        for ds in names
    ]

    print(f"Model: {model_id}  (from --model {model!r})")
    print(f"Spawning {len(payload)} parallel jobs on Modal…")
    for rank, result in enumerate(
        extract_dataset_job.starmap(payload, order_outputs=False)
    ):
        print(f"[{rank + 1}/{len(payload)}] wrote {result}")

    ex = hf_model_slug(model_id)
    print(
        "\nVolume: deferral-llm-extract-v1 — artifacts under "
        f"/extract/{ex}/<dataset>/hidden_states.pt\n"
        "Example:\n"
        f"  modal volume get deferral-llm-extract-v1 "
        f"/extract/{ex}/truthfulqa/hidden_states.pt ./truthfulqa.pt"
    )
