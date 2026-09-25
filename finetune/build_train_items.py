"""Tokenize finetune/data/train.jsonl into train_items.pt, ready for Kaggle.

Runs entirely on CPU -- no GPU needed for this step. Reuses laya.common's
own build_sequence/render_options/QTYPES, the same functions the official
fine-tuning notebook's data-prep cell uses, so the tokenized format matches
exactly what train_ddp.py (unmodified) expects to load.

The test split (finetune/data/test.jsonl) is NOT pre-tokenized here -- eval
calls agent.predict(state, questions) directly on raw text, so it doesn't
need this step.

Run from the repo root: uv run python finetune/build_train_items.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options

MODEL_ID = "convaiinnovations/laya"
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# Must match src/laya_model_router/router.py's DIFFICULTY_CRITERIA exactly --
# the fine-tuned checkpoint gets evaluated against that same question.
DIFFICULTY_CRITERIA = [
    "trivial, no reasoning needed",
    "some judgment required",
    "complex reasoning or subtle bugs",
]
TIER_TO_INDEX = {"trivial": 0, "moderate": 1, "hard": 2}
QUESTION = {
    "instructions": "How difficult is this coding task?",
    "criteria": DIFFICULTY_CRITERIA,
}

# No calibrated per-item probability exists (labels came from an LLM judge,
# not a teacher with real confidence scores), so every item gets the same
# fixed soft target instead of a degenerate one-hot: 0.85 on the assigned
# tier, 0.075 on each of the other two.
CONFIDENT = 0.85
OTHER = (1.0 - CONFIDENT) / 2


def gold_probabilities(tier: str) -> dict[str, float]:
    idx = TIER_TO_INDEX[tier]
    return {str(i): (CONFIDENT if i == idx else OTHER) for i in range(3)}


def build_training_item(tok, cfg, state: str, gold_probs: dict[str, float]):
    """Same logic as the official notebook's cell 6, specialized to our one score question."""
    n_levels = len(QUESTION["criteria"])
    target = [gold_probs.get(str(i), 0.0) for i in range(n_levels)]
    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": "score", "crit": QUESTION["criteria"]}))

    seq, markers = build_sequence(
        tok,
        state,
        {"t": "score", "ins": QUESTION["instructions"], "crit": QUESTION["criteria"]},
        cfg["max_len"],
        cfg["head_max_len"],
    )
    if len(markers) != k:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "qtype": QTYPES["score"],
        "target": target,
        "label": label,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    rows = load_jsonl(DATA_DIR / "train.jsonl")

    print(f"Fetching tokenizer/config for {MODEL_ID}...")
    model_dir = snapshot_download(MODEL_ID)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    items = []
    skipped = 0
    for row in rows:
        it = build_training_item(tok, cfg, row["text"], gold_probabilities(row["tier"]))
        if it is None:
            skipped += 1
            continue
        items.append(it)

    print(f"Built {len(items)} items ({skipped} skipped due to marker-count mismatch)")
    out_path = DATA_DIR / "train_items.pt"
    torch.save(items, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
