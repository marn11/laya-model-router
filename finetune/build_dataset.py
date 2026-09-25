"""Merge the hand-labeled 30 + 4 synthetic batches into a train/test split.

Run from the repo root: uv run python finetune/build_dataset.py

Writes finetune/data/train.jsonl and finetune/data/test.jsonl, each line
{"text": ..., "tier": "trivial"|"moderate"|"hard"}. Held-out test cases are
distinct from both the training pool AND from evals/dataset.jsonl (the
existing 30-example benchmark), so evals/dataset.jsonl keeps working as the
fixed before/after comparison across v1 (zero-shot) and v2 (fine-tuned).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "synthetic_raw"
DATA_DIR = ROOT / "data"
EVAL_DATASET = ROOT.parent / "evals" / "dataset.jsonl"

TIERS = {"trivial", "moderate", "hard"}
TEST_FRACTION = 0.12
SEED = 20260925


def load_batches() -> list[dict]:
    items = []
    for f in sorted(RAW_DIR.glob("batch_*.json")):
        batch = json.loads(f.read_text())
        for row in batch:
            assert row["tier"] in TIERS, f"bad tier {row['tier']!r} in {f.name}"
            items.append({"text": row["text"].strip(), "tier": row["tier"]})
    return items


def load_hand_labeled() -> list[dict]:
    items = []
    with EVAL_DATASET.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                items.append({"text": row["text"].strip(), "tier": row["ground_truth"]})
    return items


def dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    dropped = []
    for it in items:
        key = it["text"].lower()
        if key in seen:
            dropped.append(it["text"])
            continue
        seen.add(key)
        out.append(it)
    if dropped:
        print(f"Dropped {len(dropped)} exact-duplicate texts:")
        for t in dropped:
            print(f"  - {t}")
    return out


def main() -> None:
    # evals/dataset.jsonl (the 30 hand-labeled examples) is used ONLY to filter
    # out accidental overlap -- it must never end up in train or test here, or
    # the fine-tuned model would have memorized the exact benchmark we use to
    # measure whether fine-tuning helped.
    hand_labeled = load_hand_labeled()
    synthetic = load_batches()

    hand_labeled_texts = {it["text"].lower() for it in hand_labeled}
    synthetic = [it for it in synthetic if it["text"].lower() not in hand_labeled_texts]

    pool = dedupe(synthetic)

    by_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    for it in pool:
        by_tier[it["tier"]].append(it)

    rng = random.Random(SEED)
    train, test = [], []
    for tier, rows in by_tier.items():
        rng.shuffle(rows)
        n_test = max(1, round(len(rows) * TEST_FRACTION))
        test.extend(rows[:n_test])
        train.extend(rows[n_test:])
        print(f"{tier}: {len(rows)} total -> {len(rows) - n_test} train / {n_test} test")

    rng.shuffle(train)
    rng.shuffle(test)

    DATA_DIR.mkdir(exist_ok=True)
    with (DATA_DIR / "train.jsonl").open("w") as f:
        for it in train:
            f.write(json.dumps(it) + "\n")
    with (DATA_DIR / "test.jsonl").open("w") as f:
        for it in test:
            f.write(json.dumps(it) + "\n")

    print(f"\nWrote {len(train)} train cases -> finetune/data/train.jsonl")
    print(f"Wrote {len(test)} test cases -> finetune/data/test.jsonl")
    print(f"(evals/dataset.jsonl's {len(hand_labeled)} hand-labeled examples were excluded entirely -- they stay the untouched fixed v1-vs-v2 benchmark)")


if __name__ == "__main__":
    main()
