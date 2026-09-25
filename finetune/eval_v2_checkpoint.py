"""Evaluate the fine-tuned checkpoint on the ORIGINAL evals/dataset.jsonl.

This is the fixed 30-example benchmark that was never touched by anything in
finetune/ -- running the v2 checkpoint against it (instead of only the new
57-example synthetic test set) is what makes the v1-vs-v2 comparison
apples-to-apples with the zero-shot numbers already in evals/results.md.

Run from the repo root: uv run python finetune/eval_v2_checkpoint.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import laya

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "v2"
EVAL_DATASET = ROOT.parent / "evals" / "dataset.jsonl"

DIFFICULTY_CRITERIA = [
    "trivial, no reasoning needed",
    "some judgment required",
    "complex reasoning or subtle bugs",
]
QUESTIONS = {
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is this coding task?",
        "criteria": DIFFICULTY_CRITERIA,
    }
}
TIER_NAMES = ("trivial", "moderate", "hard")


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    rows = load_dataset(EVAL_DATASET)
    agent = laya.Agent(str(CHECKPOINT_DIR), device="cpu")

    results = []
    latencies = []
    for row in rows:
        start = time.perf_counter()
        res = agent.predict(row["text"], QUESTIONS)
        latencies.append(time.perf_counter() - start)
        ans = res["answers"]["difficulty"]
        probs = [ans["probabilities"].get(str(i), 0.0) for i in range(3)]
        pred_idx = max(range(3), key=lambda i: probs[i])
        results.append(
            {
                "text": row["text"],
                "ground_truth": row["ground_truth"],
                "predicted": TIER_NAMES[pred_idx],
                "score": ans.get("score"),
                "confidence": ans.get("confidence"),
            }
        )

    correct = sum(1 for r in results if r["predicted"] == r["ground_truth"])
    accuracy = correct / len(results)

    matrix = {(gt, pred): 0 for gt in TIER_NAMES for pred in TIER_NAMES}
    for r in results:
        matrix[(r["ground_truth"], r["predicted"])] += 1

    print(f"v2 (fine-tuned) accuracy on evals/dataset.jsonl ({len(results)} examples): {accuracy:.1%}")
    print(f"Mean latency: {sum(latencies) / len(latencies) * 1000:.1f} ms")
    print("\nConfusion matrix (rows=ground truth, cols=predicted):")
    header = "            " + "  ".join(f"{t:>10}" for t in TIER_NAMES)
    print(header)
    for gt in TIER_NAMES:
        print(f"{gt:>10}  " + "  ".join(f"{matrix[(gt, pred)]:>10}" for pred in TIER_NAMES))

    for r in results:
        if r["predicted"] != r["ground_truth"]:
            print(f"\nMISS: [{r['ground_truth']} -> {r['predicted']}] {r['text']}")

    report = {
        "n_test": len(results),
        "accuracy": accuracy,
        "mean_latency_ms": sum(latencies) / len(latencies) * 1000,
        "confusion_matrix": {f"{gt}->{pred}": v for (gt, pred), v in matrix.items()},
        "predictions": results,
    }
    out_path = ROOT / "v2_eval_on_original_30.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
