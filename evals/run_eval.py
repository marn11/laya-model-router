"""Classifier-level benchmark: Laya vs. ground truth, optionally vs. an LLM judge too.

    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --with-llm --llm-model claude-haiku-4-5

Laya always runs (no credentials needed). The LLM half only runs with
--with-llm, and requires Anthropic credentials (ANTHROPIC_API_KEY,
ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile) -- if the first call
fails, the LLM half is skipped for the rest of the run with a clear message
rather than erroring out or silently producing empty numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from laya_model_router import ModelRouter, TIER_NAMES

from llm_judge import DEFAULT_MODEL as DEFAULT_LLM_MODEL
from llm_judge import llm_classify

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def accuracy(rows: list[dict], pred_key: str) -> float:
    scored = [r for r in rows if pred_key in r]
    if not scored:
        return float("nan")
    correct = sum(1 for r in scored if r[pred_key] == r["ground_truth"])
    return correct / len(scored)


def confusion_matrix(rows: list[dict], pred_key: str) -> dict[tuple[str, str], int]:
    matrix = {(gt, pred): 0 for gt in TIER_NAMES for pred in TIER_NAMES}
    for r in rows:
        if pred_key in r:
            matrix[(r["ground_truth"], r[pred_key])] += 1
    return matrix


def format_confusion_md(matrix: dict[tuple[str, str], int]) -> str:
    header = "| ground truth \\ predicted | " + " | ".join(TIER_NAMES) + " |"
    sep = "| --- | " + " | ".join("---" for _ in TIER_NAMES) + " |"
    lines = [header, sep]
    for gt in TIER_NAMES:
        row = [str(matrix[(gt, pred)]) for pred in TIER_NAMES]
        lines.append(f"| **{gt}** | " + " | ".join(row) + " |")
    return "\n".join(lines)


def agreement(rows: list[dict]) -> float | None:
    scored = [r for r in rows if "llm_tier" in r]
    if not scored:
        return None
    matches = sum(1 for r in scored if r["laya_tier"] == r["llm_tier"])
    return matches / len(scored)


def run(with_llm: bool, llm_model: str) -> None:
    dataset = load_dataset(DATASET_PATH)
    router = ModelRouter(preload=True)

    llm_broken = False
    for row in dataset:
        start = time.perf_counter()
        laya_result = router.pick_model(row["text"])
        row["laya_tier"] = laya_result.tier
        row["laya_score"] = laya_result.score
        row["laya_confidence"] = laya_result.confidence
        row["laya_latency_s"] = time.perf_counter() - start

        if with_llm and not llm_broken:
            try:
                judgment = llm_classify(row["text"], model=llm_model)
            except Exception as exc:  # noqa: BLE001 -- any auth/network failure disables the LLM half
                print(f"LLM judge failed ({exc!r}) -- skipping the LLM half for the rest of this run.")
                llm_broken = True
            else:
                row["llm_tier"] = judgment.tier
                row["llm_latency_s"] = judgment.latency_s
                row["llm_input_tokens"] = judgment.input_tokens
                row["llm_output_tokens"] = judgment.output_tokens
                row["llm_cost_usd"] = judgment.cost_usd

    report = build_report(dataset, with_llm=with_llm and not llm_broken, llm_model=llm_model)
    print(report)
    RESULTS_PATH.write_text(report)
    print(f"\nWrote {RESULTS_PATH}")


def build_report(rows: list[dict], with_llm: bool, llm_model: str) -> str:
    lines = ["# Eval results", ""]

    lines.append(f"Laya accuracy: **{accuracy(rows, 'laya_tier'):.0%}** ({len(rows)} examples)")
    lines.append("")
    lines.append("Laya confusion matrix:")
    lines.append("")
    lines.append(format_confusion_md(confusion_matrix(rows, "laya_tier")))
    lines.append("")
    laya_latencies = [r["laya_latency_s"] for r in rows]
    lines.append(f"Laya mean latency: {statistics.mean(laya_latencies) * 1000:.1f} ms")
    lines.append("Laya cost: $0 marginal (self-hosted, local inference)")
    lines.append("")

    if with_llm:
        llm_rows = [r for r in rows if "llm_tier" in r]
        lines.append(f"LLM judge ({llm_model}) accuracy: **{accuracy(rows, 'llm_tier'):.0%}** ({len(llm_rows)} examples)")
        lines.append("")
        lines.append("LLM confusion matrix:")
        lines.append("")
        lines.append(format_confusion_md(confusion_matrix(rows, "llm_tier")))
        lines.append("")
        llm_latencies = [r["llm_latency_s"] for r in llm_rows]
        total_cost = sum(r["llm_cost_usd"] for r in llm_rows)
        lines.append(f"LLM judge mean latency: {statistics.mean(llm_latencies) * 1000:.1f} ms")
        lines.append(f"LLM judge total cost: ${total_cost:.5f} for {len(llm_rows)} classifications (${total_cost / len(llm_rows):.6f} each)")
        lines.append("")
        agree = agreement(rows)
        lines.append(f"Laya/LLM agreement: **{agree:.0%}**" if agree is not None else "Laya/LLM agreement: n/a")
    else:
        lines.append("(LLM judge not run -- pass --with-llm with Anthropic credentials set to include it.)")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true", help="Also run the LLM judge baseline.")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Anthropic model ID to use as the LLM judge.")
    args = parser.parse_args()
    run(with_llm=args.with_llm, llm_model=args.llm_model)


if __name__ == "__main__":
    main()
