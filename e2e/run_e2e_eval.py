"""End-to-end test: does the model Laya v2 routes a task to actually hold up?

Not just "is the difficulty classification right" (evals/ and finetune/ already
answer that) -- this asks whether downgrading to a cheaper model on
trivial/moderate tasks actually costs quality, using real completions and a
blind pairwise grading call against an always-Opus baseline.

max_tokens is set generously (4096) because Opus 5 and Sonnet 5 both run
adaptive thinking by default -- thinking tokens count against max_tokens, so
a too-small budget can be silently consumed by thinking before any visible
text is produced (stop_reason="max_tokens", empty text). Every call's
stop_reason is recorded explicitly for this reason: a refusal
(stop_reason="refusal", Anthropic's real-time safety classifier declining
the request) also produces empty text, but means something completely
different from an actual bad completion, and both need to be distinguished
from real quality losses before treating anything as a finding.

Needs ANTHROPIC_API_KEY set. Run from the repo root:
    set -a; source .env; set +a
    uv run python e2e/run_e2e_eval.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import laya

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT.parent / "finetune" / "checkpoints" / "v2"
TASKS_PATH = ROOT / "tasks.jsonl"
REPORT_PATH = ROOT / "report.json"

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

DEFAULT_MODEL_TIERS = {
    "trivial": "claude-haiku-4-5",
    "moderate": "claude-sonnet-5",
    "hard": "claude-opus-5",
}
BASELINE_MODEL = "claude-opus-5"
COMPLETION_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 500

# USD per million tokens.
PRICING_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
}

JUDGE_MODEL = "claude-opus-5"
JUDGE_PROMPT = """You are evaluating two AI-generated solutions to the same coding task. \
Judge correctness first, then quality and clarity. If they are practically \
equivalent, say TIE -- do not force a preference.

TASK:
{task}

COMPLETION A:
{a}

COMPLETION B:
{b}

Respond in exactly this format, nothing else:
VERDICT: A|B|TIE
REASON: <one or two sentences>"""


@dataclass
class Call:
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    refusal_category: str | None = None

    @property
    def cost_usd(self) -> float:
        p = PRICING_PER_MTOK[self.model]
        return (self.input_tokens * p["input"] + self.output_tokens * p["output"]) / 1_000_000

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


@dataclass
class TaskResult:
    id: str
    tier_ground_truth: str
    tier_predicted: str
    routed_model: str
    routed_completion: str
    baseline_completion: str
    routed_call: Call
    baseline_call: Call | None
    same_model: bool
    verdict: str | None
    reason: str | None
    judge_call: Call | None = None

    @property
    def calls(self) -> list[Call]:
        return [c for c in (self.routed_call, self.baseline_call, self.judge_call) if c is not None]


def load_tasks() -> list[dict]:
    rows = []
    with TASKS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify(agent, text: str) -> str:
    res = agent.predict(text, QUESTIONS)
    ans = res["answers"]["difficulty"]
    probs = [ans["probabilities"].get(str(i), 0.0) for i in range(3)]
    return TIER_NAMES[max(range(3), key=lambda i: probs[i])]


def complete(client: anthropic.Anthropic, model: str, prompt: str) -> tuple[str, Call]:
    resp = client.messages.create(
        model=model,
        max_tokens=COMPLETION_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    category = resp.stop_details.category if resp.stop_reason == "refusal" and resp.stop_details else None
    call = Call(
        model=model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
        refusal_category=category,
    )
    return text, call


def grade(client: anthropic.Anthropic, task: str, routed: str, baseline: str) -> tuple[str, str, Call]:
    # Randomize A/B assignment per task so the judge can't learn a position bias,
    # and so it never learns "A is always the cheaper model".
    routed_is_a = random.random() < 0.5
    a, b = (routed, baseline) if routed_is_a else (baseline, routed)
    prompt = JUDGE_PROMPT.format(task=task, a=a, b=b)
    resp = client.messages.create(model=JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS, messages=[{"role": "user", "content": prompt}])
    text = "".join(blk.text for blk in resp.content if blk.type == "text")
    call = Call(model=JUDGE_MODEL, input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens, stop_reason=resp.stop_reason)

    verdict_letter = "TIE"
    reason_lines: list[str] = []
    in_reason = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            verdict_letter = stripped.split(":", 1)[1].strip().upper()
            in_reason = False
        elif stripped.upper().startswith("REASON:"):
            reason_lines.append(stripped.split(":", 1)[1].strip())
            in_reason = True
        elif in_reason and stripped:
            reason_lines.append(stripped)
    reason = " ".join(reason_lines)

    if verdict_letter == "TIE":
        verdict = "TIE"
    elif (verdict_letter == "A") == routed_is_a:
        verdict = "routed"
    else:
        verdict = "baseline"
    return verdict, reason, call


def main() -> None:
    random.seed(20260930)
    tasks = load_tasks()
    agent = laya.Agent(str(CHECKPOINT_DIR), device="cpu")
    client = anthropic.Anthropic()

    results: list[TaskResult] = []
    for row in tasks:
        tier = classify(agent, row["prompt"])
        model = DEFAULT_MODEL_TIERS[tier]

        routed_text, routed_call = complete(client, model, row["prompt"])

        if model == BASELINE_MODEL:
            baseline_text, baseline_call = routed_text, None
            same_model = True
            verdict, reason, judge_call = None, None, None
        else:
            baseline_text, baseline_call = complete(client, BASELINE_MODEL, row["prompt"])
            same_model = False

            if routed_call.refused and baseline_call.refused:
                verdict, reason, judge_call = "both_refused", None, None
            elif baseline_call.refused:
                verdict, reason, judge_call = "opus_refused", None, None
            elif routed_call.refused:
                verdict, reason, judge_call = "routed_refused", None, None
            else:
                verdict, reason, judge_call = grade(client, row["prompt"], routed_text, baseline_text)

        results.append(
            TaskResult(
                id=row["id"],
                tier_ground_truth=row["tier_ground_truth"],
                tier_predicted=tier,
                routed_model=model,
                routed_completion=routed_text,
                baseline_completion=baseline_text,
                routed_call=routed_call,
                baseline_call=baseline_call,
                same_model=same_model,
                verdict=verdict,
                reason=reason,
                judge_call=judge_call,
            )
        )
        flag = " [ROUTED REFUSED]" if routed_call.refused else ""
        print(f"{row['id']:12s} gt={row['tier_ground_truth']:8s} pred={tier:8s} model={model:18s} verdict={verdict}{flag}")

    tier_correct = sum(1 for r in results if r.tier_predicted == r.tier_ground_truth)
    routed_cost = sum(r.routed_call.cost_usd for r in results)
    always_opus_cost = sum((r.routed_call if r.same_model else r.baseline_call).cost_usd for r in results)
    grading_cost = sum(r.judge_call.cost_usd for r in results if r.judge_call is not None)
    total_cost_this_run = sum(c.cost_usd for r in results for c in r.calls)

    verdicts = Counter(r.verdict for r in results if r.verdict is not None)
    refusal_tasks = [r.id for r in results if (r.baseline_call and r.baseline_call.refused) or r.routed_call.refused]

    print(f"\nTier accuracy on these {len(results)} tasks: {tier_correct}/{len(results)}")
    print(f"Verdicts: {dict(verdicts)}")
    print(f"Tasks with at least one refusal: {refusal_tasks}")
    print(f"\nCost if routed per DEFAULT_MODEL_TIERS: ${routed_cost:.4f}")
    print(f"Cost if every task went to Opus instead: ${always_opus_cost:.4f}")
    print(f"Savings from routing: ${always_opus_cost - routed_cost:.4f} ({(1 - routed_cost / always_opus_cost) * 100:.1f}%)")
    print(f"(Grading calls cost ${grading_cost:.4f} extra, on top of both -- that's eval overhead, not part of either number above)")
    print(f"Total real spend this run: ${total_cost_this_run:.4f}")

    def call_dict(c: Call | None) -> dict | None:
        if c is None:
            return None
        return {
            "model": c.model,
            "input_tokens": c.input_tokens,
            "output_tokens": c.output_tokens,
            "stop_reason": c.stop_reason,
            "refusal_category": c.refusal_category,
            "cost_usd": c.cost_usd,
        }

    report = {
        "tier_accuracy": tier_correct / len(results),
        "verdict_counts": dict(verdicts),
        "refusal_tasks": refusal_tasks,
        "routed_cost_usd": routed_cost,
        "always_opus_cost_usd": always_opus_cost,
        "savings_usd": always_opus_cost - routed_cost,
        "savings_pct": (1 - routed_cost / always_opus_cost) * 100,
        "grading_cost_usd": grading_cost,
        "total_spend_usd": total_cost_this_run,
        "results": [
            {
                "id": r.id,
                "tier_ground_truth": r.tier_ground_truth,
                "tier_predicted": r.tier_predicted,
                "routed_model": r.routed_model,
                "same_model": r.same_model,
                "verdict": r.verdict,
                "reason": r.reason,
                "routed_completion": r.routed_completion,
                "baseline_completion": r.baseline_completion,
                "routed_call": call_dict(r.routed_call),
                "baseline_call": call_dict(r.baseline_call),
                "judge_call": call_dict(r.judge_call),
            }
            for r in results
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
