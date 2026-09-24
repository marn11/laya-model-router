"""LLM-as-classifier baseline, for comparison against Laya.

Uses the real Anthropic API (via `anthropic.Anthropic()`, which resolves
credentials from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth
login` profile). No credential source means this raises on the first call
-- run_eval.py is responsible for catching that and reporting it clearly,
not this module.

Pricing below is USD per million tokens, current as of when this was
written (via the claude-api skill, not guessed) -- re-check before trusting
cost numbers if it's been a while.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic

TIER_NAMES = ("trivial", "moderate", "hard")

_PROMPT = """Classify the difficulty of this coding task. Respond with exactly one word: trivial, moderate, or hard.

trivial: boilerplate or pattern-matching with one obvious answer, no real reasoning needed
moderate: requires understanding local context and a small judgment call
hard: requires multi-step reasoning, weighing tradeoffs, or spotting subtle bugs

Task: {text}

Answer with exactly one word."""

# USD per million tokens.
PRICING_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}

DEFAULT_MODEL = "claude-haiku-4-5"


@dataclass
class LLMJudgment:
    tier: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


def llm_classify(text: str, model: str = DEFAULT_MODEL, client: anthropic.Anthropic | None = None) -> LLMJudgment:
    client = client or anthropic.Anthropic()

    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
    )
    latency_s = time.perf_counter() - start

    raw = "".join(block.text for block in response.content if block.type == "text").strip().lower()
    tier = next((t for t in TIER_NAMES if t in raw), "moderate")

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    pricing = PRICING_PER_MTOK.get(model, PRICING_PER_MTOK[DEFAULT_MODEL])
    cost_usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    return LLMJudgment(
        tier=tier,
        latency_s=latency_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
