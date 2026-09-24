"""Difficulty-based model routing backed by Laya.

Laya's "score" question type returns a weighted value across the ordered
`criteria` list. Two things were NOT obvious until tested live against the
English-root checkpoint (see README "Library usage" section for the numbers):

- Criteria wording matters a lot. Long, paragraph-style criteria with
  embedded examples produced a backwards ordering (a trivial rename scored
  *higher* than a pattern-following test-writing task). Single-word criteria
  ("trivial"/"moderate"/"hard") gave the right ordering but almost no
  separation. Short descriptive phrases (a few words, no examples) gave a
  correctly-ordered, usably-separated score across 5 spot-check examples.
- The raw score does not land near the criterion index (0/1/2) the way you'd
  guess. Empirically it clustered between ~0.9 and ~1.6 across
  trivial-to-hard examples, so tier boundaries below are picked from that
  observed range, not from rounding to the nearest index. This is based on
  5 examples, not a real eval set — treat the thresholds as a provisional
  starting point, not a calibrated boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from laya import Router as _LayaRouter

TIER_NAMES = ("trivial", "moderate", "hard")

# Short, example-free phrases — verbose criteria with embedded examples
# produced a backwards score ordering in testing (see module docstring).
DIFFICULTY_CRITERIA = [
    "trivial, no reasoning needed",
    "some judgment required",
    "complex reasoning or subtle bugs",
]

# Provisional, hand-picked from a 5-example spot check (range was ~0.9-1.6).
# Needs a real eval set before trusting this for anything but an MVP.
_TIER_THRESHOLDS = (1.05, 1.4)  # score < 1.05 -> trivial, < 1.4 -> moderate, else hard

DEFAULT_MODEL_TIERS = {
    "trivial": "anthropic/claude-haiku-4-5",
    "moderate": "anthropic/claude-sonnet-5",
    "hard": "anthropic/claude-opus-5",
}

_QUESTIONS = {
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is this coding task?",
        "criteria": DIFFICULTY_CRITERIA,
    }
}


@dataclass
class RoutingResult:
    tier: str
    model: str
    score: float
    confidence: float
    raw: dict[str, Any]


class ModelRouter:
    """Classifies a task's difficulty with Laya and picks a model tier for it."""

    def __init__(
        self,
        model_tiers: dict[str, str] | None = None,
        preload: bool = False,
    ) -> None:
        self._laya = _LayaRouter(preload=preload)
        self.model_tiers = model_tiers or dict(DEFAULT_MODEL_TIERS)

    def pick_model(self, text: str) -> RoutingResult:
        result = self._laya.predict(text, _QUESTIONS)
        answer = result["answers"]["difficulty"]
        score = answer["score"]
        confidence = answer.get("confidence", 0.0)
        tier = _score_to_tier(score)
        return RoutingResult(
            tier=tier,
            model=self.model_tiers[tier],
            score=score,
            confidence=confidence,
            raw=result,
        )


def _score_to_tier(score: float) -> str:
    low, high = _TIER_THRESHOLDS
    if score < low:
        return TIER_NAMES[0]
    if score < high:
        return TIER_NAMES[1]
    return TIER_NAMES[2]
