# laya-model-router

An experiment in building a self-hosted, open-source request router that picks which LLM answers a request based on how hard the request actually is — using [Laya](https://huggingface.co/convaiinnovations/laya) as the difficulty classifier.

## Status

A working "plain library" MVP exists: `laya_model_router.ModelRouter.pick_model(text)` classifies difficulty with Laya and returns a model name. No server, no LiteLLM integration yet — see "Library usage" below and the open questions at the bottom.

## Background

### What is Laya?

Laya (from Convai Innovations) is a **System 1 decision model** — a small, non-autoregressive classifier, not a generative LLM. Instead of producing text, it takes a `state` (the text to evaluate) plus one or more typed `questions` and returns a calibrated structured decision in a single forward pass (~33ms):

- **Choice** — pick from a fixed set of options, with a probability per option
- **Score** — rate on ordered levels (e.g. difficulty: trivial → simple → moderate → hard), with a weighted score and per-level probabilities
- **Yes/no** — a single calibrated probability

There are three published checkpoints: an English-only root model (421M params, ModernBERT-large backbone + a 2-layer decision head), a multilingual model (322M params, mmBERT-base, 100+ languages), and a typed-decisions fine-tuned variant (421M params). All were trained with **RLCD** (Reinforcement Learning for Calibrated Decisions) using proper scoring rules, so the output probabilities are meant to reflect real accuracy rather than just being confident-sounding numbers.

Because it never generates tokens, it can't hallucinate a decision the way a prompted LLM classifier can — it can only be miscalibrated or wrong, not made up out of thin air.

### Relationship to Jev

[Jev](https://openrouter.ai/blog/insights/what-is-jev/) is TypeSafe AI's proprietary "System One model," announced 2026-09-15 alongside $40M in seed funding. It's the same category of thing as Laya — a non-generative decision model returning typed answers (Choice / Score / Noul) with calibrated probabilities instead of prose — and it's already being used for exactly this purpose: LiteLLM added Jev as a `classifier_type` inside its complexity router, and it's the kind of model that powers Cursor's "Compass" complexity predictor for Cursor Router's Auto mode.

Laya is the open-weights alternative to Jev, built by a different team, and explicitly benchmarked against it — Convai Innovations claims 0.766 vs. Jev's 0.727 accuracy on their eval set, running ~7.8x faster. Where Jev is API-only ($0.042/M input tokens, no published weights), Laya can be self-hosted, inspected, and fine-tuned.

The tradeoff called out on Laya's own model card: zero-shot accuracy on specialized tasks is modest, and high-cardinality label spaces (50+ options) need extra configuration. A small, well-defined label space (e.g. 3-4 difficulty tiers) is a good fit; anything more elaborate will likely need fine-tuning data.

## What this repo will do

Build a standalone model router: something that sits **in front of** LLM API calls, uses Laya to score each incoming request's difficulty, and dispatches the request to a backend model chosen for that difficulty tier (e.g. a fast/cheap model for simple requests, a stronger/more expensive model for hard ones) — the same shape of system as Cursor Router or LiteLLM's Jev-based complexity router, but self-hosted and built on an open classifier.

This is deliberately a separate repo from `jev-portal-spotify`. That repo is a fork of Spotify's Portal plugin for coding agents (Claude Code / Codex / Cursor); its `shunt` plugin delegates specific sub-tasks (bulk file reads, boilerplate generation) to a fixed, pre-configured worker model via Portal's AiKA feature, gated by a static line-count threshold — there's no per-request difficulty classification, and it only ever governs a plugin's own tool calls inside an existing Claude Code session, not which model answers a request in the first place. What we're building here is unrelated to that Portal/Claude Code plugin architecture: a router that decides, before any model is called, which model gets the call — which is a different problem and doesn't belong bolted onto a Spotify fork.

## Library usage

```bash
uv sync
uv run python -m laya_model_router "find the race condition in this concurrent queue implementation"
# {"tier": "hard", "model": "anthropic/claude-sonnet-5", "score": 1.573, "confidence": 0.272}
```

```python
from laya_model_router import ModelRouter

router = ModelRouter()
result = router.pick_model("write a test for UserService.create following the pattern in OrderServiceTest")
print(result.tier, result.model, result.score, result.confidence)
```

`ModelRouter(model_tiers={...})` overrides the default tier→model mapping (`DEFAULT_MODEL_TIERS` in `router.py`) if you want different backend models per tier.

### What the first real test run found

Laya's `score` question type is very sensitive to how the `criteria` are worded — this wasn't obvious from the docs and needed a live test to surface:

| Criteria style | Result on 3 spot-check examples |
| --- | --- |
| Long paragraphs with embedded examples | **Backwards.** A trivial rename scored *higher* than a pattern-following test-writing task. |
| Single words (`"trivial"`, `"moderate"`, `"hard"`) | Correct ordering, but almost no separation (scores clustered in a ~0.18 range). |
| Short phrases, no examples (`"trivial, no reasoning needed"`, ...) | Correct ordering with usable separation across 5 examples (range ~0.9–1.6). Used in `router.py`. |

With the short-phrase criteria, tier thresholds (`_TIER_THRESHOLDS` in `router.py`) were hand-picked from that 5-example range — trivial/trivial/moderate/moderate(borderline)/hard all landed where expected, but this is **not a calibrated boundary**, just an MVP starting point. Confidence values were consistently low (0.02–0.27) even when the score ordering was correct, consistent with Laya's own model-card caveat that zero-shot accuracy on specialized tasks is modest. A real eval set (labeled examples, ideally with disagreement resolved by more than one person's judgment) would be needed before trusting this for anything beyond experimentation.

## Open questions / next steps

- Build a real eval set to replace the 5-example hand-picked thresholds
- Consider whether fine-tuning (mentioned as an option on Laya's model card) is worth it before/instead of more prompt/criteria tuning
- Decide the entry point for the next step: a LiteLLM `async_pre_call_hook` integration (reuses `pick_model` from this library) vs. a standalone proxy
- Decide how Laya gets served once this moves beyond a single local process (still in-process/local for now)
