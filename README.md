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

Spotify's Portal `shunt` plugin also delegates work to a worker model, but the decision of whether to delegate is a static line-count check, and the worker model is one fixed model picked ahead of time — there's no per-request difficulty classification. What we're building here is closer to [Cursor Router](https://cursor.com/blog/router): classify each request's difficulty first, then dynamically pick which model answers it.

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

## Evals

`evals/` benchmarks the classifier itself: does Laya's predicted tier match a ground-truth label, how does that compare to just prompting an LLM to classify the same text, and what does each approach cost.

```bash
uv run python evals/run_eval.py                                          # Laya only, no credentials needed
uv run python evals/run_eval.py --with-llm --llm-model claude-haiku-4-5  # also runs the LLM judge baseline
```

- `dataset.jsonl` — 30 hand-curated coding-task descriptions, 10 per tier, each with a one-line rationale. Single-reviewer ground truth, not multi-rater validated — a first real signal, not a calibrated benchmark.
- `llm_judge.py` — prompts a real Anthropic model to output one of trivial/moderate/hard on the same text, and measures its actual token cost.
- `run_eval.py` — runs both, reports accuracy vs. ground truth, a confusion matrix, latency, cost, and Laya/LLM agreement. Writes `results.md`. Also accepts `--llm-judgments some.json` (a `{text: tier}` map) as a stand-in for the live API call — see below for why that exists and its limits.

### Latest results

**Laya accuracy: 63% (30 examples).** Mean latency ~53-57ms across runs, $0 marginal cost (local inference). Confusion matrix:

| ground truth \ predicted | trivial | moderate | hard |
| --- | --- | --- | --- |
| **trivial** | 9 | 1 | 0 |
| **moderate** | 4 | 6 | 0 |
| **hard** | 0 | 6 | 4 |

The errors aren't random: Laya is weakest on **hard** tasks (only 4/10 correct — the other 6 all get called "moderate", never "trivial"), and moderate tasks get pulled down toward "trivial" more than up toward "hard". So it never mistakes a hard task for a trivial one (the worst-case error for a routing system), but it under-calls difficulty on the harder end of the spectrum — 63% zero-shot accuracy is real signal, not noise, but not yet something to route production traffic on unsupervised. Matches Laya's own model-card caveat that zero-shot accuracy on specialized tasks is modest.

**LLM judge, attempt 1 — Claude self-judged (circular, not trustworthy).** No Anthropic/OpenAI API key was available (the OpenAI key on hand had expired), so `evals/self_judged_claude.json` holds Claude's own answers to the exact `llm_judge.py` prompt, given directly in conversation rather than through a metered call. That produced 100% "accuracy" and 63% Laya/LLM agreement — but the accuracy number means nothing here, since Claude wrote this dataset's ground-truth labels *and* was the judge being compared. Kept on file for transparency, superseded by attempt 2 below.

**LLM judge, attempt 2 — GPT-5.6 Terra via Codex (genuinely independent).** No metered OpenAI key was available either, but Codex was — authenticated via ChatGPT subscription login, not a per-token API key. Ran the same rubric through Codex with `--model gpt-5.6-terra`, batching all 30 tasks in one call (`evals/gpt_5_6_terra_judged.json`), and Terra had never seen the ground-truth labels. Result: **100% accuracy, 30/30**, and so **63% Laya/LLM agreement** (identical to Laya's ground-truth accuracy, since Terra's judgments equal ground truth exactly). This is the trustworthy accuracy number — a different model family, with no access to the answer key, still nailed every example. That also means this dataset's 30 examples are unambiguous enough that two different frontier models (Claude and Terra) both get 100% — the eval doesn't yet stress-test the *judge*, only Laya, which is exactly what we want here but is worth keeping in mind before reading too much into "100%".

**Cost.** Neither run above was a metered per-classification API call (Claude answered in-conversation; Terra ran through a ChatGPT-subscription-authenticated Codex session, batched as one call for all 30 tasks), so there's no real billed number for either. Two estimates from actual measured text length, not guesses:
- **The one batched Terra call I actually made:** ~762 input tokens, ~624 output tokens (chars/4 heuristic on the real prompt and response text) × published Terra pricing ($2.00/$12.00 per M tokens) ≈ **$0.009 for all 30 classifications in one call.** This excludes Codex's own system prompt/tool-call scaffolding and any hidden reasoning tokens Terra may have used internally (it supports extended reasoning) — a real metered call could run higher than this visible-text estimate.
- **Per-classification, matching `llm_judge.py`'s actual one-call-per-task design** (~90 input tokens, ~5 output tokens per call): **~$0.0001/call on Haiku 4.5, ~$0.0002/call on Sonnet 5, ~$0.00024/call on Terra** — for 30 classifications, all three are a fraction of a cent.

Laya's cost is genuinely $0 per call (local inference, no metering); every LLM judge's cost floor is not zero, but at this prompt size all of them round to "negligible" regardless of provider. The real cost question isn't "which is cheaper at 30 calls" — it's what this looks like at production volume, and (per attempt 2 above) whether the judge is actually right.

**Next real step:** get a working, metered API key (Anthropic or OpenAI) and run `--with-llm` for a genuinely priced comparison — that replaces the cost estimate above with a real number. Accuracy is already independently answered by the Terra run.

## v2: fine-tuning (done)

Zero-shot Laya's 63% wasn't good enough to route production traffic on unsupervised (see above). `finetune/` builds a real fine-tune, following Laya's own published recipe (RLCD with a GRPO-style policy gradient, run via `train_ddp.py` from Laya's official Kaggle notebook, unmodified except for swapping in our data):

1. **Data**: 4 parallel Codex/GPT-5.6 Terra calls generated 480 diverse synthetic (task, difficulty) examples across web/backend, data/ML, infra/security, and mobile/embedded/games — deduped, confirmed **zero overlap** with the original 30-example benchmark, split into 423 train / 57 held-out test (`finetune/build_dataset.py`).
2. **Tokenization**: `finetune/build_train_items.py` runs entirely on CPU (this Mac, no GPU needed) using `laya.common.build_sequence` directly — 423/423 items built cleanly.
3. **Training**: pushed to Kaggle (T4×2, `finetune/kaggle_kernel/`) via the `kaggle` CLI. Took two failed attempts before it actually trained: first attempt requested GPUs but got 0 (Kaggle silently ignores GPU requests on unverified accounts — needed phone verification), second attempt got GPUs but a wrong dataset-mount path (fixed by discovering the path at runtime with `glob` instead of guessing it). Training itself took a few minutes once both were fixed.
4. **Evaluation, two ways**:
   - **On 57 new held-out synthetic cases** (run inside the Kaggle notebook, on GPU): **91.2% accuracy**, 40.6ms mean latency. Confusion matrix showed the "hard" tier (zero-shot's weak spot) now **perfect: 19/19**.
   - **On the original, untouched 30-example benchmark** (run locally afterward, on CPU, via `finetune/eval_v2_checkpoint.py` — the real apples-to-apples comparison against the 63% above): **90.0% accuracy** — trivial 10/10, hard 10/10, moderate 7/10 (3 misclassified as trivial, the only remaining error pattern).

| | v1 (zero-shot) | v2 (fine-tuned) |
| --- | --- | --- |
| Accuracy on the 30-example benchmark | 63% | **90%** |
| Confusion pattern | under-calls "hard" as "moderate" | occasionally under-calls "moderate" as "trivial"; hard tier now perfect |
| Mean latency | ~53-57ms (CPU) | ~106ms (CPU, this checkpoint) / 40.6ms (GPU, during Kaggle eval) |
| Cost | $0 (local) | $0 (local) |

+27 points from fine-tuning on 423 synthetic examples — a real, measured result, not an estimate. The remaining error pattern (some "moderate" tasks read as "trivial") is a much smaller, more specific gap than zero-shot's systematic "hard" under-calling, and a plausible next lever (more moderate-tier training examples, or ones that better separate moderate from trivial) rather than a fundamental limit.

## Open questions / next steps

- Grow past 423/57: this was intentionally a first pass at the low end of Laya's documented data-scale range (Laya's official recipe uses ~1,200 train cases) — more data, especially moderate-tier examples, is the likeliest next lever for the trivial/moderate confusion above
- Run `--with-llm` with a real metered key at some point, to get a real cost number (accuracy is already answered — see the Terra result above)
- Grow the eval set past 30 examples / get a second labeler, since a single-author 30-example set is a first signal, not a validated benchmark
- Decide the entry point for the next step: a LiteLLM `async_pre_call_hook` integration (reuses `pick_model` from this library) vs. a standalone proxy
- Decide how Laya gets served once this moves beyond a single local process (still in-process/local for now)
