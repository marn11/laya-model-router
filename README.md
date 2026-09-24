# laya-model-router

An experiment in building a self-hosted, open-source request router that picks which LLM answers a request based on how hard the request actually is — using [Laya](https://huggingface.co/convaiinnovations/laya) as the difficulty classifier.

## Status

Early-stage / design phase. Nothing is wired up yet. This repo exists to scope the problem before writing code.

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

## Open questions / next steps

- Define the difficulty taxonomy (how many tiers, what distinguishes them)
- Decide the entry point: a proxy/gateway in front of existing model APIs, a library, or an integration with something like LiteLLM
- Decide how Laya gets served (local inference vs. a small hosted endpoint)
- Zero-shot vs. fine-tuned: start with zero-shot criteria and see how far that gets us before investing in labeled training data
