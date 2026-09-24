# Eval results

Laya accuracy: **63%** (30 examples)

Laya confusion matrix:

| ground truth \ predicted | trivial | moderate | hard |
| --- | --- | --- | --- |
| **trivial** | 9 | 1 | 0 |
| **moderate** | 4 | 6 | 0 |
| **hard** | 0 | 6 | 4 |

Laya mean latency: 51.8 ms
Laya cost: $0 marginal (self-hosted, local inference)

LLM judge (gpt-5.6-terra) accuracy: **100%** (30 examples)

LLM confusion matrix:

| ground truth \ predicted | trivial | moderate | hard |
| --- | --- | --- | --- |
| **trivial** | 10 | 0 | 0 |
| **moderate** | 0 | 10 | 0 |
| **hard** | 0 | 0 | 10 |

LLM judge latency/cost: **not measured** -- these judgments came from asking an LLM directly rather than a metered API call, so there's no real token/latency data. See README for a rough estimate.

Laya/LLM agreement: **63%**
