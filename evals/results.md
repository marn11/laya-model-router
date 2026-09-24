# Eval results

Laya accuracy: **63%** (30 examples)

Laya confusion matrix:

| ground truth \ predicted | trivial | moderate | hard |
| --- | --- | --- | --- |
| **trivial** | 9 | 1 | 0 |
| **moderate** | 4 | 6 | 0 |
| **hard** | 0 | 6 | 4 |

Laya mean latency: 56.5 ms
Laya cost: $0 marginal (self-hosted, local inference)

(LLM judge not run -- pass --with-llm with Anthropic credentials set to include it.)
