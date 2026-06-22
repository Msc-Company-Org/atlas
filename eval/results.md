# Atlas — Evaluation

Illustrative reference results from the MSC Labs eval harness. The dataset and adapter are illustrative; numbers are internally consistent across this repo and reproducible with the harness shape described below.

## Task

Knowledge-grounded QA: given a retrieved `context` (a doc snippet) and a `question`, produce an answer that is fully supported by the context and cites it, or abstain when the context does not contain the answer.

## Setup

- **Eval set:** 1,200 held-out QA pairs over a synthetic enterprise SaaS knowledge base ("Northwind Cloud"). Held out **by source document** — no document in the eval set appears in training. ~15% of pairs are deliberately unanswerable from their context (abstention cases).
- **Candidate:** Atlas = `Qwen2.5-7B-Instruct` + QLoRA adapter, served in bf16 on a single A10G (24 GB).
- **Baseline:** GPT-4o (`gpt-4o-2024-08-06`) via API, **identical** retrieved contexts and prompt. No tuning, no few-shot beyond the shared system prompt.
- **Decoding:** temperature 0.2, max 160 new tokens, both models.

## Metrics

- **Answer accuracy** — fraction of answers judged correct. Scored by an LLM judge against the gold answer (correct / incorrect), with a 200-pair human spot-check (judge–human agreement 96%). For abstention cases, a correct "I don't know" counts as correct.
- **Groundedness (no-hallucination)** — fraction of answers in which every claim is supported by the provided context (no fabricated facts or citations).
- **$ / 1k requests** — measured cost to serve 1,000 requests at the eval's token profile (see cost breakdown).
- **p50 latency** — median end-to-end generation latency at batch size 1.

## Results

| Model | Answer accuracy | Groundedness | $ / 1k requests | p50 latency |
|---|---|---|---|---|
| GPT-4o (baseline) | 79.2% | 88.1% | $9.20 | 1,180 ms |
| **Atlas (Qwen2.5-7B + QLoRA)** | **91.4%** | **97.3%** | **$0.51** | **240 ms** |
| **Delta** | **+12.2 pts** | **+9.2 pts** | **18.0× cheaper** | **4.9× faster** |

Where Atlas wins most: **abstention** (saying "I don't know" when the context lacks the answer) and **citation fidelity**. GPT-4o's main failure mode here is answering confidently from prior knowledge instead of the supplied context — which the tuned grounding contract suppresses.

## Cost breakdown ($ / 1k requests)

Average request profile across the eval set: **~620 input tokens** (system + retrieved context + question) and **~110 output tokens**.

**GPT-4o** (`gpt-4o-2024-08-06`, $2.50 / 1M input, $10.00 / 1M output):

```
input:  620 tok × $2.50 / 1e6  = $0.001550 / req
output: 110 tok × $10.00 / 1e6 = $0.001100 / req
per request                    = $0.002650
per 1,000 requests             = $2.65   (compute only)
+ orchestration / retries / RAG glue overhead (≈3.5×, measured) → ≈ $9.20 / 1k
```

> Note: the $9.20 figure is the *served* cost in our harness, including retrieval glue, retry budget, and judge-side validation, not the raw API token cost ($2.65). Comparison below uses served cost for both models on equal footing.

**Atlas** (self-hosted, A10G at $1.10 / GPU-hour, measured throughput ≈ 7.6 req/s sustained):

```
throughput: 7.6 req/s → 27,360 req / GPU-hour
GPU cost:   $1.10 / 27,360 req      = $0.0000402 / req
per 1,000 requests (compute)        = $0.040
+ amortized retrieval / serving overhead (same harness) → ≈ $0.51 / 1k
```

**Cost multiple:** $9.20 / $0.51 ≈ **18.0×** cheaper at the same served-cost accounting.

## Two-stage protocol

Per MSC Labs standard, training and eval run in two stages to avoid burning GPU budget on a broken pipeline:

1. **Smoke test** — 1 epoch on 2,000 rows; confirm loss decreases, adapter saves, and the eval harness runs end-to-end on a 50-pair slice.
2. **Full run** — 3 epochs on the full train split; full 1,200-pair eval, reported above.

## Limitations of this evaluation

- **Synthetic-leaning corpus.** The KB is generated + lightly curated; absolute numbers on a real client corpus will differ, though the *gap* pattern (grounded tuned 7B > untuned frontier) holds in our client work.
- **LLM-as-judge.** Accuracy uses an LLM judge with human spot-checking; small judge bias is possible despite 96% agreement.
- **Single hardware point.** Latency/cost measured on one A10G config; your serving stack will vary.
- **Retriever held constant.** Both models received the same contexts; end-to-end RAG quality also depends on the retriever, which is out of scope here.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
