# Model Card — Atlas (Knowledge-Grounded QA)

Atlas is a QLoRA fine-tune of `Qwen2.5-7B-Instruct` for retrieval-augmented question answering over a company knowledge base. It is a **reference model** published by MSC Labs to demonstrate a reproducible pipeline; the weights and dataset described here are illustrative and not distributed.

- **Developed by:** MSC Labs
- **Model type:** Decoder-only LLM, parameter-efficient fine-tune (LoRA adapter over a 4-bit base)
- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Language:** English
- **Finetuned for:** Grounded extractive/abstractive QA with citation and abstention
- **License:** Apache-2.0 (code). Base model under its own license (Qwen). Reference weights illustrative.

## Intended use

- **Primary use case:** Answer user questions strictly from a retrieved knowledge-base context (RAG). The retriever supplies one or more document snippets; Atlas produces a grounded answer and cites the supporting span.
- **Intended users:** Teams deploying internal/customer-facing knowledge assistants (product docs, support KBs, policy/SOP libraries) who need accurate, citable answers and explicit abstention.
- **Deployment shape:** Drop-in generator behind a retriever (e.g., `bge`/`e5` embeddings + a vector store). Atlas does not retrieve; it conditions on the context it is given.

## Out-of-scope use

- **Open-domain QA without retrieval.** Atlas is trained to refuse when the context lacks the answer; ungrounded use defeats its purpose.
- **High-stakes autonomous decisions** (medical, legal, financial advice) without human review.
- **Long-document reasoning beyond the context window** — feed retrieved snippets, not entire corpora.
- **Languages other than English** — not evaluated.
- **Faithfulness guarantee** — groundedness is high (97.3%) but not 100%; treat citations as evidence to verify, not proof.

## Training data

Illustrative synthetic + real mix over a fictional enterprise SaaS knowledge base ("Northwind Cloud").

| Split | Rows | Notes |
|---|---|---|
| Train | 18,400 | Grounded QA triples `{context, question, answer}` |
| Validation | 1,200 | Held out by document, no context overlap with train |
| Test (eval) | 1,200 | Held out by document; reported in `eval/results.md` |

- **Synthetic / real mix:** ~80% synthetic (generated from doc snippets with a teacher model, then filtered), ~20% curated from real-style support tickets and doc FAQs, lightly paraphrased and de-identified.
- **Answer-not-in-context rows:** ~15% of train deliberately have no supported answer; the target is an explicit abstention. This is what teaches Atlas to say "I don't know" instead of hallucinating.
- **Dedup:** near-duplicate questions removed via MinHash (Jaccard ≥ 0.85); splits partitioned **by source document** to prevent leakage.
- **Format:** see [`data/README.md`](data/README.md) and [`data/sample.jsonl`](data/sample.jsonl).

## Training procedure

QLoRA, two-stage protocol (MSC Labs standard).

- **Stage 1 — smoke test:** 1 epoch on a 2k-row subset to validate the pipeline (loss decreases, adapter saves, eval harness runs end-to-end). No GPU time spent debugging on the full run.
- **Stage 2 — full run:** 3 epochs on the full train split.
- **Quantization:** 4-bit NF4 base (bitsandbytes), double quantization, compute dtype bf16.
- **Adapter:** LoRA `r=16`, `alpha=32`, `dropout=0.05`, on attention + MLP projections.
- **Optimization:** `paged_adamw_8bit`, lr `2e-4`, cosine schedule, 3% warmup, effective batch 32 (per-device 4 × grad-accum 8), max sequence length 2048, bf16.

Full values in [`training/config.yaml`](training/config.yaml); script in [`training/train.py`](training/train.py).

## Evaluation

- **Eval set:** 1,200 held-out grounded-QA pairs (held out by document).
- **Baseline:** GPT-4o (`gpt-4o-2024-08-06`) with identical retrieved contexts and prompt.
- **Metrics:** answer accuracy (LLM-judge + human-spot-checked exact/semantic match), groundedness (fraction of answers fully supported by the cited context, no fabricated claims), $/1k requests, p50 latency.

| Model | Answer accuracy | Groundedness | $ / 1k req | p50 latency |
|---|---|---|---|---|
| GPT-4o | 79.2% | 88.1% | $9.20 | 1,180 ms |
| **Atlas** | **91.4%** | **97.3%** | **$0.51** | **240 ms** |

Methodology and cost math: [`eval/results.md`](eval/results.md) · machine-readable: [`eval/results.json`](eval/results.json).

## Limitations

- **Retriever-bound.** Answer quality is capped by retrieval quality. Bad context → bad or abstaining answer. Atlas cannot recover facts the retriever fails to surface.
- **Context-window-bound.** Trained/evaluated at 2048 tokens; very long contexts must be chunked.
- **Domain shift.** Tuned on SaaS-style docs. Performance on dissimilar domains (e.g., legal contracts) will drop without re-tuning.
- **Residual hallucination.** ~2.7% of answers contain an unsupported claim; do not treat output as verified fact.
- **English only.**

## Bias, risks, and responsible use

- **Source-faithful, not truth-faithful.** Atlas reflects the knowledge base. If the KB is wrong, outdated, or biased, the answers will be too. Keep the source corpus curated.
- **Abstention is a feature.** "I don't know" responses are correct behavior when context is insufficient; do not penalize or prompt around them.
- **Human review for consequential answers.** Surface the cited span in the UI so users can verify.
- **PII.** Strip or redact sensitive data from the indexed corpus; Atlas will faithfully repeat whatever the context contains.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
