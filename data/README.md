# Atlas — Dataset

Grounded question-answering data for the Atlas reference model. Each row is a `{context, question, answer}` triple where the answer must be supported by — and cite — the context, or explicitly abstain when the context does not contain the answer.

> Illustrative dataset. The full corpus is not distributed; [`sample.jsonl`](sample.jsonl) ships 12 representative rows.

## Format

JSONL, one object per line:

```json
{
  "context":  "<retrieved document snippet>",
  "question": "<user question>",
  "answer":   "<grounded answer with a cited span, or an explicit 'I don't know'>"
}
```

- **context** — a single retrieved doc snippet (in production this comes from the retriever).
- **question** — a natural-language user question.
- **answer** — supported by the context, with the supporting span quoted. For unanswerable rows, the answer is an explicit abstention.

## Size & splits

| Split | Rows | Purpose |
|---|---|---|
| Train | 18,400 | QLoRA fine-tuning |
| Validation | 1,200 | Checkpoint selection during training |
| Test (eval) | 1,200 | Reported in [`../eval/results.md`](../eval/results.md) |
| **Total** | **20,800** | |

Splits are partitioned **by source document**: no document that appears in train appears in validation or test. This prevents the model from memorizing answers and inflating eval scores.

## Synthetic / real mix

- **~80% synthetic.** Generated from real-style doc snippets of a fictional enterprise SaaS product ("Northwind Cloud") using a teacher model, then filtered (answerability check, citation check, length/format check). Low-quality or unsupported generations are dropped.
- **~20% curated.** Adapted from real-style support tickets and documentation FAQs, lightly paraphrased and de-identified.
- **~15% abstention rows.** Deliberately unanswerable from their context; the target is an explicit "I don't know." These teach the grounding contract and are the main driver of the groundedness gain over the untuned baseline.

## Cleaning & dedup

- Near-duplicate questions removed with MinHash (Jaccard similarity ≥ 0.85).
- Answers validated to (a) quote a span present in the context, or (b) be a clean abstention.
- PII stripped/redacted from contexts before indexing.
- Contexts truncated to fit the 2,048-token training window.

## Notes

This is a reference dataset for a public demo. For a client engagement, MSC Labs builds the corpus from your real knowledge base and retriever, with the same grounding contract and abstention handling.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://msc-labs-ai.vercel.app/assessment
> Numbers are illustrative reference results from our standard eval harness.
