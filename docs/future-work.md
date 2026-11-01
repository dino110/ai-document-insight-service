# Future work

Designs that were considered and deliberately left out of the current version. Each one
fits behind an existing protocol or endpoint, so it can be added without reshaping the
service.

## Local extractive QA before the LLM (cascade)

Factual questions ("What is the total of invoice 2026-001?") could first go to a local
extractive QA model, for example `deepset/xlm-roberta-base-squad2` (multilingual, Docker
only because of torch, see decision 010). The model returns a span from the retrieved
chunks and a confidence score. Only when the score is above a **high** threshold
(for example `local_qa_min_score = 0.9`) is that span returned; otherwise the question
goes to the LLM as today. The threshold is high on purpose: the LLM is more reliable and
has priority, and the cascade only saves cost and latency on the clear cases. The
threshold would be tuned on `sample_docs/eval_questions.json` by comparing answer
accuracy and the share of questions answered locally. It would sit behind the existing
`AnswerGenerator` protocol.

## Answers from structured invoice fields

Phase 5 extracts invoice fields (number, issuer, buyer, OIBs, dates, net, VAT, total)
into JSON. Questions that map to one field of one invoice could then be answered from
those fields without retrieval or an LLM, and they enable questions text search cannot
answer: ranges and aggregates ("all invoices from January above 5,000 EUR"). Fields would
also work as metadata filters before hybrid retrieval, falling back to plain hybrid
retrieval when a filter matches nothing (decision 015).

## Masking more personal data before calling the API

OIBs and IBANs are already masked before every call to OpenAI, for embeddings and for the
LLM (decision 021). Names, addresses and foreign IBANs still go out. Names need a reliable
NER model (decision 020 measured the current one as too noisy for this); foreign IBANs
need the length per country. Full privacy for document text would need a local embedder
and a local LLM behind the existing protocols.
