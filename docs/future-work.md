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

## Answers from structured document fields

`GET /documents/{id}/fields` extracts the type, a summary and the fields of an invoice
(number, issuer, buyer, OIBs, IBAN, dates, net, VAT, total) or a contract (parties, OIBs,
dates, fee) into JSON (decision 022). Questions that map to one field of one document
could then be answered from
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

## Names that OCR got wrong

On scans EasyOCR loses diacritics ("Elvis Antié" for "Elvis Antić"), and the extracted
fields copy the error while every check passes (decision 022). The OIB next to a name is
reliable thanks to its check digit, so the official name could be looked up by OIB in a
company register and shown next to the scanned one. For people there is no public
register; a second OCR engine or an LLM correction pass limited to names would be the
options, with the original text always kept.
