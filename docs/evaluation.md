# Evaluation of `/ask` and document fields

Question set: `sample_docs/eval_questions.json` (20 questions: 17 answerable from the
sample invoices and contracts, in Croatian and English, including paraphrases and
cross-language questions; 3 that the documents cannot answer). Script:
`scripts/evaluate.py`. LLM: `gpt-6-luna`, reasoning effort `low`, `top_k` 5. Run on 31. 10. 2026.

- **Answer accuracy:** the answer contains one of the accepted spellings of the expected value.
- **Citation accuracy:** the expected file and page are among the returned sources
  (a scan counts as its original).
- **Correct "not found":** `found=false` for the questions without an answer.

## Embeddings and retrieval mode (local, 5 digital documents)

| Embeddings | Retrieval | Answer accuracy | Citation accuracy | Correct "not found" | Retrieval p50 / p95 |
| --- | --- | --- | --- | --- | --- |
| hashing (lexical) | dense | 13/17 (76%) | 13/17 (76%) | 3/3 (100%) | 1 / 3 ms |
| hashing (lexical) | hybrid | 14/17 (82%) | 14/17 (82%) | 3/3 (100%) | 4 / 4 ms |
| OpenAI text-embedding-3-small | dense | 15/17 (88%) | 15/17 (88%) | 3/3 (100%) | 247 / 377 ms |
| OpenAI text-embedding-3-small | hybrid | 16/17 (94%) | 16/17 (94%) | 3/3 (100%) | 233 / 258 ms |
| hashing (lexical) | hybrid + chunk headers | 14/17 (82%) | 14/17 (82%) | 3/3 (100%) | 5 / 5 ms |
| OpenAI text-embedding-3-small | **hybrid + chunk headers** | **17/17 (100%)** | **17/17 (100%)** | 3/3 (100%) | 249 / 313 ms |

Findings:

- Semantic embeddings beat lexical hashing on paraphrases ("Koliko dugujem Hiniću?") and
  cross-language questions; hashing retrieval is about 100x faster (no API call).
- Hybrid retrieval (BM25 fused with the embedding ranking) fixes identifier questions:
  "When is invoice 2026-002 due?" fails with dense OpenAI embeddings and passes with hybrid.
- Without chunk headers one question failed everywhere (`c02-start-hr`): the answer sits
  in a chunk of the English contract that does not name the parties, while the question
  names the client ("ugovor s tvrtkom Šalić d.o.o."). The chunk lost its document context,
  and in Docker the model answered confidently with the start date of the *other* contract.
- **Chunk headers** fix it: every chunk is embedded, ranked by BM25 and shown to the LLM
  together with its file name and the first 200 characters of page 1 (title and parties).
  With OpenAI embeddings all 17 answerable questions pass. With hashing the right contract
  is now found, but `c02-start-hr` still fails safely ("the excerpt does not say"): a
  lexical embedder cannot match the Croatian question to the English clause.
- 17 questions are a regression check, not a precise accuracy: one question moves the
  result by about 6 percentage points.

## Follow-up questions with conversation history (local, 5 digital documents)

Two follow-ups were added to the set (22 questions, 19 answerable): "A PDV?" after a
question about invoice 2026-002, and "A do kada mora predati djelo?" after a question
about the contract with Elvis Antić. The client sends the earlier messages in `history`.

With router and history (OpenAI embeddings, hybrid retrieval, chunk headers): answer
accuracy 19/19, citation accuracy 19/19, correct "not found" 3/3, retrieval p50 / p95
245 / 330 ms, generation p50 / p95 1350 / 2191 ms. All 20 earlier questions still route
to `rag`.

The same two questions without history:

| Question | Without history | With history |
| --- | --- | --- |
| A PDV? | VAT of all three invoices, sources from all three | 1.907,50 EUR, source `racun-2026-002.pdf` only |
| A do kada mora predati djelo? | 29.10.2026., `ugovor-01.pdf` | 29.10.2026., `ugovor-01.pdf` |

Both follow-ups pass the current metrics even without history: the first because the
answer lists every invoice and so contains the expected value, the second because only
one contract uses that wording. They guard against regressions but do not measure what
history adds. A stricter check (all sources from the expected document) and a follow-up
that is ambiguous between the contracts would.

## Production setup (Docker, 9 documents including 4 scans through EasyOCR)

Before chunk headers this setup scored 16/17: `c02-start-hr` was answered with the date of
the other contract ("01.07.2026."), with a citation. After `python -m docinsight.reindex`
(25 documents, 57 chunks rebuilt from stored page text, no OCR) it scores 17/17.

### Full report: openai-hybrid-headers (Docker, digital + scanned documents, EasyOCR)

20 questions (17 answerable, 3 not), model gpt-6-luna, 9 documents uploaded.

| Metric | Value |
| --- | --- |
| Answer accuracy | 17/17 (100%) |
| Citation accuracy (expected file and page among sources) | 17/17 (100%) |
| Correct "not found" | 3/3 (100%) |
| Retrieval p50 / p95 | 228 / 421 ms |
| Generation p50 / p95 | 1412 / 1639 ms |
| Total (client) p50 / p95 | 1694 / 1911 ms |

| Question | Answer | Citation | Sources | Response |
| --- | --- | --- | --- | --- |
| inv001-total | ✓ | ✓ | racun-2026-001.pdf p1 | Ukupni iznos računa 2026-001 je 1.575,00 EUR. [1] |
| inv001-buyer | ✓ | ✓ | racun-2026-001.pdf p1 | Kupac na računu 2026-001 je Horvat d.o.o. [1] |
| inv001-net | ✓ | ✓ | racun-2026-001.pdf p1 | Osnovica (iznos bez PDV-a) na računu 2026-001 iznosi 1.260,00 EUR. [1] |
| inv001-issuer-oib | ✓ | ✓ | racun-2026-001.pdf p1 | The issuer’s OIB is 28196001332. [1] |
| inv002-vat | ✓ | ✓ | racun-2026-002.pdf p1 | PDV na računu 2026-002 iznosi 1.907,50 EUR. [1] |
| inv002-buyer-oib | ✓ | ✓ | racun-2026-002.pdf p1 | OIB kupca na računu 2026-002 je 17816184958. [1] |
| inv002-due-en | ✓ | ✓ | racun-2026-002.pdf p1 | Invoice 2026-002 is due on 19.01.2026. [1] |
| inv002-owe-paraphrase | ✓ | ✓ | racun-2026-002.pdf p1 | Hiniću dugujete ukupno 9.537,50 EUR prema računu br. 2026-002. [1] |
| inv003-issuer | ✓ | ✓ | racun-2026-003.pdf p1 | Račun 2026-003 izdao je Fabijanić Maloča d.d. [1] |
| inv003-total-en | ✓ | ✓ | racun-2026-003.pdf p1 | The total amount of invoice 2026-003 is 4.437,13 EUR. [1] |
| inv003-due-paraphrase | ✓ | ✓ | racun-2026-003.pdf p1 | Račun tvrtke Fabijanić Maloča dospijeva 20.02.2026. [1] |
| c01-fee | ✓ | ✓ | ugovor-01.pdf p1 | Ukupna naknada iznosi 6.300,00 EUR (bruto). [2] |
| c01-deadline | ✓ | ✓ | ugovor-01.pdf p1 | Izvršitelj mora predati djelo najkasnije do 29.10.2026. [3] |
| c01-court-page2 | ✓ | ✓ | ugovor-01.pdf p2 | Ako se spor ne može riješiti sporazumno, nadležan je stvarno nadležni sud u Zagrebu. [1] |
| c01-client-en | ✓ | ✓ | ugovor-01.pdf p1 | The client (Naručitelj) is Načinović d.o.o. [1] |
| c02-fee-en | ✓ | ✓ | ugovor-02.pdf p1 | The fee is 8.800,00 EUR (gross). [1] |
| c02-start-hr | ✓ | ✓ | ugovor-02.pdf p1 | Izvršitelj počinje s radom 20.02.2026. [5] |
| none-interest | ✓ | - | - | Dokumenti ne navode kamatnu stopu za zakašnjelo plaćanje. |
| none-iban-en | ✓ | - | - | The documents do not provide the issuer’s IBAN. |
| none-director | ✓ | - | - | Dokumenti ne navode tko je direktor tvrtke Car d.o.o. |

## Document fields (local, 5 digital documents)

`scripts/evaluate_fields.py` uploads the sample invoices and contracts, calls `GET /documents/{id}/fields` once per document and compares the type and every field with `sample_docs/expected.json` (amounts as numbers, dates as dates, OIBs and IBANs without spaces, names exactly). For contracts, expected.json knows the parties, the fee and the start and end dates. OIBs and IBANs were masked in the LLM call (decision 021): the model returned placeholders and the API put the real values back, also in the summaries. The NER row shows whether local NER found the expected OIBs, IBAN, total and fee among the entities of each document, which `values_in_document` relies on. All five summaries were in Croatian and stated only facts from the documents; after this run the prompt asks for amounts and dates in the summary as the document writes them (1.575,00 EUR, 29.03.2026.), which a spot check on two documents confirmed, with the fields unchanged.

Note on the question set: since the sample invoices have an IBAN (phase 5), the unanswerable question `none-iban-en` in the reports above is no longer unanswerable. It is replaced by `none-swift-en` (SWIFT/BIC code) in `eval_questions.json`; the reports above are kept as they were measured.

### Report: OpenAI gpt-6-luna, 5 digital documents, local

5 documents, 5 answered with 200.

| Field | Correct |
| --- | --- |
| invoice: number | 3/3 (100%) |
| invoice: issuer | 3/3 (100%) |
| invoice: issuer_oib | 3/3 (100%) |
| invoice: issuer_iban | 3/3 (100%) |
| invoice: buyer | 3/3 (100%) |
| invoice: buyer_oib | 3/3 (100%) |
| invoice: issue_date | 3/3 (100%) |
| invoice: due_date | 3/3 (100%) |
| invoice: net | 3/3 (100%) |
| invoice: vat | 3/3 (100%) |
| invoice: total | 3/3 (100%) |
| contract: client | 2/2 (100%) |
| contract: contractor | 2/2 (100%) |
| contract: start_date | 2/2 (100%) |
| contract: end_date | 2/2 (100%) |
| contract: fee | 2/2 (100%) |
| **all fields** | 43/43 (100%) |

| Metric | Value |
| --- | --- |
| Document type | 5/5 (100%) |
| Summary returned | 5/5 (100%) |
| Documents with a failed check | 0 |
| Identifiers and amounts among the entities (local NER) | 14/14 (100%) |
| Extraction p50 / max | 2813 / 3738 ms |

| Document | Fields | Failed checks | Wrong fields |
| --- | --- | --- | --- |
| racun-2026-001.pdf | 11/11 | - | - |
| racun-2026-002.pdf | 11/11 | - | - |
| racun-2026-003.pdf | 11/11 | - | - |
| ugovor-01.pdf | 5/5 | - | - |
| ugovor-02.pdf | 5/5 | - | - |

## Document fields in Docker (11 digital documents and 4 scans through EasyOCR)

The sample set now has four more invoices (2026-004 to 2026-007) and two more contracts (`ugovor-03`, `ugovor-04`); the earlier documents are unchanged, so the reports above stay valid for them. Earlier `/ask` reports were measured with the 9 documents of that time; a new `/ask` run searches all 15. The run below was made in Docker against the same API, before and after OCR text started to have legal forms fixed (`extraction/ocr_cleanup.py`).

| | Before the fix | After the fix |
| --- | --- | --- |
| All fields | 121/129 (94%) | 125/129 (97%) |
| Digital documents | 97/97 | 97/97 |
| Scans | 24/32 | 28/32 |
| Documents with a failed check | 4 (every scan, `names_in_entities`) | 0 |
| Identifiers and amounts among the entities | 42/42 | 42/42 |

EasyOCR read every "d.o.o." on the scans as "d.0.0.", and the model copied it as written, so all four scanned documents had wrong company names. The company rule did not recognise "d.0.0." as a legal form, so `names_in_entities` failed on exactly those four documents and on no other. After the fix the invoice scans are correct. The contract scans still have "Nacinoviá d.o.o." and "Elvis Antié" for "Načinović d.o.o." and "Elvis Antić": OCR lost the diacritics, and the model copied the text faithfully.

These four wrong fields pass every check. The checks compare the fields with the document text, and OCR wrote the wrong names into that text: the rules find "Nacinoviá d.o.o." as a company, and spaCy finds "Elvis Antié" as a person. The checks catch a model that invents or changes values; they cannot catch an error in the source text. OIBs, IBANs, amounts and dates were right on every scan.

### Report: Docker, after OCR legal form fix

15 documents, 15 answered with 200.

| Field | Correct |
| --- | --- |
| invoice: number | 9/9 (100%) |
| invoice: issuer | 9/9 (100%) |
| invoice: issuer_oib | 9/9 (100%) |
| invoice: issuer_iban | 9/9 (100%) |
| invoice: buyer | 9/9 (100%) |
| invoice: buyer_oib | 9/9 (100%) |
| invoice: issue_date | 9/9 (100%) |
| invoice: due_date | 9/9 (100%) |
| invoice: net | 9/9 (100%) |
| invoice: vat | 9/9 (100%) |
| invoice: total | 9/9 (100%) |
| contract: client | 4/6 (67%) |
| contract: contractor | 4/6 (67%) |
| contract: start_date | 6/6 (100%) |
| contract: end_date | 6/6 (100%) |
| contract: fee | 6/6 (100%) |
| **all fields** | 125/129 (97%) |

| Metric | Value |
| --- | --- |
| Document type | 15/15 (100%) |
| Summary returned | 15/15 (100%) |
| Documents with a failed check | 0 |
| Identifiers and amounts among the entities (local NER) | 42/42 (100%) |
| Extraction p50 / max | 3862 / 4740 ms |

| Document | Fields | Failed checks | Wrong fields |
| --- | --- | --- | --- |
| racun-2026-001.pdf | 11/11 | - | - |
| racun-2026-002.pdf | 11/11 | - | - |
| racun-2026-003.pdf | 11/11 | - | - |
| ugovor-01.pdf | 5/5 | - | - |
| ugovor-02.pdf | 5/5 | - | - |
| racun-2026-001-scan.jpg | 11/11 | - | - |
| racun-2026-001-scan.pdf | 11/11 | - | - |
| ugovor-01-scan.jpg | 3/5 | - | client, contractor |
| ugovor-01-scan.pdf | 3/5 | - | client, contractor |
| racun-2026-004.pdf | 11/11 | - | - |
| racun-2026-005.pdf | 11/11 | - | - |
| racun-2026-006.pdf | 11/11 | - | - |
| racun-2026-007.pdf | 11/11 | - | - |
| ugovor-03.pdf | 5/5 | - | - |
| ugovor-04.pdf | 5/5 | - | - |
