# Evaluation of `/ask`

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
