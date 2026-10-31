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

Findings:

- Semantic embeddings beat lexical hashing on paraphrases ("Koliko dugujem Hiniću?") and
  cross-language questions; hashing retrieval is about 100x faster (no API call).
- Hybrid retrieval (BM25 fused with the embedding ranking) fixes identifier questions:
  "When is invoice 2026-002 due?" fails with dense OpenAI embeddings and passes with hybrid.
- One question fails everywhere (`c02-start-hr`): the answer sits in a chunk of the
  English contract that does not name the parties, while the question names the client
  ("ugovor s tvrtkom Šalić d.o.o."). The chunk loses its document context.

## Production setup (Docker, 9 documents including 4 scans through EasyOCR)

### Full report: openai-hybrid (Docker, digital + scanned documents, EasyOCR)

20 questions (17 answerable, 3 not), model gpt-6-luna, 9 documents uploaded.

| Metric | Value |
| --- | --- |
| Answer accuracy | 16/17 (94%) |
| Citation accuracy (expected file and page among sources) | 16/17 (94%) |
| Correct "not found" | 3/3 (100%) |
| Retrieval p50 / p95 | 216 / 506 ms |
| Generation p50 / p95 | 1133 / 1640 ms |
| Total (client) p50 / p95 | 1450 / 1845 ms |

| Question | Answer | Citation | Sources | Response |
| --- | --- | --- | --- | --- |
| inv001-total | ✓ | ✓ | racun-2026-001.pdf p1 | Ukupni iznos računa 2026-001 je 1.575,00 EUR. [1] |
| inv001-buyer | ✓ | ✓ | racun-2026-001.pdf p1 | Kupac na računu 2026-001 je Horvat d.o.o. [1] |
| inv001-net | ✓ | ✓ | racun-2026-001.pdf p1 | Osnovica na računu 2026-001 iznosi 1.260,00 EUR. [1] |
| inv001-issuer-oib | ✓ | ✓ | racun-2026-001.pdf p1 | OIB izdavatelja računa 2026-001 je 28196001332. [1] |
| inv002-vat | ✓ | ✓ | racun-2026-002.pdf p1 | PDV na računu 2026-002 iznosi 1.907,50 EUR. [1] |
| inv002-buyer-oib | ✓ | ✓ | racun-2026-002.pdf p1 | OIB kupca na računu 2026-002 je 17816184958. [1] |
| inv002-due-en | ✓ | ✓ | racun-2026-002.pdf p1 | Invoice 2026-002 is due on 19.01.2026. [4] |
| inv002-owe-paraphrase | ✓ | ✓ | racun-2026-002.pdf p1 | Dugujete Hinić d.d. 9.537,50 EUR ukupno za platiti. [1] |
| inv003-issuer | ✓ | ✓ | racun-2026-003.pdf p1 | Račun 2026-003 izdao je Fabijanić Maloča d.d. [1] |
| inv003-total-en | ✓ | ✓ | racun-2026-003.pdf p1 | The total amount of invoice 2026-003 is 4.437,13 EUR. [2] |
| inv003-due-paraphrase | ✓ | ✓ | racun-2026-003.pdf p1 | Račun tvrtke Fabijanić Maloča dospijeva 20.02.2026. [1] |
| c01-fee | ✓ | ✓ | ugovor-01.pdf p1 | Naknada iznosi 6.300,00 EUR (bruto). [2] |
| c01-deadline | ✓ | ✓ | ugovor-01.pdf p1 | Izvršitelj mora predati djelo najkasnije do 29.10.2026. [1] |
| c01-court-page2 | ✓ | ✓ | ugovor-01.pdf p2 | Nadležan je stvarno nadležni sud u Zagrebu. [1] |
| c01-client-en | ✓ | ✓ | ugovor-01-scan.jpg p1, ugovor-01-scan.pdf p1 | The client is Nacinović d.o.o. [4][5] |
| c02-fee-en | ✓ | ✓ | ugovor-02.pdf p1 | The fee is 8.800,00 EUR (gross). [3] |
| c02-start-hr | ✗ | ✗ | ugovor-01.pdf p1 | Izvršitelj počinje s radom 01.07.2026. prema ugovoru s tvrtkom Šalić d.o.o. [2] |
| none-interest | ✓ | - | - | Dokument ne navodi kamatnu stopu za zakašnjelo plaćanje. |
| none-iban-en | ✓ | - | - | The documents do not contain the issuer’s IBAN for invoice 2026-001. |
| none-director | ✓ | - | - | Dokumenti ne navode tko je direktor tvrtke Car d.o.o. |


The failing question is answered here with the start date of the *other* contract (a
confident wrong answer, not "not found"): the same lost-context problem as above.
