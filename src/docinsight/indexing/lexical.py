"""Lexical retrieval: tokens, BM25 scores and rank fusion with the dense (embedding) ranking.

Embeddings capture meaning but barely notice identifiers: three invoices with the same
layout get almost the same vector whether they are "2026-001" or "2026-002". BM25 scores
exact tokens and weights rare ones (an invoice number) above common ones ("2026", "EUR").
"""

import math
import re
import unicodedata
from collections import Counter

_TOKEN = re.compile(r"\w+")

# Standard BM25 parameters: k1 limits how much repeating a term helps, b normalises length.
BM25_K1 = 1.5
BM25_B = 0.75
# Standard RRF constant: damps the difference between the first few ranks.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase words without diacritics, so "RAČUN" (typed) matches "RACUN" (OCR).

    "2026-002" becomes ["2026", "002"].
    """
    # NFKD splits "č" into "c" + a combining caron, but "đ" is its own letter.
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("đ", "d"))
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _TOKEN.findall(folded)


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """BM25 score of every document for ``query``; 0.0 when they share no token."""
    docs = [tokenize(document) for document in documents]
    if not docs:
        return []
    average_length = sum(len(doc) for doc in docs) / len(docs) or 1.0
    document_frequency = Counter(token for doc in docs for token in set(doc))
    query_tokens = set(tokenize(query))

    scores: list[float] = []
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for token in query_tokens & counts.keys():
            df = document_frequency[token]
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = counts[token]
            length_norm = 1 - BM25_B + BM25_B * len(doc) / average_length
            score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * length_norm)
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge rankings of positions into one, best first, with scores scaled to 0..1.

    Each list contributes 1 / (k + rank). Only ranks are used, so scores on different
    scales (cosine, BM25) never have to be compared. 1.0 = first in every ranking.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            fused[position] = fused.get(position, 0.0) + 1 / (k + rank)
    best_possible = len(rankings) / (k + 1)
    # Ties are broken by position, so the order is deterministic.
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return [(position, score / best_possible) for position, score in ordered]
