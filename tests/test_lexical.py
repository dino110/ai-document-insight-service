"""Tests for tokens, BM25 and reciprocal rank fusion."""

import pytest

from docinsight.indexing.lexical import RRF_K, bm25_scores, reciprocal_rank_fusion, tokenize


def test_tokenize_folds_case_and_diacritics_and_splits_identifiers() -> None:
    assert tokenize("RAČUN br. 2026-002, Đakovo") == ["racun", "br", "2026", "002", "dakovo"]


def test_bm25_rare_token_outweighs_common_ones() -> None:
    documents = [
        "RAČUN br. 2026-001 Ukupno 1.575,00 EUR",
        "RAČUN br. 2026-002 Ukupno 9.537,50 EUR",
        "RAČUN br. 2026-003 Ukupno 2.000,00 EUR",
    ]

    scores = bm25_scores("When is invoice 2026-002 due?", documents)

    # "2026" is in every invoice, "002" only in the second one.
    assert scores.index(max(scores)) == 1
    assert scores[0] == scores[2] > 0


def test_bm25_without_shared_tokens_is_zero() -> None:
    assert bm25_scores("kamatna stopa", ["Kupac: Horvat d.o.o.", ""]) == [0.0, 0.0]
    assert bm25_scores("bilo što", []) == []


def test_rrf_prefers_items_ranked_well_in_both_lists() -> None:
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2]])

    assert [position for position, _ in fused] == [1, 2, 0]
    assert all(0 < score <= 1 for _, score in fused)


def test_rrf_first_in_every_list_scores_one() -> None:
    [(position, score), *_] = reciprocal_rank_fusion([[3, 1], [3, 0]])

    assert position == 3
    assert score == pytest.approx(1.0)


def test_rrf_ties_are_ordered_by_position() -> None:
    # 0 and 1 have the same score (first in one list, second in the other).
    fused = reciprocal_rank_fusion([[0, 1], [1, 0]])

    assert [position for position, _ in fused] == [0, 1]
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert fused[0][1] == pytest.approx((1 / (RRF_K + 1) + 1 / (RRF_K + 2)) / (2 / (RRF_K + 1)))
