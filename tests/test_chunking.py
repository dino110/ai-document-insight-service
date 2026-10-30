"""Tests for page-aware chunking."""

from itertools import pairwise

from docinsight.indexing.chunking import ChunkText, chunk_pages

SIZE = 200
OVERLAP = 50


def make_words(count: int, prefix: str = "w") -> str:
    """Distinct words of equal length, so a cut word is easy to spot: "w0001 w0002 ..."."""
    return " ".join(f"{prefix}{i:04d}" for i in range(1, count + 1))


def test_short_page_is_one_chunk() -> None:
    chunks = chunk_pages([(1, "  Račun br. 2026-001\nUkupno: 1.575,00 EUR  ")], SIZE, OVERLAP)

    assert chunks == [ChunkText(1, 0, "Račun br. 2026-001\nUkupno: 1.575,00 EUR")]


def test_long_page_is_split_into_chunks_no_longer_than_size() -> None:
    chunks = chunk_pages([(1, make_words(200))], SIZE, OVERLAP)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= SIZE for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_words_are_not_cut() -> None:
    text = make_words(200)
    words = set(text.split())

    chunks = chunk_pages([(1, text)], SIZE, OVERLAP)

    for chunk in chunks:
        assert set(chunk.text.split()) <= words


def test_every_word_ends_up_in_some_chunk() -> None:
    text = make_words(200)

    chunks = chunk_pages([(1, text)], SIZE, OVERLAP)

    assert set().union(*(chunk.text.split() for chunk in chunks)) == set(text.split())


def test_chunks_do_not_cross_page_boundaries() -> None:
    pages = [(1, make_words(60, "a")), (2, make_words(60, "b"))]

    chunks = chunk_pages(pages, SIZE, OVERLAP)

    for chunk in chunks:
        assert chunk.text in dict(pages)[chunk.page_number]
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    # chunk_index keeps counting across pages.
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_empty_pages_are_skipped() -> None:
    pages = [(1, ""), (2, "  \n\t "), (3, "Datum dospijeća: 15.11.2026.")]

    chunks = chunk_pages(pages, SIZE, OVERLAP)

    assert [chunk.page_number for chunk in chunks] == [3]


def test_word_longer_than_size_is_cut_instead_of_looping() -> None:
    chunks = chunk_pages([(1, "x" * 450 + " kraj")], SIZE, OVERLAP)

    assert all(len(chunk.text) <= SIZE for chunk in chunks)
    # Nothing is lost: every "x" and the word after the long one are in the chunks.
    assert "".join(chunk.text for chunk in chunks).count("x") == 450
    assert chunks[-1].text.endswith("kraj")


def test_neighbouring_chunks_overlap() -> None:
    chunks = chunk_pages([(1, make_words(200))], SIZE, OVERLAP)

    for first, second in pairwise(chunks):
        last_word = first.text.split()[-1]
        # The end of one chunk is repeated at the start of the next one...
        assert last_word in second.text.split()
        # ...but no more than `overlap` characters of it.
        assert second.text.index(last_word) + len(last_word) <= OVERLAP
