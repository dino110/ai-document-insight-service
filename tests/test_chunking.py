"""Tests for page-aware chunking."""

from itertools import pairwise

from docinsight.indexing.chunking import (
    ChunkText,
    chunk_pages,
    contextualize,
    document_intro,
    index_name,
)

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


# --- Document header ---


def test_intro_is_the_start_of_page_one_on_one_line() -> None:
    page = "SERVICE AGREEMENT\nConcluded on 09.02.2026. in Zagreb between:\n\nŠalić d.o.o."

    assert document_intro(page) == (
        "SERVICE AGREEMENT Concluded on 09.02.2026. in Zagreb between: Šalić d.o.o."
    )


def test_long_intro_is_cut_at_a_word() -> None:
    intro = document_intro(make_words(100), max_chars=50)

    assert len(intro) <= 50
    assert intro == make_words(8)  # 8 words of 5 characters + 7 spaces = 47


def test_blank_first_page_gives_no_intro() -> None:
    assert document_intro(" \n\t ") == ""


def test_contextualized_text_starts_with_the_header() -> None:
    assert contextualize("ugovor-02.pdf", "SERVICE AGREEMENT", "Start: 20.02.2026.") == (
        "ugovor-02.pdf\nSERVICE AGREEMENT\n\nStart: 20.02.2026."
    )
    assert contextualize("scan.jpg", "", "Tekst") == "scan.jpg\n\nTekst"


def test_index_name_contains_embedder_and_text_format() -> None:
    assert index_name("text-embedding-3-small") == "text-embedding-3-small+header-v1"
