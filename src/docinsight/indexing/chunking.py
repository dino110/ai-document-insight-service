"""Split page text into overlapping chunks that never cross a page boundary.

Every chunk is embedded and searched together with a short header of its document (file
name and the start of page 1), because a chunk from the middle of a contract often does
not say which contract it belongs to.
"""

from dataclasses import dataclass

from docinsight.nlp.patterns import find_ibans

# Part of the index name: chunks indexed with another text format are re-indexed.
# "masked": OIBs and IBANs are masked in the embedded text, and no chunk splits an IBAN.
INDEX_FORMAT = "header-masked-v1"
INTRO_CHARS = 200


@dataclass(frozen=True)
class ChunkText:
    page_number: int  # 1-based, same as PageText.page
    chunk_index: int  # position in the whole document, not within the page
    text: str


def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[ChunkText]:
    """Split each page into windows of at most ``size`` characters.

    Windows end at whitespace, so words are not cut (unless a single word is longer
    than ``size``). Each window starts about ``overlap`` characters before the end of
    the previous one, at the start of a word. Empty pages produce no chunks.
    """
    chunks: list[ChunkText] = []
    for page_number, text in pages:
        for piece in _split_page(text, size, overlap):
            chunks.append(ChunkText(page_number, len(chunks), piece))
    return chunks


def _split_page(text: str, size: int, overlap: int) -> list[str]:
    # Every cut is decided on `view`: the same text without breaks inside IBANs. It has
    # the same length, so the pieces are cut from `text` at the positions found in it.
    view = _unbreakable(text)
    pieces: list[str] = []
    start = _word_start(view, 0)
    while start < len(view):
        end = min(start + size, len(view))
        if end < len(view) and not view[end].isspace():
            # The window ends inside a word: move the end back to the last whitespace.
            end = _last_space(view, start, end)
        pieces.append(text[start:end].rstrip())
        if end >= len(view):
            break
        # max(..., start + 1) guarantees progress even when overlap is close to size.
        start = _word_start(view, max(end - overlap, start + 1))
        if not view[end].isspace():
            # A word longer than the window was cut at `end`: go on from the cut,
            # otherwise _word_start would skip the rest of that word.
            start = min(start, end)
    return pieces


def _unbreakable(text: str) -> str:
    """``text`` with the spaces inside IBANs replaced by "_": no cut falls between groups.

    Masking recognises whole IBANs only, so half of one in each of two chunks would be
    sent to the embedding API and the LLM unmasked.
    """
    chars = list(text)
    for start, end in find_ibans(text):
        for i in range(start, end):
            if chars[i].isspace():
                chars[i] = "_"
    return "".join(chars)


def _last_space(text: str, start: int, end: int) -> int:
    """Index of the last whitespace in ``text[start + 1:end]``, or ``end`` if there is none."""
    for i in range(end - 1, start, -1):
        if text[i].isspace():
            return i
    return end  # one word longer than the window: cut it


def _word_start(text: str, i: int) -> int:
    """First index at or after ``i`` where a word starts."""
    if 0 < i < len(text) and not text[i - 1].isspace():
        # ``i`` is inside a word: skip the rest of it.
        while i < len(text) and not text[i].isspace():
            i += 1
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def index_name(embedder_name: str) -> str:
    """What ``DocumentChunk.embedding_model`` stores: vectors are only comparable when both
    the embedder and the indexed text format match."""
    return f"{embedder_name}+{INDEX_FORMAT}"


def document_intro(first_page: str, max_chars: int = INTRO_CHARS) -> str:
    """The start of page 1 on one line (title, parties, invoice number), cut at a word."""
    intro = " ".join(first_page.split())
    if len(intro) <= max_chars:
        return intro
    cut = _unbreakable(intro).rfind(" ", 0, max_chars + 1)  # never inside an IBAN
    return intro[: cut if cut > 0 else max_chars]


def contextualize(filename: str, intro: str, text: str) -> str:
    """The text that is embedded and ranked by BM25 for one chunk."""
    header = f"{filename}\n{intro}" if intro else filename
    return f"{header}\n\n{text}"
