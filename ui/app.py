"""Streamlit UI for the Document Insight API.

    uv run streamlit run ui/app.py        # http://localhost:8501, API at $API_URL

Streamlit reruns this whole script on every interaction. State that must survive a rerun
lives in st.session_state (per browser tab) and st.query_params (the URL, survives F5).
"""

import html
import itertools
import os
from collections.abc import Sequence
from typing import Literal, TypedDict

import httpx2
import streamlit as st
from api_client import (
    ApiError,
    AskResponse,
    DocumentOut,
    EntityLabel,
    EntityOut,
    HistoryMessage,
    SourceOut,
    ask,
    get_document,
    list_documents,
    make_client,
    upload,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")

STATUS_LABELS = {"pending": "u obradi", "processed": "obrađen", "failed": "greška"}
METHOD_LABELS = {"text_layer": "tekstualni sloj", "ocr": "OCR"}

# Name in the legend and background colour of each entity label. Semi-transparent, so the
# text stays readable in the light and the dark theme.
ENTITY_STYLES: dict[EntityLabel, tuple[str, str]] = {
    "PERSON": ("osoba", "rgba(66, 133, 244, 0.30)"),
    "ORG": ("organizacija", "rgba(156, 39, 176, 0.28)"),
    "LOC": ("mjesto", "rgba(0, 150, 136, 0.30)"),
    "MISC": ("ostalo", "rgba(158, 158, 158, 0.35)"),
    "OIB": ("OIB", "rgba(255, 152, 0, 0.35)"),
    "IBAN": ("IBAN", "rgba(255, 87, 34, 0.30)"),
    "DATE": ("datum", "rgba(76, 175, 80, 0.30)"),
    "MONEY": ("iznos", "rgba(255, 235, 59, 0.45)"),
}


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceOut]
    caption: str  # model and timings under an answer, "" otherwise
    error: bool  # the API call failed; content is the error message
    route: str | None  # the API's route for an answer; None for questions and errors
    entities: list[EntityOut]  # named entities in an answer, [] otherwise


HISTORY_PAIRS = 3  # the API accepts at most 6 messages: 3 questions and 3 answers
HISTORY_ANSWER_CHARS = 500


@st.cache_resource
def get_client() -> httpx2.Client:
    """One HTTP client (connection pool) for the whole Streamlit process."""
    return make_client(API_URL)


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = st.query_params.get("session")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "messages" not in st.session_state:
        # A plain list would be empty again on the next rerun: every interaction
        # runs this script from the top, only st.session_state survives.
        st.session_state.messages = []


def set_session(session_id: str | None) -> None:
    """Keep the session in both places: session_state for reruns, the URL for refreshes."""
    st.session_state.session_id = session_id
    st.session_state.processed_ids = ()
    st.session_state.messages = []  # answers are about the previous session's documents
    if session_id:
        st.query_params["session"] = session_id
    else:
        st.query_params.clear()


def processed_ids(documents: list[DocumentOut]) -> tuple[str, ...]:
    return tuple(d["id"] for d in documents if d["status"] == "processed")


def sidebar() -> None:
    session_id: str | None = st.session_state.session_id
    with st.sidebar:
        st.header("Upload")
        files = st.file_uploader(
            "PDF, PNG ili JPEG",
            type=["pdf", "png", "jpg", "jpeg"],  # convenience only; the API validates
            accept_multiple_files=True,
            # A widget with a new key is a new, empty widget: this clears it after upload.
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Upload", type="primary", disabled=not files):
            payload = [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in files]
            try:
                result = upload(get_client(), payload, session_id)
            except ApiError as exc:
                st.error(exc.detail)
            else:
                set_session(result["session_id"])
                st.session_state.uploader_key += 1
                st.toast(f"Uploadano: {len(result['documents'])}")
                st.rerun()

        st.divider()
        st.caption(f"Sesija: `{session_id[:8]}…`" if session_id else "Još nema sesije.")
        if st.button("Nova sesija", disabled=session_id is None):
            set_session(None)
            st.rerun()


@st.fragment(run_every="2s")
def documents_table(session_id: str) -> None:
    """Re-runs on its own every 2 s, so statuses update without reloading the page."""
    try:
        documents = list_documents(get_client(), session_id)
    except ApiError as exc:
        st.error(exc.detail)
        return

    if not documents:
        st.info("Uploadaj prvi dokument.")
    else:
        rows = [
            {
                "Datoteka": d["filename"],
                "Status": STATUS_LABELS[d["status"]],
                "Stranice": d["page_count"],
                "Greška": d["error"] or "",
                "Uploadano (UTC)": d["created_at"][:19].replace("T", " "),
            }
            for d in documents
        ]
        st.dataframe(rows, hide_index=True)

    # A newly processed document must also appear in the picker below, which is
    # outside this fragment: rerun the whole page once when the set changes.
    current = processed_ids(documents)
    if current != st.session_state.get("processed_ids"):
        st.session_state.processed_ids = current
        st.rerun(scope="app")


def document_detail(session_id: str, documents: list[DocumentOut]) -> None:
    processed = [d for d in documents if d["status"] == "processed"]
    if not processed:
        st.caption("Tekst se pojavi kad je dokument obrađen.")
        return

    names = {d["id"]: d["filename"] for d in processed}
    document_id = st.selectbox("Dokument", options=list(names), format_func=names.__getitem__)
    if document_id is None:
        return
    try:
        detail = get_document(get_client(), session_id, document_id)
    except ApiError as exc:
        st.error(exc.detail)
        return

    # .get: an API from before phase 5 sends neither field.
    summary = detail.get("entity_summary") or {}
    if summary:
        st.markdown("**Entiteti** (lokalni NER, bez AI modela)")
        st.table(summary_rows(summary), hide_index=True)
        all_entities = [e for page in detail["pages"] for e in page.get("entities") or []]
        st.caption(entity_legend(all_entities), unsafe_allow_html=True)

    for page in detail["pages"]:
        label = f"Stranica {page['page_number']} · {METHOD_LABELS[page['method']]}"
        with st.expander(label, expanded=page["page_number"] == 1):
            if not page["text"]:
                st.text("(prazna stranica)")
                continue
            # st.html, not st.markdown: document text must not be read as Markdown
            # ("# " would become a heading). page_html escapes the text itself.
            st.html(page_html(page["text"], page.get("entities") or []))


def show_message(message: ChatMessage) -> None:
    with st.chat_message(message["role"]):
        if message["error"]:
            st.error(message["content"])
            return
        if message["role"] == "user":
            st.markdown(_no_latex(message["content"]))  # no HTML: Streamlit escapes it
        else:
            # .get: messages kept in session_state from before this field existed have none.
            entities = message.get("entities") or []
            # Our <mark> tags need unsafe_allow_html; highlight() escapes all of the text.
            st.markdown(highlight(message["content"], entities), unsafe_allow_html=True)
            if entities:
                st.caption(entity_legend(entities), unsafe_allow_html=True)
        for source in message["sources"]:
            label = f"{source['filename']}, str. {source['page_number']}"
            with st.expander(label, icon=":material/description:"):
                st.text(source["snippet"])  # document text, not markup
        if message["caption"]:
            st.caption(message["caption"])


def highlight(text: str, entities: Sequence[EntityOut], *, markdown: bool = True) -> str:
    """The text with every entity wrapped in a coloured <mark>, for Markdown or plain HTML.

    The text comes from the LLM or from a document, so it is untrusted: every piece is
    escaped and the <mark> tags are the only HTML in the result. Offsets refer to the
    original text, so it is cut first and each piece is escaped afterwards. Only Markdown
    needs "$" escaped (it would start LaTeX); in plain HTML "\\$" would show up as is.
    """
    escape = _escape if markdown else html.escape
    parts: list[str] = []
    position = 0
    for entity in sorted(entities, key=lambda e: e["start"]):
        start, end = entity["start"], entity["end"]
        if start < position or text[start:end] != entity["text"]:
            continue  # overlapping or stale offsets: leave that part as plain text
        name, colour = ENTITY_STYLES[entity["label"]]
        parts.append(escape(text[position:start]))
        parts.append(
            f'<mark style="background-color: {colour}; color: inherit; padding: 0 2px;"'
            f' title="{name}">{escape(text[start:end])}</mark>'
        )
        position = end
    parts.append(escape(text[position:]))
    return "".join(parts)


def page_html(text: str, entities: Sequence[EntityOut]) -> str:
    """A page for st.html: no Markdown, line breaks kept like st.text, entities marked."""
    return (
        '<div style="white-space: pre-wrap; line-height: 1.6;">'
        f"{highlight(text, entities, markdown=False)}</div>"
    )


def summary_rows(summary: dict[EntityLabel, list[str]]) -> list[dict[str, str]]:
    """The API's entity summary as table rows, in the API's (fixed) label order."""
    return [
        {"Oznaka": ENTITY_STYLES[label][0], "Vrijednosti": ", ".join(values)}
        for label, values in summary.items()
    ]


def entity_legend(entities: Sequence[EntityOut]) -> str:
    """The labels that occur in the answer, in a fixed order, each in its own colour."""
    present = {entity["label"] for entity in entities}
    return " ".join(
        f'<mark style="background-color: {colour}; color: inherit; padding: 0 4px;">{name}</mark>'
        for label, (name, colour) in ENTITY_STYLES.items()
        if label in present
    )


def _escape(text: str) -> str:
    return _no_latex(html.escape(text))


def _no_latex(text: str) -> str:
    """Markdown treats text between two "$" as LaTeX: "$100 or $200" would be mangled."""
    return text.replace("$", "\\$")


def answer_caption(result: AskResponse) -> str:
    """How the answer was produced: the model and timings, or that no AI model was used."""
    if result["route"] == "meta":
        return "Popis iz baze · odgovoreno bez AI modela"
    if result["route"] != "rag" or result["model"] is None:
        return "Predefinirani odgovor · odgovoreno bez AI modela"
    timings = result["timings_ms"]
    return (
        f"{result['model']} · pretraga {timings.get('retrieval', 0) / 1000:.1f} s"
        f" · odgovor {timings.get('generation', 0) / 1000:.1f} s"
    )


def conversation_history(messages: list[ChatMessage]) -> list[HistoryMessage]:
    """The last question-answer pairs answered from the documents, oldest first.

    Refusals, small talk, the document list and errors are left out: they do not help to
    understand a follow-up question. Long answers are cut, the history is only context.
    """
    # Each answer with the message right before it, so a question left without an answer
    # (an interrupted run) does not shift the pairs after it.
    pairs = [
        (question, answer)
        for question, answer in itertools.pairwise(messages)
        if question["role"] == "user"
        and answer["role"] == "assistant"
        and answer.get("route") == "rag"
        and not answer["error"]
    ]
    history: list[HistoryMessage] = []
    for question, answer in pairs[-HISTORY_PAIRS:]:
        history.append({"role": "user", "text": question["content"]})
        history.append({"role": "assistant", "text": _shorten(answer["content"])})
    return history


def _shorten(text: str) -> str:
    if len(text) <= HISTORY_ANSWER_CHARS:
        return text
    return text[:HISTORY_ANSWER_CHARS].rsplit(maxsplit=1)[0] + " …"


def _new_conversation() -> None:
    st.session_state.messages = []


def chat(session_id: str) -> None:
    st.button(
        "Novi razgovor",
        icon=":material/restart_alt:",
        on_click=_new_conversation,  # runs before the rerun, so the page is already empty
        disabled=not st.session_state.messages,
    )
    for message in st.session_state.messages:
        show_message(message)

    ready = bool(st.session_state.processed_ids)
    question = st.chat_input(
        "Pitaj nešto o dokumentima" if ready else "Pitanja su moguća kad je dokument obrađen",
        disabled=not ready,
        max_chars=1000,  # same limit as the API
        submit_mode="disable",  # no second question while the first is being answered
    )
    if not question:
        return

    # Built before the new question is added: the history is what came before it.
    history = conversation_history(st.session_state.messages)
    asked = ChatMessage(
        role="user",
        content=question,
        sources=[],
        caption="",
        error=False,
        route=None,
        entities=[],
    )
    st.session_state.messages.append(asked)
    show_message(asked)
    with st.spinner("Tražim odgovor u dokumentima…"):
        try:
            result = ask(get_client(), session_id, question, history)
        except ApiError as exc:
            reply = ChatMessage(
                role="assistant",
                content=exc.detail,
                sources=[],
                caption="",
                error=True,
                route=None,
                entities=[],
            )
        else:
            reply = ChatMessage(
                role="assistant",
                content=result["answer"],
                sources=result["sources"],
                caption=answer_caption(result),
                error=False,
                route=result["route"],
                entities=result["entities"],
            )
    st.session_state.messages.append(reply)
    show_message(reply)


def main() -> None:
    st.set_page_config(page_title="Document Insight", layout="wide")
    init_state()
    sidebar()

    st.title("Document Insight")
    session_id: str | None = st.session_state.session_id
    if session_id is None:
        st.info("Uploadaj prvi dokument (lijevo).")
        return

    # Loaded once per full run for the picker; the table below polls on its own.
    try:
        documents = list_documents(get_client(), session_id)
    except ApiError:
        documents = []  # the table shows the error and keeps retrying
    st.session_state.processed_ids = processed_ids(documents)

    st.subheader("Dokumenti")
    documents_table(session_id)

    st.subheader("Tekst po stranicama")
    document_detail(session_id, documents)

    st.subheader("Pitanja")
    chat(session_id)


if __name__ == "__main__":
    main()
