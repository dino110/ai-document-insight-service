"""Streamlit UI for the Document Insight API.

    uv run streamlit run ui/app.py        # http://localhost:8501, API at $API_URL

Streamlit reruns this whole script on every interaction. State that must survive a rerun
lives in st.session_state (per browser tab) and st.query_params (the URL, survives F5).
"""

import os
from typing import Literal, TypedDict

import httpx2
import streamlit as st
from api_client import (
    ApiError,
    AskResponse,
    DocumentOut,
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


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceOut]
    caption: str  # model and timings under an answer, "" otherwise
    error: bool  # the API call failed; content is the error message


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

    for page in detail["pages"]:
        label = f"Stranica {page['page_number']} · {METHOD_LABELS[page['method']]}"
        with st.expander(label, expanded=page["page_number"] == 1):
            # st.text, not st.markdown: document text must not be rendered as markup.
            st.text(page["text"] or "(prazna stranica)")


def show_message(message: ChatMessage) -> None:
    with st.chat_message(message["role"]):
        if message["error"]:
            st.error(message["content"])
            return
        st.markdown(_no_latex(message["content"]))
        for source in message["sources"]:
            label = f"{source['filename']}, str. {source['page_number']}"
            with st.expander(label, icon=":material/description:"):
                st.text(source["snippet"])  # document text, not markup
        if message["caption"]:
            st.caption(message["caption"])


def _no_latex(text: str) -> str:
    """Markdown treats text between two "$" as LaTeX: "$100 or $200" would be mangled."""
    return text.replace("$", "\\$")


def _answer_caption(result: AskResponse) -> str:
    timings = result["timings_ms"]
    return (
        f"{result['model']} · pretraga {timings.get('retrieval', 0) / 1000:.1f} s"
        f" · odgovor {timings.get('generation', 0) / 1000:.1f} s"
    )


def chat(session_id: str) -> None:
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

    asked = ChatMessage(role="user", content=question, sources=[], caption="", error=False)
    st.session_state.messages.append(asked)
    show_message(asked)
    with st.spinner("Tražim odgovor u dokumentima…"):
        try:
            result = ask(get_client(), session_id, question)
        except ApiError as exc:
            reply = ChatMessage(
                role="assistant", content=exc.detail, sources=[], caption="", error=True
            )
        else:
            reply = ChatMessage(
                role="assistant",
                content=result["answer"],
                sources=result["sources"],
                caption=_answer_caption(result),
                error=False,
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
