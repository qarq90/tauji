
import os
import tempfile

import streamlit as st

from rag import (
    create_vector_store,
    retrieve_documents,
    stream_answer,
)


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Tauji",
    page_icon="📚",
    layout="wide",
)


# -----------------------------
# SESSION STATE
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# -----------------------------
# SUMMARY DETECTION
# -----------------------------

SUMMARY_KEYWORDS = (
    "what is it about",
    "what is this about",
    "what is this document about",
    "what is the document about",
    "summarize the document",
    "summarise the document",
    "give me a summary",
    "summary of this document",
    "summarize this",
    "summarise this",
    "overview of the document",
)


def is_summary_question(question):
    normalized = question.lower().strip().rstrip("?.! ")
    return any(
        phrase in normalized
        for phrase in SUMMARY_KEYWORDS
    )


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.title("Tauji")
    st.caption("Your local AI document assistant")

    st.write(
        "Upload a document and ask questions "
        "about its contents."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
    )

    if uploaded_file:
        if st.button(
            "Process Document",
            use_container_width=True,
        ):
            progress_placeholder = st.empty()
            progress_messages = []

            def update_progress(message):
                progress_messages.append(message)
                progress_placeholder.info(
                    "\n\n".join(progress_messages)
                )

            suffix = os.path.splitext(
                uploaded_file.name
            )[1].lower()

            temp_path = None

            try:
                with st.spinner("Processing document..."):
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:
                        temp_file.write(
                            uploaded_file.getbuffer()
                        )
                        temp_path = temp_file.name

                    new_vector_store = create_vector_store(
                        temp_path,
                        progress_callback=update_progress,
                    )

                    st.session_state.vector_store = new_vector_store
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []

                progress_placeholder.success(
                    f"{uploaded_file.name} is ready!"
                )

            except Exception as error:
                progress_placeholder.error(
                    f"Processing failed: {error}"
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    if st.session_state.document_name:
        st.caption(
            f"Indexed: {st.session_state.document_name}"
        )

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# MAIN AREA
# -----------------------------

st.title("Chat with your document")

st.caption(
    "Ask questions, summarize sections, extract information, "
    "and explore your document."
)

if st.session_state.vector_store is None:
    st.info("Upload and process a document to get started.")


# -----------------------------
# CHAT HISTORY
# -----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View Sources"):
                for i, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    page = source.metadata.get("page")

                    page_label = (
                        f"Page {page + 1}"
                        if page is not None
                        else "Document section"
                    )

                    st.markdown(
                        f"**Source {i} — {page_label}**"
                    )
                    st.write(source.page_content[:500])


# -----------------------------
# CHAT INPUT
# -----------------------------

question = st.chat_input(
    "Ask something about your document..."
)

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a document first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status(
            "Searching your document...",
            expanded=True,
        )

        try:
            summary_mode = is_summary_question(question)

            if summary_mode:
                # Use the first few chunks for a quick overview.
                # For long documents, this is not a full-document summary.
                sources = st.session_state.vector_store["chunks"][:4]
                status.write("Preparing document overview...")
            else:
                sources = retrieve_documents(
                    st.session_state.vector_store,
                    question,
                    k=4,
                )
                status.write(
                    f"Found {len(sources)} matching sections"
                )

            status.write("Generating answer with Qwen3...")

            answer = st.write_stream(
                stream_answer(
                    sources,
                    question,
                    is_summary=summary_mode,
                )
            )

            status.update(
                label="Answer generated",
                state="complete",
                expanded=False,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            status.update(
                label="Something went wrong",
                state="error",
            )

            answer = f"Something went wrong: {error}"
            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
            })
