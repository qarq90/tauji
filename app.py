import os
import tempfile

import streamlit as st

from rag import (
    create_vector_store,
    retrieve_documents,
    stream_answer
)


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Tauji",
    page_icon="",
    layout="wide"
)


# -----------------------------
# SESSION STATE
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.title("Tauji")

    st.caption(
        "Your local AI document assistant"
    )

    st.write(
        "Upload a document and ask questions "
        "about its contents."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"]
    )

    if uploaded_file:

        if st.button(
            "Process Document",
            use_container_width=True
        ):

            progress_placeholder = st.empty()

            progress_messages = []

            def update_progress(message):

                progress_messages.append(message)

                progress_placeholder.info(
                    "\n\n".join(progress_messages)
                )


            with st.spinner(
                "Processing document..."
            ):

                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name


                try:

                    st.session_state.vector_store = (
                        create_vector_store(
                            temp_path,
                            progress_callback=update_progress
                        )
                    )

                    st.session_state.messages = []

                    progress_placeholder.success(
                        "Document ready!"
                    )

                except Exception as e:

                    progress_placeholder.error(
                        f"Error: {e}"
                    )

                finally:

                    if os.path.exists(temp_path):
                        os.remove(temp_path)


    st.divider()


    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# -----------------------------
# MAIN AREA
# -----------------------------

st.title(
    "Chat with your document"
)

st.caption(
    "Ask questions, summarize sections, "
    "extract information, and explore your document."
)


# -----------------------------
# CHAT HISTORY
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# -----------------------------
# CHAT INPUT
# -----------------------------

question = st.chat_input(
    "Ask something about your document..."
)


if question:

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process a document first."
        )

        st.stop()


    # -------------------------
    # USER MESSAGE
    # -------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    with st.chat_message("user"):

        st.markdown(question)


    # -------------------------
    # ASSISTANT
    # -------------------------

    with st.chat_message("assistant"):

        # Retrieval status
        status = st.status(
            "Searching your document...",
            expanded=True
        )

        try:

            # Retrieve sources
            sources = retrieve_documents(
                st.session_state.vector_store,
                question
            )

            status.write(
                f"✓ Found {len(sources)} relevant sections"
            )

            status.write(
                "Generating answer with Qwen3..."
            )


            # -------------------------
            # STREAM ANSWER
            # -------------------------

            answer = st.write_stream(
                stream_answer(
                    st.session_state.vector_store,
                    question
                )
            )


            status.update(
                label="Answer generated",
                state="complete",
                expanded=False
            )


            # -------------------------
            # SOURCES
            # -------------------------

            if sources:

                with st.expander(
                    "View Sources"
                ):

                    for i, source in enumerate(
                        sources,
                        start=1
                    ):

                        page = source.metadata.get(
                            "page"
                        )

                        if page is not None:

                            page_text = (
                                f"Page {page + 1}"
                            )

                        else:

                            page_text = "Document"


                        st.markdown(
                            f"**Source {i} — "
                            f"{page_text}**"
                        )

                        st.write(
                            source.page_content[:500]
                        )


            # Save assistant response

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


        except Exception as e:

            status.update(
                label="Something went wrong",
                state="error"
            )

            answer = f"Something went wrong: {e}"

            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })