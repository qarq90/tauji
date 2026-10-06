import os
from typing import Generator

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# OLLAMA SETTINGS
# -----------------------------

LLM_MODEL = "qwen3:4b"
EMBEDDING_MODEL = "nomic-embed-text"


# -----------------------------
# LOAD DOCUMENT
# -----------------------------

def load_document(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF or DOCX."
        )

    return loader.load()


# -----------------------------
# CREATE VECTOR STORE
# -----------------------------

def create_vector_store(file_path, progress_callback=None):

    # 1. Load document
    if progress_callback:
        progress_callback("Reading document...")

    documents = load_document(file_path)

    if progress_callback:
        progress_callback(
            f"✓ Extracted {len(documents)} document pages/sections"
        )


    # 2. Split document
    if progress_callback:
        progress_callback("Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if progress_callback:
        progress_callback(
            f"Created {len(chunks)} text chunks"
        )


    # 3. Create embeddings
    if progress_callback:
        progress_callback(
            "Creating local embeddings..."
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # 4. Create FAISS database
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    if progress_callback:
        progress_callback(
            "Document indexed successfully"
        )

    return vector_store


# -----------------------------
# CREATE LLM
# -----------------------------

def get_llm():

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )


# -----------------------------
# RETRIEVE DOCUMENTS
# -----------------------------

def retrieve_documents(
    vector_store,
    question
):

    documents = vector_store.similarity_search(
        question,
        k=4
    )

    return documents


# -----------------------------
# STREAM ANSWER
# -----------------------------

def stream_answer(
    vector_store,
    question
):

    # Retrieve relevant chunks
    documents = retrieve_documents(
        vector_store,
        question
    )

    # Build context
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are Tauji, an AI assistant that answers
questions about uploaded documents.

IMPORTANT RULES:

1. Answer ONLY using the provided document context.
2. Do not invent information.
3. If the answer cannot be found in the context,
   say:

"I couldn't find that information in the document."

4. Be clear and concise.
5. If the user asks for a list, provide a list.
6. If the user asks for a summary, provide a summary.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""
    )

    llm = get_llm()

    chain = prompt | llm

    # Stream response
    for chunk in chain.stream({
        "context": context,
        "question": question
    }):

        if chunk.content:
            yield chunk.content


# -----------------------------
# NORMAL ANSWER
# -----------------------------

def ask_question(
    vector_store,
    question
):

    documents = retrieve_documents(
        vector_store,
        question
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are Tauji, an AI assistant that answers
questions about uploaded documents.

Answer ONLY using the document context.

If the answer cannot be found, say:

"I couldn't find that information in the document."

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""
    )

    llm = get_llm()

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, documents