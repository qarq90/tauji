
import os
from time import perf_counter

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LLM_MODEL = "qwen3:4b"


def load_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Upload a PDF or DOCX file.")

    return loader.load()


def create_vector_store(file_path, progress_callback=None):
    start_time = perf_counter()

    def report(message):
        if progress_callback:
            progress_callback(message)

    report("Reading document...")
    documents = load_document(file_path)

    if not documents:
        raise ValueError("No document content was extracted.")

    report(f"Extracted {len(documents)} pages/sections")
    report("Splitting document into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
    )

    chunks = [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No readable text was found.")

    report(f"Created {len(chunks)} text chunks")
    report("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    texts = [chunk.page_content for chunk in chunks]

    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as error:
        raise ValueError(
            "Could not build the index. The document may contain "
            "too little usable text."
        ) from error

    report(
        f"Index ready: {matrix.shape[0]} chunks, "
        f"{matrix.shape[1]} features"
    )
    report(f"Indexing completed in {perf_counter() - start_time:.2f}s")

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def retrieve_documents(vector_store, question, k=4):
    chunks = vector_store["chunks"]
    vectorizer = vector_store["vectorizer"]
    matrix = vector_store["matrix"]

    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        matrix,
    ).flatten()

    ranked_indices = scores.argsort()[::-1]
    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            break

        results.append(chunks[index])

        if len(results) >= k:
            break

    return results


def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


def stream_answer(documents, question, is_summary=False):
    if not documents:
        yield "I couldn't find relevant information in the document."
        return

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    if is_summary:
        instructions = """
Summarize the provided document excerpts.
Explain the main topic and important points.
Use only information supported by the excerpts.
If the excerpts cover only part of the document,
make that limitation clear.
"""
    else:
        instructions = """
Answer the user's question using only the provided context.
Do not invent facts.
If the answer is not supported by the context, say:
"I couldn't find that information in the document."
Be clear and concise.
Use lists when appropriate.
"""

    prompt = ChatPromptTemplate.from_template(
        """
You are Tauji, a document question-answering assistant.

INSTRUCTIONS:
{instructions}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
    )

    chain = prompt | get_llm()

    for chunk in chain.stream({
        "instructions": instructions,
        "context": context,
        "question": question,
    }):
        if chunk.content:
            yield chunk.content
