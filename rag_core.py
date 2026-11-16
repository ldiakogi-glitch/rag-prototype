"""Shared config + retrieval/generation logic for the RAG prototype."""
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---- Tunables (M3: test these per the risk-mitigation plan) ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3                      # try 3 vs 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
OLLAMA_MODEL = "llama3"        # fallback; run `ollama pull llama3`
DB_DIR = str(Path(__file__).parent / "chroma_db")

SYSTEM_PROMPT = (
    "Answer using only the context provided to you. "
    "If the answer is not present in the context, explicitly state that you do not know. "
    "Cite sources inline like [source: filename, p.N]."
)


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name="course_docs",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


def format_context(docs) -> str:
    parts = []
    for d in docs:
        src = Path(d.metadata.get("source", "unknown")).name
        page = d.metadata.get("page", 0) + 1
        parts.append(f"[source: {src}, p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _groq():
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=0)  # needs GROQ_API_KEY


def _ollama():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


def generate(question: str, context: str) -> tuple[str, str]:
    """Try Groq first; fall back to local Ollama (Risk 3 mitigation)."""
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    try:
        return _groq().invoke(messages).content, "groq"
    except Exception as e:
        print(f"[warn] Groq failed ({type(e).__name__}); falling back to Ollama")
        return _ollama().invoke(messages).content, "ollama"


def answer(question: str, k: int = TOP_K) -> dict:
    db = get_vectorstore()
    docs = db.similarity_search(question, k=k)
    text, backend = generate(question, format_context(docs))
    return {"answer": text, "sources": docs, "backend": backend}
