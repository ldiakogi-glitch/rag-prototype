"""Offline phase: PDFs -> chunks -> embeddings -> ChromaDB.
Usage: python ingest.py [docs_dir]
Each run wipes and rebuilds chroma_db/, so keep ALL your PDFs in docs/.
"""
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core import CHUNK_OVERLAP, CHUNK_SIZE, DB_DIR, get_vectorstore


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)        # ligatures: ﬁ -> fi, ﬀ -> ff
    text = re.sub(r"CS 385 Fall 20\d\d\s*", "", text)  # repeated page header
    text = re.sub(r"Page \d+\s*", "", text)            # repeated page footer
    return text


def main(docs_dir: str = "docs"):
    pdfs = sorted(Path(docs_dir).rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {docs_dir}/ - add some and rerun.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for pdf in pdfs:
        pages = PyPDFLoader(str(pdf)).load()
        for p in pages:
            p.page_content = clean(p.page_content)
        chunks.extend(splitter.split_documents(pages))
        print(f"loaded {pdf.name}: {len(pages)} pages")

    shutil.rmtree(DB_DIR, ignore_errors=True)  # start clean: no duplicate chunks
    db = get_vectorstore()
    db.add_documents(chunks)
    print(f"stored {len(chunks)} chunks in ChromaDB")


if __name__ == "__main__":
    main(*sys.argv[1:2])
