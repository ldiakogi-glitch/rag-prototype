"""Retrieval hit-rate harness.
Questions with "answerable": false are skipped here - they test hallucination,
so run them through `python ask.py` and check that the model says it doesn't know.
A 'hit' = at least one of the top-k chunks comes from expected_source
AND contains expected_keyword (whitespace-insensitive, case-insensitive).

Usage: python evaluate.py [k ...] [-v]
  e.g. python evaluate.py 3 5
       python evaluate.py 3 -v     # explain each MISS
"""
import json
import sys
import unicodedata
from pathlib import Path

from rag_core import get_vectorstore


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).lower().split())


def explain_miss(item, docs, all_chunks):
    kw = norm(item["expected_keyword"])
    homes = [
        f"p.{m.get('page', 0) + 1}"
        for text, m in all_chunks
        if kw in norm(text)
    ]
    if homes:
        print(f"         keyword '{item['expected_keyword']}' IS in the DB ({', '.join(homes)})"
              " -> real retrieval miss")
    else:
        print(f"         keyword '{item['expected_keyword']}' NOT in any chunk"
              " -> extraction/keyword problem, check the wording")
    for d in docs:
        preview = " ".join(d.page_content.split())[:90]
        print(f"         got p.{d.metadata.get('page', 0) + 1}: {preview}...")


def hit_rate(questions, k, verbose=False):
    db = get_vectorstore()
    data = db.get()
    all_chunks = list(zip(data["documents"], data["metadatas"]))
    hits = 0
    for item in questions:
        docs = db.similarity_search(item["question"], k=k)
        ok = any(
            Path(d.metadata["source"]).name == item["expected_source"]
            and norm(item["expected_keyword"]) in norm(d.page_content)
            for d in docs
        )
        hits += ok
        print(f"  [{'HIT ' if ok else 'MISS'}] {item['question']}")
        if verbose and not ok:
            explain_miss(item, docs, all_chunks)
    return hits / len(questions)


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    ks = [int(a) for a in sys.argv[1:] if a.isdigit()] or [3, 5]
    all_q = json.loads(Path("questions.json").read_text())
    questions = [q for q in all_q if q.get("answerable", True)]
    print(f"{len(questions)} answerable questions "
          f"({len(all_q) - len(questions)} unanswerable skipped)")
    for k in ks:
        print(f"\n== top-k = {k} ==")
        print(f"hit rate: {hit_rate(questions, k, verbose):.0%}")
