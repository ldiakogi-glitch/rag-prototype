"""Online phase: ask a question, get a grounded answer with sources.
Usage: python ask.py "your question"   (or run with no args for a prompt loop)
"""
import sys
from pathlib import Path

from rag_core import answer


def show(q: str):
    r = answer(q)
    print(f"\n{r['answer']}\n")
    print(f"(answered by {r['backend']}) Retrieved chunks:")
    for d in r["sources"]:
        print(f"  - {Path(d.metadata['source']).name}, p.{d.metadata.get('page', 0) + 1}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        while (q := input("\nQuestion (blank to quit): ").strip()):
            show(q)
