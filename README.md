# RAG Q&A Prototype

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here     # free key at console.groq.com
```

## Run
1. Drop PDFs (syllabi, papers, manuals) into `docs/`
2. `python ingest.py`            # builds chroma_db/
3. `python ask.py "What is the late policy?"`
4. Fill in `questions.json`, then `python evaluate.py 3 5` to compare k=3 vs k=5

## Fallback
If Groq errors or rate-limits, `rag_core.generate` falls back to Ollama
(`ollama pull llama3` first).

## Files
- `rag_core.py` - config (chunk size, k, models), retrieval, generation
- `ingest.py` - PDF -> chunks -> embeddings -> ChromaDB
- `ask.py` - CLI Q&A with source citations
- `evaluate.py` - retrieval hit-rate harness
