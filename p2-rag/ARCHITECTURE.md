# ApexxTech RAG — Architecture

A retrieval-augmented generation system over the ApexxTech knowledge base.
Refactored from a single script into five swappable stages (Week 3).

## Pipeline

load → chunk → retrieve → rerank → generate

| Stage | Module | Responsibility |
|-------|--------|----------------|
| 1. Load | `src/loader.py` | Read .md/.txt docs from `docs/` |
| 2. Chunk | `src/chunker.py` | Split into 800-char chunks, 150 overlap |
| 3. Retrieve | `src/retriever.py` | Find relevant chunks — Chroma (semantic) or BM25 (keyword) |
| 4. Rerank | `src/reranker.py` | STUB (passthrough). Real reranker in Week 6 |
| 5. Generate | `src/generator.py` | LLM answer with source citations |

`main.py` wires the stages and runs the interactive loop.

## Design decisions

- **Two retrievers behind one interface.** ChromaRetriever and BM25Retriever
  share the same `.retrieve(question, k)` signature, so they're swappable.
  This lets the eval harness (Week 4) compare them and enables hybrid
  search (Week 5).
- **Reranker as a stub.** The slot and interface exist now so Week 6 can
  drop in a real cross-encoder without touching the rest of the pipeline.
- **Config in the entry point.** `main.py` loads `.env`; modules stay
  pure and don't reach for environment or filesystem beyond their job.

## Where evals plug in

The eval harness (Week 4) imports the pipeline, swaps the retriever,
runs a labeled question set, and scores precision@k + LLM-judge.
Modular stages are what make that measurement possible.

## Deployed variant

`deploy/` runs a serverless version on Vercel using BM25 only (the
embedding model is too large for serverless). Live at apexxtech-rag.vercel.app.