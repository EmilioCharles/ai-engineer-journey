"""Retriever adapters — lets run_evals.py score either backend.

chroma : p2-rag/rag.py            (local dev, dense embeddings, 95 chunks)
bm25   : p2-rag/deploy/api/index.py (production on Vercel, frozen chunks.json)

Both expose the same pair of callables, so the runner stays backend-agnostic:
    retrieve(question, k) -> list[chunk dict]
    answer(question, retrieved) -> str

Note: the two paths do NOT see the same corpus. chunks.json is a frozen
snapshot and is currently missing pricing.md. That difference is the point
of running both.
"""
import sys
from pathlib import Path

P2 = Path(__file__).parent.parent


def get_backend(name):
    """Return (retrieve_fn, answer_fn, meta_dict) for the named retriever."""
    if name == "chroma":
        sys.path.insert(0, str(P2))
        from rag import load_documents, build_chunks, build_index
        from rag import retrieve as _retrieve, answer as _answer

        chunks = build_chunks(load_documents())
        col = build_index(chunks)

        def retrieve(question, k=4):
            return _retrieve(col, question, k=k)

        meta = {"retriever": "chroma", "chunks": len(chunks), "source": "rag.py"}
        return retrieve, _answer, meta

    if name == "bm25":
        sys.path.insert(0, str(P2 / "deploy" / "api"))
        import index as prod

        def retrieve(question, k=4):
            return prod.retrieve(question, k=k)

        meta = {"retriever": "bm25", "chunks": len(prod.CHUNKS),
                "source": "deploy/api/index.py"}
        return retrieve, prod.answer, meta

    raise SystemExit(f"unknown retriever: {name!r} (expected 'chroma' or 'bm25')")


if __name__ == "__main__":
    for name in ("chroma", "bm25"):
        try:
            _, _, meta = get_backend(name)
            print(meta)
        except Exception as e:
            print(f"{name}: FAILED — {e}")