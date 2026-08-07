"""Retriever adapters — lets run_evals.py score any backend.

chroma : p2-rag/rag.py              (local dev, Chroma dense, live docs/)
bm25   : p2-rag/deploy/api/index.py (production on Vercel, frozen chunks.json)
hybrid : src/retriever.py           (BM25 + precomputed dense, fused via RRF)

All three expose the same pair of callables, so the runner stays
backend-agnostic:
    retrieve(question, k) -> list[chunk dict]
    answer(question, retrieved) -> str

chunks.json is a build artifact, not a live read — regenerate it with
build_chunks_snapshot.py after any change to docs/. It drifted to 93 chunks
once while docs/ was at 95, which is why that script exists.
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

    if name == "hybrid":
        sys.path.insert(0, str(P2))
        import json as _json
        from src.retriever import HybridRetriever
        from src.generator import Generator

        chunks = _json.loads(
            (P2 / "deploy" / "api" / "chunks.json").read_text(encoding="utf-8")
        )
        r = HybridRetriever(chunks)
        gen = Generator()

        def retrieve(question, k=4):
            return r.retrieve(question, k=k)

        def answer(question, retrieved):
            return gen.answer(question, retrieved)

        meta = {"retriever": "hybrid", "chunks": len(chunks),
                "source": "src/retriever.py HybridRetriever (RRF)"}
        return retrieve, answer, meta

    raise SystemExit(
        f"unknown retriever: {name!r} (expected 'chroma', 'bm25' or 'hybrid')"
    )


if __name__ == "__main__":
    for name in ("chroma", "bm25", "hybrid"):
        try:
            _, _, meta = get_backend(name)
            print(meta)
        except Exception as e:
            print(f"{name}: FAILED — {e}")