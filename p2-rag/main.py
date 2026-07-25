"""RAG pipeline — wires the five stages together.

load -> chunk -> retrieve -> rerank (stub) -> generate
"""
from src.loader import load_documents
from src.chunker import build_chunks
from src.retriever import ChromaRetriever, BM25Retriever
from src.reranker import PassthroughReranker
from src.generator import Generator

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def build_pipeline(retriever_kind="chroma"):
    chunks = build_chunks(load_documents())
    print(f"loaded {len(chunks)} chunks")
    if retriever_kind == "bm25":
        retriever = BM25Retriever(chunks)
    else:
        retriever = ChromaRetriever(chunks)
    return retriever, PassthroughReranker(), Generator()


def ask(question, retriever, reranker, generator, k=4):
    hits = retriever.retrieve(question, k=k)
    hits = reranker.rerank(question, hits, k=k)
    return hits, generator.answer(question, hits)


if __name__ == "__main__":
    retriever, reranker, generator = build_pipeline("chroma")
    while True:
        q = input("\nask> ").strip()
        if q in ("exit", "quit", ""):
            break
        hits, response = ask(q, retriever, reranker, generator)
        print("retrieved:", [h["source"] for h in hits])
        print("\n" + response)