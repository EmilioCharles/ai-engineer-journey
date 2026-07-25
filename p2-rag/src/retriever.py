"""Stage 3: Retrieve relevant chunks. Two backends behind one interface."""
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


class ChromaRetriever:
    """Semantic retrieval using vector embeddings."""

    def __init__(self, chunks):
        client = chromadb.PersistentClient(
            path=str(Path(__file__).parent.parent / "chroma_db")
        )
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = client.get_or_create_collection(
            "apexxtech", embedding_function=embed_fn
        )
        if self.collection.count() == 0:
            self.collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[{"source": c["source"]} for c in chunks],
            )

    def retrieve(self, question, k=4):
        results = self.collection.query(query_texts=[question], n_results=k)
        return [
            {"text": doc, "source": meta["source"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]


class BM25Retriever:
    """Keyword retrieval using BM25 ranking."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([self._tok(c["text"]) for c in chunks])

    def _tok(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def retrieve(self, question, k=4):
        scores = self.bm25.get_scores(self._tok(question))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            {"text": self.chunks[i]["text"], "source": self.chunks[i]["source"]}
            for i in top
        ]