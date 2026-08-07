"""Stage 3: Retrieve relevant chunks. Three backends behind one interface."""
import re
from pathlib import Path

import numpy as np
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


class HybridRetriever:
    """BM25 + dense vectors, fused with Reciprocal Rank Fusion.

    RRF needs no score normalization and no tuning: each result gets
    1/(rrf_k + rank) from each ranker, summed. It only uses rank order,
    not raw scores, which matters because BM25 scores and cosine
    similarities aren't on comparable scales.

    Over-retrieves `pool` from each ranker before fusing down to k. That
    also addresses the crowding problem seen when pricing.md was added:
    two new chunks reshuffled the top-4 on 22 of 50 eval pairs, because a
    4-slot budget has no slack. A 10-deep pool absorbs that.

    Uses precomputed "vec" fields when present (as built by
    build_chunks_snapshot.py) so the serverless path needs no vector DB.
    """

    def __init__(self, chunks, pool=10, rrf_k=60):
        self.chunks = chunks
        self.pool = pool
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([self._tok(c["text"]) for c in chunks])
        self._embed = embedding_functions.DefaultEmbeddingFunction()

        if "vec" in chunks[0]:
            self.vecs = np.array([c["vec"] for c in chunks], dtype=np.float32)
        else:
            self.vecs = np.array(
                self._embed([c["text"] for c in chunks]), dtype=np.float32
            )

        # normalize once so cosine similarity is just a dot product
        norms = np.linalg.norm(self.vecs, axis=1, keepdims=True)
        self.vecs = self.vecs / np.clip(norms, 1e-9, None)

    def _tok(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def _bm25_ranking(self, question):
        scores = self.bm25.get_scores(self._tok(question))
        return sorted(range(len(scores)), key=lambda i: -scores[i])[: self.pool]

    def _dense_ranking(self, question):
        q = np.array(self._embed([question])[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = self.vecs @ q
        return sorted(range(len(sims)), key=lambda i: -sims[i])[: self.pool]

    def retrieve(self, question, k=4):
        fused = {}
        for ranking in (self._bm25_ranking(question), self._dense_ranking(question)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            {"text": self.chunks[i]["text"], "source": self.chunks[i]["source"]}
            for i in top
        ]