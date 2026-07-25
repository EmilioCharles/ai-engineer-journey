"""Stage 4: Rerank retrieved chunks. STUB — passthrough for now.

Week 6 replaces this with a real cross-encoder reranker
(e.g. bge-reranker-base or Cohere rerank). For now it returns
the chunks unchanged, so the pipeline has the slot wired in.
"""


class PassthroughReranker:
    """Returns chunks in the order retrieval gave them. No-op."""

    def rerank(self, question, chunks, k=4):
        return chunks[:k]