"""Build deploy/api/chunks.json from the live docs/ corpus.

Production can't chunk or embed at request time on Vercel, so it ships a
pre-built JSON snapshot. That snapshot had no build step, which is how it
drifted to 93 chunks while docs/ moved to 95 (pricing.md was never included).

Now also precomputes dense vectors (all-MiniLM-L6-v2, 384 dims) so the
serverless function can do hybrid retrieval with numpy — no vector DB,
no network call, no cold-start model load.

Run from p2-rag/ after ANY change to docs/:
    python build_chunks_snapshot.py
"""
import json
from collections import Counter
from pathlib import Path

from chromadb.utils import embedding_functions

from src.chunker import build_chunks
from src.loader import load_documents

HERE = Path(__file__).parent
OUT = HERE / "deploy" / "api" / "chunks.json"

chunks = build_chunks(load_documents())
print(f"chunked {len(chunks)} chunks, embedding...")

embed_fn = embedding_functions.DefaultEmbeddingFunction()
vectors = embed_fn([c["text"] for c in chunks])

payload = [
    {
        "id": c["id"],
        "text": c["text"],
        "source": c["source"],
        "vec": [round(float(x), 5) for x in v],
    }
    for c, v in zip(chunks, vectors)
]

old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

print(f"\nwas {len(old)} chunks -> now {len(payload)} chunks")
print(f"vector dims: {len(payload[0]['vec'])}")
for src, n in Counter(c["source"] for c in payload).most_common():
    print(f"  {n:3d}  {src}")
print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")