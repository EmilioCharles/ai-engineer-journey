"""Build deploy/api/chunks.json from the live docs/ corpus.

Production can't chunk at request time on Vercel, so it ships a pre-built
JSON snapshot. That snapshot has no build step, which is how it drifted to
93 chunks while docs/ moved to 95 (pricing.md was never included).

Run from p2-rag/ after ANY change to docs/:
    python build_chunks_snapshot.py
"""
import json
from pathlib import Path

from src.loader import load_documents
from src.chunker import build_chunks

HERE = Path(__file__).parent
OUT = HERE / "deploy" / "api" / "chunks.json"

chunks = build_chunks(load_documents())

# production reads c["text"] and c["source"] only
payload = [{"id": c["id"], "text": c["text"], "source": c["source"]} for c in chunks]

old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

from collections import Counter
print(f"was {len(old)} chunks -> now {len(payload)} chunks")
for src, n in Counter(c["source"] for c in payload).most_common():
    print(f"  {n:3d}  {src}")
print(f"\nwrote {OUT}")