import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DOCS_DIR = Path(__file__).parent / "docs"


# ---- 1. LOAD ----
def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix in [".md", ".txt"]:
            text = path.read_text(encoding="utf-8")
            docs.append({"source": path.name, "text": text})
    return docs


# ---- 2. CHUNK ----
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step back so chunks share an overlap
    return chunks


def build_chunks(docs):
    all_chunks = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append({
                "id": f'{doc["source"]}-{i}',
                "source": doc["source"],
                "text": chunk,
            })
    return all_chunks


# ---- 3. EMBED + STORE ----
def build_index(chunks):
    client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db"))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()  # local model, no API cost
    collection = client.get_or_create_collection("apexxtech", embedding_function=embed_fn)
    if collection.count() == 0:  # only embed once; delete chroma_db/ to rebuild
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        print(f"embedded {collection.count()} chunks")
    else:
        print(f"index already built ({collection.count()} chunks)")
    return collection


# ---- 4. RETRIEVE ----
def retrieve(collection, question, k=4):
    results = collection.query(query_texts=[question], n_results=k)
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


# ---- 5. ANSWER ----
def answer(question, retrieved):
    context = "\n\n---\n\n".join(
        f"[source: {r['source']}]\n{r['text']}" for r in retrieved
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "Answer the question using ONLY the context below. "
                "Cite the source filename for claims. "
                "If the context doesn't contain the answer, say so.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION: {question}"
            ),
        }],
    )
    return msg.content[0].text


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No corpus files in {DOCS_DIR}")
    chunks = build_chunks(docs)
    print(f"loaded {len(docs)} documents, {len(chunks)} chunks")
    collection = build_index(chunks)

    while True:
        q = input("\nask> ").strip()
        if q in ("exit", "quit", ""):
            break
        hits = retrieve(collection, q)
        print("retrieved:", [h["source"] for h in hits])
        print("\n" + answer(q, hits))