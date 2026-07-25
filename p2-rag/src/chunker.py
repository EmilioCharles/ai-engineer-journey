"""Stage 2: Split documents into overlapping chunks."""


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
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