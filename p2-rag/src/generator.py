"""Stage 5: Generate a cited answer from retrieved chunks using an LLM."""
import anthropic


class Generator:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def answer(self, question, chunks):
        context = "\n\n---\n\n".join(
            f"[source: {c['source']}]\n{c['text']}" for c in chunks
        )
        msg = self.client.messages.create(
            model=self.model,
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