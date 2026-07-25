"""Stage 1: Load documents from the docs folder."""
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent / "docs"


def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.glob("*")):
        if path.suffix in [".md", ".txt"]:
            text = path.read_text(encoding="utf-8")
            docs.append({"source": path.name, "text": text})
    return docs