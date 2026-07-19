import json
import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

app = FastAPI()

# ---- 1. LOAD (precomputed chunks) ----
CHUNKS = json.loads((Path(__file__).parent / "chunks.json").read_text(encoding="utf-8"))


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---- 2. INDEX (BM25 keyword retrieval — serverless-friendly) ----
bm25 = BM25Okapi([tokenize(c["text"]) for c in CHUNKS])


class Question(BaseModel):
    question: str


# ---- 3. RETRIEVE ----
def retrieve(question: str, k: int = 4):
    scores = bm25.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [CHUNKS[i] for i in top]


# ---- 4. ANSWER ----
def answer(question: str, retrieved):
    context = "\n\n---\n\n".join(f"[source: {r['source']}]\n{r['text']}" for r in retrieved)
    import anthropic

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


@app.post("/api/ask")
def ask(q: Question):
    hits = retrieve(q.question)
    result = {
        "retrieved": [{"source": h["source"], "preview": h["text"][:250]} for h in hits],
        "answer": None,
    }
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            result["answer"] = answer(q.question, hits)
        except Exception as e:
            result["answer"] = f"Retrieval OK; answer generation failed: {e}"
    else:
        result["answer"] = "ANTHROPIC_API_KEY not configured — showing retrieved passages only."
    return result


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ApexxTech Knowledge Base — RAG v0</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;background:#0f1117;color:#e6e6e6}
  h1{font-size:1.4rem} .sub{color:#8a8f98;margin-bottom:24px}
  form{display:flex;gap:8px} input{flex:1;padding:12px;border-radius:8px;border:1px solid #2a2f3a;background:#1a1e28;color:#fff;font-size:1rem}
  button{padding:12px 20px;border-radius:8px;border:0;background:#4f7cff;color:#fff;font-size:1rem;cursor:pointer}
  button:disabled{opacity:.5}
  .card{background:#171b24;border:1px solid #2a2f3a;border-radius:10px;padding:16px;margin-top:16px;white-space:pre-wrap;line-height:1.5}
  .src{color:#7ab7ff;font-size:.85rem;margin-top:12px} .chip{display:inline-block;background:#1f2733;border-radius:6px;padding:2px 8px;margin:2px}
  .ex{color:#8a8f98;font-size:.9rem;margin-top:10px} .ex span{cursor:pointer;text-decoration:underline}
</style></head><body>
<h1>ApexxTech Knowledge Base — RAG v0</h1>
<div class="sub">Ask questions about the MKA Agribusiness platform: spec, build plan, SOW, MSA, NDA, DPA. Built as part of the AI Engineer Roadmap (P2, Week 2).</div>
<form id="f"><input id="q" placeholder="e.g. What is out of scope for the MKA project?" autofocus>
<button id="b">Ask</button></form>
<div class="ex">Try: <span onclick="go('What happens if a feature fails its Definition of Done checklist?')">acceptance rules</span> ·
<span onclick="go('How does the offline mobile app handle sync conflicts?')">offline sync</span> ·
<span onclick="go('What are the client dependencies for the MKA project?')">client dependencies</span></div>
<div id="out"></div>
<script>
const f=document.getElementById('f'),q=document.getElementById('q'),b=document.getElementById('b'),out=document.getElementById('out');
function go(t){q.value=t;f.requestSubmit();}
f.onsubmit=async(e)=>{e.preventDefault();if(!q.value.trim())return;
b.disabled=true;out.innerHTML='<div class="card">Thinking…</div>';
try{const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})});
const d=await r.json();
let srcs=[...new Set(d.retrieved.map(x=>x.source))].map(s=>`<span class="chip">${s}</span>`).join('');
out.innerHTML=`<div class="card">${d.answer}</div><div class="src">Retrieved from: ${srcs}</div>`;
}catch(err){out.innerHTML=`<div class="card">Error: ${err}</div>`;}
b.disabled=false;};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
