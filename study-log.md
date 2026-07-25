\## Mon May 18

\- \*\*Did:\*\* Initialized ai-engineer-journey repo. Python 3.11 venv, gitignore-first discipline, pinned 160 dependencies in requirements.txt, scaffolded hello\_world.py with proper secrets loading and error handling. Verified .env is gitignored. Three commits on main.

\- \*\*Learned:\*\* \[your one-sentence takeaway from whatever you read today — even if it's just "AI engineering's central question is application-level reliability, not model performance" or similar from your own thinking, since you haven't done the Chip Huyen reading yet]

\- \*\*Stuck:\*\* API credits not yet funded — Anthropic + OpenAI setup deferred. Will catch up Friday or Saturday once funds land. Nothing blocking conceptual/reading work.

## Tue May 19

* **Did:** Read Chip Huyen Ch. 1. Self-tested three core concepts: ML vs AI engineering distinction, central engineering challenge of foundation models, and the first tradeoff I'll face on this journey. Integrated ByteByteAI course outline as Phase 2 checklist (no restructure, surgical addition only).
* **Learned:** ML engineering trains models on raw data; AI engineering builds systems *around* trained models to solve real-world problems reliably. The central challenge foundation models introduce is unpredictability — same input can produce different outputs — which breaks traditional software engineering's determinism. This forces a new discipline of measurement, evaluation, and architectural levers (retrieval, prompting, fine-tuning) to engineer reliability into a probabilistic system. My EMU coursework gives me the foundation (algebra, algorithms, Python, Django, HCI) but not the application layer — that's what these 28 weeks are filling. HCI specifically applies directly: a model that performs on benchmarks but confuses users in production is a failed AI system. First tradeoff I'll face is cost vs. performance — with a $10 budget I can't afford flagship models (Opus, GPT-4o), so I'm constrained to cheaper ones (Haiku, GPT-4o-mini) and have to learn better retrieval, prompting, and evals as the levers instead of bigger models.
* **Stuck:** None on the conceptual level. Ch. 1 landed cleanly. Ready for Ch. 2 with three anchor questions: what a foundation model is mechanically, the data/architecture/objectives trio, and why post-training matters.













\## Jul 18 — Week 2, Saturday

Did: Built P2 RAG v0 end-to-end — 9-doc ApexxTech corpus (pulled from Drive,

sanitized), loader → 800-char chunker w/ overlap → Chroma embeddings (95 chunks,

local MiniLM) → top-4 retrieval → Haiku answers with source citations. Verified:

pricing question retrieved pricing.md + SOW and answered correctly with cites.

Also: repo went public, P2 domain locked, API credits funded.

Learned: venv activation is per-terminal; retrieval and generation are

independently swappable stages; empty files / wrong directories caused every

error today — check the prompt path first.

Stuck: Vercel deploy not attempted — moved to Sunday morning by design.



\## Jul 19 — Week 2, Sunday (pre-audit)

Did: Deployed RAG v0 to Vercel — live at https://apexxtech-rag.vercel.app

Serverless variant uses BM25 retrieval (rank\_bm25) instead of Chroma vectors

(80MB embed model unfit for serverless); local version keeps semantic search.

Verified live: DoD question answered with citations from SOW+MSA. Env var flow,

CLI deploy, cold starts — all new today.



\## July 20 — Week 2 audit, Monday

Did: Drew the transformer from memory. First attempt came out as RAG

architecture instead of the transformer — I had blurred the system I

built around the model with the model itself. Redrew it correctly

after the correction.



The flow: TOKENS (breaking input into chunks, e.g. words/characters)

\--> EMBEDDING (converting chunks into numbers positioned in a

higher-dimensional space, plus position info so order is known)

\--> \[ATTENTION + MLP] x N (attention: each token takes a weighted

average of the other tokens, weighted by relevance — communication;

MLP: each token then processes what it gathered — computation.

Each sub-layer has a skip arrow: its result is ADDED to an untouched

copy of its input (x + f(x)), so layers make edits, not rewrites,

and the signal + training gradient survive N blocks deep)

\--> LOGITS (final layer; unnormalised scores per vocabulary token —

positive, negative or zero)

\--> SOFTMAX (scores converted into probabilities)

\--> NEXT TOKEN (sampled; append and repeat).



\--> ATTENTION MECHANISM

This is the layer where tokens exchange information. Each token looks

back at the tokens before it and asks which of them are relevant to

understanding itself in this context. It does this by broadcasting what

it is looking for (its query) and comparing that against what every

earlier token advertises that it contains (its keys); how well those

match becomes a relevance score. Those scores decide the mix: the token

takes a weighted average of the other tokens' content (their values),

absorbing a lot from the relevant ones and almost nothing from the

irrelevant ones. It leaves the layer with the same shape it came in

with, but now enriched by context — "bank" after "river" carries

different meaning than "bank" after "money". Attention itself never

picks a word; it only sharpens each token's representation. The actual

next-token choice happens later at logits + softmax.





\## July 26th— Week 3, eval seed pairs

Read Hamel "Your AI Product Needs Evals" + cross-examined. Then wrote 5

eval questions and tested them against the live RAG:



1\. What is ApexxTech? — PASS (correct)

2\. What services does ApexxTech offer? — WEAK: answer incomplete,

&#x20;  didn't retrieve services.md (retrieval miss)

3\. Who founded MKA? — PASS (correct: "two brothers, unnamed" — good faithfulness)

4\. Is there a discount? — PASS (correctly refused, no hallucination)

5\. What are the payment terms? — FAIL: said "not in documents" but MSA

&#x20;  Section 5 has them (14-day payment, 1.5%/mo interest, 21-day suspension).

&#x20;  Retrieval miss, not generation.



Finding: 2 retrieval misses (Q2, Q5), both on cross-document questions.

Single-fact and refusal questions pass. This is what hybrid search /

reranking (Weeks 5-6) should fix. These 5 become Week 4 labeled pairs.

