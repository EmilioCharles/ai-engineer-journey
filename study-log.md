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

