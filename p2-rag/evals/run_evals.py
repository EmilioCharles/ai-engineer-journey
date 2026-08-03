"""P2 eval harness — runs the 50-pair set against the RAG and logs a scored run.

Usage from p2-rag/:   python evals\run_evals.py bm25

No LLM judge yet, deliberately. Rows that attempt an answer come back as
REVIEW for human labelling; those labels become the alignment set for a
judge in a later week.
"""
import json
import re
import sys
import time
from pathlib import Path
from retrievers import get_backend
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag import load_documents, build_chunks, build_index, retrieve, answer

EVALS = Path(__file__).parent
RUNS = EVALS / "runs"

REFUSAL = [
    "does not contain", "doesn't contain", "not contain",
    "no information", "not included", "not provided",
    "not available in", "cannot answer", "would need to see",
]


def validate(pairs):
    """Fail the whole run at load time rather than burning 50 calls on garbage."""
    errs = []
    for p in pairs:
        if "<" in p["question"] or ">" in p["question"]:
            errs.append(f"{p['id']}: unfilled placeholder in question")
        if not p.get("expected_behavior"):
            errs.append(f"{p['id']}: missing expected_behavior")
        if isinstance(p.get("expected_answer"), str) and "<" in p["expected_answer"]:
            errs.append(f"{p['id']}: unfilled placeholder in expected_answer")
    if errs:
        raise SystemExit("VALIDATION FAILED\n" + "\n".join(errs))


def looks_like_refusal(text):
    low = text.lower()
    return any(m in low for m in REFUSAL)


def score(pair, out):
    beh = pair["expected_behavior"]

    for bad in pair.get("must_not_contain", []):
        if bad.lower() in out.lower():
            return "FAIL", f"forbidden string: {bad}", None

    if beh == "refuse":
        return ("PASS", "refused", None) if looks_like_refusal(out) else \
               ("FAIL", "did not refuse", None)

    if pair.get("scoring") == "recall_at_item_level":
        want = {re.match(r"F\d+", i).group() for i in pair["expected_answer"]}
        got = set(re.findall(r"\bF\d+\b", out))
        r = len(want & got) / len(want)
        return ("PASS" if r == 1.0 else "FAIL"), f"recall {len(want & got)}/{len(want)}", round(r, 3)

    # expected an answer
    if looks_like_refusal(out):
        return "FAIL", "refused when answer expected", None
    return "REVIEW", "answered — needs human label", None


def main(retriever):
    pairs = json.loads((EVALS / "eval_set.json").read_text(encoding="utf-8"))
    validate(pairs)

    retrieve, answer, meta = get_backend(retriever)
    print(f"backend: {meta}")
    rows, t0 = [], time.time()

    for p in pairs:
        got = retrieve(p["question"], k=4)
        out = answer(p["question"], got)
        verdict, reason, metric = score(p, out)
        rows.append({
            "id": p["id"],
            "category": p.get("category", "uncategorised"),
            "retriever": retriever,
            "question": p["question"],
            "expected_behavior": p["expected_behavior"],
            "sources": [c.get("source") for c in got],
            "output": out,
            "verdict": verdict,
            "reason": reason,
            "metric": metric,
        })
        print(f"{p['id']:>5}  {verdict:<6}  {reason}")

    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS / f"{stamp}_{retriever}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], {"PASS": 0, "FAIL": 0, "REVIEW": 0})
        c[r["verdict"]] += 1

    print(f"\nretriever={retriever}  {len(rows)} pairs  {time.time() - t0:.0f}s")
    print(f"{'category':<32} {'P':>3} {'F':>3} {'R':>3}")
    for c, v in sorted(cats.items()):
        print(f"{c:<32} {v['PASS']:>3} {v['FAIL']:>3} {v['REVIEW']:>3}")
    tot = {k: sum(v[k] for v in cats.values()) for k in ("PASS", "FAIL", "REVIEW")}
    print(f"{'TOTAL':<32} {tot['PASS']:>3} {tot['FAIL']:>3} {tot['REVIEW']:>3}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "unknown")