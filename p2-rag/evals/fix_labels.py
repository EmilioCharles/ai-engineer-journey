"""Relabel expected_behavior using a narrow, auditable rule.

repair.py scanned gold answers for refusal phrases anywhere in the text and
flipped ~35 answerable pairs to "refuse". This uses a much tighter rule:
a pair is only "refuse" if its gold answer is ENTIRELY a statement that the
corpus lacks the information — not if it merely mentions a gap in passing.

Prints every change for review. Run from p2-rag/:
    python evals\fix_labels.py
"""
import json
from pathlib import Path

ABSENCE_OPENERS = (
    "the documents do not contain",
    "the documents don't contain",
    "the context does not contain",
    "the documents do not include",
    "no information",
)

p = Path(__file__).parent / "eval_set.json"
pairs = json.loads(p.read_text(encoding="utf-8"))

changed, needs_gold, still_refuse = [], [], []

for pair in pairs:
    # leave the hand-written Q41-Q50 behaviors alone
    if str(pair["id"]).startswith("Q"):
        continue

    gold = pair.get("gold_answer")

    if gold in (None, "", [], {}):
        needs_gold.append((pair["id"], pair["question"]))
        continue

    g = str(gold).strip().lower()
    # refuse only if the WHOLE answer is a statement of absence:
    # starts with an absence phrase and is short enough to be nothing else
    is_refusal = g.startswith(ABSENCE_OPENERS) and len(g) < 160

    new = "refuse" if is_refusal else "answer"
    if pair["expected_behavior"] != new:
        changed.append((pair["id"], pair["expected_behavior"], new, pair["question"]))
    pair["expected_behavior"] = new
    if new == "refuse":
        still_refuse.append((pair["id"], pair["question"]))

p.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"changed: {len(changed)}\n")
for pid, old, new, q in changed:
    print(f"  {str(pid):>4}  {old:<7} -> {new:<7}  {q[:60]}")

print(f"\nstill marked refuse ({len(still_refuse)}) — verify these are truly unanswerable:")
for pid, q in still_refuse:
    print(f"  {str(pid):>4}  {q[:70]}")

print(f"\nEMPTY gold_answer ({len(needs_gold)}) — these need writing before they can score:")
for pid, q in needs_gold:
    print(f"  {str(pid):>4}  {q[:70]}")