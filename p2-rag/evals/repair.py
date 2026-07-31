"""Repair the bad backfill and unify the two schemas in eval_set.json.

Pairs 1-40 use: gold_answer / expected_sources / type
Pairs Q41-Q50 use: expected_answer / source_docs / category

This adds the new-style keys as aliases on the old pairs (originals kept,
nothing deleted) and recomputes expected_behavior from the correct field.

Run once from p2-rag/:   python evals\repair.py
"""
import json
from pathlib import Path

REFUSAL_MARKERS = [
    "does not contain", "doesn't contain", "not contain",
    "no information", "not included", "not provided",
    "not available", "cannot answer", "not in the",
    "out of scope", "declines", "should refuse",
]

p = Path(__file__).parent / "eval_set.json"
pairs = json.loads(p.read_text(encoding="utf-8"))

changed = []
for pair in pairs:
    # --- alias old keys to new ones so one runner reads all 50 ---
    if "gold_answer" in pair and "expected_answer" not in pair:
        pair["expected_answer"] = pair["gold_answer"]
    if "expected_sources" in pair and "source_docs" not in pair:
        pair["source_docs"] = pair["expected_sources"]
    if "type" in pair and "category" not in pair:
        pair["category"] = pair["type"]

    # --- recompute expected_behavior from the real answer field ---
    ea = pair.get("expected_answer")
    if ea in (None, "", [], {}):
        behavior = "refuse"
    elif isinstance(ea, str) and any(m in ea.lower() for m in REFUSAL_MARKERS):
        behavior = "refuse"
    else:
        behavior = "answer"

    # never clobber the hand-written compound behaviorspatch_q42.py
    if pair.get("expected_behavior") in ("answer_partial_and_flag_gap",):
        behavior = pair["expected_behavior"]

    if pair.get("expected_behavior") != behavior:
        changed.append((pair["id"], pair.get("expected_behavior"), behavior,
                        pair["question"][:60]))
    pair["expected_behavior"] = behavior

p.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

counts = {}
for pair in pairs:
    counts[pair["expected_behavior"]] = counts.get(pair["expected_behavior"], 0) + 1

print(f"{len(pairs)} pairs, {len(changed)} corrected\n")
for pid, old, new, q in changed:
    print(f"{str(pid):>5}  {str(old):<8} -> {new:<8}  {q}")
print("\nfinal distribution:", counts)
print("\nREVIEW the 'refuse' rows above — any that should be answerable means")
print("the gold_answer text tripped a refusal marker and needs a hand fix.")