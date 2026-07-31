"""One-off: add expected_behavior to any eval pair missing it.

Run once from p2-rag/:   python evals\backfill.py

The inference is crude on purpose — read the printed list and hand-fix
anything that should be answer_partial_and_flag_gap or recall_at_item_level.
"""
import json
from pathlib import Path

p = Path(__file__).parent / "eval_set.json"
pairs = json.loads(p.read_text(encoding="utf-8"))

review = []
for pair in pairs:
    if "expected_behavior" not in pair:
        ea = pair.get("expected_answer")
        pair["expected_behavior"] = "refuse" if ea in (None, "", []) else "answer"
        review.append(f"{pair['id']}: -> {pair['expected_behavior']}  |  {pair['question'][:70]}")

p.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"{len(pairs)} pairs, {len(review)} backfilled\n")
print("\n".join(review))