"""Patch Q42: fill the placeholder expected_answer and fix must_not_contain.

The SOW defines acceptance criteria AS the Definition of Done checklist, so
"Definition of Done" and "DoD" in must_not_contain would fail a correct answer.

Run once from p2-rag/:   python evals\patch_q42.py
"""
import json
from pathlib import Path

p = Path(__file__).parent / "eval_set.json"
pairs = json.loads(p.read_text(encoding="utf-8"))

found = False
for pair in pairs:
    if str(pair.get("id")) == "Q42":
        pair["expected_answer"] = (
            "Acceptance criteria for each feature are its Definition of Done "
            "checklist in Exhibit B of the SOW. Acceptance is binary: either every "
            "checklist item is verified, or the feature is rejected with the "
            "specific unmet items listed. The Review Period and deemed-acceptance "
            "rules of the MSA apply."
        )
        pair["must_not_contain"] = ["compose up runs locally", "CI green"]
        pair["notes"] = (
            "Distractor is mka-build-plan.md, which carries per-feature DoD language "
            "across 27 chunks vs the SOW's 9. Engineering done-ness is not contractual "
            "acceptance. Tests precision, not recall. Verified PASS at baseline on BM25."
        )
        pair["expected_behavior"] = "answer"
        found = True

if not found:
    raise SystemExit("Q42 not found in eval_set.json")

p.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

# sanity sweep for any other unfilled placeholders
bad = []
for pair in pairs:
    for key in ("question", "expected_answer"):
        v = pair.get(key)
        if isinstance(v, str) and "<" in v and ">" in v:
            bad.append(f"{pair['id']}: {key}")

print("Q42 patched.")
print(f"remaining placeholders: {bad if bad else 'none'}")