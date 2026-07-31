"""Compare two eval runs to locate the source of run-to-run variance.

Splits disagreements into two buckets:
  - retrieval drift  : the same question pulled different chunks
  - generation drift : identical chunks, different verdict

Run from p2-rag/ with the two most recent run files:
    python evals\compare_runs.py evals\runs\A.json evals\runs\B.json

With no arguments it picks the two newest runs automatically.
"""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).parent / "runs"


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in rows}


def main():
    if len(sys.argv) >= 3:
        a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    else:
        files = sorted(RUNS.glob("*.json"))
        if len(files) < 2:
            raise SystemExit("need at least two run files in evals/runs/")
        a_path, b_path = files[-2], files[-1]

    print(f"A = {a_path.name}")
    print(f"B = {b_path.name}\n")

    a, b = load(a_path), load(b_path)

    retrieval_drift, generation_drift, stable = [], [], 0

    for pid in sorted(a.keys() & b.keys()):
        ra, rb = a[pid], b[pid]
        same_sources = ra.get("sources") == rb.get("sources")
        same_verdict = (ra["verdict"], ra.get("metric")) == (rb["verdict"], rb.get("metric"))

        if same_verdict and same_sources:
            stable += 1
        elif not same_sources:
            retrieval_drift.append((pid, ra, rb))
        else:
            generation_drift.append((pid, ra, rb))

    print(f"stable            : {stable}")
    print(f"retrieval drift   : {len(retrieval_drift)}")
    print(f"generation drift  : {len(generation_drift)}\n")

    if retrieval_drift:
        print("--- RETRIEVAL DRIFT (different chunks pulled) ---")
        for pid, ra, rb in retrieval_drift:
            print(f"{pid:>5}  {ra['verdict']}/{ra.get('metric')} -> {rb['verdict']}/{rb.get('metric')}")
            print(f"       A sources: {ra.get('sources')}")
            print(f"       B sources: {rb.get('sources')}")

    if generation_drift:
        print("\n--- GENERATION DRIFT (same chunks, different answer) ---")
        for pid, ra, rb in generation_drift:
            print(f"{pid:>5}  {ra['verdict']}/{ra.get('metric')} -> {rb['verdict']}/{rb.get('metric')}")
            print(f"       {ra['question'][:70]}")

    print("\nreading: retrieval drift means the index or query path is "
          "non-deterministic. generation drift means temperature.")


if __name__ == "__main__":
    main()