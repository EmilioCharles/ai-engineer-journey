"""Walk through REVIEW rows one at a time and label them by hand.

The runner can auto-score refusals and forbidden strings, but it cannot judge
whether a free-text answer is correct. This tool shows you the question, the
gold answer, and what the system actually said, and records your verdict.

Those verdicts are the alignment set: later, an LLM judge gets scored against
them, so you know whether the judge agrees with you before you trust it.

Run from p2-rag/:
    python evals\label.py                 # newest run
    python evals\label.py bm25            # newest bm25 run
    python evals\label.py evals\runs\X.json

Keys:  p = PASS   f = FAIL   s = skip   b = back   q = save and quit

Progress saves after every single label, so quitting mid-way loses nothing.
Labels are written to evals/labels.json, keyed by run file + pair id, and
merged back into the run file as `human_verdict` / `human_note`.
"""
import json
import sys
import glob
from pathlib import Path

EVALS = Path(__file__).parent
LABELS = EVALS / "labels.json"


def pick_run(arg):
    if arg and arg.endswith(".json"):
        return Path(arg)
    pattern = f"evals/runs/*{arg}*.json" if arg else "evals/runs/*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no run files match {pattern}")
    return Path(files[-1])


def load_labels():
    if LABELS.exists():
        return json.loads(LABELS.read_text(encoding="utf-8"))
    return {}


def save_labels(labels):
    LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")


def wrap(text, width=88, indent="    "):
    if text is None:
        return indent + "(none)"
    text = str(text)
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(indent + line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(indent + line)
    return "\n".join(out)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_path = pick_run(arg)
    rows = json.loads(run_path.read_text(encoding="utf-8"))
    key_prefix = run_path.name

    labels = load_labels()
    todo = [r for r in rows if r["verdict"] == "REVIEW"]

    if not todo:
        raise SystemExit(f"{run_path.name}: no REVIEW rows to label")

    print(f"run: {run_path.name}")
    print(f"{len(todo)} rows to label\n")
    print("p = PASS   f = FAIL   s = skip   b = back   q = save and quit\n")

    i = 0
    while 0 <= i < len(todo):
        r = todo[i]
        key = f"{key_prefix}::{r['id']}"
        existing = labels.get(key, {}).get("verdict")

        print("=" * 92)
        print(f"[{i+1}/{len(todo)}]  id={r['id']}  category={r.get('category')}"
              + (f"   (already labeled {existing})" if existing else ""))
        print(f"\n  QUESTION")
        print(wrap(r["question"]))
        print(f"\n  EXPECTED")
        print(wrap(r.get("expected_answer") or r.get("gold_answer")))
        print(f"\n  SYSTEM SAID")
        print(wrap(r["output"]))
        print(f"\n  retrieved: {r.get('sources')}")
        print()

        choice = input("  [p/f/s/b/q] > ").strip().lower()

        if choice == "q":
            break
        if choice == "b":
            i = max(0, i - 1)
            continue
        if choice == "s":
            i += 1
            continue
        if choice not in ("p", "f"):
            print("  ? use p, f, s, b or q\n")
            continue

        note = input("  why (optional) > ").strip()
        labels[key] = {
            "id": r["id"],
            "run": key_prefix,
            "verdict": "PASS" if choice == "p" else "FAIL",
            "note": note,
            "question": r["question"],
        }
        save_labels(labels)
        i += 1

    # merge labels back into the run file
    n_merged = 0
    for r in rows:
        lab = labels.get(f"{key_prefix}::{r['id']}")
        if lab:
            r["human_verdict"] = lab["verdict"]
            r["human_note"] = lab["note"]
            n_merged += 1
    run_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # report
    auto_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    auto_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
    h_pass = sum(1 for r in rows if r.get("human_verdict") == "PASS")
    h_fail = sum(1 for r in rows if r.get("human_verdict") == "FAIL")
    unlabeled = sum(1 for r in rows if r["verdict"] == "REVIEW"
                    and "human_verdict" not in r)

    print("\n" + "=" * 92)
    print(f"labeled this session : {n_merged} of {len(todo)}")
    print(f"still unlabeled      : {unlabeled}")
    print()
    print(f"auto   PASS {auto_pass:3d}   FAIL {auto_fail:3d}")
    print(f"human  PASS {h_pass:3d}   FAIL {h_fail:3d}")
    total_pass, total = auto_pass + h_pass, len(rows) - unlabeled
    if total:
        print(f"\nTOTAL  {total_pass}/{total}  ({100*total_pass/total:.0f}%)")
    print(f"\nlabels -> {LABELS}")
    print(f"run    -> {run_path}")


if __name__ == "__main__":
    main()