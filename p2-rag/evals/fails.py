"""List every FAIL label across all runs, with its note."""
import json
from pathlib import Path

L = json.loads((Path(__file__).parent / "labels.json").read_text(encoding="utf-8"))

fails = [v for v in L.values() if v["verdict"] == "FAIL"]
print(f"{len(fails)} FAILs across all labeled runs\n")
for v in fails:
    print(f"  {v['run'][:22]}  id={v['id']}")
    print(f"      Q: {v['question'][:70]}")
    print(f"      note: {v['note'] or '(none)'}")