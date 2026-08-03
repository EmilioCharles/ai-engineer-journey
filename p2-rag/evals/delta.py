"""Summarise what changed between the two most recent runs of one retriever.

Run from p2-rag/:   python evals\delta.py bm25
"""
import json
import sys
import glob
from collections import Counter

kind = sys.argv[1] if len(sys.argv) > 1 else "bm25"
files = sorted(glob.glob(f"evals/runs/*{kind}*.json"))[-2:]
if len(files) < 2:
    raise SystemExit(f"need two {kind} runs")

print(f"A = {files[0]}")
print(f"B = {files[1]}\n")

a = {r["id"]: r for r in json.load(open(files[0], encoding="utf-8"))}
b = {r["id"]: r for r in json.load(open(files[1], encoding="utf-8"))}

changes = [(i, a[i], b[i]) for i in a if a[i]["verdict"] != b[i]["verdict"]]

gains = [c for c in changes if c[2]["verdict"] == "PASS"]
losses = [c for c in changes if c[1]["verdict"] == "PASS"]

print(f"verdict changes: {len(changes)}  (gained {len(gains)}, lost {len(losses)})\n")
for i, x, y in changes:
    has_pricing = "pricing.md" in (y["sources"] or [])
    print(f"  {str(i):>5}  {x['verdict']:<6} -> {y['verdict']:<6}  pricing_in_sources={has_pricing}")
    print(f"         {x['question'][:65]}")

picked_up = [i for i in a
             if "pricing.md" in (b[i]["sources"] or [])
             and "pricing.md" not in (a[i]["sources"] or [])]
print(f"\npairs that newly retrieve pricing.md: {len(picked_up)}")
print(" ", picked_up)

reordered = sum(1 for i in a if a[i]["sources"] != b[i]["sources"])
print(f"pairs whose retrieved sources changed at all: {reordered}")