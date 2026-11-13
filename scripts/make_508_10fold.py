#!/usr/bin/env python3
"""10-fold split for Dataset508 (MCSTU dev 218), nested in the existing 5-fold split.

Each 5-fold validation set is shuffled (fixed seed) and cut in two halves -> 10 validation sets.
So 10-fold val k is a subset of 5-fold val k//2, and the 5-fold and 10-fold results can be
compared case by case. The 18 independent MCSTU test cases are not in 508 at all.

    python scripts/make_508_10fold.py
"""
import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", default="outputs/upload508/Dataset508_MCSTUEdge/splits_final.json")
ap.add_argument("--out", default="outputs/upload508/Dataset508_MCSTUEdge/splits_10fold.json")
ap.add_argument("--seed", type=int, default=12345)
a = ap.parse_args()

five = json.load(open(a.src))
allc = sorted(c for f in five for c in f["val"])
assert len(allc) == len(set(allc)), "5-fold val overlaps"
rng = random.Random(a.seed)
ten = []
for f in five:
    v = sorted(f["val"])
    rng.shuffle(v)
    h = (len(v) + 1) // 2
    for part in (v[:h], v[h:]):
        val = sorted(part)
        ten.append({"train": sorted(set(allc) - set(val)), "val": val})
assert sorted(c for f in ten for c in f["val"]) == allc, "every case must be in exactly one val fold"
json.dump(ten, open(a.out, "w"), indent=2)
print(f"{len(allc)} cases -> {a.out}")
for i, f in enumerate(ten):
    print(f"  fold {i}: train={len(f['train'])} val={len(f['val'])}  (inside 5-fold val {i // 2})")
