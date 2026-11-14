#!/usr/bin/env python3
"""
Integrity validation for the released score matrix.

Runs every check that could invalidate the resource, and prints PASS/FAIL per
check so failures are visible rather than buried. The strongest check is the
last one: the table must reproduce the published ICMLDE Table 2 accuracies,
because that ties this artifact to an independently peer-reviewed number.
"""
import csv, os, sys, hashlib
from collections import Counter, defaultdict

T = "data/training_table.csv"
FAILS = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)

rows = list(csv.DictReader(open(T)))
dets = ["npr", "ufd", "cliplora", "clipbaseline", "resnet"]
print(f"\n=== STRUCTURE ({len(rows)} rows) ===")
check("row count == 72,300", len(rows) == 72300, f"got {len(rows)}")
ids = Counter(r["image_id"] for r in rows)
check("image_id unique", max(ids.values()) == 1, f"max dup={max(ids.values())}")
paths = Counter(r["path"] for r in rows)
check("path unique", max(paths.values()) == 1, f"max dup={max(paths.values())}")

print("\n=== LABELS & SPLITS ===")
lab = Counter(r["label"] for r in rows)
check("labels binary", set(lab) == {"0", "1"}, str(dict(lab)))
# label must agree with provenance directory
bad = 0
for r in rows:
    p = r["path"]
    is_fake_dir = ("/ai/" in p) or ("1_fake" in p)
    is_real_dir = ("nature" in p) or ("val2017" in p)
    if is_fake_dir and r["label"] != "1": bad += 1
    if is_real_dir and r["label"] != "0": bad += 1
check("label matches source directory", bad == 0, f"{bad} mismatches")
sp = Counter(r["split"] for r in rows)
check("splits present", set(sp) == {"train", "val", "test"}, str(dict(sp)))
# an image must be in exactly one split (guaranteed by unique id, re-assert)
per_id_split = defaultdict(set)
for r in rows: per_id_split[r["image_id"]].add(r["split"])
check("no image in two splits", all(len(v) == 1 for v in per_id_split.values()))
# generator labels only on fakes
check("generator empty for reals",
      all(r["generator"] == "" for r in rows if r["label"] == "0"))

print("\n=== SCORES ===")
for d in dets:
    vals = [float(r[f"score_{d}"]) for r in rows]
    finite = all(v == v and 0.0 <= v <= 1.0 for v in vals)
    check(f"score_{d} in [0,1], no NaN", finite,
          f"min={min(vals):.4f} max={max(vals):.4f}")
    # correct_ column must be consistent with score and label at 0.5
    incons = sum(1 for r in rows
                 if int(r[f"correct_{d}"]) != int((float(r[f"score_{d}"]) > 0.5) == (r["label"] == "1")))
    check(f"correct_{d} consistent with score@0.5", incons == 0, f"{incons} bad")

print("\n=== FEATURES ===")
meta = {"image_id","path","label","generator","gen_fold","split","source","human_slice",
        "router_detector","router_score","router_correct"}
drop = meta | {f"{p}_{d}" for p in ("score","correct","origin") for d in dets}
fcols = [c for c in rows[0] if c not in drop]
check("feature count == 29", len(fcols) == 29, f"got {len(fcols)}")
badf = []
for c in fcols:
    try:
        vs = [float(r[c]) for r in rows]
        if any(v != v or v in (float("inf"), float("-inf")) for v in vs):
            badf.append(c)
    except ValueError:
        badf.append(c)
check("all features numeric & finite", not badf, str(badf))

print("\n=== FILE INTEGRITY (sample) ===")
import random
random.seed(0)
samp = random.sample(rows, 300)
missing = [r["path"] for r in samp if not os.path.exists(r["path"])]
check("sampled files exist on disk", not missing, f"{len(missing)} missing")

print("\n=== PROVENANCE OF SCORES ===")
orig = Counter(r[f"origin_{d}"] for r in rows for d in dets)
check("origin labelled for every score", set(orig) <= {"harvested", "computed"}, str(dict(orig)))

print("\n=== REPRODUCES PUBLISHED ICMLDE TABLE 2 ===")
# fake-accuracy (%) from the submitted paper, Table 2
PUB = {
 "ufd":          {"BigGan":83.0,"Glide":27.4,"flux":7.0, "sd3":11.7,"sdxl":38.5},
 "npr":          {"BigGan":84.1,"Glide":94.3,"flux":97.7,"sd3":90.1,"sdxl":54.6},
 "resnet":       {"BigGan":0.4, "Glide":6.0, "flux":54.9,"sd3":66.1,"sdxl":15.2},
 "clipbaseline": {"BigGan":9.6, "Glide":69.1,"flux":86.5,"sd3":88.6,"sdxl":77.0},
 "cliplora":     {"BigGan":99.7,"Glide":99.9,"flux":96.5,"sd3":98.3,"sdxl":99.2},
}
worst = 0.0
for d, cells in PUB.items():
    for g, published in cells.items():
        sub = [r for r in rows if r["generator"] == g]
        ours = 100.0 * sum(int(r[f"correct_{d}"]) for r in sub) / len(sub)
        diff = abs(ours - published)
        worst = max(worst, diff)
        if diff > 0.05:
            print(f"      MISMATCH {d}/{g}: ours={ours:.2f} published={published}")
check("all 25 fake-accuracy cells match published paper (<=0.05pp)", worst <= 0.05,
      f"max deviation {worst:.3f}pp")

print(f"\n{'='*60}")
print("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
