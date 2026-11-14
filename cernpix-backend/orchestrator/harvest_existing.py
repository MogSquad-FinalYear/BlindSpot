#!/usr/bin/env python3
"""
Stage 2 -- harvest detector scores that already exist on disk.

The BlindSpot evaluation already ran all five detectors over the generator
cells and dumped per-image CSVs. Those scores were verified reproducible
against a fresh GPU pass (max |delta| 4.9e-05, i.e. nothing beyond the 4-dp
rounding in the CSVs), so re-running them would burn hours to recompute
numbers we already have.

Only the canonical `*_full.csv` runs are read. The `_pad`, `_nearest`,
`_jpeg75`, `_clean` and `*2000` variants are preprocessing ablations that
deliberately feed the detector something other than its official
preprocessing; mixing them into the training table would silently teach the
orchestrator on inputs the deployed pipeline never produces.

An image can legitimately appear in several CSVs for one detector (the same
real image was reused across generator cells). Those duplicates must agree,
since it is the same model on the same file -- disagreement means the CSVs
came from different checkpoints or preprocessing, so it is reported rather
than silently averaged.

Usage:
    python3 harvest_existing.py [--manifest data/manifest.csv]
                                [--out data/scores_harvested.csv]
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

DATA_ROOT = "/home/student/data"

# detector key -> glob of canonical full-run CSVs
SOURCES = {
    "npr":          f"{DATA_ROOT}/NPR-DeepfakeDetection/predictions_*_full.csv",
    "ufd":          f"{DATA_ROOT}/UniversalFakeDetect/predictions_*_full.csv",
    "cliplora":     f"{DATA_ROOT}/genimagepp_detectors/predictions_cliplora_*_full.csv",
    "clipbaseline": f"{DATA_ROOT}/genimagepp_detectors/predictions_clipbaseline_*_full.csv",
    "resnet":       f"{DATA_ROOT}/genimagepp_detectors/predictions_resnet_*_full.csv",
}
# Ablation runs that must never be treated as canonical scores.
EXCLUDE = ("_pad", "_nearest", "_jpeg", "_clean", "2000")

TOL = 1e-3  # CSVs store 4 dp; anything past this is a real disagreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data/scores_harvested.csv")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = {r["path"]: r for r in csv.DictReader(f)}
    print(f"manifest: {len(manifest)} images")

    # detector -> path -> [scores seen]
    seen = {d: defaultdict(list) for d in SOURCES}
    label_conflicts = 0

    for det, pattern in SOURCES.items():
        files = [p for p in sorted(glob.glob(pattern))
                 if not any(x in os.path.basename(p) for x in EXCLUDE)]
        for fp in files:
            with open(fp) as f:
                for r in csv.DictReader(f):
                    p = r["filepath"]
                    m = manifest.get(p)
                    if m is None:
                        continue
                    if int(r["true_label"]) != int(m["label"]):
                        label_conflicts += 1
                        continue
                    seen[det][p].append(float(r["pred_score"]))
        print(f"  {det:13s} {len(files)} files -> {len(seen[det])} manifest images")

    if label_conflicts:
        print(f"  WARNING: {label_conflicts} rows whose CSV label disagreed with the "
              f"manifest label (skipped)")

    # consistency check on duplicates
    disagree = 0
    rows = []
    for det, d in seen.items():
        for p, vals in d.items():
            if len(vals) > 1 and (max(vals) - min(vals)) > TOL:
                disagree += 1
                if disagree <= 5:
                    print(f"  DISAGREE {det} {os.path.basename(p)}: {vals}")
            rows.append({"image_id": manifest[p]["image_id"], "detector": det,
                         "score": f"{sum(vals)/len(vals):.6f}", "origin": "harvested"})
    if disagree:
        print(f"  WARNING: {disagree} images had inconsistent duplicate scores (>{TOL})")
    else:
        print("  duplicate-score consistency: OK")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "detector", "score", "origin"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  rows={len(rows)}")

    # coverage: how many manifest images have all 5?
    have = defaultdict(set)
    for r in rows:
        have[r["image_id"]].add(r["detector"])
    full = sum(1 for v in have.values() if len(v) == len(SOURCES))
    print(f"images with all {len(SOURCES)} detectors: {full}")
    missing = defaultdict(int)
    for m in manifest.values():
        got = have.get(m["image_id"], set())
        if len(got) < len(SOURCES):
            missing[m["cell"]] += 1
    print("\nstill needing inference, by cell:")
    for c in sorted(missing):
        print(f"  {c:24s} {missing[c]:6d}")
    print(f"  {'TOTAL':24s} {sum(missing.values()):6d}")


if __name__ == "__main__":
    main()
