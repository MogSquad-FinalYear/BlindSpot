#!/usr/bin/env python3
"""
Stage 1 -- build the orchestrator dataset manifest.

Emits one row per image with a stable id, its label, provenance, and the two
split columns the evaluation needs:

  split     random stratified train/val/test (70/15/15), for fitting and
            tuning the orchestrator in-distribution.
  gen_fold  the generator for fakes, "real" for reals. Enables
            leave-one-generator-out evaluation, which is the question this
            project actually cares about: does the orchestrator still help on
            a generator it has never seen? Aggregate in-distribution accuracy
            can look excellent while LOGO collapses, so both are reported.

Stratification is on (label, generator/source) so every cell keeps its
70/15/15 proportions, and the RNG is seeded, so re-running reproduces the
same manifest byte-for-byte.

Usage:
    python3 build_manifest.py [--out data/manifest.csv] [--seed 0]
"""
import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from pools import (FAKE_POOLS, SCORED_REAL_POOLS, NEW_REAL_POOLS,
                   HUMAN_SLICE_DIR, list_images)

SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def image_id(path):
    """Stable 16-hex id derived from the absolute path."""
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


def assign_splits(rows, seed=0):
    """Stratified split, applied within each (label, cell) group."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["label"], r["cell"])].append(r)
    rng = random.Random(seed)
    for key, group in buckets.items():
        group = sorted(group, key=lambda r: r["image_id"])
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(n * SPLIT_FRACS["train"]))
        n_va = int(round(n * SPLIT_FRACS["val"]))
        for i, r in enumerate(group):
            r["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    human_names = set()
    if HUMAN_SLICE_DIR.exists():
        human_names = {p.name for p in HUMAN_SLICE_DIR.iterdir() if p.is_file()}

    rows = []
    seen = {}

    def add(path, label, cell, source, generator, prescored):
        key = str(path)
        if key in seen:                     # defensive: never list an image twice
            return
        seen[key] = True
        rows.append({
            "image_id": image_id(path),
            "path": key,
            "label": label,                 # 0 = real, 1 = fake
            "cell": cell,                   # stratification cell
            "source": source,               # which pool it came from
            "generator": generator,         # generator for fakes, "" for reals
            "gen_fold": generator if label == 1 else "real",
            "prescored": int(prescored),    # 1 = all 5 detector scores already exist
            "human_slice": int(label == 0 and Path(path).name in human_names),
        })

    for gen, d, cap in FAKE_POOLS:
        for p in list_images(d, cap, args.seed):
            add(p, 1, f"fake:{gen}", gen, gen, prescored=True)
    for src, d, cap in SCORED_REAL_POOLS:
        for p in list_images(d, cap, args.seed):
            add(p, 0, f"real:{src}", src, "", prescored=True)
    for src, d, cap in NEW_REAL_POOLS:
        for p in list_images(d, cap, args.seed):
            add(p, 0, f"real:{src}", src, "", prescored=False)

    assign_splits(rows, args.seed)

    ids = Counter(r["image_id"] for r in rows)
    dupes = [i for i, n in ids.items() if n > 1]
    if dupes:
        raise SystemExit(f"FATAL: {len(dupes)} duplicate image_ids -- hash collision or repeated path")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "path", "label", "cell", "source", "generator",
            "gen_fold", "prescored", "human_slice", "split"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["cell"], r["image_id"])))

    print(f"wrote {out}  n={len(rows)}")
    print(f"  real={sum(1 for r in rows if r['label']==0)}  fake={sum(1 for r in rows if r['label']==1)}")
    print(f"  need GPU inference: {sum(1 for r in rows if not r['prescored'])}")
    print(f"  human slice (COCO people): {sum(r['human_slice'] for r in rows)}")
    print("\n  cell                      train    val   test  total")
    for cell in sorted({r["cell"] for r in rows}):
        c = Counter(r["split"] for r in rows if r["cell"] == cell)
        print(f"  {cell:24s} {c['train']:6d} {c['val']:6d} {c['test']:6d} {sum(c.values()):6d}")


if __name__ == "__main__":
    main()
