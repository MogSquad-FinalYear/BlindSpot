#!/usr/bin/env python3
"""
Optional stage -- build a re-encoded control slice to measure the format
confound, and the router's documented failure mode, directly.

Why this exists: in the collected corpus every fake is a .png and every real
is a .jpeg. A detector or orchestrator can score well by keying on JPEG
quantization rather than on generation artifacts, and that is exactly the bug
rule_engine.py's docstring records in the field -- an AI-generated portrait
saved as a 348x456 screenshot scored 0.0014 ("confidently real"). Most
real-world uploads are screenshots or re-saves, so the deployed system meets
re-encoded fakes constantly and almost never meets pristine PNGs.

This script takes a balanced sample of the manifest, re-encodes every image
through the same JPEG quality and optional downscale, and writes a second
manifest pointing at the re-encoded copies. Scoring the detectors on it
(run_inference.py --manifest ...) answers two questions the clean corpus
cannot:

  1. How much of any detector's / orchestrator's accuracy was format, not
     content?  (accuracy on clean vs re-encoded fakes)
  2. Does the orchestrator degrade more gracefully than the single detector
     and the router it replaces?

Deliberately NOT applied to the training corpus by default. Training on
re-encoded images too would be the stronger fix, but it changes what the
harvested BlindSpot scores mean -- those were computed on the originals -- so
it is a separate, explicit experiment rather than a silent rewrite.

Usage:
    python3 augment.py --n-per-cell 700 --quality 75 [--max-side 512]
    python3 run_inference.py --all --manifest data/manifest_jpeg75.csv \
                             --out data/scores_jpeg75.csv
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="data/recompressed")
    ap.add_argument("--out-manifest", default=None)
    ap.add_argument("--n-per-cell", type=int, default=700,
                    help="images sampled per cell; keeps the slice balanced")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--max-side", type=int, default=None,
                    help="also downscale so max(w,h) <= this, mimicking a screenshot")
    ap.add_argument("--split", default="test",
                    help="which split to draw from; test keeps the control honest")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="several splits at once, e.g. --splits train val (for "
                         "building augmented TRAINING data)")
    ap.add_argument("--mode", choices=["fixed", "mixed"], default="fixed",
                    help="fixed: one quality, for a clean paired control. "
                         "mixed: randomized quality/downscale per image, for "
                         "training -- a model trained on one fixed quality just "
                         "learns that quality")
    ap.add_argument("--id-suffix", default="",
                    help="appended to image_id so augmented rows coexist with "
                         "their clean originals in one table (training use)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_manifest = args.out_manifest or f"data/manifest_jpeg{args.quality}.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.splits) if args.splits else {args.split}
    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in wanted]
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    rng = random.Random(args.seed)
    picked = []
    for cell, group in sorted(by_cell.items()):
        group = sorted(group, key=lambda r: r["image_id"])
        picked.extend(rng.sample(group, min(args.n_per_cell, len(group))))
    print(f"re-encoding {len(picked)} images from split={args.split} "
          f"at JPEG q={args.quality}" + (f", max_side={args.max_side}" if args.max_side else ""))

    # Mixed mode mimics the spread of what actually reaches an upload endpoint:
    # a range of JPEG qualities, frequent downscaling (screenshots, messaging
    # apps), and a minority left untouched so the model still sees pristine
    # input. Fixed mode keeps one quality, which is what a paired control needs.
    QUALITIES = [50, 60, 70, 75, 80, 85, 90, 95]
    SCALES = [None, None, 1024, 768, 640, 512, 384]

    out_rows, failed = [], 0
    for i, r in enumerate(picked):
        try:
            img = Image.open(r["path"]).convert("RGB")
            if args.mode == "mixed":
                quality = rng.choice(QUALITIES)
                max_side = rng.choice(SCALES)
            else:
                quality, max_side = args.quality, args.max_side
            if max_side:
                w, h = img.size
                if max(w, h) > max_side:
                    sc = max_side / max(w, h)
                    img = img.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.BICUBIC)
            dest = out_dir / f"{r['image_id']}{args.id_suffix}.jpg"
            img.save(dest, format="JPEG", quality=quality)
        except Exception as exc:                            # noqa: BLE001
            failed += 1
            print(f"SKIP {r['path']}: {exc}")
            continue
        nr = dict(r)
        nr["path"] = str(dest)
        # With no suffix the image_id is kept, so results join back to the clean
        # row for the same image -- that pairing is the whole point of a control.
        # With a suffix the augmented row becomes its own training example
        # alongside the original.
        nr["image_id"] = r["image_id"] + args.id_suffix
        nr["source"] = r["source"] + ("_aug" if args.mode == "mixed"
                                      else f"_jpeg{args.quality}")
        nr["prescored"] = 0
        nr["variant"] = f"aug_q{quality}_s{max_side or 'orig'}"
        out_rows.append(nr)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(picked)}", flush=True)

    with open(out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_manifest}  n={len(out_rows)}  failed={failed}")
    print(f"\nnext:\n  python3 features.py --manifest {out_manifest} --out data/features_jpeg{args.quality}.csv")
    print(f"  python3 run_inference.py --all --manifest {out_manifest} "
          f"--out data/scores_jpeg{args.quality}.csv --harvested /dev/null")


if __name__ == "__main__":
    main()
