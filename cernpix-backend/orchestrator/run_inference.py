#!/usr/bin/env python3
"""
Stage 3 -- score the manifest images that no existing CSV covers.

Only the ~18k freshly-added real images need this; the 54k images that the
BlindSpot evaluation already covered are picked up by harvest_existing.py.
Every wrapper in detectors_bulk.py was verified to reproduce the stored CSVs
to <1e-4, so harvested and freshly-computed scores sit on one scale.

Runs ONE detector per process invocation. Three of the five wrappers cache
their model in a module global, and three CLIP ViT-L/14 towers at fp32 do not
fit beside each other on an 8 GB card; a fresh process is the simplest
guarantee that each run starts from a clean allocator.

Resumable by construction: results are appended and flushed every batch, and
on startup every (image_id, detector) pair already present in the output --
or already harvested -- is skipped. Killing and restarting loses at most one
batch.

fp32 is deliberate. fp16 autocast measured *slower* on this T1000 (2.0 vs 7.4
img/s): TU117 has no tensor cores, so autocast only adds cast overhead.

Usage:
    python3 run_inference.py --detector cliplora [--batch-size 32]
    python3 run_inference.py --all          # every detector, in cost order
"""
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors_bulk as db
from pools import DETECTORS

# cheapest first, so a partial run still leaves a usable table
COST_ORDER = ["npr", "resnet", "ufd", "cliplora", "clipbaseline"]


def load_done(paths, detector):
    """(image_id) already scored for `detector`, across harvested + new files."""
    done = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["detector"] == detector and r.get("score"):
                    done.add(r["image_id"])
    return done


def run_detector(detector, manifest, out_path, harvested, batch_size, device):
    done = load_done([harvested, out_path], detector)
    todo = [r for r in manifest if r["image_id"] not in done]
    if not todo:
        print(f"[{detector}] nothing to do ({len(done)} already scored)")
        return
    print(f"[{detector}] {len(todo)} images to score ({len(done)} already done)")

    model, tf = db.load(detector, device)
    new_file = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=["image_id", "detector", "score", "origin"])
    if new_file:
        w.writeheader()

    t0 = time.time()
    n_done = 0
    n_fail = 0
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        tensors, keep = [], []
        for r in chunk:
            try:
                tensors.append(tf(Image.open(r["path"]).convert("RGB")))
                keep.append(r)
            except Exception as exc:                      # noqa: BLE001
                n_fail += 1
                print(f"[{detector}] SKIP {r['path']}: {exc}", file=sys.stderr)
        if not tensors:
            continue
        with torch.no_grad():
            x = torch.stack(tensors).to(device)
            scores = torch.sigmoid(model(x)).flatten().cpu().tolist()
        for r, s in zip(keep, scores):
            w.writerow({"image_id": r["image_id"], "detector": detector,
                        "score": f"{s:.6f}", "origin": "computed"})
        fh.flush()
        n_done += len(keep)
        if (i // batch_size) % 20 == 0 or i + batch_size >= len(todo):
            el = time.time() - t0
            rate = n_done / el if el else 0
            eta = (len(todo) - n_done) / rate / 60 if rate else 0
            print(f"[{detector}] {n_done}/{len(todo)}  {rate:.1f} img/s  ETA {eta:.1f} min",
                  flush=True)
    fh.close()
    print(f"[{detector}] done: {n_done} scored, {n_fail} failed, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    db.unload()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--harvested", default="data/scores_harvested.csv")
    ap.add_argument("--out", default="data/scores_new.csv")
    ap.add_argument("--detector", choices=DETECTORS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="score at most N images (smoke tests)")
    args = ap.parse_args()

    if not args.detector and not args.all:
        ap.error("pass --detector NAME or --all")

    with open(args.manifest) as f:
        manifest = list(csv.DictReader(f))
    if args.limit:
        manifest = manifest[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  manifest={len(manifest)}")

    targets = COST_ORDER if args.all else [args.detector]
    for det in targets:
        run_detector(det, manifest, args.out, args.harvested, args.batch_size, device)


if __name__ == "__main__":
    main()
