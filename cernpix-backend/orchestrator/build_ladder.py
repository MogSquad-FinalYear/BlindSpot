#!/usr/bin/env python3
"""
Build a JPEG quality ladder: the SAME images re-encoded at several qualities.

The existing re-encoded slice is a single quality (q75), which gives one point.
Measuring how detector complementarity changes *as a function of* degradation
needs the same images at several qualities, so each image contributes a curve
rather than a point and the diversity measure is paired throughout.

Drawn from the test split only, so nothing here was ever trained on.
"""
import argparse, csv, random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

QUALITIES = [95, 85, 65, 55, 45, 35]   # q75 already exists; clean is the anchor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out-dir", default="data/ladder")
    ap.add_argument("--out-manifest", default="data/manifest_ladder.csv")
    ap.add_argument("--n-per-cell", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    by_cell = defaultdict(list)
    for r in rows: by_cell[r["cell"]].append(r)

    rng = random.Random(args.seed)
    picked = []
    for cell, grp in sorted(by_cell.items()):
        grp = sorted(grp, key=lambda r: r["image_id"])
        picked.extend(rng.sample(grp, min(args.n_per_cell, len(grp))))
    print(f"{len(picked)} base images x {len(QUALITIES)} qualities = "
          f"{len(picked)*len(QUALITIES)} encodes")

    out, failed = [], 0
    for i, r in enumerate(picked):
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception as e:
            failed += 1; print("SKIP", r["path"], e); continue
        for q in QUALITIES:
            dest = out_dir / f"{r['image_id']}_q{q}.jpg"
            img.save(dest, format="JPEG", quality=q)
            nr = dict(r)
            nr["image_id"] = f"{r['image_id']}q{q}"
            nr["path"] = str(dest)
            nr["source"] = f"{r['source']}_q{q}"
            nr["prescored"] = 0
            nr["base_id"] = r["image_id"]     # links back to the clean row
            nr["quality"] = q
            out.append(nr)
        if (i+1) % 500 == 0: print(f"  {i+1}/{len(picked)}", flush=True)

    with open(args.out_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote {args.out_manifest}  n={len(out)}  failed={failed}")

if __name__ == "__main__":
    main()
