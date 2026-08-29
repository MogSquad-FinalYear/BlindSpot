#!/usr/bin/env python3
"""
Pulls the N most-confidently-misclassified images per generator (paper item 6:
qualitative failure figure) from a predictions CSV, copies the actual image
files into an output folder, and writes a metadata CSV alongside them.

"Most confident" = fake images with the lowest pred_score (confidently called
real) and real images with the highest pred_score (confidently called fake),
matching this project's main finding that failures are confident misses, not
borderline uncertainty.

Usage:
    python3 failure_examples.py \
        --csv predictions_sd3_full.csv --label sd3 --n 6 \
        --out_dir /home/student/data/paper/universalfakedetect/failure_examples
"""
import argparse
import csv
import os
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=6, help="number of failure examples per direction (fake-as-real / real-as-fake)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["true_label"] = int(row["true_label"])
            row["pred_score"] = float(row["pred_score"])
            rows.append(row)

    fakes_missed = sorted([r for r in rows if r["true_label"] == 1], key=lambda r: r["pred_score"])[:args.n]
    reals_missed = sorted([r for r in rows if r["true_label"] == 0], key=lambda r: -r["pred_score"])[:args.n]

    out_dir = os.path.join(args.out_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)

    meta_rows = []
    for direction, items in [("fake_confidently_called_real", fakes_missed),
                              ("real_confidently_called_fake", reals_missed)]:
        for rank, r in enumerate(items):
            src = r["filepath"]
            ext = os.path.splitext(src)[1]
            dst_name = f"{direction}_{rank+1:02d}{ext}"
            dst = os.path.join(out_dir, dst_name)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"WARNING: could not copy {src}: {e}")
                continue
            meta_rows.append(dict(
                generator=args.label, direction=direction, rank=rank + 1,
                copied_as=dst_name, true_label=r["true_label"],
                pred_score=r["pred_score"], original_path=src,
            ))

    meta_path = os.path.join(out_dir, "metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"Copied {len(meta_rows)} failure examples to {out_dir}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
