#!/usr/bin/env python3
"""
Quantifies the "generator has less high-frequency energy than real photos"
claim numerically (paper item 5), instead of relying on the spectrum plots
alone. For each generator, computes the mean log-magnitude FFT energy inside
the top-X% highest-frequency band, separately for real and fake images, plus
the paired difference and percent change -- not a single collapsed gap
number (that was the earlier mistake: a scalar "hf/lf gap" correlated
against accuracy with only 3-5 points isn't meaningful, and doesn't let you
say "real has how much more energy than fake" in the generator's own band).

Usage:
    python3 highpass_energy.py \
        --pairs BigGan=real_dir:fake_dir Glide=real_dir:fake_dir ... \
        --n_samples 300 --top_frac 0.2 \
        --out /home/student/data/paper/universalfakedetect/highpass_energy/highpass_energy.csv
"""
import argparse
import os
import random

import numpy as np
from PIL import Image
import csv


def load_gray(path, size=256):
    img = Image.open(path).convert("L").resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0


def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".JPEG", ".webp", ".bmp")):
    out = []
    for r, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(r, f))
    return out


def high_freq_mask(size, top_frac):
    """Boolean mask selecting the outer top_frac of the frequency radius (highest frequencies)."""
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(2) * size / 2
    threshold = (1 - top_frac) * max_dist
    return dist >= threshold


def per_image_highfreq_energy(paths, n_samples, size, mask, seed=0):
    rng = random.Random(seed)
    paths = paths[:]
    rng.shuffle(paths)
    paths = paths[:n_samples]
    energies = []
    for p in paths:
        try:
            img = load_gray(p, size)
        except Exception:
            continue
        f = np.fft.fftshift(np.fft.fft2(img))
        mag = np.log(np.abs(f) + 1e-8)
        energies.append(mag[mask].mean())
    return np.array(energies)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", required=True,
                    help="label=real_dir:fake_dir triples")
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--top_frac", type=float, default=0.2,
                    help="fraction of the frequency radius counted as 'high frequency' (0.2 = outer 20%%)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    mask = high_freq_mask(args.size, args.top_frac)
    rows = []
    for entry in args.pairs:
        label, dirs = entry.split("=", 1)
        real_dir, fake_dir = dirs.split(":", 1)
        real_paths = list_images(real_dir)
        fake_paths = list_images(fake_dir)

        real_e = per_image_highfreq_energy(real_paths, args.n_samples, args.size, mask)
        fake_e = per_image_highfreq_energy(fake_paths, args.n_samples, args.size, mask)

        diff = fake_e.mean() - real_e.mean()
        pct = 100 * diff / abs(real_e.mean())

        rows.append(dict(
            generator=label,
            n_real=len(real_e), n_fake=len(fake_e),
            real_highfreq_energy_mean=round(float(real_e.mean()), 4),
            real_highfreq_energy_std=round(float(real_e.std()), 4),
            fake_highfreq_energy_mean=round(float(fake_e.mean()), 4),
            fake_highfreq_energy_std=round(float(fake_e.std()), 4),
            fake_minus_real=round(float(diff), 4),
            pct_change_vs_real=round(float(pct), 2),
        ))
        print(f"{label}: real={real_e.mean():.4f}±{real_e.std():.4f} (n={len(real_e)})  "
              f"fake={fake_e.mean():.4f}±{fake_e.std():.4f} (n={len(fake_e)})  "
              f"diff={diff:.4f} ({pct:+.2f}%)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
