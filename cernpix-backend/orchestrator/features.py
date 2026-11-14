#!/usr/bin/env python3
"""
Stage 4 -- per-image features for the orchestrator.

These are the cheap, image-only signals the orchestrator can see *before*
deciding what to do (which detector to trust, whether to escalate). They must
be computable in milliseconds on CPU, because the whole point of a routing
decision is that it costs less than the detectors it is choosing between.

`hf_energy` is imported from rule_engine rather than reimplemented, so the
orchestrator's feature is bit-identical to the one the deployed router keys
on and "orchestrator vs router" stays a fair comparison.

--------------------------------------------------------------------------
THE FORMAT CONFOUND -- read before using these features
--------------------------------------------------------------------------
In this corpus every fake is a .png and every real is a .jpeg/.jpg (42300 /
30000, no exceptions). Any feature that senses JPEG quantization therefore
separates the classes almost perfectly *on this data* while carrying no
generalizable evidence about whether an image was generated.

That is not a hypothetical worry here. The deployed router's own docstring
records the confirmed failure it causes: an AI-generated portrait saved as a
348x456 screenshot scored 0.0014 ("confidently real") because re-encoding
moved it into real-photo territory. A meta-model that learns "JPEG => real"
reproduces that bug and makes it worse.

So compression-sensitive features are computed but quarantined in
SHORTCUT_FEATURES, excluded from the default training set. build_table.py
reports how much accuracy they would add, which measures the size of the
confound rather than the quality of the model. The durable fix is the
recompression control in augment.py -- re-encode everything to a common
format so format carries no class information at all.
"""
import argparse
import csv
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rule_engine import FFT_SIZE, _HF_MASK          # noqa: E402  (router parity)

N_BANDS = 8

# Features excluded from the default orchestrator input: they mostly measure
# "was this JPEG-compressed", which in this corpus is a proxy for the label.
SHORTCUT_FEATURES = ["blockiness_8", "jpeg_grid_ratio"]

# Native-geometry features. Kept in the default set because an orchestrator is
# allowed to know how big an image is, but grouped so the ablation is one flag
# away: resolution correlates hard with generator in this corpus (BigGAN emits
# 128x128, DiT generators 1024x1024, ImageNet reals vary), and rule_engine.py
# deleted resolution from the routing decision for exactly that reason -- any
# resize, screenshot or re-save destroys it while leaving generator content
# intact. If --no-geometry barely moves the numbers, the orchestrator is not
# leaning on it; if it collapses, it learned resolution, not generation.
GEOMETRY_FEATURES = ["width", "height", "min_side", "max_side", "aspect", "log_pixels"]


def _radial_bins(size=FFT_SIZE, n=N_BANDS):
    cy = cx = size // 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = math.sqrt(2) * size / 2
    edges = np.linspace(0, rmax, n + 1)
    return [(dist >= edges[i]) & (dist < edges[i + 1]) for i in range(n)], dist / rmax


_BAND_MASKS, _NORM_R = _radial_bins()


def _axis_diag_masks(size=FFT_SIZE):
    """Split the high-frequency corner region into axis-aligned vs diagonal.

    GAN upsampling leaves checkerboard artifacts that are strongly
    axis-aligned, so the ratio between the two carries information that a
    single isotropic average (hf_energy) throws away.
    """
    cy = cx = size // 2
    yy, xx = np.ogrid[:size, :size]
    dy, dx = np.abs(yy - cy), np.abs(xx - cx)
    ang = np.arctan2(np.minimum(dy, dx) + 0.0, np.maximum(dy, dx) + 1e-9)
    axis = (ang < math.radians(20))
    diag = (ang > math.radians(35))
    return axis, diag


_AXIS_MASK, _DIAG_MASK = _axis_diag_masks()


def compute(path):
    """All features for one image path. Returns a dict, or None if unreadable."""
    try:
        img = Image.open(path)
        img.load()
    except Exception:                                     # noqa: BLE001
        return None
    return compute_from_image(img)


def compute_from_image(img):
    """Same features, from an already-decoded PIL image.

    This is the entry point serve.py uses: at request time the app already
    holds the decoded image, and re-opening it from disk would both cost a
    decode and risk diverging from what the detectors are actually scoring.
    """
    width, height = img.size
    rgb = img.convert("RGB")
    gray_full = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    g = np.asarray(
        img.convert("L").resize((FFT_SIZE, FFT_SIZE)), dtype=np.float32
    ) / 255.0
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log(np.abs(f) + 1e-8)

    out = {
        # --- geometry (native, pre-resize) ---
        "width": width,
        "height": height,
        "min_side": min(width, height),
        "max_side": max(width, height),
        "aspect": round(max(width, height) / max(min(width, height), 1), 4),
        "log_pixels": round(math.log(max(width * height, 1)), 4),
        # --- router parity: identical to rule_engine.compute_features ---
        "hf_energy": round(float(mag[_HF_MASK].mean()), 4),
    }

    # --- radial spectrum: what hf_energy collapses into one number ---
    band_means = [float(mag[m].mean()) for m in _BAND_MASKS]
    for i, v in enumerate(band_means):
        out[f"band{i}"] = round(v, 4)
    lf = float(np.mean(band_means[:2]))
    hf = float(np.mean(band_means[-2:]))
    out["hf_lf_gap"] = round(hf - lf, 4)

    # spectral slope: log-magnitude vs normalized radius, least squares
    r = _NORM_R.ravel()
    sel = r > 0.05
    slope = np.polyfit(np.log(r[sel]), mag.ravel()[sel], 1)[0]
    out["spectral_slope"] = round(float(slope), 4)

    # anisotropy inside the high-frequency region
    ax = float(mag[_HF_MASK & _AXIS_MASK].mean()) if (_HF_MASK & _AXIS_MASK).any() else 0.0
    dg = float(mag[_HF_MASK & _DIAG_MASK].mean()) if (_HF_MASK & _DIAG_MASK).any() else 0.0
    out["hf_axis"] = round(ax, 4)
    out["hf_diag"] = round(dg, 4)
    out["hf_anisotropy"] = round(ax - dg, 4)

    # --- intensity / colour statistics ---
    out["gray_mean"] = round(float(gray_full.mean()), 4)
    out["gray_std"] = round(float(gray_full.std()), 4)
    hist, _ = np.histogram(gray_full, bins=64, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    out["entropy"] = round(float(-(p * np.log2(p)).sum()), 4)

    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    out["sat_mean"] = round(float(sat.mean()), 4)
    out["sat_std"] = round(float(sat.std()), 4)

    # --- local noise: high-pass residual on the native-resolution luma ---
    if gray_full.shape[0] > 3 and gray_full.shape[1] > 3:
        lap = (gray_full[1:-1, 2:] + gray_full[1:-1, :-2] +
               gray_full[2:, 1:-1] + gray_full[:-2, 1:-1] -
               4 * gray_full[1:-1, 1:-1])
        out["residual_std"] = round(float(lap.std()), 5)
        out["residual_kurt"] = round(float(((lap - lap.mean()) ** 4).mean() /
                                           max(lap.var() ** 2, 1e-12)), 4)
    else:
        out["residual_std"] = 0.0
        out["residual_kurt"] = 0.0

    # --- QUARANTINED: compression-sensitive, see module docstring ---
    gy = gray_full
    if gy.shape[0] >= 16 and gy.shape[1] >= 16:
        dv = np.abs(np.diff(gy, axis=0))
        dh = np.abs(np.diff(gy, axis=1))
        on_v = dv[7::8].mean() if dv[7::8].size else 0.0
        on_h = dh[:, 7::8].mean() if dh[:, 7::8].size else 0.0
        off_v, off_h = dv.mean(), dh.mean()
        out["blockiness_8"] = round(float((on_v + on_h) / 2), 5)
        out["jpeg_grid_ratio"] = round(float((on_v + on_h) / max(off_v + off_h, 1e-9)), 4)
    else:
        out["blockiness_8"] = 0.0
        out["jpeg_grid_ratio"] = 1.0

    return out


FEATURE_NAMES = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data/features.csv")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {r["image_id"] for r in csv.DictReader(f)}
    todo = [r for r in rows if r["image_id"] not in done]
    print(f"features: {len(todo)} to compute ({len(done)} cached)")
    if not todo:
        return

    probe = compute(todo[0]["path"])
    if probe is None:
        raise SystemExit(f"cannot read first image: {todo[0]['path']}")
    cols = ["image_id"] + list(probe)

    new_file = not os.path.exists(args.out)
    import time
    t0 = time.time()
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new_file:
            w.writeheader()
        with ThreadPoolExecutor(args.threads) as ex:
            for i in range(0, len(todo), 512):
                chunk = todo[i:i + 512]
                for r, feat in zip(chunk, ex.map(lambda x: compute(x["path"]), chunk)):
                    if feat is None:
                        print(f"SKIP unreadable {r['path']}", file=sys.stderr)
                        continue
                    w.writerow({"image_id": r["image_id"], **feat})
                fh.flush()
                el = time.time() - t0
                n = i + len(chunk)
                print(f"  {n}/{len(todo)}  {n/el:.0f} img/s  ETA {(len(todo)-n)/max(n/el,1e-9)/60:.1f} min",
                      flush=True)
    print(f"wrote {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
