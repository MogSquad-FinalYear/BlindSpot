#!/usr/bin/env python3
"""
Produces the extended evaluation numbers requested in review:
  (1) real-accuracy / overall-accuracy / AP / AUC for EVERY detector x generator
      cell, not only the BigGAN preprocessing case study;
  (2) paired McNemar tests between detector pairs on the identical fake-image
      sets;
  (3) calibration metrics (Brier score, Expected Calibration Error) per
      detector x generator, to put a number on the "confidently wrong" claim.

Pairing note: all four detectors scored the SAME fake images for every
generator (verified: 6000/6000 shared for BigGan/Glide/flux/sd3, 18300/18300
for sdxl), so McNemar is a legitimate paired test on fakes. The REAL images
differ for flux/sd3/sdxl because each inference script drew its own random
subsample from the shared 162k ImageNet pool, so cross-detector real-accuracy
on those three is an unpaired comparison of equal-sized draws from one pool.
McNemar is therefore restricted to fake images throughout.
"""
import argparse
import csv
import os
from itertools import combinations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

DETECTORS = {
    "UniversalFakeDetect": "/home/student/data/UniversalFakeDetect/predictions_{g}_full.csv",
    "NPR": "/home/student/data/NPR-DeepfakeDetection/predictions_{g}_full.csv",
    "GenImage++ ResNet-50 baseline": "/home/student/data/genimagepp_detectors/predictions_resnet_{g}_full.csv",
    "GenImage++ CLIP baseline": "/home/student/data/genimagepp_detectors/predictions_clipbaseline_{g}_full.csv",
    "GenImage++ CLIP-LoRA (OMAT)": "/home/student/data/genimagepp_detectors/predictions_cliplora_{g}_full.csv",
}
GENERATORS = ["BigGan", "Glide", "flux", "sd3", "sdxl"]
THRESHOLD = 0.5


def load(path):
    """basename -> (true_label, pred_score). Basenames are unique within a run."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[os.path.basename(row["filepath"])] = (
                int(row["true_label"]),
                float(row["pred_score"]),
            )
    return out


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Standard equal-width ECE on the predicted-class confidence."""
    y_hat = (y_prob > THRESHOLD).astype(int)
    conf = np.where(y_hat == 1, y_prob, 1.0 - y_prob)
    correct = (y_hat == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return ece


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out_dir", default="/home/student/data/paper/comparison")
    args = ap_.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- (1) full metrics per detector x generator ----------
    rows = []
    cache = {}
    for gen in GENERATORS:
        for det, tpl in DETECTORS.items():
            path = tpl.format(g=gen)
            if not os.path.exists(path):
                continue
            data = load(path)
            cache[(det, gen)] = data
            y_true = np.array([v[0] for v in data.values()])
            y_prob = np.array([v[1] for v in data.values()])
            y_hat = (y_prob > THRESHOLD).astype(int)

            rows.append(dict(
                generator=gen,
                detector=det,
                n_real=int((y_true == 0).sum()),
                n_fake=int((y_true == 1).sum()),
                acc_real=round(100 * (y_hat[y_true == 0] == 0).mean(), 2),
                acc_fake=round(100 * (y_hat[y_true == 1] == 1).mean(), 2),
                acc_overall=round(100 * (y_hat == y_true).mean(), 2),
                AP=round(100 * average_precision_score(y_true, y_prob), 2),
                AUC=round(100 * roc_auc_score(y_true, y_prob), 2),
                brier=round(float(np.mean((y_prob - y_true) ** 2)), 4),
                ECE=round(float(expected_calibration_error(y_true, y_prob)), 4),
            ))

    metrics_path = os.path.join(args.out_dir, "full_metrics_all_cells.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[1] wrote {metrics_path} ({len(rows)} cells)")
    for r in rows:
        print(f"    {r['generator']:7s} {r['detector'][:28]:28s} "
              f"real={r['acc_real']:6.2f} fake={r['acc_fake']:6.2f} "
              f"acc={r['acc_overall']:6.2f} AP={r['AP']:6.2f} AUC={r['AUC']:6.2f} "
              f"Brier={r['brier']:.4f} ECE={r['ECE']:.4f}")

    # ---------- (2) paired McNemar on identical fake images ----------
    mc_rows = []
    for gen in GENERATORS:
        avail = [d for d in DETECTORS if (d, gen) in cache]
        for a, b in combinations(avail, 2):
            da, db = cache[(a, gen)], cache[(b, gen)]
            shared = [k for k in da if k in db and da[k][0] == 1 and db[k][0] == 1]
            if not shared:
                continue
            a_ok = np.array([(da[k][1] > THRESHOLD) for k in shared])
            b_ok = np.array([(db[k][1] > THRESHOLD) for k in shared])
            n01 = int((a_ok & ~b_ok).sum())   # A right, B wrong
            n10 = int((~a_ok & b_ok).sum())   # A wrong, B right
            disc = n01 + n10
            if disc == 0:
                pval = 1.0
            else:
                pval = binomtest(min(n01, n10), disc, 0.5).pvalue
            mc_rows.append(dict(
                generator=gen, detector_A=a, detector_B=b,
                n_fake_shared=len(shared),
                A_correct_B_wrong=n01, A_wrong_B_correct=n10,
                p_value=pval,
                p_str=("<1e-300" if pval == 0 else f"{pval:.3g}"),
            ))

    mc_path = os.path.join(args.out_dir, "mcnemar_paired_tests.csv")
    with open(mc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mc_rows[0].keys()))
        w.writeheader()
        w.writerows(mc_rows)
    print(f"\n[2] wrote {mc_path} ({len(mc_rows)} pairwise tests, fake images only)")
    for r in mc_rows:
        if r["generator"] == "flux":
            print(f"    flux: {r['detector_A'][:26]:26s} vs {r['detector_B'][:26]:26s} "
                  f"b={r['A_correct_B_wrong']:5d} c={r['A_wrong_B_correct']:5d} p={r['p_str']}")


if __name__ == "__main__":
    main()
