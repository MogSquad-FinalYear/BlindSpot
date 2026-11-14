#!/usr/bin/env python3
"""
Does label-free threshold estimation survive CLASS IMBALANCE?

The closest prior work (Chen et al., "When EER Hides Deployment Failure",
arXiv:2606.21584, speech) reports that its best unlabeled calibrator --
temperature/shift with symmetric quartile pseudo-labels -- "collapsed on DF21:
its symmetric quartile pseudo-labels implicitly assume a roughly balanced
target, but DF21 is 96% spoof."

That failure mode matters more for images than for speech: an upload endpoint
sees overwhelmingly real photographs, so the deployed prevalence is nothing
like the 50/50 of a benchmark. A quantile-based estimator places the threshold
at a fixed rank, so it must drift as prevalence moves; a two-component mixture
estimates the two modes and their weights, so in principle it should not.

This tests that directly by resampling the same scored images to a range of
fake prevalences and measuring how far each estimator's threshold drifts from
the oracle, and what that costs in balanced accuracy.
"""
import csv, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.mixture import GaussianMixture

DETS = ["cliplora"]
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))
PREVALENCES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]


def gmm2(x):
    g = GaussianMixture(2, random_state=0, n_init=3).fit(x.reshape(-1, 1))
    mus = np.sort(g.means_.ravel())
    grid = np.linspace(mus[0], mus[1], 400).reshape(-1, 1)
    post = g.predict_proba(grid)
    order = np.argsort(g.means_.ravel())
    return float(grid[np.argmin(np.abs(post[:, order[0]] - post[:, order[1]])), 0])


def quartile_shift(x):
    """Stand-in for the prior work's symmetric quartile pseudo-labelling:
    threshold midway between the lower and upper quartile means."""
    lo, hi = np.quantile(x, 0.25), np.quantile(x, 0.75)
    return float((x[x <= lo].mean() + x[x >= hi].mean()) / 2)


def bal_acc(x, y, t):
    p = (x > t)
    tpr = p[y == 1].mean() if (y == 1).any() else 0.0
    tnr = (~p[y == 0]).mean() if (y == 0).any() else 0.0
    return float((tpr + tnr) / 2)


def main():
    with open("data/training_table.csv") as f:
        clean = {r["image_id"]: r for r in csv.DictReader(f)}
    with open("data/manifest_ladder.csv") as f:
        lad = list(csv.DictReader(f))
    base = sorted({r["base_id"] for r in lad if r["base_id"] in clean})
    S = defaultdict(dict)
    with open("data/scores_ladder.csv") as f:
        for r in csv.DictReader(f):
            S[r["image_id"]][r["detector"]] = float(r["score"])

    det = "cliplora"
    rng = np.random.RandomState(0)
    print(f"\n{'='*96}")
    print(f"LABEL-FREE THRESHOLD UNDER CLASS IMBALANCE  (detector {det}, JPEG q65)")
    print("=" * 96)
    q = 65
    ids = [b for b in base if det in S[f"{b}q{q}"]]
    x_all = np.array([S[f"{b}q{q}"][det] for b in ids])
    y_all = np.array([int(clean[b]["label"]) for b in ids])
    x_all = lg(x_all)
    fake, real = np.where(y_all == 1)[0], np.where(y_all == 0)[0]

    print(f"  {'prevalence':12s}{'oracle_t':>10s}{'GMM-2_t':>10s}{'quartile_t':>12s}"
          f"{'|drift| GMM':>13s}{'|drift| quart':>15s}{'balAcc GMM':>12s}{'balAcc quart':>14s}")
    rows = []
    for prev in PREVALENCES:
        gs, qs, dg, dq, bg, bq = [], [], [], [], [], []
        for trial in range(20):
            n = 1200
            nf = int(n * prev); nr = n - nf
            fi = rng.choice(fake, nf, replace=nf > len(fake))
            ri = rng.choice(real, nr, replace=nr > len(real))
            idx = np.concatenate([fi, ri])
            x, y = x_all[idx], y_all[idx]
            ths = np.unique(np.quantile(x, np.linspace(0.005, 0.995, 200)))
            t_or = max(ths, key=lambda t: bal_acc(x, y, t))
            t_g, t_q = gmm2(x), quartile_shift(x)
            gs.append(t_g); qs.append(t_q)
            dg.append(abs(t_g - t_or)); dq.append(abs(t_q - t_or))
            bg.append(bal_acc(x, y, t_g)); bq.append(bal_acc(x, y, t_q))
        print(f"  {prev*100:9.0f}%  {np.mean([t_or]):10.2f}{np.mean(gs):10.2f}"
              f"{np.mean(qs):12.2f}{np.mean(dg):13.2f}{np.mean(dq):15.2f}"
              f"{np.mean(bg)*100:12.2f}{np.mean(bq)*100:14.2f}")
        rows.append([prev, f"{np.mean(gs):.4f}", f"{np.mean(qs):.4f}",
                     f"{np.mean(dg):.4f}", f"{np.mean(dq):.4f}",
                     f"{np.mean(bg):.6f}", f"{np.mean(bq):.6f}"])

    with open("data/imbalance_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fake_prevalence","gmm2_threshold","quartile_threshold",
                    "gmm2_drift_from_oracle","quartile_drift_from_oracle",
                    "gmm2_balanced_acc","quartile_balanced_acc"])
        w.writerows(rows)
    print("\n  drift = |estimated threshold - oracle threshold| in logit units,")
    print("  averaged over 20 resampled batches of 1,200 images each.")
    print("  wrote data/imbalance_test.csv")


if __name__ == "__main__":
    main()
