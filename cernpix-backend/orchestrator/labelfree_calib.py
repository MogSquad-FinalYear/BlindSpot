#!/usr/bin/env python3
"""
Can the +21pp recalibration gain be recovered WITHOUT labels?

recalib_control.py showed that re-fitting one detector's threshold at the
target degradation recovers ~21pp, and that combining detectors adds nothing.
But that threshold was fitted with labels at the target condition, which a
deployed system does not have. If no label-free estimator recovers the gain,
the finding stays an observation; if one does, it is a deployable procedure.

Estimators, all computed from the incoming scores alone (plus, for one of
them, real images only -- which are cheap to obtain and need no fake labels):

  fixed 0.5        what the literature reports
  Otsu             classic bimodal split on the logit histogram
  GMM-2            two-component Gaussian mixture, threshold at the crossing
  KDE valley       lowest-density point between the two modes
  median           assumes a balanced batch; threshold at the 50th percentile
  real-ref FPR1%   99th percentile of scores on a DEGRADATION-MATCHED pool of
                   known-real images (no fake labels used)
  oracle           best achievable threshold, fitted with full labels
"""
import csv, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

DETS = ["npr", "ufd", "cliplora"]
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))


def otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    p = hist / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    best_t, best_v = centers[0], -np.inf
    for i in range(1, bins):
        w0, w1 = p[:i].sum(), p[i:].sum()
        if w0 <= 0 or w1 <= 0: continue
        m0 = (p[:i] * centers[:i]).sum() / w0
        m1 = (p[i:] * centers[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best_v: best_v, best_t = v, centers[i]
    return best_t


def gmm2(x):
    g = GaussianMixture(2, random_state=0, n_init=3).fit(x.reshape(-1, 1))
    mus = g.means_.ravel()
    lo, hi = sorted(mus)
    grid = np.linspace(lo, hi, 400).reshape(-1, 1)
    post = g.predict_proba(grid)
    order = np.argsort(mus)
    diff = np.abs(post[:, order[0]] - post[:, order[1]])
    return float(grid[np.argmin(diff), 0])


def kde_valley(x):
    lo, hi = np.quantile(x, [0.02, 0.98])
    if hi <= lo: return float(np.median(x))
    grid = np.linspace(lo, hi, 400)
    d = gaussian_kde(x)(grid)
    # restrict to the interior so the valley is between modes, not at an edge
    lo_i, hi_i = int(0.15 * len(grid)), int(0.85 * len(grid))
    return float(grid[lo_i + int(np.argmin(d[lo_i:hi_i]))])


def main():
    with open("data/training_table.csv") as f:
        clean = {r["image_id"]: r for r in csv.DictReader(f)}
    with open("data/manifest_ladder.csv") as f:
        lad = list(csv.DictReader(f))
    base = sorted({r["base_id"] for r in lad if r["base_id"] in clean})
    quals = sorted({int(r["quality"]) for r in lad}, reverse=True)
    S_lad = defaultdict(dict)
    with open("data/scores_ladder.csv") as f:
        for r in csv.DictReader(f):
            S_lad[r["image_id"]][r["detector"]] = float(r["score"])

    det = "cliplora"
    print(f"\n{'='*104}")
    print(f"LABEL-FREE THRESHOLD RECOVERY  (detector: {det})")
    print("=" * 104)
    print(f"  {'level':7s}{'@0.5':>8s}{'Otsu':>8s}{'GMM-2':>8s}{'KDE':>8s}"
          f"{'median':>9s}{'realFPR1':>10s}{'oracle':>9s}   {'best label-free recovers':>26s}")
    rows = []
    for q in ["clean"] + quals:
        if q == "clean":
            ids = base
            s = np.array([float(clean[b][f"score_{det}"]) for b in ids])
        else:
            ids = [b for b in base if det in S_lad[f"{b}q{q}"]]
            s = np.array([S_lad[f"{b}q{q}"][det] for b in ids])
        y = np.array([int(clean[b]["label"]) for b in ids])
        x = lg(s)

        acc = lambda t: float(((x > t) == (y == 1)).mean())
        a05 = acc(0.0)                                  # logit 0 == score 0.5
        ths = np.unique(np.quantile(x, np.linspace(0.005, 0.995, 400)))
        orc = max(acc(t) for t in ths)

        # split reals in half: one half is the reference pool, the other is evaluated
        real_idx = np.where(y == 0)[0]
        rng = np.random.RandomState(0); rng.shuffle(real_idx)
        ref = real_idx[:len(real_idx)//2]
        keep = np.ones(len(y), bool); keep[ref] = False
        t_ref = float(np.quantile(x[ref], 0.99))
        a_ref = float(((x[keep] > t_ref) == (y[keep] == 1)).mean())

        cand = {"Otsu": otsu(x), "GMM-2": gmm2(x), "KDE": kde_valley(x),
                "median": float(np.median(x))}
        accs = {k: acc(t) for k, t in cand.items()}
        accs["realFPR1"] = a_ref
        bestlf = max(accs, key=accs.get)
        rec = (accs[bestlf] - a05) / (orc - a05) * 100 if orc > a05 + 1e-9 else float("nan")
        print(f"  {str(q):7s}{a05*100:8.2f}{accs['Otsu']*100:8.2f}{accs['GMM-2']*100:8.2f}"
              f"{accs['KDE']*100:8.2f}{accs['median']*100:9.2f}{a_ref*100:10.2f}"
              f"{orc*100:9.2f}   {bestlf:>10s} {rec:6.1f}%")
        rows.append([q, f"{a05:.6f}"] + [f"{accs[k]:.6f}" for k in
                    ["Otsu","GMM-2","KDE","median","realFPR1"]] +
                    [f"{orc:.6f}", bestlf, f"{rec:.2f}"])

    with open("data/labelfree_calib.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level","acc_at_0.5","acc_otsu","acc_gmm2","acc_kde","acc_median",
                    "acc_realref_fpr1","acc_oracle_threshold","best_labelfree",
                    "pct_of_oracle_gain_recovered"])
        w.writerows(rows)
    print("\n  wrote data/labelfree_calib.csv")
    print("  NOTE: this batch is 55.6% fake (5 fake cells vs 4 real cells), which")
    print("  favours estimators that assume a balanced batch. Flagged, not hidden.")


if __name__ == "__main__":
    main()
