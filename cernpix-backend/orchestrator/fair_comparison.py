#!/usr/bin/env python3
"""
FAIRNESS CHECK on the paper's central claim.

recalib_control.py compared a single detector whose threshold was FITTED
against a stacker left at its default 0.5 cut. That is not a fair comparison:
it grants recalibration to one side only, and the whole finding of this study
is that threshold placement dominates under compression. If a stacker is also
allowed a fitted threshold and then beats the recalibrated single detector,
the claim "calibration beats ensembling" is wrong and must be withdrawn.

Every method below is fitted on the training folds only -- model, weights and
threshold alike -- and evaluated on held-out folds, 5-fold CV:

  single@0.5        best detector, literature-standard cut
  single@fit        best detector, threshold fitted on train folds
  mean-logit@fit    simplest ensemble: average the logits, fitted threshold
  stack@0.5         gradient-boosted stacker, default cut  (the earlier setup)
  stack@fit         SAME stacker, output threshold also fitted   <-- the control

Balanced accuracy is reported alongside accuracy because the batch is 55.6%
fake, and a threshold shift trades the two classes off against each other.
"""
import csv, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

DETS = ["npr", "ufd", "cliplora"]
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))


def fit_thresh(s, y, balanced=False):
    ths = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 300)))
    def sc(t):
        p = s > t
        if not balanced:
            return (p == (y == 1)).mean()
        tpr = p[y == 1].mean() if (y == 1).any() else 0
        tnr = (~p[y == 0]).mean() if (y == 0).any() else 0
        return (tpr + tnr) / 2
    return max(ths, key=sc)


def bal(p, y):
    tpr = p[y == 1].mean() if (y == 1).any() else 0
    tnr = (~p[y == 0]).mean() if (y == 0).any() else 0
    return float((tpr + tnr) / 2)


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

    print(f"\n{'='*104}")
    print("FAIR COMPARISON: every method gets a fitted threshold (5-fold CV, "
          "all fitting on train folds)")
    print("=" * 104)
    print(f"  {'level':7s}{'single@0.5':>12s}{'single@fit':>12s}{'meanlogit@fit':>15s}"
          f"{'stack@0.5':>11s}{'stack@fit':>11s}   {'stack@fit - single@fit':>24s}")
    rows = []
    for q in ["clean"] + quals:
        if q == "clean":
            ids = base
            S = np.array([[float(clean[b][f"score_{d}"]) for d in DETS] for b in ids])
        else:
            ids = [b for b in base if all(d in S_lad[f"{b}q{q}"] for d in DETS)]
            S = np.array([[S_lad[f"{b}q{q}"][d] for d in DETS] for b in ids])
        y = np.array([int(clean[b]["label"]) for b in ids])
        L = lg(S)

        n = len(y)
        p_s05 = np.zeros(n, bool); p_sfit = np.zeros(n, bool)
        p_mfit = np.zeros(n, bool); p_k05 = np.zeros(n, bool); p_kfit = np.zeros(n, bool)

        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(L, y):
            # --- best single detector, chosen on the training fold ---
            accs = [((S[tr][:, k] > 0.5) == (y[tr] == 1)).mean() for k in range(len(DETS))]
            k = int(np.argmax(accs))
            p_s05[te] = S[te][:, k] > 0.5
            t = fit_thresh(S[tr][:, k], y[tr])
            p_sfit[te] = S[te][:, k] > t

            # --- simplest ensemble: mean of logits, fitted threshold ---
            m_tr, m_te = L[tr].mean(axis=1), L[te].mean(axis=1)
            tm = fit_thresh(m_tr, y[tr])
            p_mfit[te] = m_te > tm

            # --- stacker, default cut AND fitted cut on its own output ---
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(L[tr], y[tr])
            pr_tr = clf.predict_proba(L[tr])[:, 1]
            pr_te = clf.predict_proba(L[te])[:, 1]
            p_k05[te] = pr_te > 0.5
            tk = fit_thresh(pr_tr, y[tr])
            p_kfit[te] = pr_te > tk

        a = lambda p: float((p == (y == 1)).mean())
        d = a(p_kfit) - a(p_sfit)
        print(f"  {str(q):7s}{a(p_s05)*100:12.2f}{a(p_sfit)*100:12.2f}{a(p_mfit)*100:15.2f}"
              f"{a(p_k05)*100:11.2f}{a(p_kfit)*100:11.2f}   {d*100:+24.2f}")
        rows.append([str(q), f"{a(p_s05):.6f}", f"{a(p_sfit):.6f}", f"{a(p_mfit):.6f}",
                     f"{a(p_k05):.6f}", f"{a(p_kfit):.6f}", f"{d:.6f}",
                     f"{bal(p_sfit,y):.6f}", f"{bal(p_kfit,y):.6f}"])

    print(f"\n  {'level':7s}{'balAcc single@fit':>20s}{'balAcc stack@fit':>20s}{'diff':>10s}")
    for r in rows:
        print(f"  {r[0]:7s}{float(r[7])*100:20.2f}{float(r[8])*100:20.2f}"
              f"{(float(r[8])-float(r[7]))*100:+10.2f}")

    with open("data/fair_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level","single_at_0.5","single_fitted","meanlogit_fitted",
                    "stack_at_0.5","stack_fitted","stack_minus_single_fitted",
                    "balacc_single_fitted","balacc_stack_fitted"])
        w.writerows(rows)
    print("\n  wrote data/fair_comparison.csv")


if __name__ == "__main__":
    main()
