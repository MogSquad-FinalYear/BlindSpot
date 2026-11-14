#!/usr/bin/env python3
"""
Control: is the in-place combiner's gain COMBINATION, or just RECALIBRATION?

realizable_gain.py shows a stacker retrained at a degradation level beating the
best single detector by up to +20pp. Two very different mechanisms produce that:

  recalibration  each detector still RANKS images correctly under compression
                 (AUC holds) but its fixed 0.5 threshold sits in the wrong
                 place. Simply moving one detector's threshold recovers the
                 accuracy, and no combination is needed.
  combination    the detectors carry genuinely complementary evidence that only
                 a joint model can exploit.

The distinction decides what the paper can claim, so it is measured rather than
assumed: a single detector with a threshold fitted on the SAME held-out folds
is compared against the 3-detector stacker fitted on those folds.
"""
import csv
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DETS = ["npr", "ufd", "cliplora"]
EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))


def best_thresh(s, y):
    ths = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 200)))
    return max(ths, key=lambda t: ((s > t) == (y == 1)).mean())


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

    print(f"\n{'='*100}")
    print("RECALIBRATION vs COMBINATION  (5-fold CV, threshold and stacker fitted on the same folds)")
    print("=" * 100)
    print(f"  {'level':8s}{'best@0.5':>10s}{'best@fitted-thr':>17s}{'gain_recal':>12s}"
          f"{'stacker3':>11s}{'gain_over_recal':>17s}{'AUC_best':>10s}")
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

        acc05 = [((S[:, k] > 0.5) == (y == 1)).mean() for k in range(len(DETS))]
        bi = int(np.argmax(acc05))
        best05 = acc05[bi]
        auc_best = roc_auc_score(y, S[:, bi])

        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        p_recal = np.zeros(len(y), dtype=int)
        p_stack = np.zeros(len(y), dtype=int)
        for a, b_ in skf.split(L, y):
            # recalibration: pick the best detector AND its threshold on the training fold
            accs_a = [((S[a][:, k] > 0.5) == (y[a] == 1)).mean() for k in range(len(DETS))]
            k = int(np.argmax(accs_a))
            t = best_thresh(S[a][:, k], y[a])
            p_recal[b_] = (S[b_][:, k] > t).astype(int)
            m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(L[a], y[a])
            p_stack[b_] = m.predict(L[b_])
        recal = (p_recal == y).mean()
        stack = (p_stack == y).mean()
        print(f"  {str(q):8s}{best05*100:10.2f}{recal*100:17.2f}{(recal-best05)*100:+12.2f}"
              f"{stack*100:11.2f}{(stack-recal)*100:+17.2f}{auc_best*100:10.2f}")
        rows.append([q, f"{best05:.6f}", f"{recal:.6f}", f"{recal-best05:.6f}",
                     f"{stack:.6f}", f"{stack-recal:.6f}", f"{auc_best:.6f}"])

    with open("data/recalib_control.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level","best_at_0.5","best_recalibrated","gain_from_recalibration",
                    "stacker_3det","gain_of_stacker_over_recalibration","auc_best_detector"])
        w.writerows(rows)
    print("\n  gain_recal      = what you get by only moving ONE detector's threshold")
    print("  gain_over_recal = what COMBINING adds on top of that")
    print("  wrote data/recalib_control.csv")


if __name__ == "__main__":
    main()
