#!/usr/bin/env python3
"""
Is the oracle headroom under degradation actually REACHABLE by a combiner?

diversity_curve.py shows two things pulling in opposite directions as JPEG
quality falls: error correlation rises sharply (Q from -0.15 to +0.90, and the
both-wrong rate from 1.8% to 33%), yet the ORACLE headroom over the best single
detector grows (2.4pp to 10.9pp). The headroom grows only because the best
detector collapses, not because the detectors became more complementary -- so
the oracle overstates what any real combiner can do.

This measures what is actually realizable, in the two regimes that matter:

  trained-clean     a stacker fitted on pristine images, then met with
                    degraded input. This is how every published ensemble is
                    built and how it would be deployed.
  trained-in-place  a stacker fitted AND tested at the same degradation level,
                    with a held-out split. Optimistic upper bound: it assumes
                    you knew the degradation in advance and had labels for it.

Scores only (no image features), deliberately: features carry the resolution
shortcut, and re-encoding changes them, which would confound the comparison.
"""
import csv
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

DETS = ["npr", "ufd", "cliplora"]
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    with open("data/training_table.csv") as f:
        clean = {r["image_id"]: r for r in csv.DictReader(f)}
    with open("data/manifest_ladder.csv") as f:
        lad = list(csv.DictReader(f))
    base_ids = sorted({r["base_id"] for r in lad if r["base_id"] in clean})
    quals = sorted({int(r["quality"]) for r in lad}, reverse=True)

    # --- stacker trained on CLEAN train-split rows, scores only ---
    tr = [r for r in clean.values() if r["split"] == "train"]
    Xtr = logit([[float(r[f"score_{d}"]) for d in DETS] for r in tr])
    ytr = np.array([int(r["label"]) for r in tr])
    stacker = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, ytr)
    print(f"stacker trained on {len(tr)} clean images (scores only)")

    lad_s = defaultdict(dict)
    with open("data/scores_ladder.csv") as f:
        for r in csv.DictReader(f):
            lad_s[r["image_id"]][r["detector"]] = float(r["score"])
    q75 = defaultdict(dict)
    try:
        with open("data/scores_jpeg75.csv") as f:
            for r in csv.DictReader(f):
                q75[r["image_id"]][r["detector"]] = float(r["score"])
    except FileNotFoundError:
        pass

    levels = [("clean", None)] + [("q75", "q75")] + [(f"q{q}", q) for q in quals]
    print(f"\n{'='*84}")
    print("REALIZABLE COMBINER GAIN vs JPEG DEGRADATION")
    print("=" * 84)
    print(f"  {'level':10s}{'n':>6s}{'best1':>9s}{'oracle':>9s}"
          f"{'trained-clean':>15s}{'delta':>8s}{'trained-in-place':>18s}{'delta':>8s}")
    out_rows = []
    for name, q in levels:
        if q is None:
            ids = base_ids
            S = np.array([[float(clean[b][f"score_{d}"]) for d in DETS] for b in ids])
        elif q == "q75":
            ids = [b for b in base_ids if all(d in q75[b] for d in DETS)]
            if len(ids) < 200:
                continue
            S = np.array([[q75[b][d] for d in DETS] for b in ids])
        else:
            ids = [b for b in base_ids if all(d in lad_s[f"{b}q{q}"] for d in DETS)]
            if len(ids) < 200:
                continue
            S = np.array([[lad_s[f"{b}q{q}"][d] for d in DETS] for b in ids])
        y = np.array([int(clean[b]["label"]) for b in ids])
        L = logit(S)

        accs = [((S[:, k] > 0.5) == (y == 1)).mean() for k in range(len(DETS))]
        best = float(max(accs))
        oracle = float(np.any([(S[:, k] > 0.5) == (y == 1) for k in range(len(DETS))], axis=0).mean())

        tc = float((stacker.predict(L) == y).mean())

        # trained-in-place: 5-fold CV within this level
        preds = np.zeros(len(y), dtype=int)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for a, b_ in skf.split(L, y):
            m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(L[a], y[a])
            preds[b_] = m.predict(L[b_])
        tip = float((preds == y).mean())

        print(f"  {name:10s}{len(ids):6d}{best*100:9.2f}{oracle*100:9.2f}"
              f"{tc*100:15.2f}{(tc-best)*100:+8.2f}{tip*100:18.2f}{(tip-best)*100:+8.2f}")
        out_rows.append([name, len(ids), f"{best:.6f}", f"{oracle:.6f}",
                         f"{tc:.6f}", f"{tc-best:.6f}", f"{tip:.6f}", f"{tip-best:.6f}"])

    with open("data/realizable_gain.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "n", "best_single", "oracle", "trained_clean",
                    "delta_trained_clean", "trained_in_place", "delta_trained_in_place"])
        w.writerows(out_rows)
    print("\n  wrote data/realizable_gain.csv")


if __name__ == "__main__":
    main()
