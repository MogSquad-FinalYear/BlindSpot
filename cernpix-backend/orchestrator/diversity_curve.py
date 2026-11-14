#!/usr/bin/env python3
"""
Detector complementarity as a continuous function of JPEG degradation.

Ensemble theory says combining classifiers helps only to the extent their
errors are *decorrelated* (Kuncheva & Whitaker 2003; the
bias-variance-covariance decomposition). Every ensemble result in
AI-generated-image detection is reported on pristine benchmark images. This
measures what happens to that decorrelation as images are compressed the way
real uploads are.

Reported per quality level, over the SAME images at every level so the
comparison is paired throughout:

  Q          Yule's Q-statistic, averaged over detector pairs. +1 = errors
             perfectly correlated (no diversity, ensembling cannot help);
             0 = independent; negative = complementary.
  DF         double-fault rate: fraction of images where BOTH detectors of a
             pair are wrong. The errors an ensemble provably cannot fix.
  disagree   fraction where exactly one of the pair is right.
  oracle     accuracy if an omniscient selector always picked a correct
             detector -- the ceiling on any combination strategy.
  headroom   oracle minus best-single. This is the quantity that matters:
             it is the most any orchestrator could ever gain.

Usage: python3 diversity_curve.py
"""
import csv
from collections import defaultdict

import numpy as np

DETS = ["npr", "ufd", "cliplora"]
TABLE = "data/training_table.csv"
LADDER_MANIFEST = "data/manifest_ladder.csv"
LADDER_SCORES = "data/scores_ladder.csv"
Q75_SCORES = "data/scores_jpeg75.csv"


def q_statistic(a, b):
    """Yule's Q on two boolean correctness vectors."""
    n11 = int(np.sum(a & b)); n00 = int(np.sum(~a & ~b))
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b))
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    return float(num / den) if den else 0.0


def summarize(correct, label_note, n):
    """correct: dict detector -> boolean array (True = detector was right)."""
    C = np.stack([correct[d] for d in DETS])          # (n_det, n_img)
    accs = {d: float(correct[d].mean()) for d in DETS}
    best = max(accs, key=accs.get)
    oracle = float(C.any(axis=0).mean())
    qs, dfs, dis, rhos = [], [], [], []
    for i in range(len(DETS)):
        for j in range(i + 1, len(DETS)):
            a, b = C[i], C[j]
            qs.append(q_statistic(a, b))
            dfs.append(float(np.mean(~a & ~b)))
            dis.append(float(np.mean(a ^ b)))
            ea, eb = (~a).astype(float), (~b).astype(float)
            if ea.std() > 0 and eb.std() > 0:
                rhos.append(float(np.corrcoef(ea, eb)[0, 1]))
    return {
        "level": label_note, "n": n,
        "best": accs[best], "best_det": best, "oracle": oracle,
        "headroom": oracle - accs[best],
        "Q": float(np.mean(qs)), "DF": float(np.mean(dfs)),
        "disagree": float(np.mean(dis)),
        "rho": float(np.mean(rhos)) if rhos else float("nan"),
        "accs": accs,
    }


def main():
    # --- clean anchor, restricted to the ladder's base images ---
    with open(LADDER_MANIFEST) as f:
        lad = list(csv.DictReader(f))
    base_ids = sorted({r["base_id"] for r in lad})
    qualities = sorted({int(r["quality"]) for r in lad}, reverse=True)

    with open(TABLE) as f:
        clean = {r["image_id"]: r for r in csv.DictReader(f)}
    base_ids = [b for b in base_ids if b in clean]
    y = np.array([int(clean[b]["label"]) for b in base_ids])

    rows = []
    corr_clean = {d: np.array([(float(clean[b][f"score_{d}"]) > 0.5) == (y[k] == 1)
                               for k, b in enumerate(base_ids)]) for d in DETS}
    rows.append(summarize(corr_clean, "clean (no re-encode)", len(base_ids)))

    # --- q75, already computed, same image_id as the clean row ---
    q75 = defaultdict(dict)
    try:
        with open(Q75_SCORES) as f:
            for r in csv.DictReader(f):
                q75[r["image_id"]][r["detector"]] = float(r["score"])
        ids75 = [b for b in base_ids if all(d in q75[b] for d in DETS)]
        if len(ids75) > 200:
            y75 = np.array([int(clean[b]["label"]) for b in ids75])
            c75 = {d: np.array([(q75[b][d] > 0.5) == (y75[k] == 1)
                                for k, b in enumerate(ids75)]) for d in DETS}
            rows.append(summarize(c75, "q75", len(ids75)))
    except FileNotFoundError:
        pass

    # --- the ladder ---
    lad_scores = defaultdict(dict)
    with open(LADDER_SCORES) as f:
        for r in csv.DictReader(f):
            lad_scores[r["image_id"]][r["detector"]] = float(r["score"])
    for q in qualities:
        ids = [b for b in base_ids if all(d in lad_scores[f"{b}q{q}"] for d in DETS)]
        if len(ids) < 200:
            print(f"  (q{q}: only {len(ids)} scored, skipping)")
            continue
        yq = np.array([int(clean[b]["label"]) for b in ids])
        cq = {d: np.array([(lad_scores[f"{b}q{q}"][d] > 0.5) == (yq[k] == 1)
                           for k, b in enumerate(ids)]) for d in DETS}
        rows.append(summarize(cq, f"q{q}", len(ids)))

    order = {"clean (no re-encode)": 0}
    rows.sort(key=lambda r: order.get(r["level"], 1000 - int(r["level"][1:])
                                      if r["level"].startswith("q") else 999))

    print(f"\n{'='*94}")
    print("DETECTOR COMPLEMENTARITY vs JPEG DEGRADATION  "
          f"(detectors: {', '.join(DETS)}; paired images)")
    print("=" * 94)
    hdr = (f"  {'level':22s}{'n':>6s}{'best':>8s}{'oracle':>8s}{'headroom':>10s}"
           f"{'Q':>8s}{'DF':>8s}{'disagree':>10s}{'rho(err)':>10s}")
    print(hdr)
    for r in rows:
        print(f"  {r['level']:22s}{r['n']:6d}{r['best']*100:8.2f}{r['oracle']*100:8.2f}"
              f"{r['headroom']*100:10.2f}{r['Q']:8.3f}{r['DF']*100:8.2f}"
              f"{r['disagree']*100:10.2f}{r['rho']:10.3f}")
    print("\n  Q -> +1 means the detectors fail on the same images, so no combination")
    print("  strategy can help. headroom is the ceiling on any orchestrator's gain.")

    print(f"\n  per-detector accuracy by level:")
    print(f"  {'level':22s}" + "".join(f"{d:>13s}" for d in DETS))
    for r in rows:
        print(f"  {r['level']:22s}" + "".join(f"{r['accs'][d]*100:13.2f}" for d in DETS))

    with open("data/diversity_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "n", "best_detector", "best_acc", "oracle_acc",
                    "headroom", "Q_statistic", "double_fault", "disagreement",
                    "error_correlation"] + [f"acc_{d}" for d in DETS])
        for r in rows:
            w.writerow([r["level"], r["n"], r["best_det"], f"{r['best']:.6f}",
                        f"{r['oracle']:.6f}", f"{r['headroom']:.6f}", f"{r['Q']:.6f}",
                        f"{r['DF']:.6f}", f"{r['disagree']:.6f}", f"{r['rho']:.6f}"]
                       + [f"{r['accs'][d]:.6f}" for d in DETS])
    print("\n  wrote data/diversity_curve.csv")


if __name__ == "__main__":
    main()
