#!/usr/bin/env python3
"""
Evaluate the format confound: same images, clean vs JPEG-re-encoded.

Leave-one-generator-out measures robustness to an unseen *generator*. It
cannot measure robustness to unseen *preprocessing*, because every image in
every fold is the pristine original. That gap matters here for two reasons:

  1. In this corpus every fake is a PNG and every real a JPEG, so anything
     sensing compression separates the classes for free.
  2. The orchestrator's measured advantage depends partly on geometry
     features (LOGO cascade drops 99.17 -> 96.89 without them), and
     resolution is precisely what a screenshot or re-save destroys -- the
     failure rule_engine.py documents in the field.

augment.py re-encodes a balanced sample of the TEST split, keeping image_id
so each re-encoded row pairs with its clean original. This script scores both
and reports the drop. A detector or strategy that was reading format rather
than generation artifacts loses accuracy here; one that was not, does not.

Usage:
    python3 compare_recompressed.py [--model data/orchestrator_model_3det.pkl]
"""
import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_table import router_choice, THRESH                # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/training_table.csv")
    ap.add_argument("--rc-features", default="data/features_jpeg75.csv")
    ap.add_argument("--rc-scores", default="data/scores_jpeg75.csv")
    ap.add_argument("--model", default="data/orchestrator_model_3det.pkl")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        art = pickle.load(f)
    dets = art["detectors"]
    feat_cols = art["feat_cols"]
    band = art["cascade_band"]
    primary = art["cascade_primary"]
    pi = dets.index(primary)

    clean = {}
    with open(args.table) as f:
        for r in csv.DictReader(f):
            clean[r["image_id"]] = r

    rc_feat = {}
    with open(args.rc_features) as f:
        for r in csv.DictReader(f):
            rc_feat[r["image_id"]] = r
    rc_score = defaultdict(dict)
    with open(args.rc_scores) as f:
        for r in csv.DictReader(f):
            rc_score[r["image_id"]][r["detector"]] = float(r["score"])

    ids = [i for i in rc_feat
           if i in clean and all(d in rc_score[i] for d in dets)]
    print(f"paired images: {len(ids)}")
    if not ids:
        raise SystemExit("no paired rows -- has the re-encoded inference finished?")

    y = np.array([int(clean[i]["label"]) for i in ids])
    # training_table.csv carries generator/source rather than the manifest's
    # combined `cell`; rebuild it so the per-cell breakdown lines up.
    cell = np.array([
        ("fake:" + clean[i]["generator"]) if clean[i]["label"] == "1"
        else ("real:" + clean[i]["source"])
        for i in ids])

    S_clean = logit(np.array([[float(clean[i][f"score_{d}"]) for d in dets] for i in ids]))
    S_rc = logit(np.array([[rc_score[i][d] for d in dets] for i in ids]))
    F_clean = np.array([[float(clean[i][c]) for c in feat_cols] for i in ids])
    F_rc = np.array([[float(rc_feat[i][c]) for c in feat_cols] for i in ids])

    def cascade(S, F):
        X = np.hstack([S, F])
        pred = (S[:, pi] > 0).astype(int)
        unsure = np.abs(S[:, pi]) < band
        if unsure.any():
            pred[unsure] = (art["fusion"].predict_proba(X[unsure])[:, 1] > 0.5).astype(int)
        return pred, float(unsure.mean())

    def router(S, F):
        pred = np.zeros(len(ids), dtype=int)
        for k, i in enumerate(ids):
            src = clean[i] if F is F_clean else rc_feat[i]
            rd = router_choice(float(src["min_side"]), float(src["hf_energy"]))
            pred[k] = int(S[k, dets.index(rd)] > 0) if rd in dets else 0
        return pred

    print(f"\n{'='*72}")
    print(f"CLEAN vs JPEG-75 RE-ENCODED  (same {len(ids)} images, paired)")
    print("=" * 72)
    print(f"  {'system':24s}{'clean':>9s}{'re-enc':>9s}{'delta':>9s}")
    rows_out = []
    for k, d in enumerate(dets):
        a = float(((S_clean[:, k] > 0) == (y == 1)).mean())
        b = float(((S_rc[:, k] > 0) == (y == 1)).mean())
        rows_out.append((d, a, b))
    pr_c, pr_r = router(S_clean, F_clean), router(S_rc, F_rc)
    rows_out.append(("router (rule)", float((pr_c == y).mean()), float((pr_r == y).mean())))
    pc, esc_c = cascade(S_clean, F_clean)
    pr_, esc_r = cascade(S_rc, F_rc)
    rows_out.append(("ORCH cascade", float((pc == y).mean()), float((pr_ == y).mean())))
    for name, a, b in rows_out:
        print(f"  {name:24s}{a*100:9.2f}{b*100:9.2f}{(b-a)*100:+9.2f}")
    print(f"\n  cascade escalation rate: clean {esc_c*100:.1f}%  re-encoded {esc_r*100:.1f}%")

    print(f"\n  per-cell delta for ORCH cascade (negative = hurt by re-encoding):")
    for c in sorted(set(cell)):
        m = cell == c
        a = float((pc[m] == y[m]).mean())
        b = float((pr_[m] == y[m]).mean())
        print(f"    {c:26s}{a*100:8.2f}{b*100:8.2f}{(b-a)*100:+8.2f}   n={int(m.sum())}")

    print(f"\n  FAKE-side recall (the side re-encoding is expected to break):")
    mf = y == 1
    for k, d in enumerate(dets):
        a = float((S_clean[mf][:, k] > 0).mean())
        b = float((S_rc[mf][:, k] > 0).mean())
        print(f"    {d:24s}{a*100:9.2f}{b*100:9.2f}{(b-a)*100:+9.2f}")
    print(f"    {'ORCH cascade':24s}{float((pc[mf]==1).mean())*100:9.2f}"
          f"{float((pr_[mf]==1).mean())*100:9.2f}"
          f"{(float((pr_[mf]==1).mean())-float((pc[mf]==1).mean()))*100:+9.2f}")


if __name__ == "__main__":
    main()
