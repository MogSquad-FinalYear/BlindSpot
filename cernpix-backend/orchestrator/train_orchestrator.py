#!/usr/bin/env python3
"""
Stage 6 -- train and evaluate the orchestrator that replaces the rule router.

Three strategies are trained from the same table, because they trade accuracy
against inference cost differently and the choice should be made on measured
numbers rather than on which one sounds more sophisticated:

  fusion     every detector runs; their scores (in logit space) plus the
             cheap image features feed a meta-classifier that makes the final
             real/fake call.                         cost: N detector calls
  selection  features only decide WHICH single detector to trust; that
             detector's score is the answer. This is the learned version of
             what rule_engine.py does by hand.       cost: 1 detector call
  cascade    cliplora runs first; if its logit lands inside an uncertainty
             band the remaining detectors run and the fusion head decides,
             otherwise cliplora's own call stands.   cost: 1 + p*(N-1)

Two evaluations are reported, and they answer different questions:

  in-distribution   the stratified test split. Flatters every model, because
                    the same generators appear in training.
  leave-one-generator-out   train with one generator's fakes entirely absent,
                    test on them. This is the question BlindSpot is about --
                    a detector stack that only works on generators it has
                    already seen is not a detector stack, and an orchestrator
                    can overfit to generator identity even more easily than a
                    detector can.

SHORTCUT_FEATURES are excluded by default; --with-shortcuts reports what they
would add, which measures the PNG/JPEG confound rather than model quality.

Usage:
    python3 train_orchestrator.py
    python3 train_orchestrator.py --with-shortcuts
    python3 train_orchestrator.py --detectors npr ufd cliplora
"""
import argparse
import csv
import json
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pools import DETECTORS, ROUTER_DETECTORS               # noqa: E402
from features import SHORTCUT_FEATURES, GEOMETRY_FEATURES   # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
META = {"image_id", "path", "label", "generator", "gen_fold", "split",
        "source", "human_slice", "router_detector", "router_score",
        "router_correct"}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_table(path, detectors, with_shortcuts, no_geometry=False):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    score_cols = [f"score_{d}" for d in detectors]
    correct_cols = [f"correct_{d}" for d in detectors]
    drop = META | set(f"score_{d}" for d in DETECTORS) | \
        set(f"correct_{d}" for d in DETECTORS) | set(f"origin_{d}" for d in DETECTORS)
    feat_cols = [c for c in rows[0] if c not in drop]
    if not with_shortcuts:
        feat_cols = [c for c in feat_cols if c not in SHORTCUT_FEATURES]
    if no_geometry:
        feat_cols = [c for c in feat_cols if c not in GEOMETRY_FEATURES]

    X_feat = np.array([[float(r[c]) for c in feat_cols] for r in rows])
    X_score = logit(np.array([[float(r[c]) for c in score_cols] for r in rows]))
    y = np.array([int(r["label"]) for r in rows])
    correct = np.array([[int(r[c]) for c in correct_cols] for r in rows])
    gen_fold = np.array([r["gen_fold"] for r in rows])
    split = np.array([r["split"] for r in rows])
    router_correct = np.array([int(r["router_correct"]) for r in rows])
    return dict(rows=rows, feat_cols=feat_cols, X_feat=X_feat, X_score=X_score,
                y=y, correct=correct, gen_fold=gen_fold, split=split,
                router_correct=router_correct, detectors=detectors)


# ----------------------------------------------------------------- strategies
def fit_fusion(Xtr, ytr):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         max_leaf_nodes=31, random_state=0)
    clf.fit(Xtr, ytr)
    return clf


def fit_selection(F_tr, correct_tr, detectors):
    """Learned router: predict which detector is right, from features alone.

    Target is the index of a detector that got this image right (preferring
    the globally strongest one when several did, so the model is not asked to
    break ties arbitrarily). Images no detector got right carry no usable
    supervision and are dropped.
    """
    order = np.argsort(-correct_tr.mean(axis=0))           # strongest first
    target, keep = [], []
    for i, row in enumerate(correct_tr):
        for d in order:
            if row[d]:
                target.append(d)
                keep.append(i)
                break
    if not keep:
        return None, None
    clf = HistGradientBoostingClassifier(max_iter=250, random_state=0)
    clf.fit(F_tr[keep], np.array(target))
    return clf, order[0]


def split_acc(pred, y):
    """Overall plus class-conditional accuracy.

    LOGO folds mix one generator's fakes with a fixed real test set, and the
    fake:real ratio differs per generator (6k:4.5k for BigGan, 18.3k:4.5k for
    sdxl), so a single overall number is not comparable across folds.
    acc_fake is recall on the unseen generator -- the quantity the held-out
    evaluation is actually asking about.
    """
    pred = np.asarray(pred).astype(int)
    m_f, m_r = (y == 1), (y == 0)
    return {
        "acc": float((pred == y).mean()),
        "acc_fake": float((pred[m_f] == 1).mean()) if m_f.any() else float("nan"),
        "acc_real": float((pred[m_r] == 0).mean()) if m_r.any() else float("nan"),
    }


def run_fold(data, tr, te, args):
    """Train every strategy on `tr`, evaluate on `te`. Returns a result dict."""
    dets = data["detectors"]
    F, S, y, C = data["X_feat"], data["X_score"], data["y"], data["correct"]
    Xfull = np.hstack([S, F])
    out = {}

    # --- baselines -----------------------------------------------------
    acc_single = {d: float((C[te][:, i] == 1).mean()) for i, d in enumerate(dets)}
    best_det = max(acc_single, key=acc_single.get)
    bi = dets.index(best_det)
    out["best_single"] = {"detector": best_det,
                          **split_acc((S[te][:, bi] > 0).astype(int), y[te])}
    out["per_detector"] = acc_single
    out["router"] = {"acc": float(data["router_correct"][te].mean())}
    mv_idx = [dets.index(d) for d in ROUTER_DETECTORS if d in dets]
    if len(mv_idx) >= 3:
        votes = (S[te][:, mv_idx] > 0).sum(axis=1) >= 2
        out["majority_vote"] = {"acc": float((votes == (y[te] == 1)).mean())}
    out["oracle"] = {"acc": float(C[te].any(axis=1).mean())}

    # --- fusion --------------------------------------------------------
    fus = fit_fusion(Xfull[tr], y[tr])
    p_fus = fus.predict_proba(Xfull[te])[:, 1]
    out["fusion"] = {**split_acc(p_fus > 0.5, y[te]),
                     "auc": float(roc_auc_score(y[te], p_fus)),
                     "cost_calls": float(len(dets))}

    # fusion on scores alone -- isolates how much the image features add
    fus_s = fit_fusion(S[tr], y[tr])
    p_fs = fus_s.predict_proba(S[te])[:, 1]
    out["fusion_scores_only"] = {"acc": float(((p_fs > 0.5) == (y[te] == 1)).mean()),
                                 "auc": float(roc_auc_score(y[te], p_fs))}

    # --- selection -----------------------------------------------------
    sel, fallback = fit_selection(F[tr], C[tr], dets)
    if sel is not None:
        pick = sel.predict(F[te])
        chosen = S[te][np.arange(len(te)), pick]
        out["selection"] = {
            **split_acc(chosen > 0, y[te]),
            "cost_calls": 1.0,
            "pick_dist": {dets[i]: int((pick == i).sum()) for i in range(len(dets))},
        }

    # --- cascade -------------------------------------------------------
    # cliplora first; escalate to fusion inside an uncertainty band. The band
    # is tuned on a held-out slice of the TRAINING fold, scored by a fusion
    # model that never saw that slice -- tuning it against `fus`'s predictions
    # on `fus`'s own training rows would read the in-sample (near-perfect)
    # error rate and pick an artificially narrow band.
    if "cliplora" in dets:
        li = dets.index("cliplora")
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(tr))
        cut = int(len(tr) * 0.8)
        fit_idx, tune_idx = tr[perm[:cut]], tr[perm[cut:]]
        fus_tune = fit_fusion(Xfull[fit_idx], y[fit_idx])
        best_band, best_acc, best_rate = 0.0, -1.0, 0.0
        for band in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
            unsure_t = np.abs(S[tune_idx][:, li]) < band
            pred_t = (S[tune_idx][:, li] > 0).astype(int)
            if unsure_t.any():
                pred_t[unsure_t] = (fus_tune.predict_proba(
                    Xfull[tune_idx][unsure_t])[:, 1] > 0.5).astype(int)
            a = float((pred_t == y[tune_idx]).mean())
            if a > best_acc:
                best_acc, best_band, best_rate = a, band, float(unsure_t.mean())
        unsure = np.abs(S[te][:, li]) < best_band
        pred = (S[te][:, li] > 0).astype(int)
        if unsure.any():
            pred[unsure] = (fus.predict_proba(Xfull[te][unsure])[:, 1] > 0.5).astype(int)
        out["cascade"] = {
            **split_acc(pred, y[te]),
            "band": best_band,
            "escalation_rate": float(unsure.mean()),
            "cost_calls": float(1 + unsure.mean() * (len(dets) - 1)),
        }
    return out


def fmt(out, title, n):
    lines = [f"\n{'='*70}", f"{title}  (n_test={n})", "=" * 70]
    b = out["best_single"]
    lines.append(f"  {'router (deployed rule)':32s} {out['router']['acc']*100:6.2f}%")
    lines.append(f"  {'best single: '+b['detector']:32s} {b['acc']*100:6.2f}%   <- bar to beat")
    if "majority_vote" in out:
        lines.append(f"  {'majority vote of 3':32s} {out['majority_vote']['acc']*100:6.2f}%")
    lines.append(f"  {'oracle (ceiling, not a target)':32s} {out['oracle']['acc']*100:6.2f}%")
    lines.append("  " + "-" * 62)
    for key, label in [("selection", "ORCH selection (1 call)"),
                       ("cascade", "ORCH cascade"),
                       ("fusion_scores_only", "ORCH fusion, scores only"),
                       ("fusion", "ORCH fusion (all calls)")]:
        if key not in out:
            continue
        r = out[key]
        delta = (r["acc"] - b["acc"]) * 100
        cost = r.get("cost_calls")
        cost_s = f"  cost={cost:.2f} calls" if cost else ""
        extra = ""
        if key == "cascade":
            extra = f"  band=+-{r['band']:.1f}  escalate={r['escalation_rate']*100:.1f}%"
        lines.append(f"  {label:32s} {r['acc']*100:6.2f}%  ({delta:+.2f}pp){cost_s}{extra}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/training_table.csv")
    ap.add_argument("--detectors", nargs="+", default=None)
    ap.add_argument("--with-shortcuts", action="store_true")
    ap.add_argument("--no-geometry", action="store_true",
                    help="drop native width/height/min_side features; tests whether "
                         "the orchestrator is leaning on resolution as a generator proxy")
    ap.add_argument("--out", default="data/orchestrator_results.json")
    ap.add_argument("--skip-logo", action="store_true")
    ap.add_argument("--save-model", default="data/orchestrator_model.pkl",
                    help="persist the fitted in-distribution orchestrator for serving")
    args = ap.parse_args()

    dets = args.detectors or DETECTORS
    data = load_table(args.table, dets, args.with_shortcuts, args.no_geometry)
    n = len(data["y"])
    print(f"table: n={n}  detectors={dets}")
    print(f"features ({len(data['feat_cols'])}): {', '.join(data['feat_cols'])}")
    if args.with_shortcuts:
        print(f"  !! including shortcut features {SHORTCUT_FEATURES} -- "
              f"numbers below measure the PNG/JPEG confound, not generalization")

    results = {}
    # ---- in-distribution ----
    tr = np.where(np.isin(data["split"], ["train", "val"]))[0]
    te = np.where(data["split"] == "test")[0]
    r = run_fold(data, tr, te, args)
    results["in_distribution"] = r
    print(fmt(r, "IN-DISTRIBUTION (stratified test split)", len(te)))

    if args.save_model:
        # Refit on train+val and persist everything serve.py needs to
        # reproduce the decision: the fusion head, the selection head, the
        # tuned cascade band, and -- critically -- the exact feature order,
        # since a column permutation at serve time fails silently.
        S, F, C, y = data["X_score"], data["X_feat"], data["correct"], data["y"]
        Xfull = np.hstack([S, F])
        fus = fit_fusion(Xfull[tr], y[tr])
        sel, _ = fit_selection(F[tr], C[tr], dets)
        artifact = {
            "detectors": dets,
            "feat_cols": data["feat_cols"],
            "fusion": fus,
            "selection": sel,
            "cascade_band": r.get("cascade", {}).get("band"),
            "cascade_primary": "cliplora" if "cliplora" in dets else dets[0],
            "shortcuts_included": args.with_shortcuts,
            "geometry_excluded": args.no_geometry,
            "trained_on": {"n": int(len(tr)), "table": args.table},
            "in_distribution_acc": {k: r[k]["acc"] for k in
                                    ("fusion", "cascade", "selection", "router",
                                     "best_single") if k in r},
        }
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_model, "wb") as f:
            pickle.dump(artifact, f)
        print(f"\nsaved deployable orchestrator -> {args.save_model}")

    # ---- leave-one-generator-out ----
    if not args.skip_logo:
        gens = sorted({g for g in data["gen_fold"] if g != "real"})
        logo = {}
        for g in gens:
            te_g = np.where(data["gen_fold"] == g)[0]
            # hold out this generator's fakes entirely; reals stay (they are
            # generator-agnostic), but only the ones in the test split, so the
            # real side is never trained and tested on the same images.
            real_te = np.where((data["gen_fold"] == "real") & (data["split"] == "test"))[0]
            te_idx = np.concatenate([te_g, real_te])
            tr_idx = np.where((data["gen_fold"] != g) & (data["gen_fold"] != "real") |
                              ((data["gen_fold"] == "real") & (data["split"] != "test")))[0]
            rr = run_fold(data, tr_idx, te_idx, args)
            logo[g] = rr
            print(fmt(rr, f"LEAVE-ONE-GENERATOR-OUT: held out {g}", len(te_idx)))
        results["logo"] = logo

        print(f"\n{'='*70}\nLOGO SUMMARY (accuracy on the unseen generator + real test set)\n{'='*70}")
        print(f"  {'held-out':10s}{'router':>9s}{'best1':>9s}{'select':>9s}{'cascade':>9s}{'fusion':>9s}")
        for g, rr in logo.items():
            print(f"  {g:10s}{rr['router']['acc']*100:9.2f}{rr['best_single']['acc']*100:9.2f}"
                  f"{rr.get('selection',{}).get('acc',float('nan'))*100:9.2f}"
                  f"{rr.get('cascade',{}).get('acc',float('nan'))*100:9.2f}"
                  f"{rr['fusion']['acc']*100:9.2f}")
        print(f"\n  recall on the UNSEEN generator's fakes (acc_fake):")
        print(f"  {'held-out':10s}{'best1':>9s}{'select':>9s}{'cascade':>9s}{'fusion':>9s}")
        for g, rr in logo.items():
            gv = lambda k: rr.get(k, {}).get("acc_fake", float("nan")) * 100
            print(f"  {g:10s}{gv('best_single'):9.2f}{gv('selection'):9.2f}"
                  f"{gv('cascade'):9.2f}{gv('fusion'):9.2f}")
        mean = lambda k, sub=None: np.mean([
            (rr[k][sub] if sub else rr[k]["acc"]) for rr in logo.values() if k in rr])
        print(f"  {'MEAN':10s}{mean('router')*100:9.2f}{mean('best_single')*100:9.2f}"
              f"{mean('selection')*100:9.2f}{mean('cascade')*100:9.2f}{mean('fusion')*100:9.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
