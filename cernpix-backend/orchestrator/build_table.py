#!/usr/bin/env python3
"""
Stage 5 -- join manifest + features + detector scores into the training table,
and report what an orchestrator could actually gain over the current router.

Emits data/training_table.csv, one row per image:

  image_id, path, label, generator, gen_fold, split, source, human_slice
  <28 feature columns>                          (see features.py)
  score_<det>       sigmoid output per detector, [0,1]
  correct_<det>     1 if that detector alone would have been right at 0.5
  origin_<det>      "harvested" or "computed"
  router_detector   what rule_engine.select_detector picks for this image
  router_score      that detector's score -- the deployed system's output
  router_correct    whether the deployed system would have been right

Rows missing any detector score are dropped, and the count is reported: a
ragged table would silently bias every comparison toward whichever detectors
happen to cover the easy images.

The report at the end is the decision input for the orchestrator design:
  * best single detector -- the bar to beat
  * the router as deployed -- what we are replacing
  * oracle over all detectors -- the ceiling, and deliberately NOT a target
    (with 5 detectors, "at least one was right" is high even when each is
    near chance on that image; it bounds the gain, it does not promise it)
  * majority vote -- the naive fusion baseline
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pools import DETECTORS, ROUTER_DETECTORS
from features import SHORTCUT_FEATURES          # noqa: F401  (re-exported for trainers)
from config import NPR_HF_ENERGY_MAX, UFD_MAX_MIN_SIDE, UFD_HF_ENERGY_MAX

THRESH = 0.5          # app.py's current flag threshold, and the paper's tables
ALT_THRESH = 0.2      # the committed value app.py used before the working-tree
                      # edit; kept as a sensitivity probe, see the report below


def router_choice(min_side, hf_energy):
    """rule_engine.select_detector, reduced to the detector key it returns.

    Reimplemented on raw feature values rather than imported, because
    select_detector builds an explanatory string per call and we evaluate it
    over ~72k rows. The branch structure is copied verbatim; if rule_engine's
    thresholds move, this reads them from the same config constants.
    """
    if hf_energy < NPR_HF_ENERGY_MAX:
        return "npr"
    if min_side < UFD_MAX_MIN_SIDE and hf_energy < UFD_HF_ENERGY_MAX:
        return "ufd"
    return "cliplora"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--scores", nargs="+",
                    default=["data/scores_harvested.csv", "data/scores_new.csv"])
    ap.add_argument("--out", default="data/training_table.csv")
    ap.add_argument("--detectors", nargs="+", default=None,
                    help="require only this subset of detectors per row. The "
                         "augmented rows are scored with the 3 that actually "
                         "ship, so demanding all 5 would silently drop them.")
    ap.add_argument("--manifest-extra", nargs="+", default=[],
                    help="additional manifests (e.g. augmented rows) to append")
    ap.add_argument("--features-extra", nargs="+", default=[],
                    help="additional feature files matching --manifest-extra")
    args = ap.parse_args()
    detectors = args.detectors or DETECTORS

    manifest = {}
    for mp in [args.manifest] + args.manifest_extra:
        with open(mp) as f:
            for r in csv.DictReader(f):
                manifest[r["image_id"]] = r
    feats = {}
    for fp in [args.features] + args.features_extra:
        with open(fp) as f:
            for r in csv.DictReader(f):
                feats[r["image_id"]] = r
    feat_cols = [c for c in (next(iter(feats.values())).keys()) if c != "image_id"]

    scores = defaultdict(dict)
    origins = defaultdict(dict)
    for sp in args.scores:
        if not os.path.exists(sp):
            print(f"  (no {sp}, skipping)")
            continue
        with open(sp) as f:
            for r in csv.DictReader(f):
                scores[r["image_id"]][r["detector"]] = float(r["score"])
                origins[r["image_id"]][r["detector"]] = r["origin"]

    rows, dropped_feat, dropped_score = [], 0, 0
    for iid, m in manifest.items():
        if iid not in feats:
            dropped_feat += 1
            continue
        s = scores.get(iid, {})
        if not all(d in s for d in detectors):
            dropped_score += 1
            continue
        y = int(m["label"])
        row = {k: m[k] for k in ("image_id", "path", "label", "generator",
                                 "gen_fold", "split", "source", "human_slice")}
        row.update({c: feats[iid][c] for c in feat_cols})
        for d in detectors:
            row[f"score_{d}"] = f"{s[d]:.6f}"
            row[f"correct_{d}"] = int((s[d] > THRESH) == (y == 1))
            row[f"origin_{d}"] = origins[iid][d]
        rd = router_choice(float(feats[iid]["min_side"]), float(feats[iid]["hf_energy"]))
        row["router_detector"] = rd
        row["router_score"] = f"{s[rd]:.6f}"
        row["router_correct"] = int((s[rd] > THRESH) == (y == 1))
        rows.append(row)

    print(f"joined {len(rows)} rows "
          f"(dropped: {dropped_feat} missing features, {dropped_score} missing scores)")

    cols = (["image_id", "path", "label", "generator", "gen_fold", "split",
             "source", "human_slice"] + feat_cols +
            [f"score_{d}" for d in detectors] +
            [f"correct_{d}" for d in detectors] +
            [f"origin_{d}" for d in detectors] +
            ["router_detector", "router_score", "router_correct"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}  ({len(cols)} columns)")

    # ---------------- headroom report ----------------
    if not rows:
        return
    n = len(rows)
    print(f"\n{'='*66}\nHEADROOM REPORT  (n={n}, threshold={THRESH})\n{'='*66}")
    acc = {d: sum(int(r[f"correct_{d}"]) for r in rows) / n for d in detectors}
    for d in sorted(acc, key=acc.get, reverse=True):
        print(f"  {d:14s} {acc[d]*100:6.2f}%")
    best = max(acc, key=acc.get)
    router = sum(int(r["router_correct"]) for r in rows) / n
    oracle5 = sum(any(int(r[f"correct_{d}"]) for d in detectors) for r in rows) / n
    rdets = [d for d in ROUTER_DETECTORS if d in detectors]
    oracle3 = sum(any(int(r[f"correct_{d}"]) for d in rdets) for r in rows) / n
    mv = sum(((sum(float(r[f"score_{d}"]) > THRESH for d in rdets) >= 2)
              == (int(r["label"]) == 1)) for r in rows) / n
    print(f"\n  router (deployed)   {router*100:6.2f}%")
    print(f"  best single ({best:<12s}) {acc[best]*100:6.2f}%   <- the bar to beat")
    print(f"  majority vote of 3  {mv*100:6.2f}%   ({(mv-acc[best])*100:+.2f}pp vs best)")
    print(f"  ORACLE over 3       {oracle3*100:6.2f}%   ceiling, not a target")
    print(f"  ORACLE over 5       {oracle5*100:6.2f}%")
    print(f"\n  headroom best->oracle5: {(oracle5-acc[best])*100:+.2f}pp")
    print(f"  router is {(acc[best]-router)*100:+.2f}pp vs just always calling {best}")

    err = [r for r in rows if not int(r[f"correct_{best}"])]
    rec = sum(any(int(r[f"correct_{d}"]) for d in detectors if d != best) for r in err)
    print(f"  {best} errors: {len(err)}  recoverable by another detector: {rec} "
          f"({100*rec/max(len(err),1):.1f}%)")

    # app.py currently flags at 0.5 (matching THRESH above), but HEAD still
    # has 0.2 -- a more sensitive point trading false positives for recall.
    # Reporting both shows how much of any accuracy claim is threshold choice
    # rather than model quality, and the orchestrator's own operating point
    # should be tuned against an explicit cost preference rather than
    # inherited from either value.
    r_app = sum(int((float(r["router_score"]) > ALT_THRESH) == (int(r["label"]) == 1))
                for r in rows) / n
    best_app = sum(int((float(r[f"score_{best}"]) > ALT_THRESH) == (int(r["label"]) == 1))
                   for r in rows) / n
    print(f"\n  threshold sensitivity (score > {ALT_THRESH}, the committed value):")
    print(f"    router        {r_app*100:6.2f}%   ({(r_app-router)*100:+.2f}pp vs its 0.5 number)")
    print(f"    {best:<13s} {best_app*100:6.2f}%   ({(best_app-acc[best])*100:+.2f}pp)")

    print("\n  router detector usage:")
    use = defaultdict(int)
    for r in rows:
        use[r["router_detector"]] += 1
    for d, c in sorted(use.items(), key=lambda kv: -kv[1]):
        print(f"    {d:14s} {c:6d}  ({100*c/n:5.1f}%)")

    print("\n  per-generator accuracy (fakes) / per-source (reals):")
    cells = sorted({r["gen_fold"] + "|" + r["source"] for r in rows})
    hdr = "    " + f"{'cell':26s}" + "".join(f"{d[:9]:>10s}" for d in detectors) + f"{'router':>10s}"
    print(hdr)
    for c in cells:
        sub = [r for r in rows if r["gen_fold"] + "|" + r["source"] == c]
        line = "    " + f"{c[:26]:26s}"
        for d in detectors:
            line += f"{100*sum(int(r[f'correct_{d}']) for r in sub)/len(sub):10.1f}"
        line += f"{100*sum(int(r['router_correct']) for r in sub)/len(sub):10.1f}"
        print(line + f"   n={len(sub)}")


if __name__ == "__main__":
    main()
