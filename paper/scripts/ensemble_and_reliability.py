#!/usr/bin/env python3
"""Reviewer-requested additions, computed offline from the per-image
prediction dumps already produced by the main sweep (no GPU, no re-inference).

  (1) Reliability diagrams for every detector (Reviewer 3).
  (2) A direct test of the ensembling hypothesis stated in the conclusion of
      the submitted version: pairing detectors with different *training
      lineages* should remove a blind spot that pairing for *architectural*
      variety does not (Reviewer 2).

Fake-image sets are identical across all five detectors for every generator,
so fake-accuracy is properly paired everywhere.  Real-image sets are shared
across detectors only for BigGAN and GLIDE (each has its own paired real
split); for FLUX.1/SD3/SDXL each detector drew its own sample from the
162k shared real pool, so ensemble real-accuracy is reported on BigGAN and
GLIDE only and flagged as such.
"""
import os, csv, itertools, json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths resolve against the repository by default, so the script runs from a
# clean clone with no editing.  Override either with an environment variable.
REPO = Path(__file__).resolve().parents[2]
PRED = Path(os.environ.get("BLINDSPOT_PREDICTIONS", REPO / "paper" / "predictions"))
OUT = Path(os.environ.get("BLINDSPOT_OUT", REPO / "paper" / "review_additions"))
OUT.mkdir(parents=True, exist_ok=True)

GENS = ["BigGan", "Glide", "flux", "sd3", "sdxl"]
GEN_LABEL = {"BigGan": "BigGAN", "Glide": "GLIDE", "flux": "FLUX.1",
             "sd3": "SD3", "sdxl": "SDXL"}

# detector -> (score-file subdirectory, training lineage, representation family)
DET = OrderedDict([
    ("UniversalFakeDetect",  ("universalfakedetect",  "ProGAN", "CLIP")),
    ("NPR",                  ("npr",                  "ProGAN", "CNN")),
    ("GenImage++ ResNet-50", ("genimagepp_resnet",    "SDv1.4", "CNN")),
    ("GenImage++ CLIP",      ("genimagepp_clip",      "SDv1.4", "CLIP")),
    ("GenImage++ CLIP-LoRA", ("genimagepp_cliplora",  "SDv1.4", "CLIP")),
])

SHORT = {"UniversalFakeDetect": "UFD", "NPR": "NPR",
         "GenImage++ ResNet-50": "GI++ RN50", "GenImage++ CLIP": "GI++ CLIP",
         "GenImage++ CLIP-LoRA": "GI++ LoRA"}


def load(det, gen):
    """-> dict(basename -> (true_label, predicted P(fake)))."""
    path = PRED / DET[det][0] / f"{gen}.csv"
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[os.path.basename(row["filepath"])] = (
                int(row["true_label"]), float(row["pred_score"]))
    return out


# --------------------------------------------------------------- calibration
def ece(scores, labels, nbins=15):
    """Expected calibration error, 15 equal-width bins, matching the main table."""
    conf = np.where(scores >= 0.5, scores, 1.0 - scores)
    pred = (scores >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, nbins + 1)
    total, n = 0.0, len(scores)
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return total


def reliability_curve(scores, labels, nbins=10):
    """Reliability curve in the same coordinates the reported ECE uses:
    confidence = max(p, 1-p) on [0.5, 1], accuracy = fraction correct in bin.
    Returns bin confidence, bin accuracy, and bin mass so that sparsely
    occupied bins can be drawn smaller rather than read as if they carried
    the same weight as the extremes (most mass sits near confidence 1)."""
    conf = np.where(scores >= 0.5, scores, 1.0 - scores)
    pred = (scores >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, nbins + 1)
    xs, ys, ws = [], [], []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum() < 25:          # too few images to read as a rate
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
        ws.append(m.sum() / len(scores))
    return np.array(xs), np.array(ys), np.array(ws)


def main():
    data = {d: {g: load(d, g) for g in GENS} for d in DET}

    # ---------------------------------------------------- (1) reliability
    # Sized so that, scaled to a Procedia text width of ~6.9in, the labels
    # still render near 9pt: the panels are drawn large and the type larger.
    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.9), sharex=True, sharey=True)
    colors = ["#4C3B8F", "#2E7FA6", "#2FA37C", "#8DB032", "#E0A52E"]
    rel_rows = []
    for ax, det in zip(axes, DET):
        ax.fill_between([0.5, 1.0], [0.30, 0.30], [0.5, 1.0],
                        color="#C94F4F", alpha=0.055, zorder=0, linewidth=0)
        ax.plot([0.5, 1], [0.5, 1], ls=(0, (5, 4)), lw=1.3, color="0.3", zorder=2)
        for c, g in zip(colors, GENS):
            d = data[det][g]
            lab = np.array([v[0] for v in d.values()])
            sc = np.array([v[1] for v in d.values()])
            x, y, w = reliability_curve(sc, lab)
            ax.plot(x, y, "-", lw=2.0, color=c, label=GEN_LABEL[g], zorder=3)
            ax.scatter(x, y, s=22 + 230 * w, color=c, edgecolor="white",
                       linewidth=0.9, zorder=4)
            rel_rows.append(dict(detector=det, generator=GEN_LABEL[g],
                                 ece=round(ece(sc, lab), 4)))
        ax.set_title(SHORT[det], fontsize=16, pad=7)
        ax.set_xlim(0.5, 1.0); ax.set_ylim(0.30, 1.02)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xlabel("confidence", fontsize=14)
        ax.grid(alpha=0.18, lw=0.8)
        ax.tick_params(labelsize=12.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("accuracy in bin", fontsize=14)
    axes[0].legend(fontsize=12, loc="upper left", frameon=False,
                   handlelength=1.3, labelspacing=0.28, borderpad=0.1)
    axes[2].text(0.75, 0.335, "overconfident", fontsize=12.5, color="#A85252",
                 style="italic", ha="center")
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(str(OUT), "fig_reliability_diagrams.png"), dpi=220)
    fig.savefig(os.path.join(str(OUT), "fig_reliability_diagrams.pdf"))
    plt.close(fig)

    with open(os.path.join(str(OUT), "reliability_ece.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "generator", "ece"])
        w.writeheader(); w.writerows(rel_rows)

    # ------------------------------------------------------- (2) ensembles
    # Mean-probability fusion at the same fixed 0.5 threshold used throughout.
    ens_rows = []
    for a, b in itertools.combinations(DET, 2):
        la, ra = DET[a][1], DET[a][2]
        lb, rb = DET[b][1], DET[b][2]
        same_lineage = (la == lb)
        same_repr = (ra == rb)
        per_gen_fake, per_gen_real, per_gen_auc = {}, {}, {}
        for g in GENS:
            da, db = data[a][g], data[b][g]
            # fakes: identical sets across detectors -> always paired
            kf = [k for k, v in da.items() if v[0] == 1 and k in db]
            sf = np.array([(da[k][1] + db[k][1]) / 2.0 for k in kf])
            per_gen_fake[g] = 100.0 * (sf >= 0.5).mean()
            # reals: shared across detectors for BigGAN/GLIDE only
            kr = [k for k, v in da.items() if v[0] == 0 and k in db]
            if g in ("BigGan", "Glide") and len(kr) > 5000:
                sr = np.array([(da[k][1] + db[k][1]) / 2.0 for k in kr])
                per_gen_real[g] = 100.0 * (sr < 0.5).mean()
                y = np.r_[np.zeros(len(sr)), np.ones(len(sf))]
                s = np.r_[sr, sf]
                order = np.argsort(s)
                ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
                # average ranks over ties
                ss = s[order]
                i = 0
                while i < len(ss):
                    j = i
                    while j + 1 < len(ss) and ss[j + 1] == ss[i]:
                        j += 1
                    if j > i:
                        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
                    i = j + 1
                n1 = y.sum(); n0 = len(y) - n1
                per_gen_auc[g] = 100.0 * ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))
        fakes = [per_gen_fake[g] for g in GENS]
        ens_rows.append(dict(
            pair=f"{SHORT[a]} + {SHORT[b]}",
            lineage=("same" if same_lineage else "different"),
            representation=("same" if same_repr else "different"),
            **{f"fake_{GEN_LABEL[g]}": round(per_gen_fake[g], 2) for g in GENS},
            mean_fake=round(float(np.mean(fakes)), 2),
            worst_fake=round(float(np.min(fakes)), 2),
            spread_fake=round(float(np.max(fakes) - np.min(fakes)), 2),
            real_BigGAN=round(per_gen_real.get("BigGan", float("nan")), 2),
            real_GLIDE=round(per_gen_real.get("Glide", float("nan")), 2),
            auc_BigGAN=round(per_gen_auc.get("BigGan", float("nan")), 2),
            auc_GLIDE=round(per_gen_auc.get("Glide", float("nan")), 2),
        ))

    # single detectors, same metrics, for the baseline column
    single_rows = []
    for d in DET:
        fakes = []
        for g in GENS:
            dd = data[d][g]
            sf = np.array([v[1] for v in dd.values() if v[0] == 1])
            fakes.append(100.0 * (sf >= 0.5).mean())
        single_rows.append(dict(detector=SHORT[d], lineage=DET[d][1],
                                representation=DET[d][2],
                                mean_fake=round(float(np.mean(fakes)), 2),
                                worst_fake=round(float(np.min(fakes)), 2),
                                spread_fake=round(float(np.max(fakes) - np.min(fakes)), 2)))

    with open(os.path.join(str(OUT), "ensemble_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ens_rows[0].keys()))
        w.writeheader(); w.writerows(sorted(ens_rows, key=lambda r: -r["worst_fake"]))
    with open(os.path.join(str(OUT), "single_detectors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(single_rows[0].keys()))
        w.writeheader(); w.writerows(single_rows)

    # group summary: does lineage diversity beat representation diversity?
    def grp(rows, lin, rep):
        sel = [r for r in rows if r["lineage"] == lin and r["representation"] == rep]
        if not sel: return None
        return dict(n=len(sel),
                    mean_worst=round(float(np.mean([r["worst_fake"] for r in sel])), 2),
                    mean_mean=round(float(np.mean([r["mean_fake"] for r in sel])), 2),
                    pairs=[r["pair"] for r in sel])
    summary = {f"lineage_{l}__repr_{r}": grp(ens_rows, l, r)
               for l in ("same", "different") for r in ("same", "different")}
    with open(os.path.join(str(OUT), "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== SINGLE DETECTORS (fake-acc, %) ===")
    for r in single_rows:
        print(f"  {r['detector']:<10} {r['lineage']:<7} {r['representation']:<5} "
              f"mean={r['mean_fake']:>6}  worst={r['worst_fake']:>6}  spread={r['spread_fake']:>6}")
    print("\n=== ENSEMBLE PAIRS (mean-probability fusion, threshold 0.5) ===")
    hdr = f"  {'pair':<24}{'lineage':<11}{'repr':<11}{'mean':>7}{'worst':>7}{'spread':>8}{'realBG':>8}{'realGL':>8}"
    print(hdr)
    for r in sorted(ens_rows, key=lambda r: -r["worst_fake"]):
        print(f"  {r['pair']:<24}{r['lineage']:<11}{r['representation']:<11}"
              f"{r['mean_fake']:>7}{r['worst_fake']:>7}{r['spread_fake']:>8}"
              f"{r['real_BigGAN']:>8}{r['real_GLIDE']:>8}")
    print("\n=== GROUP MEANS ===")
    for k, v in summary.items():
        if v: print(f"  {k:<34} n={v['n']}  mean_worst={v['mean_worst']:>6}  mean_mean={v['mean_mean']:>6}")
    print("\nwrote ->", OUT)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Paired significance test for the headline ensemble contrast, run separately
# so the main pass stays readable.  Both ensembles contain GenImage++ CLIP;
# they differ only in whether the partner shares its training lineage.
def ensemble_mcnemar():
    data = {d: {g: load(d, g) for g in GENS} for d in DET}
    anchor = "GenImage++ CLIP"
    A, B = "NPR", "GenImage++ ResNet-50"          # cross-lineage vs same-lineage partner
    rows = []
    for g in GENS:
        da, db, dc = data[anchor][g], data[A][g], data[B][g]
        keys = [k for k, v in da.items() if v[0] == 1 and k in db and k in dc]
        ca = np.array([((da[k][1] + db[k][1]) / 2.0 >= 0.5) for k in keys])
        cb = np.array([((da[k][1] + dc[k][1]) / 2.0 >= 0.5) for k in keys])
        n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
        n = n01 + n10
        if n == 0:
            p = 1.0
        else:
            # exact two-sided binomial on the discordant pairs; computed in
            # log space because n runs into the thousands here
            from scipy.stats import binomtest
            p = binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue
        rows.append(dict(generator=GEN_LABEL[g], n_fake=len(keys),
                         crosslineage_correct_only=n01, samelineage_correct_only=n10,
                         p_value=f"{p:.3g}"))
    with open(os.path.join(str(OUT), "ensemble_mcnemar.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n=== McNEMAR: (GI++CLIP + NPR) vs (GI++CLIP + GI++RN50), fake images ===")
    for r in rows:
        print(f"  {r['generator']:<8} n={r['n_fake']:>6}  n01={r['crosslineage_correct_only']:>6}"
              f"  n10={r['samelineage_correct_only']:>6}  p={r['p_value']}")
