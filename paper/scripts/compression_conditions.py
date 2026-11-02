#!/usr/bin/env python3
"""Reviewer 2's request: a quantitative estimate of the PNG-versus-JPEG
confound in these benchmarks.

Three conditions on the SAME 2,000-real / 2,000-fake FLUX.1 sample per
detector, so all three are paired within a detector:

  clean    generated images as distributed (PNG), real images as
           distributed (ImageNet JPEG).  This is the confounded condition
           every published number on these benchmarks is measured in.
  matched  generated images re-encoded to JPEG q=75, real images untouched.
           Compression history is now the same kind on both sides, so the
           format asymmetry is removed while the reals stay exactly as the
           benchmark ships them.  The clean-to-matched drop is our estimate
           of how much apparent skill the asymmetry was supplying.
  both     every image re-encoded at q=75, the deployment-style robustness
           check already reported in the submitted version.
"""
import os, csv
from pathlib import Path
import numpy as np

# Paths resolve against the repository by default; override with an
# environment variable to point at score files elsewhere.
REPO = Path(__file__).resolve().parents[2]
PRED = Path(os.environ.get("BLINDSPOT_PREDICTIONS", REPO / "paper" / "predictions"))
COND = PRED / "conditions"
OUT = Path(os.environ.get("BLINDSPOT_OUT", REPO / "paper" / "review_additions"))
OUT.mkdir(parents=True, exist_ok=True)

# detector -> (clean, fakes-only JPEG q75, both sides JPEG q75) score files,
# all three measured on the identical 2,000-per-class FLUX.1 sample.
DET = [
    ("UniversalFakeDetect",  "ufd_flux_clean2000.csv",
                             "ufd_flux_matched2000.csv",
                             "ufd_flux_jpeg752000.csv"),
    ("NPR",                  "npr_flux_clean2000.csv",
                             "npr_flux_matched2000.csv",
                             "npr_flux_jpeg752000.csv"),
    ("GenImage++ ResNet-50", "resnet_flux_clean2000.csv",
                             "resnet_flux_matched2000.csv",
                             "resnet_flux_jpeg752000.csv"),
    ("GenImage++ CLIP",      "clipbaseline_flux_clean2000.csv",
                             "clipbaseline_flux_matched2000.csv",
                             "clipbaseline_flux_jpeg752000.csv"),
    ("GenImage++ CLIP-LoRA", "cliplora_flux_clean2000.csv",
                             "cliplora_flux_matched2000.csv",
                             "cliplora_flux_jpeg752000.csv"),
]


def read(path):
    lab, sc = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            lab.append(int(r["true_label"])); sc.append(float(r["pred_score"]))
    return np.array(lab), np.array(sc)


def auc(lab, sc):
    order = np.argsort(sc); ranks = np.empty(len(sc), float)
    ranks[order] = np.arange(1, len(sc) + 1)
    ss = sc[order]; i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = lab.sum(); n0 = len(lab) - n1
    return 100.0 * ((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def metrics(path):
    lab, sc = read(path)
    pred = (sc >= 0.5).astype(int)
    return dict(n_real=int((lab == 0).sum()), n_fake=int((lab == 1).sum()),
                fake=100.0 * (pred[lab == 1] == 1).mean(),
                real=100.0 * (pred[lab == 0] == 0).mean(),
                auc=auc(lab, sc))


rows = []
for name, cf, mf, bf in DET:
    c, m, b = COND / cf, COND / mf, COND / bf
    if not all(p.exists() for p in (c, m, b)):
        print(f"[skip] {name}: missing a condition"); continue
    mc, mm, mb = metrics(c), metrics(m), metrics(b)
    rows.append(dict(detector=name,
                     fake_clean=round(mc["fake"], 2), fake_matched=round(mm["fake"], 2),
                     fake_both=round(mb["fake"], 2),
                     auc_clean=round(mc["auc"], 2), auc_matched=round(mm["auc"], 2),
                     auc_both=round(mb["auc"], 2),
                     real_clean=round(mc["real"], 2), real_matched=round(mm["real"], 2),
                     d_fake_confound=round(mm["fake"] - mc["fake"], 2),
                     d_auc_confound=round(mm["auc"] - mc["auc"], 2)))

with open(os.path.join(str(OUT), "compression_conditions.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"{'detector':<22}{'fake clean':>11}{'matched':>9}{'both':>8}"
      f"{'AUC clean':>11}{'matched':>9}{'both':>8}{'dAUC':>8}")
for r in rows:
    print(f"{r['detector']:<22}{r['fake_clean']:>11}{r['fake_matched']:>9}{r['fake_both']:>8}"
          f"{r['auc_clean']:>11}{r['auc_matched']:>9}{r['auc_both']:>8}{r['d_auc_confound']:>8}")
print("\nwrote ->", OUT / "compression_conditions.csv")
