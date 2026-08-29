#!/usr/bin/env python3
"""Emit LaTeX table bodies straight from the computed CSVs, so no number in
the manuscript is transcribed by hand."""
import csv
import os

COMP = "/home/student/data/paper/comparison"
GENS = ["BigGan", "Glide", "flux", "sd3", "sdxl"]
GEN_LABEL = {"BigGan": "BigGAN", "Glide": "GLIDE", "flux": "FLUX.1", "sd3": "SD3", "sdxl": "SDXL"}
ORDER = [
    "UniversalFakeDetect",
    "NPR",
    "GenImage++ ResNet-50 baseline",
    "GenImage++ CLIP baseline",
    "GenImage++ CLIP-LoRA (OMAT)",
]
SHORT = {
    "UniversalFakeDetect": "UniversalFakeDetect",
    "NPR": "NPR",
    "GenImage++ ResNet-50 baseline": "GenImage++ ResNet-50",
    "GenImage++ CLIP baseline": "GenImage++ CLIP",
    "GenImage++ CLIP-LoRA (OMAT)": "GenImage++ CLIP-LoRA",
}


def load_metrics():
    rows = list(csv.DictReader(open(os.path.join(COMP, "full_metrics_all_cells.csv"))))
    d = {}
    for r in rows:
        d[(r["detector"], r["generator"])] = r
    return d


def main():
    m = load_metrics()
    present = [det for det in ORDER if any((det, g) in m for g in GENS)]

    print("%" * 60)
    print("% TABLE: fake-acc / real-acc per cell")
    print("%" * 60)
    for det in present:
        cells = []
        for g in GENS:
            r = m.get((det, g))
            cells.append(f"{float(r['acc_fake']):.1f} / {float(r['acc_real']):.1f}" if r else "--")
        print(f"{SHORT[det]} & " + " & ".join(cells) + r" \\")

    print()
    print("%" * 60)
    print("% TABLE: AUC per cell")
    print("%" * 60)
    for det in present:
        cells = []
        for g in GENS:
            r = m.get((det, g))
            cells.append(f"{float(r['AUC']):.2f}" if r else "--")
        print(f"{SHORT[det]} & " + " & ".join(cells) + r" \\")

    print()
    print("%" * 60)
    print("% TABLE: ECE per cell (calibration)")
    print("%" * 60)
    for det in present:
        cells = []
        for g in GENS:
            r = m.get((det, g))
            cells.append(f"{float(r['ECE']):.3f}" if r else "--")
        print(f"{SHORT[det]} & " + " & ".join(cells) + r" \\")

    print()
    print("%" * 60)
    print("% Mean fake-acc / mean real-acc / mean AUC per detector")
    print("%" * 60)
    for det in present:
        rs = [m[(det, g)] for g in GENS if (det, g) in m]
        if not rs:
            continue
        mf = sum(float(r["acc_fake"]) for r in rs) / len(rs)
        mr = sum(float(r["acc_real"]) for r in rs) / len(rs)
        ma = sum(float(r["AUC"]) for r in rs) / len(rs)
        me = sum(float(r["ECE"]) for r in rs) / len(rs)
        print(f"{SHORT[det]:24s} meanfake={mf:6.2f} meanreal={mr:6.2f} meanAUC={ma:6.2f} meanECE={me:.3f}")

    # McNemar highlights
    mc_path = os.path.join(COMP, "mcnemar_paired_tests.csv")
    if os.path.exists(mc_path):
        print()
        print("%" * 60)
        print("% TABLE: McNemar (all pairs, per generator)")
        print("%" * 60)
        rows = list(csv.DictReader(open(mc_path)))
        for r in rows:
            a = SHORT.get(r["detector_A"], r["detector_A"])
            b = SHORT.get(r["detector_B"], r["detector_B"])
            p = r["p_str"]
            ptex = r"$<10^{-300}$" if p == "<1e-300" else f"${p}$"
            print(f"{GEN_LABEL[r['generator']]} & {a} vs.\\ {b} & {r['A_correct_B_wrong']} & "
                  f"{r['A_wrong_B_correct']} & {ptex} \\\\")


if __name__ == "__main__":
    main()
