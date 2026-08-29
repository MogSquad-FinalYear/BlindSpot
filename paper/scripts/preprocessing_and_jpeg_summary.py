#!/usr/bin/env python3
"""Summarises the two review-requested robustness analyses:

  (a) padding vs. upsampling preprocessing, on BOTH non-224-native generators
      (BigGAN at 128x128 and GLIDE at 256x256), for every checkpoint that has
      both runs on disk;
  (b) JPEG re-encoding robustness on FLUX.1, comparing each detector's clean
      and quality-75 runs on the same image sample.

Everything is read from the prediction CSVs so no number is transcribed.
"""
import csv
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

GP = "/home/student/data/genimagepp_detectors/"
NPRD = "/home/student/data/NPR-DeepfakeDetection/"
UFD = "/home/student/data/UniversalFakeDetect/"


def stats(path):
    if not os.path.exists(path):
        return None
    yt, yp = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            yt.append(int(r["true_label"]))
            yp.append(float(r["pred_score"]))
    yt, yp = np.array(yt), np.array(yp)
    yh = (yp > 0.5).astype(int)
    return dict(
        n=len(yt),
        real=100 * (yh[yt == 0] == 0).mean(),
        fake=100 * (yh[yt == 1] == 1).mean(),
        overall=100 * (yh == yt).mean(),
        AP=100 * average_precision_score(yt, yp),
        AUC=100 * roc_auc_score(yt, yp),
    )


def block(title, rows):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'setting':38s} {'fake':>7s} {'real':>7s} {'overall':>8s} {'AP':>7s} {'AUC':>7s}")
    for label, path in rows:
        s = stats(path)
        if s is None:
            print(f"{label:38s}   [not available yet]")
            continue
        print(f"{label:38s} {s['fake']:7.2f} {s['real']:7.2f} {s['overall']:8.2f} "
              f"{s['AP']:7.2f} {s['AUC']:7.2f}")


def main():
    block("(a) PREPROCESSING: padding vs upsampling, BigGAN (128x128 native)", [
        ("CLIP baseline  upsample", GP + "predictions_clipbaseline_BigGan_full.csv"),
        ("CLIP baseline  pad",      GP + "predictions_clipbaseline_BigGan_pad.csv"),
        ("CLIP-LoRA      upsample", GP + "predictions_cliplora_BigGan_full.csv"),
        ("CLIP-LoRA      pad",      GP + "predictions_cliplora_BigGan_pad.csv"),
    ])

    block("(a) PREPROCESSING: padding vs upsampling, GLIDE (256x256 native)", [
        ("CLIP baseline  upsample", GP + "predictions_clipbaseline_Glide_full.csv"),
        ("CLIP baseline  pad",      GP + "predictions_clipbaseline_Glide_pad.csv"),
        ("CLIP-LoRA      upsample", GP + "predictions_cliplora_Glide_full.csv"),
        ("CLIP-LoRA      pad",      GP + "predictions_cliplora_Glide_pad.csv"),
    ])

    N = 2000
    block(f"(b) JPEG q=75 robustness on FLUX.1 ({N}/class, paired clean vs jpeg)", [
        ("UniversalFakeDetect clean", UFD + f"predictions_flux_clean{N}.csv"),
        ("UniversalFakeDetect jpeg75", UFD + f"predictions_flux_jpeg75{N}.csv"),
        ("NPR                 clean", NPRD + f"predictions_flux_clean{N}.csv"),
        ("NPR                 jpeg75", NPRD + f"predictions_flux_jpeg75{N}.csv"),
        ("ResNet-50           clean", GP + f"predictions_resnet_flux_clean{N}.csv"),
        ("ResNet-50           jpeg75", GP + f"predictions_resnet_flux_jpeg75{N}.csv"),
        ("CLIP baseline       clean", GP + f"predictions_clipbaseline_flux_clean{N}.csv"),
        ("CLIP baseline       jpeg75", GP + f"predictions_clipbaseline_flux_jpeg75{N}.csv"),
        ("CLIP-LoRA           clean", GP + f"predictions_cliplora_flux_clean{N}.csv"),
        ("CLIP-LoRA           jpeg75", GP + f"predictions_cliplora_flux_jpeg75{N}.csv"),
    ])

    block("ResNet-50 fifth detector, full sweep", [
        (f"ResNet-50 {g}", GP + f"predictions_resnet_{g}_full.csv")
        for g in ["BigGan", "Glide", "flux", "sd3", "sdxl"]
    ])


if __name__ == "__main__":
    main()
