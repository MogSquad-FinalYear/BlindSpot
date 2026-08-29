#!/usr/bin/env python3
"""
Builds the core Accuracy/AUC table (paper item 1) and per-generator confusion
matrices (item 2) from one or more predictions CSVs (filepath,true_label,
pred_score,pred_label -- the format produced by dump_predictions.py /
run_and_dump.py / any detector's inference wrapper in this project).

Generic across detectors: point it at any set of label=csv pairs.

Usage:
    python3 compute_metrics_table.py \
        --csv BigGan=.../predictions_BigGan_full.csv Glide=... flux=... sd3=... sdxl=... \
        --out_dir /home/student/data/paper/universalfakedetect \
        --detector_name "UniversalFakeDetect (CLIP ViT-L/14 linear probe)"
"""
import argparse
import csv
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    y_true, y_pred = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["true_label"]))
            y_pred.append(float(row["pred_score"]))
    return np.array(y_true), np.array(y_pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--detector_name", default="")
    args = p.parse_args()

    os.makedirs(os.path.join(args.out_dir, "accuracy"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "confusion_matrices"), exist_ok=True)

    entries = []
    for item in args.csv:
        label, path = item.split("=", 1)
        entries.append((label, path))

    rows = []
    for label, path in entries:
        y_true, y_pred = load_csv(path)
        n_real = int((y_true == 0).sum())
        n_fake = int((y_true == 1).sum())

        ap = average_precision_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        y_hat = (y_pred > 0.5).astype(int)
        acc_real = (y_hat[y_true == 0] == 0).mean() if n_real else float("nan")
        acc_fake = (y_hat[y_true == 1] == 1).mean() if n_fake else float("nan")
        acc_overall = (y_hat == y_true).mean()

        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

        rows.append(dict(
            generator=label, n_real=n_real, n_fake=n_fake,
            AP=round(ap * 100, 2), AUC=round(auc * 100, 2),
            acc_real=round(acc_real * 100, 2), acc_fake=round(acc_fake * 100, 2),
            acc_overall=round(acc_overall * 100, 2),
            TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
        ))

        # confusion matrix plot
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred real", "pred fake"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true real", "true fake"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_title(f"{label} (n={n_real + n_fake})")
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        cm_path = os.path.join(args.out_dir, "confusion_matrices", f"{label}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)

    # combined CSV
    csv_path = os.path.join(args.out_dir, "accuracy", "accuracy_auc_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # markdown table
    md_path = os.path.join(args.out_dir, "accuracy", "accuracy_auc_table.md")
    header = list(rows[0].keys())
    with open(md_path, "w") as f:
        if args.detector_name:
            f.write(f"# {args.detector_name} — Accuracy / AUC table\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in header) + " |\n")
        overall_acc = np.mean([r["acc_overall"] for r in rows])
        overall_auc = np.mean([r["AUC"] for r in rows])
        overall_ap = np.mean([r["AP"] for r in rows])
        f.write(f"\n**Mean across generators:** AP={overall_ap:.2f}, AUC={overall_auc:.2f}, acc={overall_acc:.2f}\n")

    print(f"Wrote {csv_path} and {md_path}")
    print(f"Wrote confusion matrix plots to {os.path.join(args.out_dir, 'confusion_matrices')}/")
    for r in rows:
        print(f"  {r['generator']}: AP={r['AP']} AUC={r['AUC']} acc={r['acc_overall']} "
              f"(real={r['acc_real']} fake={r['acc_fake']}) TN={r['TN']} FP={r['FP']} FN={r['FN']} TP={r['TP']}")


if __name__ == "__main__":
    main()
