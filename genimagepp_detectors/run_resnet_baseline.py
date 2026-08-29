#!/usr/bin/env python
"""
Standalone inference for the GenImage++ ResNet-50 baseline detector
(attack_code/discriminators.py :: load_discriminator('resnet50'),
checkpoint attack_code/weights/resnet_epoch_20.pth).

Why this detector matters for the study: it was trained on the SAME data as
the GenImage++ CLIP baseline (GenImage SDv1.4 subset) but uses a completely
different architecture (a timm ResNet-50 CNN, no CLIP backbone). That makes
it the cleanest available way to separate "what the training distribution
did" from "what the CLIP backbone did", which the CLIP-only detector set
cannot do on its own.

Normalization note: the released attack code routes every detector, including
resnet50, through discriminator_preprocess(), which applies CLIP mean/std.
The source paper's Appendix A.2 instead describes ImageNet mean/std for the
ResNet-50 fine-tuning runs. The released code and the paper text therefore
disagree for this checkpoint, so --normalize exposes both and the choice is
resolved empirically (see paper Section 3.1).

CSV columns: filepath,true_label,pred_score,pred_label
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pad_or_crop_native(img, size=224):
    """Preserve native pixel scale instead of interpolating (see
    run_clip_baseline.py for the full rationale)."""
    w, h = img.size
    if w > size or h > size:
        left = max(0, (w - size) // 2)
        top = max(0, (h - size) // 2)
        img = img.crop((left, top, left + min(w, size), top + min(h, size)))
        w, h = img.size
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


def list_images(dir_path, max_sample=None):
    dir_path = Path(dir_path)
    files = sorted(
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    if max_sample is not None:
        files = files[:max_sample]
    return files


class ImageListDataset(Dataset):
    def __init__(self, filepaths, labels, mean, std, preprocess="upsample",
                 jpeg_quality=None):
        self.filepaths = filepaths
        self.labels = labels
        self.mean = mean
        self.std = std
        self.preprocess = preprocess
        self.jpeg_quality = jpeg_quality

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.jpeg_quality is not None:
                import io
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=self.jpeg_quality)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
            if self.preprocess == "pad":
                img = pad_or_crop_native(img, 224)
            else:
                img = img.resize((224, 224), Image.BICUBIC)
            arr = torch.from_numpy(np.array(img, dtype="float32") / 255.0).permute(2, 0, 1)
            for c in range(3):
                arr[c] = (arr[c] - self.mean[c]) / self.std[c]
            ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] failed to load {path}: {e}", file=sys.stderr)
            arr = torch.zeros(3, 224, 224)
            ok = False
        return arr, label, str(path), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_path", required=True)
    ap.add_argument("--fake_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_sample", type=int, default=None,
                    help="Max images PER CLASS. Omit to use ALL images found.")
    ap.add_argument("--normalize", choices=["clip", "imagenet"], default="clip")
    ap.add_argument("--preprocess", choices=["upsample", "pad"], default="upsample")
    ap.add_argument("--jpeg_quality", type=int, default=None,
                    help="If set, re-encode each image at this JPEG quality first "
                         "(robustness check against social-media re-encoding).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    mean, std = (CLIP_MEAN, CLIP_STD) if args.normalize == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    print(f"[INFO] device={device} normalize={args.normalize} preprocess={args.preprocess} "
          f"jpeg_quality={args.jpeg_quality}")

    real_files = list_images(args.real_path, args.max_sample)
    fake_files = list_images(args.fake_path, args.max_sample)
    print(f"[INFO] found {len(real_files)} real, {len(fake_files)} fake")
    if not real_files or not fake_files:
        print("[ERROR] no images found", file=sys.stderr)
        sys.exit(1)

    filepaths = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    ds = ImageListDataset(filepaths, labels, mean, std, args.preprocess, args.jpeg_quality)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"[INFO] loading resnet50 from {args.ckpt}")
    model = timm.create_model("resnet50", num_classes=1, checkpoint_path=args.ckpt)
    model.to(device).eval()

    rows, n_done = [], 0
    with torch.no_grad():
        for imgs, blabels, bpaths, oks in loader:
            logits = model(imgs.to(device)).squeeze(-1)
            probs = torch.sigmoid(logits).cpu()
            for i in range(len(bpaths)):
                if not bool(oks[i]):
                    continue
                score = float(probs[i].item())
                rows.append((bpaths[i], int(blabels[i]), score, int(score > 0.5)))
            n_done += len(bpaths)
            print(f"[INFO] processed {n_done}/{len(ds)}", end="\r")
    print()

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true_label", "pred_score", "pred_label"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", r[3]])

    n_correct = sum(1 for r in rows if r[1] == r[3])
    fakes = [r for r in rows if r[1] == 1]
    reals = [r for r in rows if r[1] == 0]
    fa = sum(1 for r in fakes if r[3] == 1) / max(len(fakes), 1)
    ra = sum(1 for r in reals if r[3] == 0) / max(len(reals), 1)
    scores = [r[2] for r in rows]
    print(f"[SUMMARY] wrote {len(rows)} rows to {args.out}")
    print(f"[SUMMARY] overall={n_correct/len(rows):.4f} real_acc={ra:.4f} fake_acc={fa:.4f}")
    print(f"[SUMMARY] score min/max/mean: {min(scores):.4f}/{max(scores):.4f}/{sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
