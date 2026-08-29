#!/usr/bin/env python
"""
Standalone inference script for the GenImage++ CLIP baseline detector
(attack_code/discriminators.py :: clip_detector, checkpoint clip_epoch_20.pth).

Architecture: openai/clip-vit-large-patch14 vision tower (frozen at inference
time, no LoRA) + a single Linear(hidden_size, 1) classifier head on the CLS
token of the last hidden state. Checkpoint is a plain state_dict for this
whole module (see discriminators.py: load_discriminator -> clip_detector()
-> discriminator.load_state_dict(torch.load(ckpt))).

Preprocessing (from discriminators.py:discriminator_preprocess): resize to
224x224, center crop 224 (a no-op after the resize, kept for fidelity with
the original code), then normalize with CLIP's mean/std. We reimplement this
with plain PIL/torch ops instead of kornia so this script has no kornia
dependency.

Usage:
    python run_clip_baseline.py \
        --real_path /path/to/real/dir \
        --fake_path /path/to/fake/dir \
        --ckpt attack_code/weights/clip_epoch_20.pth \
        --out results_clip_baseline.csv \
        --batch_size 16 \
        [--max_sample 20]

CSV columns: filepath,true_label,pred_score,pred_label
  true_label: 0=real, 1=fake (from which directory the image was read)
  pred_score: sigmoid(logit), in [0, 1]
  pred_label: 1 if pred_score > 0.5 else 0
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}


class clip_detector(nn.Module):
    """Mirrors attack_code/discriminators.py::clip_detector exactly, so the
    checkpoint's state_dict keys line up 1:1."""

    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.classifier = nn.Linear(self.clip.config.hidden_size, 1)

    def forward(self, inputs):
        outputs = self.clip(pixel_values=inputs)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits


def list_images(dir_path, max_sample=None):
    dir_path = Path(dir_path)
    files = sorted(
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in {e.lower() for e in IMG_EXTENSIONS}
    )
    if max_sample is not None:
        files = files[:max_sample]
    return files


def pad_or_crop_native(img, size=224):
    """Preserve native pixel scale instead of interpolating: center-crop down
    if larger than `size` in either dimension (no resample), then center-paste
    onto a black size x size canvas. Used to test whether force-upsampling
    small-native-resolution generator outputs (e.g. BigGAN's 128x128) is
    itself responsible for detector miscalibration, vs. the official resize
    pipeline -- see paper/comparison/data_leakage_verification.md."""
    w, h = img.size
    if w > size or h > size:
        left = max(0, (w - size) // 2)
        top = max(0, (h - size) // 2)
        img = img.crop((left, top, left + min(w, size), top + min(h, size)))
        w, h = img.size
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


class ImageListDataset(Dataset):
    """Resize->224x224, ToTensor, CLIP normalize. Equivalent to
    discriminator_preprocess() in discriminators.py (resize+centercrop to
    224 is a no-op after resizing to exactly 224x224, then CLIP normalize).

    preprocess="upsample" (default): official behavior, resize to 224x224
    regardless of source size (upsamples small-native-resolution generators).
    preprocess="pad": diagnostic alternative, preserves native pixel scale
    via pad_or_crop_native() instead of interpolating."""

    def __init__(self, filepaths, labels, preprocess="upsample", jpeg_quality=None):
        self.filepaths = filepaths
        self.labels = labels
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
            elif self.preprocess == "nearest":
                # Upsample to 224 with no interpolation and no border: isolates
                # "smooth interpolation" from "zero-padding border" as the cause
                # of a preprocessing effect.
                img = img.resize((224, 224), Image.NEAREST)
            else:
                img = img.resize((224, 224), Image.BICUBIC)
            arr = torch.from_numpy(
                __import__("numpy").array(img, dtype="float32") / 255.0
            ).permute(2, 0, 1)
            for c in range(3):
                arr[c] = (arr[c] - CLIP_MEAN[c]) / CLIP_STD[c]
            ok = True
        except Exception as e:
            print(f"[WARN] failed to load {path}: {e}", file=sys.stderr)
            arr = torch.zeros(3, 224, 224)
            ok = False
        return arr, label, str(path), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_path", required=True, help="Directory of real images (label 0)")
    ap.add_argument("--fake_path", required=True, help="Directory of fake images (label 1)")
    ap.add_argument("--ckpt", required=True, help="Path to clip_epoch_20.pth")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--max_sample", type=int, default=None,
        help="Max images PER CLASS to evaluate. If omitted, use ALL images found.",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--jpeg_quality", type=int, default=None,
                    help="Re-encode each image at this JPEG quality before inference "
                         "(social-media re-encoding robustness check).")
    ap.add_argument(
        "--preprocess", choices=["upsample", "pad", "nearest"], default="upsample",
        help="upsample (default, official discriminator_preprocess behavior) or "
             "pad (diagnostic: preserve native pixel scale instead of interpolating)",
    )
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] device: {device}")
    print(f"[INFO] preprocess mode: {args.preprocess}")

    real_files = list_images(args.real_path, args.max_sample)
    fake_files = list_images(args.fake_path, args.max_sample)
    print(f"[INFO] found {len(real_files)} real images, {len(fake_files)} fake images")
    if len(real_files) == 0 or len(fake_files) == 0:
        print("[ERROR] no images found in one of the input directories.", file=sys.stderr)
        sys.exit(1)

    filepaths = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    dataset = ImageListDataset(filepaths, labels, preprocess=args.preprocess, jpeg_quality=args.jpeg_quality)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[INFO] loading model + checkpoint from {args.ckpt}")
    model = clip_detector()
    state_dict = torch.load(args.ckpt, map_location="cpu")
    # The checkpoint was trained against a transformers version where
    # CLIPVisionModel wraps its submodules under `vision_model.` (e.g.
    # "clip.vision_model.encoder...."). The installed transformers version
    # here has flattened CLIPVisionModel so those submodules hang directly
    # off the model (e.g. "clip.encoder...."). Strip the now-absent
    # "vision_model." path segment so the checkpoint's keys line up with
    # the currently-instantiated model's state_dict keys.
    if any(".vision_model." in k for k in state_dict.keys()):
        state_dict = {k.replace(".vision_model.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rows = []
    n_done = 0
    with torch.no_grad():
        for imgs, batch_labels, batch_paths, oks in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(-1)
            probs = torch.sigmoid(logits).cpu()
            for i in range(len(batch_paths)):
                if not bool(oks[i]):
                    continue
                score = float(probs[i].item())
                true_label = int(batch_labels[i])
                pred_label = int(score > 0.5)
                rows.append((batch_paths[i], true_label, score, pred_label))
            n_done += len(batch_paths)
            print(f"[INFO] processed {n_done}/{len(dataset)}", end="\r")
    print()

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "true_label", "pred_score", "pred_label"])
        for r in rows:
            writer.writerow([r[0], r[1], f"{r[2]:.4f}", r[3]])

    n_correct = sum(1 for r in rows if r[1] == r[3])
    acc = n_correct / len(rows) if rows else float("nan")
    scores = [r[2] for r in rows]
    print(f"[SUMMARY] wrote {len(rows)} rows to {args.out}")
    print(f"[SUMMARY] accuracy: {acc:.4f} ({n_correct}/{len(rows)})")
    print(f"[SUMMARY] pred_score min/max/mean: {min(scores):.4f}/{max(scores):.4f}/{sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
