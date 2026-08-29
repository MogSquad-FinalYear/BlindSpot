#!/usr/bin/env python
"""
Standalone inference script for the GenImage++ CLIP-LoRA (rank 4) detector,
adversarially trained with On-Manifold Adversarial Training (OMAT).
Source: attack_code/Clip_lora_rank4_model.py :: CLIPLoRADetector,
checkpoint attack_code/weights/best_model_low_rank.pt.

Architecture: openai/clip-vit-large-patch14 vision tower wrapped with a PEFT
LoraConfig (task_type=FEATURE_EXTRACTION, r=4, lora_alpha=8, lora_dropout=0.1,
target_modules=[q_proj, v_proj, k_proj, out_proj]), base weights frozen, only
LoRA + a Linear(hidden_size, 1) classifier head are trained. The checkpoint is
a dict with keys {'epoch', 'model_state_dict', ...}; we load
checkpoint['model_state_dict'] (see Clip_lora_rank4_model.py:_load_model).

Preprocessing (from Clip_lora_rank4_model.py):
  - PIL image -> Resize((224,224)) -> ToTensor()  [0,1] range, NOT centercropped
  - then discriminator_preprocess(): CLIP normalize (mean/std below)
  (Note: unlike the baseline discriminators.py preprocessing, this file's
  discriminator_preprocess only normalizes; the resize happens earlier in
  CLIPLoRADetectorWrapper.transform. Net effect on a directly-loaded PIL
  image is the same as the baseline script: resize 224x224 + CLIP normalize.)

Usage:
    python run_clip_lora.py \
        --real_path /path/to/real/dir \
        --fake_path /path/to/fake/dir \
        --ckpt attack_code/weights/best_model_low_rank.pt \
        --out results_clip_lora.csv \
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
from peft import get_peft_model, LoraConfig, TaskType

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}


class CLIPLoRADetector(nn.Module):
    """Mirrors attack_code/Clip_lora_rank4_model.py::CLIPLoRADetector exactly
    so the checkpoint's model_state_dict keys line up 1:1."""

    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", lora_rank=4):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        )
        self.clip = get_peft_model(self.clip, peft_config)

        for param in self.clip.parameters():
            param.requires_grad = False
        for name, param in self.clip.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        self.classifier = nn.Linear(self.clip.config.hidden_size, 1)

    def forward(self, pixel_values):
        try:
            outputs = self.clip(pixel_values)
        except Exception:
            try:
                outputs = self.clip.base_model(pixel_values=pixel_values)
            except Exception:
                if hasattr(self.clip, "model"):
                    clip_model = self.clip.model
                elif hasattr(self.clip, "base_model"):
                    clip_model = self.clip.base_model
                else:
                    clip_model = self.clip
                outputs = clip_model.forward(pixel_values=pixel_values)

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
    """Preserve native pixel scale instead of interpolating -- see
    run_clip_baseline.py's copy of this function for the full rationale
    (paper/comparison/data_leakage_verification.md)."""
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
    """Resize->224x224, ToTensor([0,1]), CLIP normalize -- matches
    CLIPLoRADetectorWrapper.transform + discriminator_preprocess().

    preprocess="upsample" (default): official behavior.
    preprocess="pad": diagnostic alternative, preserves native pixel scale."""

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
                img = img.resize((224, 224), Image.BILINEAR)
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
    ap.add_argument("--ckpt", required=True, help="Path to best_model_low_rank.pt")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument(
        "--max_sample", type=int, default=None,
        help="Max images PER CLASS to evaluate. If omitted, use ALL images found.",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--jpeg_quality", type=int, default=None,
                    help="Re-encode each image at this JPEG quality before inference "
                         "(social-media re-encoding robustness check).")
    ap.add_argument(
        "--preprocess", choices=["upsample", "pad", "nearest"], default="upsample",
        help="upsample (default, official behavior) or pad (diagnostic: preserve native pixel scale)",
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
    model = CLIPLoRADetector(lora_rank=args.lora_rank)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    # Same transformers-version key-layout shift as in run_clip_baseline.py:
    # the checkpoint's CLIPVisionModel submodules were saved under
    # `vision_model.` (e.g. "...vision_model.encoder...."); the installed
    # transformers version flattens that away. Strip it so keys line up.
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
