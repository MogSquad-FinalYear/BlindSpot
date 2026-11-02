#!/usr/bin/env python3
"""DIRE (Wang et al., ICCV 2023) inference on a single GPU.

The released pipeline is two stages driven by mpiexec across four GPUs:
compute_dire.py writes DIRE images to disk, then test.py classifies them.
This collapses both into one pass on one GPU and writes the same per-image
CSV the other detectors in this study produce, so DIRE drops straight into
the existing comparison without a second format.

DIRE itself is the reconstruction error of a diffusion model: invert the
image to a latent with DDIM, sample back down, and take the absolute
difference.  Generated images reconstruct more faithfully than real ones,
so the error map is the detector's input rather than the pixels.

Model flags and the DDIM step count follow guided-diffusion/compute_dire.sh
from the official repository; the classifier preprocessing follows
utils/datasets.py (resize 256, center-crop 224, ImageNet normalization).
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC / "guided-diffusion"))
sys.path.insert(0, str(SRC))

from guided_diffusion.script_util import (  # noqa: E402
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# guided-diffusion/compute_dire.sh, for both lsun_bedroom.pt and
# 256x256_diffusion_uncond.pt
ADM_FLAGS = dict(
    attention_resolutions="32,16,8",
    class_cond=False,
    diffusion_steps=1000,
    dropout=0.1,
    image_size=256,
    learn_sigma=True,
    noise_schedule="linear",
    num_channels=256,
    num_head_channels=64,
    num_res_blocks=2,
    resblock_updown=True,
    use_fp16=True,
    use_scale_shift_norm=True,
)


def list_images(root, limit=None):
    files = sorted(p for p in Path(root).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXT)
    return files[:limit] if limit else files


def load_batch(paths, size=256):
    """-> float tensor in [-1, 1], center-cropped square then resized.

    Mirrors reshape_image() in compute_dire.py: crop to square only when the
    image is not square, then bicubic to the model's resolution.
    """
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = transforms.functional.to_tensor(img)
        if t.shape[1] != t.shape[2]:
            t = transforms.functional.center_crop(t, min(t.shape[1], t.shape[2]))
        if t.shape[-1] != size:
            t = F.interpolate(t.unsqueeze(0), size=(size, size),
                              mode="bicubic", align_corners=False).squeeze(0)
        out.append(t * 2.0 - 1.0)
    return torch.stack(out)


def build_classifier(ckpt_path, device):
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


CLS_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


def dire_to_classifier_input(dire):
    """DIRE map -> classifier input.

    The released pipeline saves DIRE to PNG and reloads it, so the uint8
    quantisation is part of the measured signal and is reproduced here.
    """
    d = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8).float() / 255.0
    d = transforms.functional.center_crop(d, 224)
    return CLS_NORM(d)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_path", required=True)
    ap.add_argument("--fake_path", required=True)
    ap.add_argument("--diffusion_ckpt", required=True)
    ap.add_argument("--classifier_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sample", type=int, default=None,
                    help="images PER CLASS; omit to use everything found")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = model_and_diffusion_defaults()
    cfg.update(ADM_FLAGS)
    cfg["timestep_respacing"] = f"ddim{args.ddim_steps}"

    print(f"[INFO] loading diffusion model {args.diffusion_ckpt}", flush=True)
    model, diffusion = create_model_and_diffusion(**cfg)
    model.load_state_dict(torch.load(args.diffusion_ckpt, map_location="cpu"))
    model.to(device)
    if cfg["use_fp16"]:
        model.convert_to_fp16()
    model.eval()

    print(f"[INFO] loading DIRE classifier {args.classifier_ckpt}", flush=True)
    clf = build_classifier(args.classifier_ckpt, device)

    reals = list_images(args.real_path, args.max_sample)
    fakes = list_images(args.fake_path, args.max_sample)
    print(f"[INFO] {len(reals)} real, {len(fakes)} fake", flush=True)
    items = [(p, 0) for p in reals] + [(p, 1) for p in fakes]

    rows, t0, done = [], time.time(), 0
    for i in range(0, len(items), args.batch_size):
        chunk = items[i:i + args.batch_size]
        paths = [c[0] for c in chunk]
        labels = [c[1] for c in chunk]
        try:
            imgs = load_batch(paths).to(device)
            shape = (imgs.shape[0], 3, cfg["image_size"], cfg["image_size"])
            latent = diffusion.ddim_reverse_sample_loop(
                model, shape, noise=imgs, clip_denoised=True, model_kwargs={})
            recons = diffusion.ddim_sample_loop(
                model, shape, noise=latent, clip_denoised=True, model_kwargs={})
            dire = torch.abs(imgs - recons)
            logits = clf(dire_to_classifier_input(dire)).flatten()
            scores = torch.sigmoid(logits).float().cpu().numpy()
        except Exception as e:  # keep the sweep alive, record the gap
            print(f"[WARN] batch at {i} failed: {e}", file=sys.stderr, flush=True)
            continue
        for p, lab, s in zip(paths, labels, scores):
            rows.append((str(p), lab, f"{s:.6f}", int(s >= 0.5)))
        del imgs, latent, recons, dire, logits
        torch.cuda.empty_cache()
        done += len(chunk)
        if (i // args.batch_size) % 10 == 0:
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (len(items) - done) / rate / 60 if rate else 0
            print(f"[{done}/{len(items)}] {rate:.2f} img/s  eta {eta:.1f} min",
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true_label", "pred_score", "pred_label"])
        w.writerows(rows)

    y = np.array([r[1] for r in rows])
    s = np.array([float(r[2]) for r in rows])
    if len(y):
        fa = 100.0 * ((s >= 0.5)[y == 1]).mean() if (y == 1).any() else float("nan")
        ra = 100.0 * ((s < 0.5)[y == 0]).mean() if (y == 0).any() else float("nan")
        print(f"\nwrote {len(rows)} rows -> {args.out}")
        print(f"fake-acc {fa:.2f}%   real-acc {ra:.2f}%   "
              f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
