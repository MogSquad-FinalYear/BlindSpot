# DIRE: single-GPU harness

DIRE (Wang et al., ICCV 2023) is the one detector family in reach that is
structurally independent of everything else in the study: its signal is
neither a CLIP embedding nor a pixel residual, but the error a diffusion
model makes reconstructing the image. Generated images reconstruct more
faithfully than real ones, so the error map is what the classifier sees.

**This harness is released but not yet run at the study's scale.** The paper
reports the measured cost, not DIRE accuracy numbers, and Section 8 explains
why. The code is here so the experiment is reproducible on larger hardware
rather than merely deferred.

## Why it is not in the results tables

DIRE scores one image by inverting it to a latent with 20-step DDIM and
sampling back down, roughly 40 diffusion forward passes. On the 8 GB GPU used
throughout this study, shared with a desktop session and therefore limited to
batch size 1, that measured at **171 seconds per image**.

| Scope | Cost at that rate |
|---|---|
| 100 images per class, one generator | 9.5 GPU-hours |
| 500 per class, three generators | 142 GPU-hours |
| the 6,000-per-class protocol the other five detectors use, five generators | ~2,800 GPU-hours |

Reporting a handful of images instead would be the partial sweep the paper
says it will not report. On a less constrained GPU, larger batches and fp16
throughput should move this by more than an order of magnitude.

## Setup

```
git submodule update --init dire_eval/src
pip install numpy torch torchvision pillow
bash dire_eval/download_weights.sh
```

`download_weights.sh` pulls two files from the SIDBench mirror on Hugging
Face, which is public and ungated, unlike the Baidu and OneDrive links in
DIRE's own (now archived) repository:

| File | Size | Role |
|---|---|---|
| `weights/lsun_adm.pth` | 269 MB | the DIRE classifier, trained on LSUN-Bedroom + ADM |
| `weights/lsun_bedroom.pt` | 2.1 GB | the diffusion model the reconstruction runs against |

Weights are gitignored. Their licences are those of the original DIRE
release, not this repository's.

## Running

```
python dire_eval/run_dire.py \
  --real_path  <dir of real images> \
  --fake_path  <dir of generated images> \
  --diffusion_ckpt  dire_eval/weights/lsun_bedroom.pt \
  --classifier_ckpt dire_eval/weights/lsun_adm.pth \
  --out results/dire_flux.csv \
  --max_sample 500 --batch_size 1
```

Output is the same four-column schema every other detector in this study
writes (`filepath,true_label,pred_score,pred_label`), so results drop into
`paper/predictions/` and the existing analysis scripts without conversion.

`run_sweep.sh N BS` runs one generator per architecture family (FLUX.1,
BigGAN, SDXL) against the same real pools the rest of the study uses.

## Differences from the official pipeline

The released code is two stages driven by `mpiexec` across four GPUs:
`compute_dire.py` writes DIRE images to disk, then `test.py` classifies them.
This harness collapses both into one pass on one GPU and never touches disk
between them. Three things are kept deliberately identical:

- the ADM model flags and the 20-step DDIM schedule, from
  `guided-diffusion/compute_dire.sh`
- the square-crop-then-bicubic input handling, from `reshape_image()` in
  `compute_dire.py`
- the classifier preprocessing: resize 256, center-crop 224, ImageNet
  normalization, from `utils/datasets.py`

The uint8 quantisation of the DIRE map is also reproduced, because the
official pipeline saves it as PNG and reloads it, which makes the rounding
part of the measured signal rather than an implementation detail.

One deviation worth stating: the released classifiers are all trained on
LSUN-Bedroom, so running them over the ImageNet-derived generators in this
study is a cross-distribution test. That is the same position
UniversalFakeDetect and NPR are in, both ProGAN/LSUN-trained, and is
consistent with how the rest of the study is set up.
