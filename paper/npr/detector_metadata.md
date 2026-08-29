# Detector metadata — NPR (Neighboring Pixel Relationships)

**Paper:** Tan, Chuangchuang; Liu, Huan; Zhao, Yao; Wei, Shikui; Gu, Guanghua;
Liu, Ping; Wei, Yunchao. *Rethinking the Up-Sampling Operations in
CNN-based Generative Network for Generalizable Deepfake Detection.* CVPR 2024.
[arXiv:2312.10461](https://arxiv.org/abs/2312.10461) ·
[Code](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)

**Domain:** Pixel/up-sampling-artifact detector — frequency-adjacent (targets
the checkerboard-like artifacts CNN-based generator up-sampling layers leave
in neighboring-pixel relationships), distinct from UniversalFakeDetect's
CLIP-embedding approach.

**Architecture:** ResNet-50 trained from scratch on NPR-transformed inputs
(the "neighboring pixel relationship" preprocessing described in the paper),
single sigmoid output = P(fake).

**Checkpoint used in this project:** `NPR.pth`, the repo's main released
checkpoint (referenced directly in the README's "Testing the detector"
section) — trained on the full ForenSynths ProGAN training set (20 LSUN
object categories). A second checkpoint, `model_epoch_last_3090.pth`
("ProGAN-4class"), ships in the repo too but is only used for a specific
benchmark table (AIGCDetectBenchmark) in the original paper and was **not**
used here.

**Note on the checkpoint file format:** `NPR.pth` is saved as a full training
checkpoint dict (`{'model': state_dict, 'optimizer': ..., 'total_steps':
...}`) with `module.`-prefixed keys from DataParallel training — the repo's
own `test.py` loads checkpoints with `strict=True` and does **not** unwrap
this, so it would actually error on `NPR.pth` as shipped. This project's
`run_inference.py` auto-detects and unwraps both the outer dict and the
`module.` prefix.

**Training data:** same ProGAN-only lineage as UniversalFakeDetect (both
descend from the Wang et al. 2020 CNNDetection dataset) — **no diffusion
model images** in training. Same "generalization gradient from GAN-era to
2025-era DiT/UNet generators" framing applies here as for UniversalFakeDetect
(see `../universalfakedetect/detector_metadata.md`), not "trained on X,
tested on Y."

**Preprocessing at inference (GenImage protocol, per README):** real-world
eval sets mix large, variably-sized real photos with small, canonical-size
generator outputs (e.g. 128×128 BigGAN crops). A naive `Resize()` to a fixed
size would inject its own resize/upsampling artifacts on top of (or instead
of) the generator's own — confounding exactly the signal NPR targets. This
project instead replicates the repo's documented GenImage protocol:
`translate_duplicate` (tile the image with itself, no interpolation, only if
smaller than the crop size) + `CenterCrop(224)`. Images already ≥224 on both
sides are center-cropped at native resolution, no resampling. Normalization:
ImageNet mean/std (not CLIP stats).

**Inference conventions in this project:** batch size 16 (T1000 8GB
headroom), no `--max_sample` cap beyond balancing real/fake pool sizes to
match the methodology used for the other three detectors (see
`../README.md`).
