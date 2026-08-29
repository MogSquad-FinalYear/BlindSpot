# Detector metadata — UniversalFakeDetect (CLIP ViT-L/14 linear probe)

**Paper:** Ojha, Utkarsh; Li, Yuheng; Lee, Yong Jae. *Towards Universal Fake Image
Detectors that Generalize Across Generative Models.* CVPR 2023.
[arXiv:2302.10174](https://arxiv.org/abs/2302.10174) ·
[Project page](https://utkarshojha.github.io/universal-fake-detection/) ·
[Code](https://github.com/Yuheng-Li/UniversalFakeDetect)

**Architecture:** Frozen CLIP ViT-L/14 image encoder + a single trained linear
layer (`fc`) on top of the CLIP embedding, sigmoid output = P(fake). The CLIP
backbone itself is never fine-tuned (`--fix_backbone` during training) — only
the linear probe's weights are learned.

**Checkpoint used in this project:** `pretrained_weights/fc_weights.pth`,
shipped in the official repo — this is the authors' main released checkpoint.

**Training data (per the paper/repo):** the official training set is the
same one used by Wang et al. 2020 (CNNDetection), consisting **only of
ProGAN-generated images** (20 LSUN object categories, real + ProGAN fake, no
diffusion-model images at all). The repo's `train.py --data_mode=wang2020`
call and dataset layout confirm this.

**What this means for "trained on X, tested on Y" framing:** none of the five
generators evaluated in this project (BigGAN, GLIDE, FLUX.1, SD3, SDXL) match
the checkpoint's training distribution. BigGAN is a GAN like ProGAN
(architecturally closer, same broad era) but was never seen during training;
GLIDE/FLUX/SD3/SDXL are diffusion-based and further still. **This is not an
in-distribution vs. OOD comparison** — it is zero-shot generalization from a
single 2020-era GAN family to everything else, including other GANs. Present
results as "generalization gradient from GAN-era to 2025-era DiT/UNet
generators," not as "trained on GAN, tested on diffusion."

**Preprocessing at inference:** center-crop to 224×224, CLIP normalization
(`mean=[0.48145466, 0.4578275, 0.40821073]`,
`std=[0.26862954, 0.26130258, 0.27577711]`). No resize before crop — images
are cropped directly from their native resolution.

**Inference conventions in this project:**
- `--data_mode=ours` (direct `--real_path`/`--fake_path`, no filename filtering)
- Threshold-0.5 accuracy (`acc0`) and best-threshold accuracy (`acc1`, threshold
  chosen post-hoc on the eval set itself — an oracle upper bound, not a
  deployment-realistic number; report `acc0`/AUC as the primary metrics and
  `acc1` only as a "best case" reference).
- Batch size 16 (default 128 OOMs on the shared 8GB NVIDIA T1000 used for this
  project).
