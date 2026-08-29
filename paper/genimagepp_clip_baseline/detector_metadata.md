# Detector metadata — GenImage++ CLIP baseline

**Source paper:** *Breaking Latent Prior Bias in Detectors for Generalizable
AIGC Image Detection.* NeurIPS 2025 (Poster). arXiv:2506.00874. Introduces
both the GenImage++ benchmark and the On-Manifold Adversarial Training
(OMAT) method this checkpoint is the *un-hardened baseline* comparison
point for.
[HF dataset/code repo](https://huggingface.co/datasets/Lunahera/genimagepp)

**Domain:** CLIP-embedding classifier — architecturally similar to
UniversalFakeDetect (frozen CLIP ViT-L/14 vision tower + linear head), but
trained by a different group specifically as the "before OMAT" baseline in
their generalization study, not by the UniversalFakeDetect authors.

**Architecture:** `openai/clip-vit-large-patch14` vision tower (frozen at
inference, no LoRA) + `Linear(hidden_size, 1)` classifier head on the CLS
token of the last hidden state. Source: `attack_code/discriminators.py ::
clip_detector`.

**Checkpoint used in this project:** `attack_code/weights/clip_epoch_20.pth`
— the repo's released "baseline CLIP detector" checkpoint, epoch 20. This is
the paper's own conventionally-trained baseline (i.e. *not* adversarially
hardened) — the direct counterpart to `genimagepp_clip_lora` in this
project's "what improves detection" comparison.

**Training data:** not explicitly stated in the released README beyond
"baseline CLIP detector" — the source paper's broader methodology trains
baseline detectors on standard AIGC-detection training data (real ImageNet
images + earlier-generation diffusion fakes, e.g. SD-family) and evaluates
zero-shot on GenImage++'s held-out modern-generator subsets (FLUX.1, SD3,
and the multi-style/realistic/amateur variants) — GenImage++ is explicitly a
**test-only** benchmark, no GenImage++ images were used to train this
checkpoint. Confirm exact training-set composition against the arXiv paper
(2506.00874, Section on baseline training setup) before stating this
precisely in the paper draft.

**Preprocessing at inference:** resize to 224×224 (not center-cropped after
— resize-to-224 is a no-op crop), CLIP normalization
(`mean=[0.48145466, 0.4578275, 0.40821073]`,
`std=[0.26862954, 0.26130258, 0.27577711]`). Source:
`discriminators.py :: discriminator_preprocess`. Unlike
UniversalFakeDetect's inference, this **does resize** rather than
center-crop at native resolution. (An earlier note here claimed this meant
the shared frequency-spectrum analysis in `../shared_spectra/` wouldn't
apply to this detector — that was wrong: that analysis resizes every image
to a standardized 256×256 grid for its own FFT computation regardless of
which detector it's paired with, so it was never "native resolution" to
begin with. It's the same generator-level data used for all four
detectors; see `../spectra/README.md`.)

**Known compatibility fix applied:** the checkpoint was trained against an
older `transformers` layout where `CLIPVisionModel` wraps submodules under
`vision_model.*`; the installed version here (5.14.1) has since flattened
that structure. This project's inference script strips the
`.vision_model.` path segment from checkpoint keys before loading — without
this the state_dict wouldn't load at all, not merely miscalibrate.

**Inference conventions in this project:** batch size 16, real/fake pools
balanced to match the other three detectors (see `../README.md`).
