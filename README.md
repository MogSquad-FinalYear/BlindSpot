# BlindSpot

**Why deepfake detectors fail on modern diffusion generators.**

BlindSpot studies how well existing fake-image detectors generalize once you move
past the generators they were originally evaluated on. Most published detectors
report strong numbers on GAN-era fakes (ProGAN, BigGAN, StyleGAN); we test the same
detectors, unmodified, against a spread of generators from GAN-era to 2025-era
diffusion transformers (DiT) — BigGAN, GLIDE, Flux, SD3, and SDXL — and measure
where accuracy actually holds up versus where it quietly collapses.

## What's in this repo

```
BlindSpot/
├── paper/                      # paper draft + all generated figures/tables
│   ├── paper_draft.tex
│   ├── universalfakedetect/    # per-detector results: accuracy, confusion
│   ├── npr/                    #   matrices, score distributions, spectra
│   ├── genimagepp_clip_baseline/
│   ├── genimagepp_clip_lora/
│   ├── shared_spectra/         # generator-level frequency analysis (not detector-specific)
│   ├── comparison/             # cross-detector summary once all runs are in
│   └── scripts/                # reusable, detector-agnostic analysis scripts
├── genimagepp_detectors/
│   ├── run_resnet_baseline.py  # ResNet baseline detector
│   ├── run_clip_baseline.py    # CLIP linear-probe baseline
│   ├── run_clip_lora.py        # CLIP-LoRA, adversarially hardened variant
│   ├── attack_code/            # adversarial robustness evaluation
│   └── predictions_*.csv       # per-generator, per-detector raw predictions
├── NPR-DeepfakeDetection/       # baseline (submodule, chuangchuangtan/NPR-DeepfakeDetection)
├── UniversalFakeDetect/         # baseline (submodule, WisconsinAIVision/UniversalFakeDetect)
├── paper.txt                   # working notes on framing/results
└── Court_room_videos.txt       # misc reference links (unrelated to the detection work)
```

## Detectors evaluated

| Detector | Type | Source |
|---|---|---|
| UniversalFakeDetect | CLIP ViT-L/14 linear probe (Ojha et al., CVPR 2023) | submodule |
| NPR | Neighboring Pixel Relationships (Tan et al.) | submodule |
| CLIP baseline | CLIP linear probe, trained on modern DiT generators | `genimagepp_detectors/run_clip_baseline.py` |
| CLIP-LoRA | CLIP-LoRA, adversarially/OOD-hardened variant | `genimagepp_detectors/run_clip_lora.py` |
| ResNet baseline | Standard CNN baseline | `genimagepp_detectors/run_resnet_baseline.py` |

## Generators evaluated

GAN-era: **BigGAN**, **GLIDE**. Modern DiT-era: **Flux**, **SD3**, **SDXL**. Real
images are sampled from MS COCO (`val2017`).

Each detector folder under `paper/` reports, per generator: accuracy/AUC,
confusion matrices, score distributions, and frequency-domain spectra (high-pass
vs. low-pass log-energy gap) — plus a folder of failure examples where a detector
called a fake image confidently real.

## What's intentionally not in this repo

Two datasets and one set of model checkpoints are excluded via `.gitignore`
because they're far too large for git (tens of gigabytes) and, in COCO's case,
freely available from the original source rather than something to vendor here:

- `genimagepp/` — the generated-image sets (SDXL/SD3/Flux), ~43GB
- `real_coco/` — MS COCO val2017 + annotations, ~2.7GB, download from
  [cocodataset.org](https://cocodataset.org)
- `genimagepp_detectors/attack_code/weights/` — trained detector checkpoints, ~2.4GB

## Baselines

`NPR-DeepfakeDetection` and `UniversalFakeDetect` are included as git submodules
pointing at their original repositories, rather than vendored copies, since they're
published baselines from other research groups, not original contributions of this
project. Run `git submodule update --init --recursive` after cloning to pull them.

## Paper

The accompanying paper was submitted to **ICMLDE** (International Conference on
Machine Learning and Data Engineering). Draft source is at
[`paper/paper_draft.tex`](paper/paper_draft.tex).
