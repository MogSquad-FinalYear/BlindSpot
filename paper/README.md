# Paper artifacts — Why Deepfake Detectors Fail on Modern Diffusion Generators

Organized outputs for the 8 requested artifact types, per detector. Generated
from existing datasets/checkpoints only (no training, no image generation),
consistent with the project's inference-only, modest-hardware constraint
(NVIDIA T1000 8GB, shared with the desktop GUI session).

## Folder layout

```
paper/
├── universalfakedetect/     # CLIP ViT-L/14 linear probe (Ojha et al., CVPR 2023)
├── npr/                     # Neighboring Pixel Relationships (Tan et al.)
├── genimagepp_clip_baseline/  # CLIP baseline trained on modern DiT generators
├── genimagepp_clip_lora/    # CLIP-LoRA, adversarially/OMAT-hardened variant
├── shared_spectra/          # items 4 & 5 base data — generator property, not
│                             #   detector property (see shared_spectra/README.md)
├── scripts/                 # reusable, detector-agnostic generation scripts
└── comparison/              # cross-detector summary (once all 4 are in)
```

**Status: complete.** All 4 detectors × 5 generators finished. Every detector
folder is fully populated — no empty subfolders anywhere under `paper/`.

Each detector folder contains:
- `accuracy/accuracy_auc_table.{csv,md}` — item 1
- `confusion_matrices/<generator>.png` — item 2
- `score_distributions/score_dist_all.png` — item 3
- `spectra/`, `highpass_energy/` — items 4/5 (copied from `../shared_spectra/`
  into every detector folder — same generator-level data, valid for all four;
  see `shared_spectra/README.md` for why this applies uniformly regardless of
  a given detector's own inference-time resize/crop choice)
- `failure_examples/<generator>/` — actual misclassified images + metadata.csv — item 6
- `detector_metadata.md` — item 8

`comparison/sd3_vs_others_note.md` holds item 7.
`comparison/cross_detector_summary.md` holds the full 4-detector comparison —
this is the main finding of the whole exercise: no two detectors fail on the
same generator, and OMAT adversarial training (CLIP-LoRA vs. CLIP baseline)
nearly eliminates cross-generator variance where ordinary training-distribution
differences don't. Read this file first.
`comparison/data_leakage_verification.md` — sourced (paper text + code, not
inference) confirmation that none of the 5 eval generators appear in any of
the 4 detectors' training data, plus one open provenance question on the
BigGAN eval sample specifically. Read this before citing the OMAT result.

## Important methodology note: the max_sample bug

Every UniversalFakeDetect result that existed before this pass (flux, sd3,
sdxl, BigGan, Glide — see `~/data/UniversalFakeDetect/clip_vitl14_*` without
the `_full` suffix) was computed on **only 100 real + 100 fake images per
generator**, not the full datasets. `validate.py`'s dataset loader silently
falls back to `max_sample=100` whenever the requested cap exceeds either
pool's size — which is always true here (fake pools are 6000–18301 images,
and every run passed `--max_sample=100000` intending "use everything"). The
full data was sitting on disk unused the whole time.

This is fixed in `~/data/UniversalFakeDetect/run_and_dump.py` (no silent
fallback; balances real/fake to `min(len(real), len(fake))` and uses all of
it unless `--max_sample` is explicitly passed) and `~/data/NPR-DeepfakeDetection/run_inference.py`
was written from scratch with the same no-silent-subsampling guarantee.
**Artifacts in this `paper/` folder are built from the `_full` reruns
(`predictions_<generator>_full.csv`, `clip_vitl14_<generator>_full/`), not
the original small-sample results.** The old ap.txt/acc0.txt/acc1.txt files
are left in place for reference but should not be cited in the paper.

## Balanced sampling across detectors

To keep detector comparisons apples-to-apples, every detector's full run
uses the same per-generator sample sizes: real and fake capped to the
smaller pool (BigGan/Glide: 6000/6000 — already balanced; flux/sd3: 6000
fake vs. the 162k shared real pool, real subsampled to 6000; sdxl: 18301
fake, real subsampled to 18301).
