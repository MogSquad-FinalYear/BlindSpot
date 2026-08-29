# Cross-detector comparison — all 4 detectors, full data

All numbers from `../<detector>/accuracy/accuracy_auc_table.csv`, using each
detector's official preprocessing consistently across all 5 generators.
Sample sizes: BigGAN/GLIDE/FLUX/SD3 = 6000 real + 6000 fake each; SDXL =
18300–18301 each. See `../README.md` for the max_sample-bug fix that makes
these numbers trustworthy (prior results used only 100/100 images), and
**`data_leakage_verification.md` for a sourced check confirming none of the
5 eval generators appear in any of the 4 detectors' training data** — read
that before citing the OMAT result below as a clean generalization finding.

That file also documents a resolved discrepancy worth knowing about: the
source paper's own Table 1 reports 62.66% for their CLIP baseline on
BigGAN; my official-preprocessing measurement was 9.6% fake-accuracy
(53.04% overall). Verified this wasn't a pipeline bug, then found the gap
is ~90% explained by (a) their Table 1 metric almost certainly being
overall accuracy, not fake-only, and (b) a real, full-scale-verified
preprocessing effect — BigGAN's natively-128×128 images get force-upsampled
to 224×224 by the official `discriminator_preprocess`, and using
native-resolution-preserving padding instead moves the CLIP baseline to
61.66% overall accuracy, matching the paper almost exactly. The same padding
change *hurts* CLIP-LoRA badly (97.26%→54.70% overall) since it was
fine-tuned specifically against the upsample pipeline. Both preprocessing
variants, full-scale, both checkpoints, are in that file and in
`../genimagepp_clip_baseline/accuracy/biggan_preprocessing_sensitivity/` /
`../genimagepp_clip_lora/accuracy/biggan_preprocessing_sensitivity/`. The
tables and findings below use the official upsample-preprocessing numbers
throughout (consistent methodology across all 5 generators × 4 detectors);
Finding 3 has been corrected accordingly.

## Fake-accuracy side by side

| Generator | UniversalFakeDetect | NPR | GenImage++ CLIP baseline | GenImage++ CLIP-LoRA (OMAT) |
|---|---|---|---|---|
| BigGAN | 82.95% | 84.12% | **9.60%** | 99.70% |
| GLIDE | 27.40% | 94.33% | 69.13% | 99.87% |
| FLUX.1 | 7.00% | 97.73% | 86.48% | 96.48% |
| SD3 | 11.70% | 90.08% | 88.60% | 98.27% |
| SDXL | 38.52% | 54.64% | 76.98% | 99.17% |
| **Mean acc** | 65.31% | 85.70% | 81.55% | **96.99%** |

## AP / AUC side by side

| Generator | UFD AP/AUC | NPR AP/AUC | CLIP-baseline AP/AUC | CLIP-LoRA AP/AUC |
|---|---|---|---|---|
| BigGAN | 98.21 / 98.19 | 88.74 / 91.28 | 64.26 / 67.97 | 99.55 / 99.70 |
| GLIDE | 84.26 / 85.55 | 93.11 / 95.15 | 94.09 / 94.03 | 99.67 / 99.82 |
| FLUX.1 | 59.12 / 61.37 | 94.87 / 96.90 | 98.50 / 98.59 | 99.22 / 99.31 |
| SD3 | 68.80 / 72.45 | 92.06 / 94.37 | 98.69 / 98.71 | 99.57 / 99.65 |
| SDXL | 85.61 / 85.39 | 75.45 / 76.53 | 97.05 / 97.03 | 99.13 / 99.38 |
| **Mean** | 79.20 / 80.59 | 88.85 / 90.85 | 90.52 / 91.27 | **99.43 / 99.57** |

## Finding 1 — no two detectors fail on the same generator

Fake-accuracy rankings (worst → best generator) diverge sharply by detector:

- **UniversalFakeDetect:** FLUX < SD3 < GLIDE < SDXL < BigGAN
- **NPR:** SDXL < BigGAN < SD3 < GLIDE < FLUX
- **GenImage++ CLIP baseline:** BigGAN < GLIDE < SDXL < FLUX < SD3

Three detectors, three different worst-case generators (FLUX, SDXL, and
**BigGAN** respectively). The CLIP baseline result is the sharpest
illustration: it catches 86–89% of FLUX/SD3 fakes — the exact generators
UniversalFakeDetect almost completely misses — yet catches under 10% of
BigGAN, the one generator every other detector in this project handles
reasonably well (82–84%). **Generalization failure is a property of a
specific detector's training distribution and detection signal, not a
property of "how modern the generator is."** A detector trained with more
modern generators in its own history (as the GenImage++ CLIP baseline
plausibly was, vs. UniversalFakeDetect's ProGAN-only training —
see `../universalfakedetect/detector_metadata.md`) can trade GAN-era
accuracy for diffusion-era accuracy, not simply gain one without losing the
other.

## Finding 2 — OMAT is the standout result: adversarial training on
## on-manifold latents closes the gap almost completely, everywhere

This is the project's clearest "what improves detection" result. CLIP-LoRA
(OMAT-hardened) and the CLIP baseline share the **same CLIP ViT-L/14
backbone and the same eval data** — the only difference is OMAT training.
The effect:

- Mean fake-accuracy: 81.55% → **96.99%** (+15.4 points)
- BigGAN specifically: 9.60% → **99.70%** (+90 points — the baseline's worst
  case becomes the LoRA model's near-perfect case)
- Every single generator lands at 96.3–99.9% fake-accuracy under OMAT —
  the huge cross-generator variance seen in every other detector (a 9–89
  point spread for the CLIP baseline, a 7–83 point spread for
  UniversalFakeDetect) essentially disappears (a 3.4-point spread, 96.48–99.87%).

This is a stronger and more useful "what improves detection" story than
"adversarial training helps somewhat" — it suggests OMAT isn't teaching the
detector a specific generator's artifacts (which would trade one blind spot
for another, as the CLIP baseline vs. UniversalFakeDetect comparison shows
happens with ordinary training-distribution differences) but something closer
to a generator-invariant signal. Worth stating carefully in the paper: this
is one training run of one method on one architecture family (CLIP
ViT-L/14) — NPR (a structurally different, non-CLIP detector) still shows
large cross-generator variance (54–98%) despite not being adversarially
trained, so OMAT's benefit here shouldn't be overgeneralized as "any
adversarial training would do this" without further evidence.

**Additional caveat found during the BigGAN preprocessing sensitivity check**
(`data_leakage_verification.md`): CLIP-LoRA's near-perfect BigGAN score is
*not* preprocessing-invariant — swapping the official upsample preprocessing
for a native-resolution-preserving alternative drops it from 99.70% to
16.28% fake-accuracy (the CLIP baseline, run through the same swap, moves
the *opposite* direction, 9.60%→25.30%). So part of what looks like "OMAT
learned a generator-invariant signal" is at least partly "OMAT learned a
signal that depends on being fed images through the exact preprocessing
pipeline it was fine-tuned with" — a narrower, more preprocessing-coupled
form of generalization than the headline 96.99% mean accuracy alone
suggests. Both numbers (upsample and pad) are real and reproducible; this
doesn't overturn Finding 2, but it means "generator-invariant" should be
qualified as "generator-invariant, under this specific preprocessing" in
the paper.

## Finding 3 — BigGAN is not a uniformly "easy" generator

It's tempting to treat BigGAN (GAN-era, closest to most detectors' training
lineage) as the easy baseline case in every row. It isn't, though the
original phrasing here overstated it — **corrected after a preprocessing
sensitivity check** (see `data_leakage_verification.md`): UniversalFakeDetect
on FLUX (7.00%) is unambiguously the single worst cell in this table, not
the GenImage++ CLIP baseline on BigGAN. The CLIP baseline's official-preprocessing
BigGAN number (9.60% fake-accuracy) is very close to UFD-FLUX but not lower
than it; under a diagnostic alternative preprocessing that better matches
the source paper's own reported baseline behavior (native-resolution
padding instead of upsampling BigGAN's natively-128×128 images), the CLIP
baseline's BigGAN number moves to 25.3% fake-accuracy / 61.66% overall
accuracy — no longer competitive for "worst cell" at all. What survives
this correction: **BigGAN is still the CLIP baseline's own worst generator
among its 5** (25.3% vs. 69–89% on the other four, under either
preprocessing), and it's still true that generator "easiness" tracks what a
detector was actually trained on rather than the generator's age — just not
via the specific "single worst cell" claim originally made here.

## Relationship to item 7 (SD3 vs. FLUX/SDXL spectral note)

See `sd3_vs_others_note.md`, written from UniversalFakeDetect data. Re-reading
it against this table: SD3's fake-accuracy is close to FLUX's under
UniversalFakeDetect (11.7 vs 7.0) and under NPR (90.1 vs 97.7), but is
slightly *higher* than SDXL's under NPR/UFD despite SD3 having the more
unusual (positive) spectral deviation. Under the CLIP baseline, SD3 and FLUX
remain each other's closest neighbors (88.6 vs 86.5) while SDXL sits lower
(77.0) and BigGAN far lower (9.6) — architecture-family clustering (SD3/FLUX
as DiT-based, both trained by different labs but similar text-to-image
pipeline) holds up as a better predictor than the spectral sign flip across
three of four detectors. Under OMAT/CLIP-LoRA the whole question becomes
moot — all generators are caught at 96%+ regardless of family or spectral
signature.

## Caveats

- Exact training-data composition for the two GenImage++ checkpoints
  (CLIP baseline, CLIP-LoRA) isn't fully documented in the released repo —
  see the "confirm against arXiv 2506.00874" notes in each
  `detector_metadata.md`. The *relative* baseline-vs-LoRA comparison is solid
  (same backbone, same data, controlled comparison); claims about *why* the
  CLIP baseline in particular does so poorly on BigGAN specifically would
  benefit from confirming what was and wasn't in its training set.
- All four detectors here use a CLIP or CNN-based single-image classifier
  paradigm at inference; "detector diversity" in this project spans training
  distribution and one adversarial-training method, not fundamentally
  different model classes beyond NPR's pixel-artifact signal vs. the three
  CLIP-embedding detectors.
