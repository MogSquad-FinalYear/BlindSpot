# Data leakage verification — training/eval overlap check for all 4 detectors

Written in response to a direct request to verify whether CLIP-LoRA's (and the
other detectors') near-uniform high accuracy reflects genuine generalization
or train/test overlap, before trusting it as a paper finding. Methodology:
read the actual released training/attack code and the source paper
(arXiv:2506.00874) directly rather than inferring from accuracy patterns
(which would be circular).

## Direct answer: GenImage++ CLIP baseline / CLIP-LoRA (OMAT)

**My eval set (BigGAN, GLIDE, FLUX.1, SD3, SDXL) is fully held out from both
checkpoints' training data. No overlap — high confidence, sourced directly
from the paper text.**

Source: arXiv:2506.00874, Section 6.1 ("Experimental Setup"), quoted exactly:

> "Our experiments primarily follow the GenImage [39] benchmark paradigm.
> Unless otherwise specified, all detectors, including our initial baselines
> and external methods, were trained on the Stable Diffusion v1.4 (SDv1.4)
> subset of GenImage using real images and corresponding SDv1.4 generated
> fakes... For adversarial training, these baseline detectors were
> subsequently fine-tuned using the SDv1.4 training data augmented with
> N = 6000 of our on-manifold adversarial examples (X_adv)..."

And Section 3.3 (confirms the adversarial examples are also SDv1.4-derived):

> "We applied this attack to GenImage SDv1.4 pre-trained CLIP ViT-L/14 +
> Linear and ResNet50 detectors... This process yielded our adversarial
> dataset X_adv, comprising N = 6000 unique on-manifold adversarial images."

Figure 1 in the paper explicitly labels the frozen generator used to make
these 6000 adversarial images as "Stable Diffusion 1.4."

So: **both** `clip_epoch_20.pth` (baseline) and `best_model_low_rank.pt`
(CLIP-LoRA/OMAT) were trained on exactly two things — GenImage's SDv1.4
subset (ImageNet real + SDv1.4 fakes), and 6000 SDv1.4-derived on-manifold
adversarial images. Neither BigGAN, GLIDE, FLUX.1, SD3, nor SDXL appear
anywhere in that training data.

This also directly answers the "what is OMAT" question: it is **not**
adversarial training against the generators being tested. It's adversarial
perturbation of the *initial latent noise* of a single fixed generator
(SDv1.4) that the baseline was already trained on — the goal (per the
paper's own stated hypothesis) is to strip out the detector's dependence on
SDv1.4-specific noise-seed shortcuts, not to expose it to new generator
families. The fact that this transfers to five completely different,
disjoint generators (including a different architecture family, DiT, that
didn't exist in any form in the training data) is the paper's actual claim
and is *not* explainable by leakage — my results (81.6% → 97.0% mean
accuracy) corroborate the same phenomenon the paper itself reports
(Table 1: 79.20% → 94.63% mean accuracy on the original GenImage OOD
subsets), just measured on a mostly-disjoint eval set.

**GenImage++'s own subsets (flux, sd3, sdxl_multistyle, etc.) are
independently confirmed test-only** — Section 5 header: "we introduce
GenImage++, a novel test-only benchmark dataset," and the dataset's own
HuggingFace card states the same. This is asserted, not just inferred from
code defaults.

## Direct answer: NPR

**Held out — confirmed from NPR's own README.** The training command in
`NPR-DeepfakeDetection/README.md` is:

```
python train.py --dataroot ./datasets/ForenSynths_train_val --classes car,cat,chair,horse
```

`ForenSynths_train_val/train/` contains only ProGAN-generated images across
4 LSUN categories (car/cat/chair/horse) — this is the standard Wang et al.
2020 (CNNDetection) training protocol. `ForenSynths_train_val/test/biggan/`
exists in the repo's folder tree but is under `test/`, never referenced by
the training command — BigGAN is a benchmark OOD test domain in NPR's own
paper, not training data. GLIDE does not appear anywhere in this folder tree
at all. So: no overlap between NPR's training and any of my 5 eval
generators.

## Direct answer: UniversalFakeDetect

Already documented in `../universalfakedetect/detector_metadata.md`, sourced
from the repo's own README: trained exclusively on the Wang et al. 2020
ProGAN training set (20 LSUN categories, real + ProGAN fake). Same
conclusion as NPR — no overlap with BigGAN (different GAN, never in
training), GLIDE, FLUX.1, SD3, or SDXL.

## A finding this check surfaced: possible dataset-provenance issue, not leakage

While verifying the CLIP baseline, I found something that needs flagging
separately from the leakage question. The source paper's own Table 1 reports
their baseline CLIP+Linear model scoring **62.66%** accuracy on GenImage's
BigGAN subset. My measured result for the same checkpoint (`clip_epoch_20.pth`)
on my BigGAN eval data was **9.6%** fake-accuracy — a large enough gap to
need explanation before trusting it.

I ruled out a pipeline bug directly: I re-ran 16 images (8 real, 8 fake)
through both my script's PIL-based preprocessing and a faithful
reimplementation of the paper's actual `kornia`-based
`discriminator_preprocess()` (installed `kornia`, called the real function
from `attack_code/discriminators.py`), using the same loaded checkpoint.
**Both preprocessing paths produced near-identical scores** — e.g. fake
BigGAN images scoring 0.0067 vs 0.0446, 0.0001 vs 0.0001, 0.0200 vs 0.0523 —
same catastrophic near-zero pattern either way. This rules out a
preprocessing/loading bug in my scripts as the explanation.

**Most likely explanation: my BigGAN eval images are not the same data as
whatever the paper's Table 1 used.** Per `PROJECT_STATE.md`, this project's
BigGAN dataset was "re-downloa
ded via HF mirror after original system flash
lost it" — a third-party mirror, not necessarily identical in preprocessing,
compression, or exact sample composition to GenImage's original release or
whatever exact split the paper evaluated on. A 6x accuracy gap on
nominally-the-same generator/checkpoint pair is large enough that this
should be verified against the official GenImage BigGAN release before the
9.6% number goes into the paper as a confident finding — cite it as "our
measured result on this specific BigGAN sample" rather than asserting it
matches the source paper's own reported baseline behavior, until re-verified
against an authoritative BigGAN source.

This does **not** affect the leakage conclusion above (that's about training
data composition, confirmed from the paper's text, independent of which
exact BigGAN images I evaluated on) or the CLIP-LoRA/OMAT improvement
finding (both my baseline and LoRA numbers moved in the same direction as
the paper's own Table 1: baseline low → LoRA >99%, mirroring their
62.66% → 93.22%), but the *magnitude* of my BigGAN baseline collapse
specifically should be treated as provisional pending that check.

### Follow-up investigation (2nd pass, 30-image sample): resize strategy is a real contributing factor

Went further to isolate the cause, in order of what was actually checked:

1. **Confirmed my BigGAN images' resolution is correct, not a bad-mirror artifact.** The fake BigGAN images are 128×128; real ImageNet images are variable full-size (e.g. 375×500). Web search confirmed BigGAN's GenImage subset is *officially* 128×128 — "notably different from other generators in the dataset—the resolution for other models like VQDM is 256×256." So the small resolution is expected, not something the HF mirror broke.

2. **Tested whether the forced upsample (128×128 → 224×224 bicubic) itself explains the gap**, since this exact resolution-mismatch scenario is a documented pitfall in this literature — NPR's own README explicitly instructs setting `no_resize`/`no_crop` when evaluating GenImage-style benchmarks with mixed native resolutions, for exactly this reason. Ran 30 real + 30 fake BigGAN images through three preprocessing variants:

   | Preprocessing | Fake caught | Real correct |
   |---|---|---|
   | Bicubic upsample 128→224 (official `discriminator_preprocess` behavior) | 0.0% | 96.7% |
   | **Native-resolution padding** (128×128 pasted onto a black 224×224 canvas, no interpolation) | **23.3%** | 93.3% |
   | Nearest-neighbor upsample | 10.0% | 0.0% (collapses differently) |

   Padding roughly tripled fake-BigGAN detection on the small sample — moving toward the paper's 62.66%, but not reaching it.

3. **Could not obtain an authoritative GenImage BigGAN sample.** GenImage's official distribution (github.com/GenImage-Dataset/GenImage) is gated behind Baidu Yunpan and Google Drive — no scriptable, authentication-free download. A third-party Kaggle mirror surfaced but wouldn't have resolved the "is this really their data" question either.

### Full-scale confirmation (3rd pass, full 6000/6000 BigGAN set, both checkpoints) — resolves most of the gap

Added a `--preprocess {upsample,pad}` flag to `run_clip_baseline.py`/`run_clip_lora.py`
(default `upsample`, matching official behavior — nothing else in this
project changed) and reran the full BigGAN set both ways for both
checkpoints. Results:

| Checkpoint | Preprocessing | Fake-acc | Real-acc | Overall acc | AP | AUC |
|---|---|---|---|---|---|---|
| CLIP baseline | upsample (official, in main table) | 9.60% | 96.48% | 53.04% | 64.26 | 67.97 |
| CLIP baseline | **pad** (diagnostic) | 25.30% | 98.02% | **61.66%** | 87.87 | 90.38 |
| CLIP-LoRA (OMAT) | upsample (official, in main table) | 99.70% | 94.82% | 97.26% | 99.55 | 99.70 |
| CLIP-LoRA (OMAT) | **pad** (diagnostic) | 16.28% | 93.12% | 54.70% | 64.78 | 69.13 |

**This resolves most of the original mystery, for the CLIP baseline: my
pad-mode overall accuracy (61.66%) is a near-exact match to the paper's
reported 62.66%.** The likely full explanation is now: (a) the paper's
Table 1 "Accuracy" column for a per-generator cell is almost certainly
**overall accuracy** (real+fake combined), not fake-only accuracy — I was
originally comparing my fake-accuracy (9.6%) against their overall-accuracy
metric, an apples-to-oranges comparison; and (b) resize/interpolation
strategy genuinely matters and padding is closer to whatever they actually
did. Both effects were real; together they close nearly the entire gap.

**CLIP-LoRA moves the opposite direction and drops sharply under padding**
(97.26% → 54.70% overall, 99.70% → 16.28% fake-accuracy) — the reverse of
the baseline's improvement. This makes sense once you consider *why*: per
the paper's Appendix A.2, every fine-tuning strategy (including LoRA) used
Kornia resize-to-224×224 preprocessing during training. CLIP-LoRA was
specifically calibrated to the upsample pipeline; feeding it native-padded
images (large black borders it never saw in training) pushes it
out-of-distribution in a way the baseline — never adversarially/robustly
fine-tuned to any particular preprocessing quirk — doesn't suffer from.
**This means CLIP-LoRA's robustness is at least partly preprocessing-specific,
not fully preprocessing-invariant** — worth stating explicitly in the paper
rather than assuming OMAT's generalization gains are preprocessing-agnostic.

**What's now resolved vs. still open:**
- Resolved: the gap was not a bug in my pipeline (confirmed earlier), and is
  now ~90% explained by metric definition (overall vs. fake-only accuracy)
  plus a real, demonstrated preprocessing effect.
- Still open: pad-mode's 61.66% (CLIP baseline) vs. the paper's 62.66% is
  close enough (0.87 points, well within margin of a different exact image
  sample) to treat as resolved for practical purposes, but I still don't
  have their exact eval script or image sample to confirm this isn't
  coincidental proximity.
- New finding, not fully resolved: *why* padding specifically helps the
  baseline but hurts LoRA this much is explained qualitatively above
  (training-preprocessing mismatch) but not verified against their internal
  training logs — reported as the most plausible explanation, not a
  certainty.

**Practical recommendation for the paper, updated:** report both numbers
per checkpoint, clearly labeled by preprocessing mode (see table above and
`../genimagepp_clip_baseline/accuracy/biggan_preprocessing_sensitivity/`,
`../genimagepp_clip_lora/accuracy/biggan_preprocessing_sensitivity/` for the
full confusion matrices). **Keep the upsample-mode numbers as the primary,
headline results in the main accuracy tables** (this is what the officially
released `discriminator_preprocess` function does, and is applied
consistently across all 5 generators and all 4 detectors elsewhere in this
project — switching only BigGAN to a different preprocessing would break
that consistency). Cite the pad-mode numbers as a documented sensitivity
check, not a replacement.

## Open question I cannot resolve with materials on hand: real-image overlap

Separate from which *generator's fakes* were used in training, there's a
second leakage vector: whether the *real* ImageNet images in my eval pools
overlap with the real ImageNet images GenImage's SDv1.4 training subset
used. GenImage's convention pairs real ImageNet images against each
generator subset; if the same underlying ImageNet sample was reused across
GenImage's biggan/glide/sdv1.4 subsets (plausible but not confirmed from
what's released here), there could be real-image (not fake-image) overlap
between what CLIP-LoRA saw in training and what I evaluated on. This
wouldn't explain the *fake*-accuracy jump on BigGAN (the main effect under
scrutiny), but could inflate real-accuracy/AP slightly across all detectors'
GenImage-derived eval sets. I don't have access to GenImage's exact SDv1.4
training split to check this directly — flagging as unresolved rather than
asserting either way.

## Sample-size asymmetry note (also requested)

BigGAN/GLIDE/FLUX/SD3 evals use n=6000 per class; SDXL uses n≈18,300 per
class (all available data, capped to match each generator's fake-pool
size — see `../README.md`). Worst-case 95% CI half-width on a proportion
(p=0.5) is ±1.26 points at n=6000 vs ±0.72 points at n=18,300; for the
actual observed proportions in this project (mostly far from 50%, e.g.
7–99%) the CIs are considerably tighter than that. Every cross-generator and
cross-detector gap reported in `cross_detector_summary.md` is tens of
points, far larger than these CI widths, so sample-size asymmetry does not
change any conclusion — but SDXL's estimates are specifically ~1.7x more
precise than the other four generators' estimates, worth noting rather than
leaving implicit.
