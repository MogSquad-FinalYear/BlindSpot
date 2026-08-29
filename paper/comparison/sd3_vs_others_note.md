# SD3 vs. FLUX/SDXL — why "diffusion models fail the same way" is the wrong framing

**Grounded in:** UniversalFakeDetect full-data results (`../universalfakedetect/accuracy/accuracy_auc_table.csv`)
and the high-pass energy comparison (`../universalfakedetect/highpass_energy/highpass_energy.csv`).
Should be re-checked once NPR / GenImage++ CLIP baseline / CLIP-LoRA full
results land, to see whether the pattern below is UniversalFakeDetect-specific
or holds across detector families.

## The numbers

| Generator | Architecture family | AP | AUC | Fake-acc | Fake high-freq energy vs. real |
|---|---|---|---|---|---|
| FLUX.1 | DiT | 59.12 | 61.37 | 7.0% | **-31.6%** (less high-freq than real) |
| SD3 | DiT | 68.80 | 72.45 | 11.7% | **+15.1%** (*more* high-freq than real) |
| SDXL | UNet | 85.61 | 85.39 | 38.5% | -52.4% (less high-freq than real) |

## The finding

SD3's spectral signature is qualitatively different from every other
generator tested — it is the **only** one, across all 5 (BigGAN -283.5%,
GLIDE -234.3%, FLUX -31.6%, SDXL -52.4%), where fake images carry *more*
high-frequency energy than real photos rather than less. Every other
generator, old or new, GAN or diffusion, systematically under-produces
high-frequency detail relative to real images (consistent with the
conventional "generators smooth out fine detail" story); SD3 inverts this.

**But that spectral divergence does not translate into a proportionally
different detection outcome.** SD3 and FLUX — both DiT-based, both trained
by different labs but sharing the modern text-to-image architecture family —
land close together in detectability (AP 59–69, fake-acc 7–12%), despite
having opposite-signed spectral deviations from real images. Meanwhile SDXL,
whose spectral deviation points the *same direction* as FLUX's (both show a
high-frequency deficit), is dramatically more detectable than either DiT
model (AP 85.6, fake-acc 38.5% — over 3x FLUX's and SDXL's numbers are
closer to GLIDE's 27.4% than to either DiT model).

**Read together, this says architecture family (DiT vs. UNet) predicts this
detector's failure severity better than the sign or magnitude of the
generator's spectral deviation from real-image statistics does.** SD3's
unique spectral signature is a real, measurable difference worth reporting —
but it is not the variable driving why UniversalFakeDetect catches ~12% of
its fakes instead of ~38%. Don't present flux/sd3/sdxl as "diffusion models
fail the same way, differing only in degree" — SD3 is spectrally the odd one
out, SDXL is behaviorally the odd one out, and those are two different axes.

## Caveat

This is drawn from one detector (CLIP-embedding-based). Whether
architecture-family-predicts-detectability generalizes, or whether it's an
artifact of what UniversalFakeDetect's CLIP backbone happens to key on,
needs the NPR (pixel/up-sampling-artifact) and GenImage++ CLIP-LoRA results
as a cross-check before stating this as a general claim in the paper rather
than a UniversalFakeDetect-specific observation.
