# Frequency-domain analysis (paper items 4 & 5) — shared across all 4 detectors

The FFT spectrum comparison and high-pass energy numbers depend only on the
**generator's own image statistics**, computed on a standardized 256×256
grid (see `load_gray()` in `highpass_energy.py` / `frequency_analysis.py`) —
not on which detector is evaluating them, and not on any given detector's
own inference-time preprocessing. UniversalFakeDetect and NPR center-crop
natively (no resize) at inference, while the GenImage++ CLIP baseline and
CLIP-LoRA resize to 224×224 at inference — but neither matches this
analysis's own 256×256 standardization either way, so there is no
"native-resolution version" to diverge from. One version of this analysis
is valid evidence for all four detectors; it's copied into each detector's
`spectra/` and `highpass_energy/` folder for convenience rather than only
living here.

(An earlier version of this note incorrectly claimed a resizing detector
would need a separate rerun of this analysis at its own resize resolution —
that confused "does the detector's preprocessing resize" with "does this
analysis's own FFT preprocessing resize," which are unrelated questions.
Corrected.)

## Files

- `spectra_BigGan.png` / `spectra_BigGan_diff.png` — BigGAN vs its own paired
  real split (`imagenet_ai_0419_biggan/val/nature`).
- `spectra_Glide.png` / `spectra_Glide_diff.png` — GLIDE vs its own paired
  real split (`imagenet_glide/val/nature`).
- `spectra_flux_sd3_sdxl.png` / `spectra_flux_sd3_sdxl_diff.png` — FLUX, SD3,
  SDXL vs the shared real pool (`imagenet_ai_0419_biggan/train/nature`,
  since GenImage++ ships fakes only, no matching reals).
- `spectra_comparison_all5.png` — all 8 panels (3 real baselines + 5 fakes)
  side by side, no diff overlay (just for quick visual scan).
- `master_hf_ratios.csv` — high-freq/low-freq **log-energy gap** per
  label (a single scalar summary; kept for continuity with earlier analysis,
  but see the caveat below).
- `../universalfakedetect/highpass_energy/highpass_energy.csv` — the
  item-5 numeric comparison: mean high-frequency-band energy for real vs.
  fake **separately** (not collapsed into one gap number), per generator,
  with the paired real pool matching each generator (BigGan/Glide use their
  own val/nature; flux/sd3/sdxl use the shared train/nature pool).

## Reading the numbers

`master_hf_ratios.csv` (log-energy gap, more negative = more energy
concentrated in low frequencies relative to high):

| label | gap |
|---|---|
| real_BigGan_paired | -1.748 |
| fake_BigGan | -3.288 |
| real_Glide_paired | -1.762 |
| fake_Glide | -3.025 |
| real_shared_pool_flux_sd3_sdxl | -1.837 |
| fake_flux | -1.978 |
| fake_sd3 | -1.861 |
| fake_sdxl | -1.882 |

**Key pattern:** BigGAN and GLIDE (GAN-era / early-diffusion) show a *much*
larger high-frequency deficit than real photos (gap drops by ~1.5–1.6) —
these generators are spectrally very distinguishable from real images.
FLUX/SD3/SDXL (2025-era) show a far smaller deviation (gap drops by only
~0.02–0.14) — modern generators' spectra are much closer to real-photo
statistics, which is consistent with why detectors trained on/calibrated to
older generators fail on them.

**Caveat carried over from earlier analysis:** a single scalar "gap"
correlated against per-generator fake-accuracy across only 5 points is not
statistically meaningful (too few points, not necessarily monotonic) — do
not present that specific correlation as a finding. Use the *shape* of the
diff maps (see PNGs) and the real-vs-fake **absolute** high-pass energy
numbers in `highpass_energy.csv` instead, which is what item 5 asked for.
