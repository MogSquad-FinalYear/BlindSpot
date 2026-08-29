Copies of ../../shared_spectra/ — same generator-level data used for all
four detectors. Correction to an earlier note: this frequency analysis
resizes every image to a standardized 256x256 grid for the FFT computation
itself (see load_gray() in the analysis script) — it was never "native
resolution," regardless of which detector it's paired with. So whether a
given detector internally resizes to 224x224 (this one) or center-crops
natively (UniversalFakeDetect/NPR) doesn't change which version of this
analysis applies to it — there's only one version, and it's valid evidence
for all four. See ../../shared_spectra/README.md for the full write-up.
