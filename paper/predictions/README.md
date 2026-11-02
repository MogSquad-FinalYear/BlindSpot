# Per-image prediction scores

Every number in the paper is computed from these files. Each row is one image
scored by one detector; no aggregation has been applied.

```
filepath,true_label,pred_score,pred_label
/path/to/n01440764_10026.JPEG,0,0.0000,0
```

| Column | Meaning |
|---|---|
| `filepath` | absolute path on the machine the sweep ran on; only the basename is used for joining, since the fake sets are identical across detectors |
| `true_label` | 0 real, 1 generated |
| `pred_score` | the detector's predicted probability that the image is generated |
| `pred_label` | `pred_score >= 0.5`, the fixed threshold used for every accuracy figure in the paper |

`filepath` is kept verbatim rather than rewritten, so a row can be traced back
to the exact source image in the original benchmark layout.

## Layout

The five detector directories hold the main sweep: 25 cells, one file per
detector and generator, under each detector's officially released
preprocessing.

```
universalfakedetect/   npr/   genimagepp_clip/   genimagepp_cliplora/   genimagepp_resnet/
  BigGan.csv  Glide.csv  flux.csv  sd3.csv  sdxl.csv
```

Sample sizes are 6,000 real and 6,000 generated for BigGAN, GLIDE, FLUX.1 and
SD3, and 18,300--18,301 per class for SDXL.

`conditions/` holds the ablations, named `<detector>_<generator>_<condition>`:

| Condition | What it is | Used in |
|---|---|---|
| `pad` | native pixel scale preserved: center-crop if larger than 224, zero-pad if smaller | Table 7 |
| `upsample2000` | official bicubic resize to 224, on the matched 2,000-per-class sample | Table 7, §5.1 |
| `nearest2000` | nearest-neighbour resize to 224: no smooth interpolation, no border | §5.1 |
| `pad2000` | the padding path on the same 2,000-per-class sample | §5.1 |
| `clean2000` | FLUX.1 as shipped, PNG fakes against JPEG reals | Table 8 |
| `matched2000` | generated images re-encoded at JPEG q75, reals untouched | Table 8 |
| `jpeg752000` | every image re-encoded at JPEG q75 | Table 8 |

The three FLUX.1 compression conditions use the identical 2,000-per-class
sample, so they are paired within a detector.

## Regenerating the tables

```
python paper/scripts/ensemble_and_reliability.py   # Table 6, Figure 1, ECE values
python paper/scripts/compression_conditions.py     # Table 8
python paper/scripts/full_metrics_and_tests.py     # Tables 3, 4, 5
```

Both of the first two resolve paths against the repository root and need no
editing. Set `BLINDSPOT_PREDICTIONS` to read score files from elsewhere, or
`BLINDSPOT_OUT` to write results elsewhere.

## Two caveats carried over from the paper

Real-image sets are shared across detectors only for BigGAN and GLIDE. For
FLUX.1, SD3 and SDXL each run drew its own sample from the shared
162,000-image real pool, and those samples overlap only incidentally (7 of
6,000 for FLUX.1 and SD3, 200 of 18,300 for SDXL). Fake sets are identical
across all five detectors for every generator, so joining on basename is valid
for fake-image comparisons everywhere and for real images only on BigGAN and
GLIDE.

Generated images in these benchmarks are PNG and real images are JPEG, so
absolute accuracies computed from the `clean` files carry a compression
confound. Section 6 of the paper measures its size.
