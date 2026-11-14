# Orchestrator

Replaces the hand-tuned detector **router** (`../rule_engine.py`) with a
**trained orchestrator**: a meta-model that decides how to combine — or which
to trust among — the five detectors BlindSpot evaluates.

## Why replace the router

The router picks one detector per image from one feature (`hf_energy`) and two
hand-set thresholds. Two measurements say that ceiling is low:

- **CLIP-LoRA dominates every generator** (97.3 / 97.4 / 96.3 / 97.2 / 96.9 on
  BigGan / Glide / flux / sd3 / sdxl). A router that *selects* a detector can
  therefore do no better than "always call CLIP-LoRA" unless per-image
  complementarity is real.
- **It is real.** On the 24k BigGan+Glide images where all five detectors
  already have scores, 98–99% of CLIP-LoRA's errors are recovered by at least
  one other detector, and a logistic stacker on the three router detectors
  reached **99.26% vs CLIP-LoRA's 97.33% (+1.93pp)** — while majority voting
  gained nothing. The headroom is real but only a *trained* combiner reaches it.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 1 | `build_manifest.py` | `data/manifest.csv` — 72,300 images, labels, splits |
| 2 | `harvest_existing.py` | `data/scores_harvested.csv` — scores already on disk |
| 3 | `run_inference.py` | `data/scores_new.csv` — the rest, on GPU |
| 4 | `features.py` | `data/features.csv` — 28 per-image features |
| 5 | `build_table.py` | `data/training_table.csv` + headroom report |
| 6 | `train_orchestrator.py` | `data/orchestrator_results.json` |
| — | `augment.py` | optional JPEG re-encode control (see *Confound*) |

```bash
python3 build_manifest.py
python3 harvest_existing.py
python3 run_inference.py --all          # ~2 h on a T1000
python3 features.py
python3 build_table.py
python3 train_orchestrator.py
```

Stages 3 and 4 are resumable — both skip work already present in their output.

## Dataset

72,300 images: **42,300 fake** / **30,000 real**.

| cell | n | source |
|---|---|---|
| fake:BigGan | 6,000 | `imagenet_ai_0419_biggan/val/ai` |
| fake:Glide | 6,000 | `imagenet_glide/val/ai` |
| fake:flux | 6,000 | `genimagepp/flux` |
| fake:sd3 | 6,000 | `genimagepp/sd3` |
| fake:sdxl | 18,300 | `genimagepp/sdxl` |
| real:imagenet_val_biggan | 6,000 | ImageNet val |
| real:imagenet_val_glide | 6,000 | ImageNet val |
| real:imagenet_train | 13,000 | ImageNet train |
| real:coco_val2017 | 5,000 | COCO — out-of-domain reals |

**54,317 images (75%) already carried all five detector scores** from the
BlindSpot evaluation CSVs. Those were verified reproducible against a fresh
GPU pass (max |Δ| 4.9e-05, i.e. nothing past the 4-dp rounding in the CSVs),
so only the ~18k newly added reals needed inference — about 2 GPU-hours
instead of 8.

Two split columns:

- `split` — stratified 70/15/15, for in-distribution training.
- `gen_fold` — generator name, for **leave-one-generator-out** evaluation.
  Aggregate accuracy can look excellent while LOGO collapses, so both are
  reported.

`real_coco/humans/images` is a 600-image *subset* of `val2017`, not a separate
pool; listing both would put identical images in train and test. It is carried
as the `human_slice` tag instead.

## Confound: format leaks the label

**Every fake in this corpus is a `.png`; every real is a `.jpeg`/`.jpg`.** No
exceptions, 42,300 / 30,000. Anything sensing JPEG quantization separates the
classes perfectly here while carrying no evidence about generation.

This is not hypothetical. `rule_engine.py`'s docstring records the field
failure it causes: an AI-generated portrait saved as a 348×456 screenshot
scored 0.0014 — "confidently real". Most real uploads are screenshots or
re-saves, so production meets re-encoded fakes constantly and pristine PNGs
almost never.

Handling:

1. `features.py` quarantines `blockiness_8` and `jpeg_grid_ratio` in
   `SHORTCUT_FEATURES`, excluded from training by default.
   `train_orchestrator.py --with-shortcuts` reports what they add — that
   number measures the confound, not model quality.
2. `augment.py` builds a re-encoded control slice (JPEG q75, optional
   downscale) from the **test** split, so "how much of this accuracy was
   format?" is answerable rather than assumed.

The confound is a property of the corpus, not of this pipeline — it applies
equally to every accuracy number in the BlindSpot paper.

## Detector fidelity

`detectors_bulk.py` mirrors each origin project's model construction and
preprocessing, because the table mixes harvested and freshly-computed scores;
any drift would become a seam the orchestrator learns instead of the signal.
All five verified against stored CSVs at max |Δ| < 1e-4 on two generators.

One trap found: `run_resnet_baseline.py` **defaults to `--normalize clip`, but
the stored `predictions_resnet_*_full.csv` were produced with
`--normalize imagenet`** (mean |Δ| 4.6e-06 under ImageNet vs 7.2e-04, max
5.4e-02, under CLIP). Taking the script default on faith would have put fresh
reals on a different scale from harvested fakes.

`features.py` imports `hf_energy` from `rule_engine` rather than
reimplementing it, so orchestrator-vs-router is a fair comparison (verified
bit-identical on 40 sampled images).

## Strategies trained

| strategy | decides | inference cost |
|---|---|---|
| `selection` | which single detector to trust, from features alone | 1 detector call |
| `cascade` | CLIP-LoRA first; escalate inside a tuned uncertainty band | 1 + p·(N−1) |
| `fusion` | meta-classifier over all detector logits + features | N detector calls |

`selection` is the learned form of what `rule_engine.py` does by hand, and is
the only one that keeps the router's current cost. An oracle is reported as a
ceiling, **not a target**: with five detectors, "at least one was right" is
high even when each is near chance on that image.

## Hardware notes

NVIDIA T1000 8GB. fp32 throughout — fp16 autocast measured *slower* (2.0 vs
7.4 img/s), since TU117 has no tensor cores. Per 10k images: NPR ~0.6 min,
ResNet ~1.6 min, each CLIP ViT-L/14 ~22 min. `run_inference.py` runs one
detector per process; three CLIP towers at fp32 do not fit side by side.

## Results

Full table: 72,300 images, all five detectors, 0 failures, 0 dropped rows.

### Detector baselines (threshold 0.5)

| detector | accuracy |
|---|---|
| cliplora | 97.26% |
| npr | 80.93% |
| clipbaseline | 80.59% |
| ufd | 60.90% |
| resnet | 55.25% |
| **router (deployed rule)** | **95.30%** |

The router is **1.95pp worse than simply always calling CLIP-LoRA**. It sends
17.7% of images to NPR, and on BigGan that means handing NPR (84.1%) images
CLIP-LoRA would have scored 99.7% on. Its one specialist rule is where all of
its loss comes from — the `hf_energy` feature fires correctly; the premise
that a specialist should win there is what is wrong.

### Strategy comparison

| strategy | in-dist | mean LOGO | worst fold | cost |
|---|---|---|---|---|
| **cascade (3 det)** | 99.59% | **99.26%** | **98.56%** | **1.38 calls** |
| cascade (5 det) | 99.58% | 99.17% | 98.12% | 1.75 calls |
| best single (cliplora) | 97.27% | 97.27% | 95.81% | 1.00 |
| selection | 97.29% | 97.07% | 95.55% | 1.00 |
| router | 95.25% | 94.60% | 88.80% | 1.00 |
| fusion | 99.95% | 87.73% | **42.86%** | 5.00 |

Three conclusions:

1. **Fusion must not ship.** It wins in-distribution and collapses to 42.86%
   on held-out Glide with **0.00% recall** — it calls every unseen Glide fake
   real. It had learned to identify the generator from the five-score pattern.
   In-distribution numbers alone would have shipped a system that fails
   silently on the first genuinely new generator.
2. **Selection is a dead end.** Below best-single on every fold. A one-call
   router cannot beat just calling CLIP-LoRA — that is a verdict on the
   router *shape*, not on its thresholds.
3. **Three detectors beat five.** CLIP-baseline and ResNet slightly hurt while
   adding a third CLIP tower and 21% more calls. The winning set is exactly
   what `app.py` already loads.

### Robustness: the format confound, measured

Same 6,300 test-split images, clean vs JPEG q75 re-encoded:

| system | clean | re-encoded | delta |
|---|---|---|---|
| npr | 86.00% | 45.60% | −40.40 |
| ufd | 61.67% | 59.60% | −2.06 |
| cliplora | 96.92% | 71.52% | −25.40 |
| router (rule) | 94.16% | 62.81% | **−31.35** |
| **ORCH cascade** | 99.59% | **85.71%** | **−13.87** |

Fake-side recall, which is what re-encoding actually destroys:

| detector | clean | re-encoded | delta |
|---|---|---|---|
| npr | 83.77% | 2.57% | **−81.20** |
| cliplora | 98.71% | 49.69% | −49.03 |
| ORCH cascade | 99.89% | 74.31% | −25.57 |

Real-side accuracy *rises* slightly (+0.7–0.9pp): after JPEG everything looks
more like a photo. NPR is effectively destroyed — 2.57% recall means it
detects almost nothing once compression removes the high-frequency evidence
it keys on. BigGan fakes suffer worst (100% → 49.7%), sdxl least (100% →
97.4%), consistent with GAN artifacts living in exactly the frequencies JPEG
discards.

**Read this as the honest production number.** 99.26% is a clean-corpus
figure; ~85.7% is what this stack does on re-encoded input, and most real
uploads are re-encoded. The orchestrator is still the right choice — it
degrades **2.3× less than the router** — and its escalation rate rises from
16.9% to 48.9% on re-encoded input, i.e. it correctly notices it is less sure
and spends more compute. But no number in the BlindSpot tables, this one
included, describes performance on screenshots.

The clear next step is retraining on augmented data so the stack learns
artifacts that survive compression, rather than ones that do not.
