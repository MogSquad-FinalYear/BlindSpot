# cernpix-backend

Flask service that replaces CernPix's hardcoded filename-regex flagging with
real AI-generated-image detection. For each uploaded image it picks one of
three trained detectors using a rule based on measurable image features,
runs inference, and stores the result in MongoDB.

## Why Flask, not Node

All three detectors are PyTorch models with Python-only inference code
(`NPR-DeepfakeDetection`, `UniversalFakeDetect`, `genimagepp_detectors`,
all siblings of this directory). Flask runs that code directly; a Node
backend would have to shell out to Python per request or reimplement model
loading in JS, so Flask is the direct choice here.

## Detector selection rule

See `rule_engine.py` for the full rationale. Summary: two measurable
features decide the detector -

- **Resolution** (min side in pixels) - GAN-era generators (BigGAN, GLIDE)
  top out around 256-512px; modern DiT generators (Flux, SD3, SDXL) default
  to 1024x1024.
- **High-frequency spectral energy** - mean log-magnitude FFT energy in the
  outer 20% frequency radius of a 256x256 grayscale version of the image,
  the same method used in the BlindSpot project's
  `paper/scripts/highpass_energy.py`. GAN-era fakes measure far below real
  photos and DiT-era fakes (see the numbers in `rule_engine.py`'s docstring).

Large image + real-like spectrum -> **genimagepp CLIP-LoRA** (trained on
Flux/SD3/SDXL). Small image with a strongly GAN-suppressed spectrum ->
**NPR** (built to exploit exactly that artifact). Everything else ->
**UniversalFakeDetect** (CLIP features generalize better on ambiguous cases).

Every response includes a `reason` string built from the actual computed
feature values, not a canned sentence.

## Setup

```bash
python3 -m venv --system-site-packages .venv   # reuses system torch/transformers
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in MONGODB_URI
python app.py          # listens on :8010
```

`--system-site-packages` matters: torch/torchvision/transformers/numpy/Pillow
are expected to already be installed system-wide (they were, from prior ML
environment setup on this machine). Only `requirements.txt`'s packages
install into the venv itself.

Model checkpoints are read directly from the sibling detector repos:

- `../NPR-DeepfakeDetection/NPR.pth`
- `../UniversalFakeDetect/pretrained_weights/fc_weights.pth`
- `../genimagepp_detectors/attack_code/weights/best_model_low_rank.pt`

No download step needed - these already exist on this machine from the
BlindSpot evaluation work. `CLIPVisionModel.from_pretrained(...)` and CLIP's
own `.load()` both resolve from local caches
(`~/.cache/huggingface`, `~/.cache/clip`) with no network access required.

## Endpoints

- `POST /api/detect` - multipart form, field `images` (one or more files),
  optional field `folder`. Saves each image to `uploads/`, runs the rule +
  detector, saves the result to MongoDB, returns
  `{"results": [{filename, folder, flagged, score, detector,
  detector_display_name, reason, features, inference_ms}, ...]}`.
- `GET /api/detections?folder=<name>` - returns every stored detection
  record, newest first, optionally filtered by folder.
- `GET /health` - liveness check.

## MongoDB

Stores every processed image (flagged and clean), not just flagged ones, so
the UI can show detector/reason on hover for anything and "fetch previous
responses" naturally returns full history. Database/collection names are
configurable via `.env` (default `cernpix.detections`).
