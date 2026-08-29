# Detector metadata — GenImage++ CLIP-LoRA (OMAT-hardened)

**Source paper:** *Breaking Latent Prior Bias in Detectors for Generalizable
AIGC Image Detection.* NeurIPS 2025 (Poster). arXiv:2506.00874.
[HF dataset/code repo](https://huggingface.co/datasets/Lunahera/genimagepp)

**Domain:** CLIP-embedding classifier, adversarially hardened via
**On-Manifold Adversarial Training (OMAT)** — this project's "what improves
detection" comparison point against `genimagepp_clip_baseline` (same CLIP
backbone, same eval data, the only difference is the OMAT training
procedure). The paper's core claim: detectors trained on shortcut features
tied to a generator's initial latent noise pattern ("latent prior bias")
fail to generalize; OMAT optimizes the initial latent noise itself during
training to force the detector to learn more fundamental generative
artifacts instead.

**Architecture:** same `openai/clip-vit-large-patch14` vision tower as the
baseline, wrapped with a PEFT LoRA adapter (`LoraConfig`, rank=4, alpha=8,
dropout=0.1, targeting `q_proj`/`k_proj`/`v_proj`/`out_proj`) — base CLIP
weights frozen, only the LoRA adapter + `Linear(hidden_size, 1)` classifier
head are trained. Source: `attack_code/Clip_lora_rank4_model.py ::
CLIPLoRADetector`.

**Checkpoint used in this project:**
`attack_code/weights/best_model_low_rank.pt` — dict with keys
`{'epoch': ..., 'model_state_dict': ...}`; this project loads
`checkpoint['model_state_dict']`.

**Training data:** trained with OMAT-generated on-manifold adversarial
examples (latent-space and token-embedding attacks against a grad-enabled
Stable Diffusion forward pass, see `attack_code/core_attacks/`) layered on
top of the same base training distribution as the CLIP baseline — again,
confirm exact composition against the arXiv paper before finalizing paper
language. GenImage++ itself is test-only and was not used for training
either checkpoint.

**Preprocessing at inference:** PIL image → `Resize((224,224))` → ToTensor
→ CLIP normalize (same stats as the baseline). Functionally identical net
effect to the baseline's preprocessing despite being implemented across two
source files (`Clip_lora_rank4_model.py`'s wrapper transform +
`discriminators.py`'s `discriminator_preprocess` for the normalize step).

**Known compatibility fix applied:** same `vision_model.` key-stripping fix
as the CLIP baseline (see `../genimagepp_clip_baseline/detector_metadata.md`)
— required for the checkpoint to load at all under the installed
`transformers` version.

**Inference conventions in this project:** batch size 16, real/fake pools
balanced to match the other three detectors (see `../README.md`).
