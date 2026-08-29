# OMAT Attack Code

This folder contains the cleaned release of the OMAT attack code used in the GenImage++ project.

## Contents

- `core_attacks/diffusion.py`: grad-enabled Stable Diffusion forward pass.
- `core_attacks/latent_attack.py`: latent adversarial optimization.
- `core_attacks/token_attack.py`: token-embedding adversarial optimization.
- `run_core_attack.py`: unified CLI.
- `discriminators.py`: detector loading and preprocessing utilities for the baseline ResNet-50 and CLIP detectors.
- `Clip_lora_rank4_model.py`: wrapper and model definition for the adversarially trained CLIP-LoRA detector.
- `adv_latents.py`, `adv_latents_1k.py`, `adv_prompt_generate.py`, `sd_forward.py`: compatibility wrappers.

## Install

```bash
pip install -r attack_code/requirements.txt
```

## Run

```bash
python attack_code/run_core_attack.py latent --device cuda:0
python attack_code/run_core_attack.py token --device cuda:0
python attack_code/Clip_lora_rank4_model.py --model_path attack_code/weights/best_model_low_rank.pt --device cuda:0
```

## Checkpoints

Released checkpoints included in this folder:

- `attack_code/weights/resnet_epoch_20.pth`: baseline ResNet-50 detector.
- `attack_code/weights/clip_epoch_20.pth`: baseline CLIP detector.
- `attack_code/weights/best_model_low_rank.pt`: adversarially trained CLIP-LoRA detector (rank 4).

If you want to use a different checkpoint location, set one of:

- `GENIMAGEPP_WEIGHTS_DIR`
- `GENIMAGEPP_RESNET_CKPT`
- `GENIMAGEPP_CLIP_CKPT`
