"""
GenImage++ CLIP-LoRA (rank 4) detector wrapper, adversarially trained with
On-Manifold Adversarial Training (OMAT) on modern DiT-era generators
(Flux/SD3/SDXL).

Mirrors genimagepp_detectors/run_clip_lora.py's CLIPLoRADetector class and
preprocessing exactly (resize 224x224 bilinear + CLIP normalize), so the
checkpoint's state_dict keys line up.
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from peft import get_peft_model, LoraConfig, TaskType
from transformers import CLIPVisionModel

from config import CLIPLORA_CKPT

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_model = None


class CLIPLoRADetector(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", lora_rank=4):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        )
        self.clip = get_peft_model(self.clip, peft_config)

        for param in self.clip.parameters():
            param.requires_grad = False
        for name, param in self.clip.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        self.classifier = nn.Linear(self.clip.config.hidden_size, 1)

    def forward(self, pixel_values):
        try:
            outputs = self.clip(pixel_values)
        except Exception:
            try:
                outputs = self.clip.base_model(pixel_values=pixel_values)
            except Exception:
                clip_model = getattr(self.clip, "model", None) or getattr(
                    self.clip, "base_model", self.clip
                )
                outputs = clip_model.forward(pixel_values=pixel_values)

        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits


def _preprocess(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = torch.from_numpy(np.array(img, dtype="float32") / 255.0).permute(2, 0, 1)
    for c in range(3):
        arr[c] = (arr[c] - CLIP_MEAN[c]) / CLIP_STD[c]
    return arr


def load(device):
    global _model
    if _model is not None:
        return _model

    model = CLIPLoRADetector()
    checkpoint = torch.load(CLIPLORA_CKPT, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if any(".vision_model." in k for k in state_dict.keys()):
        state_dict = {k.replace(".vision_model.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _model = model
    return model


@torch.no_grad()
def predict(image: Image.Image, device) -> float:
    model = load(device)
    x = _preprocess(image).unsqueeze(0).to(device)
    logit = model(x)
    score = torch.sigmoid(logit).flatten().item()
    return score
