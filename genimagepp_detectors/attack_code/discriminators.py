import os
from pathlib import Path

import torch
import timm
from transformers import CLIPVisionModel
from torch import nn
import kornia.augmentation as K


def discriminator_preprocess(image_tensor):
    preprocess = K.AugmentationSequential(
        K.Resize((224, 224), align_corners=False, antialias=True),
        K.CenterCrop(224),
    )
    image_tensor = preprocess(image_tensor)
    
    # Normalize to the CLIP image space.
    normalize = K.AugmentationSequential(
        K.Normalize(
            mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
            std=torch.tensor([0.26862954, 0.26130258, 0.27577711]),
        ),
    )
    image_tensor = normalize(image_tensor)
    return image_tensor
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = Path(os.environ.get("GENIMAGEPP_WEIGHTS_DIR", PROJECT_ROOT / "weights"))
DEFAULT_CLIP_WEIGHT = Path(os.environ.get("GENIMAGEPP_CLIP_CKPT", DEFAULT_WEIGHTS_DIR / "clip_epoch_20.pth"))
DEFAULT_RESNET_WEIGHT = Path(os.environ.get("GENIMAGEPP_RESNET_CKPT", DEFAULT_WEIGHTS_DIR / "resnet_epoch_20.pth"))


def resolve_checkpoint_path(checkpoint_path: str | Path | None, default_path: Path, model_name: str) -> str:
    resolved = Path(checkpoint_path) if checkpoint_path is not None else default_path
    if not resolved.exists():
        raise FileNotFoundError(
            f"Checkpoint for `{model_name}` not found at {resolved}. "
            "Set `GENIMAGEPP_WEIGHTS_DIR`, `GENIMAGEPP_RESNET_CKPT`, or `GENIMAGEPP_CLIP_CKPT` "
            "to point to the correct checkpoint location."
        )
    return str(resolved)

class clip_detector(nn.Module):
    def __init__(self):
        super(clip_detector, self).__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.classifier = nn.Linear(self.clip.config.hidden_size, 1)

    def forward(self, inputs):
        outputs = self.clip(pixel_values=inputs)
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits

def load_discriminator(model_name, device, checkpoint_path: str | Path | None = None):
    if model_name == 'resnet50':
        resolved_path = resolve_checkpoint_path(checkpoint_path, DEFAULT_RESNET_WEIGHT, model_name)
        discriminator = timm.create_model('resnet50', num_classes=1, checkpoint_path=resolved_path)
    elif model_name == 'clip':
        resolved_path = resolve_checkpoint_path(checkpoint_path, DEFAULT_CLIP_WEIGHT, model_name)
        discriminator = clip_detector()
        discriminator.load_state_dict(torch.load(resolved_path, map_location=device))
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    discriminator.to(device)
    return discriminator
