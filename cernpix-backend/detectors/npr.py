"""
NPR (Neighboring Pixel Relationships) detector wrapper.

Architecture and preprocessing mirror
NPR-DeepfakeDetection/run_inference.py exactly, so the checkpoint's
state_dict keys line up and results match that project's own evaluation.
"""
import math
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image

from config import NPR_DIR, NPR_CKPT

sys.path.insert(0, str(NPR_DIR))
from networks.resnet import resnet50  # noqa: E402

CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_model = None


def _translate_duplicate(img, crop_size=CROP_SIZE):
    width, height = img.size
    if min(width, height) < crop_size:
        new_width = width * math.ceil(crop_size / width)
        new_height = height * math.ceil(crop_size / height)
        new_img = Image.new("RGB", (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    return img


_transform = transforms.Compose([
    transforms.Lambda(_translate_duplicate),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load(device):
    global _model
    if _model is not None:
        return _model

    model = resnet50(num_classes=1)
    raw = torch.load(NPR_CKPT, map_location="cpu")
    state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    _model = model
    return model


@torch.no_grad()
def predict(image: Image.Image, device) -> float:
    model = load(device)
    x = _transform(image.convert("RGB")).unsqueeze(0).to(device)
    logit = model(x)
    score = torch.sigmoid(logit).flatten().item()
    return score
