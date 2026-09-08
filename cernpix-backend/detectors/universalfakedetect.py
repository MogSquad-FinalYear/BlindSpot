"""
UniversalFakeDetect (CLIP ViT-L/14 linear probe) detector wrapper.

Mirrors UniversalFakeDetect/validate.py's model construction and
UniversalFakeDetect/run_and_dump.py's preprocessing exactly: CenterCrop(224)
+ CLIP mean/std, checkpoint loaded into model.fc only (the CLIP backbone
itself is the pretrained, frozen openai CLIP ViT-L/14).
"""
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image

from config import UFD_DIR, UFD_CKPT

sys.path.insert(0, str(UFD_DIR))
from models import get_model  # noqa: E402

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_model = None

_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def load(device):
    global _model
    if _model is not None:
        return _model

    model = get_model("CLIP:ViT-L/14")
    state_dict = torch.load(UFD_CKPT, map_location="cpu")
    model.fc.load_state_dict(state_dict)
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
