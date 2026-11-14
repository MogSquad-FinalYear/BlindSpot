"""
Batched loaders for all five detectors, for bulk scoring of the manifest.

Each entry mirrors its origin project's model construction and preprocessing
exactly, because the training table mixes these scores with ones harvested
from the original evaluation CSVs -- if preprocessing drifted, the harvested
and freshly-computed rows would be on different scales and the orchestrator
would learn the seam instead of the signal. The npr/ufd/cliplora wrappers are
reused from cernpix-backend/detectors/ (the same code the Flask app serves
from), and were verified against the stored CSVs at max |delta| 4.9e-05.

Preprocessing, per detector:
  npr           tile-if-small -> CenterCrop(224) -> ImageNet mean/std
  ufd           CenterCrop(224) -> CLIP mean/std            (zero-pads if small)
  cliplora      resize 224 BILINEAR -> CLIP mean/std
  clipbaseline  resize 224 BICUBIC  -> CLIP mean/std
  resnet        resize 224 BICUBIC  -> ImageNet mean/std    (see note below)

resnet normalization: run_resnet_baseline.py exposes --normalize {clip,
imagenet} because the released GenImage++ attack code and the paper's
Appendix A.2 disagree for this checkpoint, and it *defaults to clip*. The
stored predictions_resnet_*_full.csv were nevertheless produced with
--normalize imagenet: matching a 128-image sample against the CSV gives
mean |delta| 4.6e-06 under ImageNet stats versus 7.2e-04 (max 5.4e-02) under
CLIP stats. ImageNet is therefore used here -- taking the script default on
faith would have put the freshly-scored reals on a different scale from the
harvested fakes and handed the orchestrator that seam as a feature.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

BACKEND = Path(__file__).resolve().parent.parent      # cernpix-backend/
DATA_ROOT = BACKEND.parent                            # /home/student/data
sys.path.insert(0, str(BACKEND))

from config import GENIMAGEPP_DIR                      # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CLIPBASE_CKPT = GENIMAGEPP_DIR / "attack_code" / "weights" / "clip_epoch_20.pth"
RESNET_CKPT = GENIMAGEPP_DIR / "attack_code" / "weights" / "resnet_epoch_20.pth"


# --- the three the router already uses -------------------------------------
def _load_npr(device):
    from detectors import npr
    return npr.load(device), npr._transform


def _load_ufd(device):
    from detectors import universalfakedetect as ufd
    return ufd.load(device), ufd._transform


def _load_cliplora(device):
    from detectors import genimagepp_cliplora as lora
    return lora.load(device), lora._preprocess


# --- the two evaluated in the paper but not wired into the app -------------
class ClipBaseline(nn.Module):
    """Mirrors genimagepp_detectors/run_clip_baseline.py::clip_detector."""

    def __init__(self):
        super().__init__()
        from transformers import CLIPVisionModel
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.classifier = nn.Linear(self.clip.config.hidden_size, 1)

    def forward(self, x):
        return self.classifier(self.clip(pixel_values=x).last_hidden_state[:, 0])


def _resize_norm(resample, mean=CLIP_MEAN, std=CLIP_STD):
    import numpy as np
    from PIL import Image

    def tf(img):
        img = img.convert("RGB").resize((224, 224), resample)
        arr = torch.from_numpy(np.array(img, dtype="float32") / 255.0).permute(2, 0, 1)
        for c in range(3):
            arr[c] = (arr[c] - mean[c]) / std[c]
        return arr
    return tf


def _load_clipbaseline(device):
    from PIL import Image
    model = ClipBaseline()
    state_dict = torch.load(CLIPBASE_CKPT, map_location="cpu")
    # Same key normalization run_clip_baseline.py applies: the checkpoint was
    # saved from a CLIPVisionModel whose submodule was still nested under
    # .vision_model., which this transformers version flattens away.
    if any(".vision_model." in k for k in state_dict):
        state_dict = {k.replace(".vision_model.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, _resize_norm(Image.BICUBIC)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_resnet(device):
    import timm
    from PIL import Image
    model = timm.create_model("resnet50", num_classes=1,
                              checkpoint_path=str(RESNET_CKPT))
    model.to(device).eval()
    return model, _resize_norm(Image.BICUBIC, IMAGENET_MEAN, IMAGENET_STD)


LOADERS = {
    "npr": _load_npr,
    "ufd": _load_ufd,
    "cliplora": _load_cliplora,
    "clipbaseline": _load_clipbaseline,
    "resnet": _load_resnet,
}


def load(name, device):
    """Return (model, transform) for `name`, both ready for batched inference."""
    if name not in LOADERS:
        raise KeyError(f"unknown detector {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name](device)


def unload():
    """Free every cached detector from GPU memory.

    The three wrappers reused from cernpix-backend/detectors/ each cache their
    model in a module-level `_model` global, which is right for a long-lived
    Flask worker serving one image at a time but wrong here: scoring five
    detectors in one process would other wise pile ~1.2 GB per CLIP tower onto
    an 8 GB card and OOM on the fourth. run_inference.py also runs one
    detector per process, so this is belt-and-braces.
    """
    import gc
    for mod_name in ("detectors.npr", "detectors.universalfakedetect",
                     "detectors.genimagepp_cliplora"):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "_model", None) is not None:
            mod._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
