import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent  # /home/student/data

NPR_DIR = DATA_ROOT / "NPR-DeepfakeDetection"
UFD_DIR = DATA_ROOT / "UniversalFakeDetect"
GENIMAGEPP_DIR = DATA_ROOT / "genimagepp_detectors"

NPR_CKPT = NPR_DIR / "NPR.pth"
UFD_CKPT = UFD_DIR / "pretrained_weights" / "fc_weights.pth"
CLIPLORA_CKPT = GENIMAGEPP_DIR / "attack_code" / "weights" / "best_model_low_rank.pt"

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MONGODB_URI = os.environ.get("MONGODB_URI", "")
MONGODB_DB = os.environ.get("MONGODB_DB", "cernpix")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "detections")

DEVICE = os.environ.get("DEVICE", "cuda")

# --- Rule-engine thresholds, grounded in paper/*/highpass_energy/*.csv from
# the BlindSpot project: real photos and DiT-era fakes (flux/sd3/sdxl) sit in
# the 0.3-0.9 high-frequency log-energy range, GAN-era fakes (BigGAN/GLIDE)
# sit far below that (-1.2 to -1.6) because their upsampling stages suppress
# high-frequency content. See rule_engine.py for how these are used. Native
# resolution is deliberately NOT used to gate routing (see rule_engine.py's
# module docstring for why).
NPR_HF_ENERGY_MAX = -1.0    # strongly GAN-suppressed spectrum -> NPR's target signal
UFD_MAX_MIN_SIDE = 300      # native small-resolution range (raw GAN thumbnails)
UFD_HF_ENERGY_MAX = -0.3    # moderate suppression, not extreme enough for NPR
