"""
Canonical dataset inventory for the orchestrator training set.

Source of truth for *which* images the orchestrator is trained and evaluated
on. Kept separate from the manifest builder so the pools can be audited and
extended without touching sampling/split logic.

Provenance of the pairings below: reconstructed from the per-image prediction
CSVs the BlindSpot evaluation already produced (NPR-DeepfakeDetection/,
UniversalFakeDetect/, genimagepp_detectors/), so the fake pools line up
exactly with images that already carry all five detector scores. See
harvest_existing.py.

Note on real pools: the original per-generator runs each drew their own
random sample of ImageNet `train/nature` for the flux/sd3/sdxl cells, so
those real draws do NOT intersect across detectors (only ~200 images do).
Only the two `val/nature` pools carry all five detector scores already;
everything under NEW_REAL_POOLS has to be scored from scratch.
"""
from pathlib import Path

DOWNLOADS = Path("/home/student/Downloads")
DATA = Path("/home/student/data")

# (generator, directory, cap) -- cap None means "take the whole pool".
# Every one of these images already has all 5 detector scores stored.
FAKE_POOLS = [
    ("BigGan", DOWNLOADS / "imagenet_ai_0419_biggan/val/ai", None),
    ("Glide",  DOWNLOADS / "imagenet_glide/val/ai", None),
    ("flux",   DATA / "genimagepp/flux/flux/val/1_fake", None),
    ("sd3",    DATA / "genimagepp/sd3/stable_diffusion_v_3_0/val/1_fake", None),
    ("sdxl",   DATA / "genimagepp/sdxl/sdxl_style/val/1_fake", None),
]

# Real images that already carry all 5 detector scores (no GPU needed).
SCORED_REAL_POOLS = [
    ("imagenet_val_biggan", DOWNLOADS / "imagenet_ai_0419_biggan/val/nature", 6000),
    ("imagenet_val_glide",  DOWNLOADS / "imagenet_glide/val/nature", 6000),
]

# Real images that need a fresh inference pass. The COCO slices are the point
# of this addition: production traffic is user uploads, not ImageNet, and a
# real-vs-fake meta-model trained only on ImageNet reals learns ImageNet's JPEG
# and resolution statistics as much as it learns "real".
NEW_REAL_POOLS = [
    ("imagenet_train", DOWNLOADS / "imagenet_ai_0419_biggan/train/nature", 13000),
    ("coco_val2017",   DATA / "real_coco/val2017", 5000),
]

# real_coco/humans/images is a 600-image SUBSET of val2017 (100% basename
# overlap), not an independent pool -- listing both would put byte-identical
# images in train and test under two different paths. It is carried as a slice
# tag on the val2017 rows instead, so "does the orchestrator hold up on photos
# of people" stays answerable without duplicating anything.
HUMAN_SLICE_DIR = DATA / "real_coco/humans/images"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Detectors the orchestrator may consume. `key` is the column name used
# everywhere downstream; `legacy` is how the detector is spelled in the
# pre-existing prediction CSVs.
DETECTORS = ["npr", "ufd", "cliplora", "clipbaseline", "resnet"]

# The three the deployed router currently chooses between.
ROUTER_DETECTORS = ["npr", "ufd", "cliplora"]


def list_images(directory, cap=None, seed=0):
    """Deterministically list images under `directory`, optionally capped.

    Sorted first so the enumeration is filesystem-order independent, then
    sampled with a fixed RNG when a cap applies, so re-running reproduces the
    same manifest.
    """
    import random
    directory = Path(directory)
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if cap is not None and len(files) > cap:
        rng = random.Random(seed)
        files = sorted(rng.sample(files, cap))
    return files
