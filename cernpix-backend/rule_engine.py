"""
Rule-based detector selection.

Revision note (2026-08-29): the original version gated the DiT-era branch on
native resolution (min_side >= 768px), on the theory that SDXL/SD3/Flux
default to 1024x1024 output. That gate turned out to be a bad proxy in
practice: any resize, screenshot, or re-save (i.e. most real-world uploads)
destroys native resolution while leaving the frequency-domain content mostly
intact, so a downscaled modern-generator image would fail the resolution
check and fall through to UniversalFakeDetect - the weakest generalizer of
the three (its checkpoint was trained only on ProGAN; see paper.txt in the
BlindSpot project). Confirmed failure case: a clearly AI-generated portrait
saved as a 348x456 screenshot scored 0.0014 (confidently "real") from
UniversalFakeDetect, while its hf_energy (0.55) showed no GAN-suppression at
all - CLIP-LoRA would have caught it.

Resolution is dropped from the routing decision entirely. The only feature
that survives resizing well enough to trust is high-frequency spectral
energy - mean log-magnitude FFT energy in the outer 20% frequency radius of
a 256x256 grayscale version of the image (same method as
paper/scripts/highpass_energy.py). Measured values from that evaluation
(paper/npr/highpass_energy/highpass_energy.csv):

    real photos        ~0.7  to  0.9
    flux / sd3 / sdxl   0.3  to  0.8   (close to real - DiT generators don't
                                        suppress high frequencies much)
    BigGAN / GLIDE     -1.2  to -1.6   (strongly suppressed - GAN upsampling
                                        stages smear out high-frequency detail)

A value below NPR_HF_ENERGY_MAX is strong, resolution-independent evidence
of GAN-style upsampling artifacts - genuine BigGAN/GLIDE output already
lands well below -1.0 even at native resolution, so this threshold alone
reliably separates real GAN output without needing a resolution check.

Everything else defaults to genimagepp CLIP-LoRA, the strongest
generalizing detector of the three per the BlindSpot evaluation
(adversarially hardened, trained specifically on modern DiT generators) -
rather than defaulting to UniversalFakeDetect, the weakest one.

UniversalFakeDetect keeps one narrow, honest niche: images at native small
resolution typical of raw GAN thumbnails (BigGAN/ProGAN output is 128-256px)
with only moderate spectral suppression - low enough to not look like a real
photo, not low enough for confident NPR routing. This case is rare among
real uploads (most are >=300px), which is an honest reflection of
UniversalFakeDetect being the least broadly useful of the three, not an
artifact of the thresholds.
"""
import numpy as np
from PIL import Image

from config import NPR_HF_ENERGY_MAX, UFD_MAX_MIN_SIDE, UFD_HF_ENERGY_MAX

FFT_SIZE = 256
TOP_FRAC = 0.2


def _high_freq_mask(size=FFT_SIZE, top_frac=TOP_FRAC):
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(2) * size / 2
    threshold = (1 - top_frac) * max_dist
    return dist >= threshold


_HF_MASK = _high_freq_mask()

def compute_features(image: Image.Image) -> dict:
    width, height = image.size
    min_side = min(width, height)

    gray = np.array(
        image.convert("L").resize((FFT_SIZE, FFT_SIZE)), dtype=np.float32
    ) / 255.0
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(f) + 1e-8)
    hf_energy = float(mag[_HF_MASK].mean())
 
    return {
        "width": width,
        "height": height,
        "min_side": min_side,
        "hf_energy": round(hf_energy, 4),
    }


def select_detector(features: dict) -> dict:
    min_side = features["min_side"]
    hf_energy = features["hf_energy"]
    width, height = features["width"], features["height"]

    if hf_energy < NPR_HF_ENERGY_MAX:
        reason = (
            f"Image is {width}x{height} with high-frequency spectral energy "
            f"{hf_energy:.2f} (strongly suppressed relative to real photos, "
            f"consistent with GAN-era upsampling artifacts - this signal survives "
            f"resizing, unlike resolution). Routed to NPR, which is built to "
            f"exploit exactly this neighboring-pixel-relationship signal."
        )
        return {"detector": "npr", "reason": reason}

    if min_side < UFD_MAX_MIN_SIDE and hf_energy < UFD_HF_ENERGY_MAX:
        reason = (
            f"Image is {width}x{height} ({min_side}px min side, native "
            f"small-resolution range typical of raw GAN thumbnails) with "
            f"moderately suppressed high-frequency energy {hf_energy:.2f} (not "
            f"suppressed enough for confident NPR routing, not real-like "
            f"enough to skip past). Routed to UniversalFakeDetect."
        )
        return {"detector": "universalfakedetect", "reason": reason}

    reason = (
        f"Image is {width}x{height} with high-frequency spectral energy "
        f"{hf_energy:.2f} (no GAN-suppression signal detected). Resolution "
        f"alone isn't used to route away from this default, since resizing or "
        f"screenshotting - true of most real-world uploads - destroys native "
        f"resolution without removing generator content. Routed to CLIP-LoRA "
        f"(genimagepp), the strongest general-purpose detector of the three "
        f"per the BlindSpot evaluation."
    )
    return {"detector": "genimagepp_cliplora", "reason": reason}
