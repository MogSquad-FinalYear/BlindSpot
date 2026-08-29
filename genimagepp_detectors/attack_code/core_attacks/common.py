from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "adv_cat_imgs"


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def build_pipeline(
    repo_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def postprocess_for_detector(image: torch.Tensor) -> torch.Tensor:
    image = (image * 0.5 + 0.5).clamp(0, 1)
    image = image * 255
    image = image + image.round().detach() - image.detach()
    return image / 255


def decode_latents(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> torch.Tensor:
    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return decoded.to(torch.float32)


def save_decoded_image(
    pipe: StableDiffusionPipeline,
    decoded_images: torch.Tensor,
    save_path: str | Path,
) -> Path:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    images = pipe.image_processor.postprocess(
        decoded_images.detach(),
        output_type="pil",
        do_denormalize=[True] * decoded_images.shape[0],
    )
    images[0].save(save_path)
    return save_path


def default_latent_shape(
    pipe: StableDiffusionPipeline,
    batch_size: int = 1,
    height: int | None = None,
    width: int | None = None,
) -> tuple[int, int, int, int]:
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
    return (
        batch_size,
        pipe.unet.config.in_channels,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor,
    )
