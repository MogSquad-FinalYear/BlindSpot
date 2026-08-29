from __future__ import annotations

from dataclasses import dataclass

import torch

from discriminators import discriminator_preprocess, load_discriminator

from .common import (
    DEFAULT_OUTPUT_DIR,
    build_pipeline,
    decode_latents,
    default_latent_shape,
    ensure_dir,
    postprocess_for_detector,
    save_decoded_image,
)
from .diffusion import forward_with_grad


@dataclass
class LatentAttackConfig:
    repo_id: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda:1"
    discriminator_name: str = "resnet50"
    prompt: str = "a photo of cat"
    max_seeds: int = 100
    max_steps: int = 100
    num_inference_steps: int = 35
    guidance_scale: float = 7.5
    lr: float = 1e-3
    target_label: float = 0.0
    height: int = 512
    width: int = 512
    pipeline_dtype: torch.dtype = torch.float16
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    save_template: str = "adv_latents_seed_{seed}_at{step}.png"
    seed_start: int = 0


def _seed_everything(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def run_latent_attack(config: LatentAttackConfig) -> dict[str, list[int]]:
    ensure_dir(config.output_dir)
    pipe = build_pipeline(config.repo_id, config.device, config.pipeline_dtype)
    discriminator = load_discriminator(config.discriminator_name, config.device)
    discriminator.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    results = {
        "seed_num": [],
        "optimization_steps": [],
    }

    latent_shape = default_latent_shape(pipe, height=config.height, width=config.width)

    for seed in range(config.seed_start, config.seed_start + config.max_seeds):
        _seed_everything(seed, config.device)
        init_latents = torch.nn.Parameter(
            torch.randn(latent_shape, device=config.device, dtype=torch.float32).detach()
        )
        optimizer = torch.optim.SGD([init_latents], lr=config.lr)
        attack_success = False

        for step in range(1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            latent_output = forward_with_grad(
                pipe,
                prompt=config.prompt,
                latents=init_latents.to(pipe.unet.dtype),
                output_type="latent",
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                height=config.height,
                width=config.width,
            ).images

            decoded_images = decode_latents(pipe, latent_output)
            detector_input = postprocess_for_detector(decoded_images)
            score = discriminator(discriminator_preprocess(detector_input))
            target = torch.full_like(score, config.target_label)
            loss = criterion(score, target)

            loss.backward()
            if init_latents.grad is None:
                raise RuntimeError("Latent gradients are missing. Please use `forward_with_grad`, not `pipe(...)`.")

            grad_mean = init_latents.grad.detach().mean().item()
            optimizer.step()

            disc_prob = torch.sigmoid(score.detach()).mean().item()
            print(
                f"[latent] seed={seed} step={step}/{config.max_steps} "
                f"loss={loss.item():.6f} grad_mean={grad_mean:.6e} disc_prob={disc_prob:.6f}"
            )

            if disc_prob < 0.5:
                save_path = save_decoded_image(
                    pipe,
                    decoded_images,
                    ensure_dir(config.output_dir) / config.save_template.format(seed=seed, step=step),
                )
                results["seed_num"].append(seed)
                results["optimization_steps"].append(step)
                print(f"[latent] success seed={seed}, saved to {save_path}")
                attack_success = True
                break

        if not attack_success:
            print(f"[latent] seed={seed} did not succeed within {config.max_steps} steps")

    print(results)
    return results
