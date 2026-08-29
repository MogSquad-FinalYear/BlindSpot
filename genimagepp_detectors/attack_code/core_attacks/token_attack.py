from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.optim as optim

from discriminators import discriminator_preprocess, load_discriminator

from .common import (
    DEFAULT_OUTPUT_DIR,
    build_pipeline,
    decode_latents,
    ensure_dir,
    postprocess_for_detector,
    save_decoded_image,
)
from .diffusion import forward_with_grad


@dataclass
class TokenAttackConfig:
    repo_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda:7"
    discriminator_name: str = "resnet50"
    prompt: str = "a photo of cat"
    random_token_length: int = 3
    max_initial_prompts: int = 100
    max_steps: int = 100
    num_inference_steps: int = 35
    guidance_scale: float = 7.5
    lr: float = 1e-7
    reg_lambda: float = 1e-3
    target_label: float = 0.0
    pipeline_dtype: torch.dtype = torch.float16
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    save_template: str = "adv_prompt_{success_idx}.png"


def text_tokens_init(token_length: int, tokenizer, device: str) -> dict[str, torch.Tensor]:
    random_token_ids = torch.randint(0, tokenizer.vocab_size, (1, token_length), device=device)
    attention_mask = torch.ones_like(random_token_ids)
    return {
        "input_ids": random_token_ids,
        "attention_mask": attention_mask,
    }


def initialize_trainable_prompt(pipe, prompt: str, token_length: int, device: str):
    tokenizer = pipe.tokenizer
    init_prompt = text_tokens_init(token_length, tokenizer, device)
    fixed_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    fixed_prompt = {name: tensor.to(device) for name, tensor in fixed_prompt.items()}

    with torch.no_grad():
        fixed_embeds = pipe.text_encoder(
            input_ids=fixed_prompt["input_ids"],
            attention_mask=fixed_prompt["attention_mask"],
        )[0].detach().float()
        random_embeds = pipe.text_encoder(
            input_ids=init_prompt["input_ids"],
            attention_mask=init_prompt["attention_mask"],
        )[0].detach().float()

    random_embeds = torch.nn.Parameter(random_embeds)
    original_random_embeds = random_embeds.detach().clone()
    return random_embeds, fixed_embeds, original_random_embeds


def run_token_attack(config: TokenAttackConfig) -> dict[str, int]:
    ensure_dir(config.output_dir)
    pipe = build_pipeline(config.repo_id, config.device, config.pipeline_dtype)
    discriminator = load_discriminator(config.discriminator_name, config.device)
    discriminator.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    success_count = 0

    for prompt_idx in range(1, config.max_initial_prompts + 1):
        print(f"[token] start prompt_init={prompt_idx}/{config.max_initial_prompts}")
        random_embeds, fixed_embeds, original_random_embeds = initialize_trainable_prompt(
            pipe,
            config.prompt,
            config.random_token_length,
            config.device,
        )
        optimizer = optim.Adam([random_embeds], lr=config.lr)
        attack_success = False

        for step in range(1, config.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)

            combined_embeds = torch.cat([random_embeds, fixed_embeds], dim=1).to(pipe.text_encoder.dtype)
            negative_prompt_embeds = torch.zeros_like(combined_embeds)
            latent_output = forward_with_grad(
                pipe,
                prompt_embeds=combined_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type="latent",
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
            ).images

            decoded_images = decode_latents(pipe, latent_output)
            detector_input = postprocess_for_detector(decoded_images)
            score = discriminator(discriminator_preprocess(detector_input))
            attack_loss = criterion(score, torch.full_like(score, config.target_label))
            reg_loss = config.reg_lambda * torch.norm(random_embeds - original_random_embeds, p=2)
            total_loss = attack_loss + reg_loss

            if torch.isnan(total_loss).any():
                print(f"[token] nan loss at prompt_init={prompt_idx} step={step}, skip")
                continue

            total_loss.backward()
            if random_embeds.grad is None:
                raise RuntimeError("Token gradients are missing. Please use `forward_with_grad`, not `pipe(...)`.")
            if torch.isnan(random_embeds.grad).any():
                print(f"[token] nan gradient at prompt_init={prompt_idx} step={step}, skip")
                optimizer.zero_grad(set_to_none=True)
                continue

            grad_mean = random_embeds.grad.detach().mean().item()
            torch.nn.utils.clip_grad_norm_([random_embeds], max_norm=0.1)
            optimizer.step()

            disc_prob = torch.sigmoid(score.detach()).mean().item()
            print(
                f"[token] prompt_init={prompt_idx} step={step}/{config.max_steps} "
                f"attack_loss={attack_loss.item():.6f} reg_loss={reg_loss.item():.6f} "
                f"total_loss={total_loss.item():.6f} grad_mean={grad_mean:.6e} disc_prob={disc_prob:.6f}"
            )

            if disc_prob < 0.5:
                success_count += 1
                save_path = save_decoded_image(
                    pipe,
                    decoded_images,
                    ensure_dir(config.output_dir)
                    / config.save_template.format(success_idx=success_count, prompt_idx=prompt_idx, step=step),
                )
                print(f"[token] success prompt_init={prompt_idx}, saved to {save_path}")
                attack_success = True
                break

        if not attack_success:
            print(f"[token] prompt_init={prompt_idx} did not succeed within {config.max_steps} steps")

    result = {"attack_successes": success_count}
    print(result)
    return result
