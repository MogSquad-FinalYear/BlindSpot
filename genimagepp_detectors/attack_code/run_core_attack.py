from __future__ import annotations

import argparse

from core_attacks.common import DEFAULT_OUTPUT_DIR, get_torch_dtype
from core_attacks.latent_attack import LatentAttackConfig, run_latent_attack
from core_attacks.token_attack import TokenAttackConfig, run_token_attack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the cleaned adversarial attack core.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    latent = subparsers.add_parser("latent", help="Run latent adversarial optimization.")
    latent.add_argument("--repo-id", default="CompVis/stable-diffusion-v1-4")
    latent.add_argument("--device", default="cuda:1")
    latent.add_argument("--discriminator", default="resnet50", choices=["resnet50", "clip"])
    latent.add_argument("--prompt", default="a photo of cat")
    latent.add_argument("--max-seeds", type=int, default=100)
    latent.add_argument("--max-steps", type=int, default=100)
    latent.add_argument("--num-inference-steps", type=int, default=35)
    latent.add_argument("--guidance-scale", type=float, default=7.5)
    latent.add_argument("--lr", type=float, default=1e-3)
    latent.add_argument("--height", type=int, default=512)
    latent.add_argument("--width", type=int, default=512)
    latent.add_argument("--dtype", default="float16")
    latent.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    latent.add_argument("--save-template", default="adv_latents_seed_{seed}_at{step}.png")
    latent.add_argument("--seed-start", type=int, default=0)

    token = subparsers.add_parser("token", help="Run token embedding adversarial optimization.")
    token.add_argument("--repo-id", default="runwayml/stable-diffusion-v1-5")
    token.add_argument("--device", default="cuda:7")
    token.add_argument("--discriminator", default="resnet50", choices=["resnet50", "clip"])
    token.add_argument("--prompt", default="a photo of cat")
    token.add_argument("--random-token-length", type=int, default=3)
    token.add_argument("--max-initial-prompts", type=int, default=100)
    token.add_argument("--max-steps", type=int, default=100)
    token.add_argument("--num-inference-steps", type=int, default=35)
    token.add_argument("--guidance-scale", type=float, default=7.5)
    token.add_argument("--lr", type=float, default=1e-7)
    token.add_argument("--reg-lambda", type=float, default=1e-3)
    token.add_argument("--dtype", default="float16")
    token.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    token.add_argument("--save-template", default="adv_prompt_{success_idx}.png")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline_dtype = get_torch_dtype(args.dtype)

    if args.mode == "latent":
        run_latent_attack(
            LatentAttackConfig(
                repo_id=args.repo_id,
                device=args.device,
                discriminator_name=args.discriminator,
                prompt=args.prompt,
                max_seeds=args.max_seeds,
                max_steps=args.max_steps,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                lr=args.lr,
                height=args.height,
                width=args.width,
                pipeline_dtype=pipeline_dtype,
                output_dir=args.output_dir,
                save_template=args.save_template,
                seed_start=args.seed_start,
            )
        )
        return

    run_token_attack(
        TokenAttackConfig(
            repo_id=args.repo_id,
            device=args.device,
            discriminator_name=args.discriminator,
            prompt=args.prompt,
            random_token_length=args.random_token_length,
            max_initial_prompts=args.max_initial_prompts,
            max_steps=args.max_steps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            lr=args.lr,
            reg_lambda=args.reg_lambda,
            pipeline_dtype=pipeline_dtype,
            output_dir=args.output_dir,
            save_template=args.save_template,
        )
    )


if __name__ == "__main__":
    main()
