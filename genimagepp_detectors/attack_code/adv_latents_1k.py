from core_attacks.latent_attack import LatentAttackConfig, run_latent_attack


if __name__ == "__main__":
    run_latent_attack(
        LatentAttackConfig(
            repo_id="CompVis/stable-diffusion-v1-4",
            device="cuda:1",
            discriminator_name="resnet50",
            prompt="a photo of cat",
            max_seeds=100,
            max_steps=100,
            num_inference_steps=35,
            lr=1e-3,
            save_template="adv_latents_seed_{seed}_at{step}.png",
        )
    )
