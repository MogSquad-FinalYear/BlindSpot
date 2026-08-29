from core_attacks.token_attack import TokenAttackConfig, run_token_attack


if __name__ == "__main__":
    run_token_attack(
        TokenAttackConfig(
            repo_id="runwayml/stable-diffusion-v1-5",
            device="cuda:7",
            discriminator_name="resnet50",
            prompt="a photo of cat",
            random_token_length=3,
            max_initial_prompts=100,
            max_steps=100,
            num_inference_steps=35,
            lr=1e-7,
            reg_lambda=1e-3,
            save_template="adv_prompt_{success_idx}.png",
        )
    )
