from __future__ import annotations

from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.deprecation_utils import deprecate


def rescale_noise_cfg(
    noise_cfg: torch.Tensor,
    noise_pred_text: torch.Tensor,
    guidance_rescale: float = 0.0,
) -> torch.Tensor:
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def forward_with_grad(
    self,
    prompt: Union[str, List[str], None] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
    **kwargs,
):
    callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs or ["latents"]
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `forward_with_grad` is deprecated. "
            "Use `callback_on_step_end` instead.",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `forward_with_grad` is deprecated. "
            "Use `callback_on_step_end` instead.",
        )

    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    if self.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        self.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=self.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    self._num_timesteps = len(timesteps)
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            return_dict=False,
        )[0]

        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        if callback_on_step_end is not None:
            callback_kwargs = {}
            for name in callback_on_step_end_tensor_inputs:
                callback_kwargs[name] = locals()[name]
            callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
            latents = callback_outputs.pop("latents", latents)
            prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
            negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

    if output_type != "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def attach_forward_with_grad(pipe, attribute_name: str = "forward_with_grad"):
    setattr(pipe, attribute_name, MethodType(forward_with_grad, pipe))
    return pipe


forward_func = forward_with_grad
