"""Anomagic single-image inference.

Contains `generate_anomagic_single` — the core generation function used by
both the training script (for periodic samples) and standalone generation
scripts (e.g. generate_cashew.py).
"""

import torch
import torch.nn.functional as F

from src.utils.mask_utils import create_latent_band_mask


def _dilate_clip_mask_16(mask: torch.Tensor) -> torch.Tensor:
    """Dilate CLIP mask by +1px at 16x16 grid, then upscale back.

    Downsamples mask to 16x16 via maxpool (matching MaskedAnomalySelfAttention),
    applies +1 cell dilation, then upscales to original resolution via nearest.
    This gives the self-attention 1 extra ring of context tokens.

    Args:
        mask: [B, 1, H, W] binary mask (e.g. [1, 1, 224, 224])

    Returns:
        Dilated mask at original resolution [B, 1, H, W]
    """
    h, w = mask.shape[-2:]
    # Downscale to 16x16 via maxpool (same as model internals)
    kernel = h // 16
    mask_16 = F.max_pool2d(mask.float(), kernel_size=kernel)
    mask_16 = (mask_16 > 0.5).float()
    # +1px dilation at 16x16
    mask_16_dilated = F.max_pool2d(mask_16, kernel_size=3, stride=1, padding=1)
    # Upscale back to original resolution
    return F.interpolate(mask_16_dilated, size=(h, w), mode="nearest")


def generate_anomagic_single(
    pipeline, ip_adapter, normal_image, mask, reference,
    anomaly_type, caption,
    num_steps=50, guidance_scale=7.5,
    noise_strength: float = 0.7,
    reference_mode: str = "full",
    band_mode: int = 1,
    t2i_adapter=None,
    seed: int = None,
    clip_mask: torch.Tensor = None,
    clip_core_mask: torch.Tensor = None,
    reference_2: torch.Tensor = None,
    clip_mask_2: torch.Tensor = None,
    clip_core_mask_2: torch.Tensor = None,
    group_valid: torch.Tensor = None,
    dilate_clip_mask: bool = False,
    cfg_mode: str = "text",
):
    """Generate anomaly using IP-Adapter + text captions.

    Starts from the normal image noised to ``noise_strength`` (default 70%)
    and denoises from that point. This preserves structure outside the mask
    and gives the model a realistic starting point.

    Args:
        noise_strength: Fraction of the diffusion schedule to noise through.
            0.0 = no noise (returns normal image), 1.0 = pure noise.
        clip_mask: Optional [B, 1, H, W] dilated mask for CLIP self-attention.
            When provided, overrides the default mask logic (useful when
            reference is a crop with its own mask separate from the UNet mask).
        clip_core_mask: Optional [B, 1, H, W] core mask for role embeddings.
        reference_2: Optional second CLIP crop [1, 3, 224, 224] in [-1, 1]
            for multi-crop mode (matches training with --multi-crop).
        clip_mask_2: Optional mask for second crop [1, 1, 224, 224].
        clip_core_mask_2: Optional core mask for second crop [1, 1, 224, 224].
        group_valid: Optional [1, 2] validity flags per crop group.
        dilate_clip_mask: If True, expand CLIP mask by +1px at 16x16 grid
            before self-attention. Use for hard/deformation anomalies.
        cfg_mode: CFG direction. "text" (default) = amplify text on visual
            baseline. "visual" = amplify visual on text baseline.
    """
    device = normal_image.device

    with torch.amp.autocast('cuda'):
        normal_image = normal_image.float()
        mask = mask.float()

        # Pathway 1: Text
        if caption:
            prompt = caption
        else:
            type_word = anomaly_type.replace("_", " ")
            prompt = f"a photo of a {type_word} defect"

        text_emb = pipeline.encode_text([prompt], enable_grad=False)

        # Pathway 2: Visual (IP-Adapter) with masked mean pooling
        ref_01 = (reference + 1.0) / 2.0
        if clip_mask is not None:
            # Explicit CLIP mask provided (e.g. cropped mask from clip_crop)
            _clip_mask = clip_mask
            _clip_core = clip_core_mask
        elif reference_mode == "full":
            # Full reference: reuse the UNet inpainting mask for CLIP
            _clip_mask = mask
            _clip_core = None
        else:
            _clip_mask = None
            _clip_core = None
        # +1px dilation at 16x16 grid level (for hard/deformation mode)
        if dilate_clip_mask and _clip_mask is not None:
            _clip_mask = _dilate_clip_mask_16(_clip_mask)

        ip_image_embeds = ip_adapter.encode_image(ref_01, mask=_clip_mask, core_mask=_clip_core)  # [1, K, dim]

        # --- Multi-crop: encode second crop and concatenate (matches training) ---
        null_token_mask = None
        if reference_2 is not None:
            ref_2_01 = (reference_2 + 1.0) / 2.0
            _clip_mask_2 = clip_mask_2 if clip_mask_2 is not None else None
            _clip_core_2 = clip_core_mask_2 if clip_core_mask_2 is not None else None
            if dilate_clip_mask and _clip_mask_2 is not None:
                _clip_mask_2 = _dilate_clip_mask_16(_clip_mask_2)
            ip_image_embeds_2 = ip_adapter.encode_image(ref_2_01, mask=_clip_mask_2, core_mask=_clip_core_2)  # [1, K, dim]

            # Concatenate: [1, K, dim] + [1, K, dim] -> [1, 2K, dim]
            K = ip_image_embeds.shape[1]
            ip_image_embeds = torch.cat([ip_image_embeds, ip_image_embeds_2], dim=1)  # [1, 2K, dim]

            # Build null_token_mask [1, 2K]: 1=valid, 0=null (from invalid groups)
            if group_valid is not None:
                mask_1 = group_valid[:, 0:1].expand(-1, K)  # [1, K]
                mask_2 = group_valid[:, 1:2].expand(-1, K)  # [1, K]
                null_token_mask = torch.cat([mask_1, mask_2], dim=1)  # [1, 2K]

        # CFG: choose which pathway to amplify
        if cfg_mode == "visual":
            # Amplify visual: uncond = (real_text, zero_IP)
            # pred = text_only + scale * (text+visual - text_only)
            uncond_text = text_emb
            uncond_ip = torch.zeros_like(ip_image_embeds)
        elif cfg_mode == "both":
            # Amplify both: uncond = (empty_text, zero_IP)
            # pred = uncond + scale * (text+visual - uncond)
            uncond_text = pipeline.encode_text([""], enable_grad=False)
            uncond_ip = torch.zeros_like(ip_image_embeds)
        else:
            # Amplify text (default): uncond = (empty_text, real_IP)
            # pred = visual_only + scale * (text+visual - visual_only)
            uncond_text = pipeline.encode_text([""], enable_grad=False)
            uncond_ip = ip_image_embeds

        # Inpainting setup — latent-space band dilation
        normal_latents = pipeline.encode_image(normal_image)
        kernel = mask.shape[-1] // normal_latents.shape[-1]
        core_mask_64 = F.max_pool2d(mask, kernel_size=kernel)
        core_mask_64 = (core_mask_64 > 0.5).float()

        dilated_binary_64, alpha_map_64, _, band_mask_64 = create_latent_band_mask(
            core_mask_64, band_mode,
        )

        mask_latents = dilated_binary_64
        unet_mask_512 = F.interpolate(dilated_binary_64, size=mask.shape[-2:], mode='nearest')
        masked_image = normal_image * (1 - unet_mask_512)
        masked_image_latents = pipeline.encode_image(masked_image)

        # T2I-Adapter features (constant across timesteps, computed ONCE outside loop)
        t2i_kwargs = {}
        if t2i_adapter is not None:
            t2i_input = torch.cat([core_mask_64, band_mask_64], dim=1)  # [1, 2, 64, 64]
            t2i_features = t2i_adapter(t2i_input, mask=dilated_binary_64)
            t2i_kwargs_single = t2i_adapter.prepare_unet_kwargs(t2i_features)
            # CFG: both branches get real T2I features (visual always present)
            t2i_kwargs = {
                k: [torch.cat([f, f]) for f in vs]
                for k, vs in t2i_kwargs_single.items()
            }

        # Start from normal image noised to noise_strength (not pure noise)
        pipeline.scheduler.set_timesteps(num_steps, device=device)
        all_timesteps = pipeline.scheduler.timesteps
        # Find the starting timestep: noise_strength=0.7 → start at 70% through schedule
        start_step = max(0, int(len(all_timesteps) * (1 - noise_strength)))
        start_timestep = all_timesteps[start_step]

        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(normal_latents.shape, generator=gen, device=device, dtype=normal_latents.dtype)
        else:
            noise = torch.randn_like(normal_latents)
        latents = pipeline.scheduler.add_noise(normal_latents, noise, start_timestep.unsqueeze(0))

        for i, t in enumerate(all_timesteps[start_step:]):
            latent_input = torch.cat([latents] * 2)
            mask_input = torch.cat([mask_latents] * 2)
            masked_input = torch.cat([masked_image_latents] * 2)

            model_input = torch.cat([latent_input, mask_input, masked_input], dim=1)

            combined_text = torch.cat([uncond_text, text_emb])
            combined_ip = torch.cat([uncond_ip, ip_image_embeds])

            cross_attn_kwargs = {"ip_adapter_image_embeds": combined_ip}
            # Soft alpha mask for cross-attention routing (duplicated for CFG)
            cross_attn_kwargs["ip_adapter_mask"] = torch.cat([alpha_map_64, alpha_map_64])
            # Multi-crop null masking (duplicated for CFG)
            if null_token_mask is not None:
                cross_attn_kwargs["null_token_mask"] = torch.cat([null_token_mask, null_token_mask])

            noise_pred = pipeline.unet(
                model_input, t,
                encoder_hidden_states=combined_text,
                cross_attention_kwargs=cross_attn_kwargs,
                **t2i_kwargs,
            ).sample.float()

            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            # Alpha-blended diffusion: blend generated x_{t-1} with correctly-noised normal
            # Normal side: noise from clean x_0 at the next timestep (exact forward process)
            # Generated side: already at x_{t-1} from scheduler.step (correct noise level)
            remaining = len(all_timesteps) - (start_step + i + 1)
            if remaining > 0:
                next_t = all_timesteps[start_step + i + 1]
                noisy_normal = pipeline.scheduler.add_noise(normal_latents, noise, next_t.unsqueeze(0))
                latents = latents * alpha_map_64 + noisy_normal * (1 - alpha_map_64)

    output = pipeline.decode_latents(latents.float())
    # Alpha-blend at pixel level
    alpha_512 = F.interpolate(alpha_map_64, size=output.shape[-2:], mode='nearest')
    output = output * alpha_512 + normal_image.float() * (1 - alpha_512)

    return output
