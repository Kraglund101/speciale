"""
Stable Diffusion XL pipeline implementation.

SDXL uses dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG).
"""
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLInpaintPipeline

from src.models.base import BaseSDPipeline, SDConfig


class SDXLPipeline(BaseSDPipeline):
    """
    Stable Diffusion XL pipeline for inpainting with textual inversion support.

    Key differences from SD 1.5:
    - Dual text encoders (CLIP ViT-L/14 + OpenCLIP ViT-bigG)
    - 1024x1024 native resolution
    - Larger UNet with cross-attention to both text encoder outputs
    """

    def __init__(self, config: SDConfig, device: str = "cuda", **kwargs):
        super().__init__(config, device)
        self.concept_token_ids: Dict[str, List[int]] = {}
        self.concept_token_ids_2: Dict[str, List[int]] = {}  # For second encoder
        self.dtype = kwargs.get("dtype", torch.float16)

        # SDXL has two text encoders
        self.text_encoder_2 = None
        self.tokenizer_2 = None

    def load_pipeline(
        self,
        pretrained_model: Optional[str] = None,
        **kwargs
    ) -> None:
        """Load the SDXL inpainting pipeline."""
        model_id = pretrained_model or self.config.pretrained_model

        self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            **kwargs
        ).to(self.device)

        # Extract components
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler

    def encode_text(
        self,
        text: Union[str, List[str]],
        enable_grad: bool = False,
    ) -> torch.Tensor:
        """
        Encode text using both SDXL text encoders.

        Args:
            text: Text prompt(s) to encode
            enable_grad: If True, enable gradient computation (for training)

        Returns:
            Concatenated embeddings from both encoders.
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize for both encoders
        tokens_1 = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        tokens_2 = self.tokenizer_2(
            text,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode with both (with or without gradients)
        if enable_grad:
            embed_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            embed_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)

            # SDXL uses penultimate hidden states
            pooled_prompt_embeds = embed_2[0]
            prompt_embeds = torch.cat([
                embed_1.hidden_states[-2],
                embed_2.hidden_states[-2]
            ], dim=-1)
        else:
            with torch.no_grad():
                embed_1 = self.text_encoder(tokens_1, output_hidden_states=True)
                embed_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)

                # SDXL uses penultimate hidden states
                pooled_prompt_embeds = embed_2[0]
                prompt_embeds = torch.cat([
                    embed_1.hidden_states[-2],
                    embed_2.hidden_states[-2]
                ], dim=-1)

        return prompt_embeds

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space."""
        image = image.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to image space."""
        latents = latents / self.vae.config.scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return image

    def get_text_embedding_dim(self) -> int:
        """Get combined text embedding dimension (2048 for SDXL)."""
        return (
            self.text_encoder.config.hidden_size +
            self.text_encoder_2.config.hidden_size
        )

    def add_concept_token(
        self,
        token_name: str,
        initializer_token: Optional[str] = None,
        num_vectors: int = 1,
    ) -> torch.Tensor:
        """
        Add concept token to BOTH text encoders.

        For SDXL, we need to add tokens to both tokenizers and
        resize both embedding layers.
        """
        placeholder_tokens = [token_name]
        if num_vectors > 1:
            placeholder_tokens = [f"{token_name}_{i}" for i in range(num_vectors)]

        # Add to first tokenizer/encoder
        self.tokenizer.add_tokens(placeholder_tokens)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_ids_1 = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)
        self.concept_token_ids[token_name] = token_ids_1

        # Add to second tokenizer/encoder
        self.tokenizer_2.add_tokens(placeholder_tokens)
        self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
        token_ids_2 = self.tokenizer_2.convert_tokens_to_ids(placeholder_tokens)
        self.concept_token_ids_2[token_name] = token_ids_2

        # Initialize embeddings in both encoders
        embeds_1 = self.text_encoder.get_input_embeddings().weight.data
        embeds_2 = self.text_encoder_2.get_input_embeddings().weight.data

        if initializer_token:
            init_id_1 = self.tokenizer.encode(initializer_token, add_special_tokens=False)
            init_id_2 = self.tokenizer_2.encode(initializer_token, add_special_tokens=False)

            if init_id_1:
                init_embed_1 = embeds_1[init_id_1[0]].clone()
                for tid in token_ids_1:
                    embeds_1[tid] = init_embed_1 + torch.randn_like(init_embed_1) * 0.01

            if init_id_2:
                init_embed_2 = embeds_2[init_id_2[0]].clone()
                for tid in token_ids_2:
                    embeds_2[tid] = init_embed_2 + torch.randn_like(init_embed_2) * 0.01

        return embeds_1[token_ids_1]

    def get_concept_embedding(self, token_name: str) -> torch.Tensor:
        """Get embeddings from first encoder (primary)."""
        if token_name not in self.concept_token_ids:
            raise ValueError(f"Unknown concept token: {token_name}")

        token_ids = self.concept_token_ids[token_name]
        embeddings = self.text_encoder.get_input_embeddings().weight.data[token_ids]
        return embeddings.clone()

    def set_concept_embedding(self, token_name: str, embedding: torch.Tensor) -> None:
        """Set embeddings in both encoders."""
        if token_name not in self.concept_token_ids:
            raise ValueError(f"Unknown concept token: {token_name}")

        # Set in first encoder
        token_ids_1 = self.concept_token_ids[token_name]
        embeds_1 = self.text_encoder.get_input_embeddings().weight.data
        for i, tid in enumerate(token_ids_1):
            embeds_1[tid] = embedding[i]

        # Set in second encoder (use same embedding, will be projected)
        token_ids_2 = self.concept_token_ids_2[token_name]
        embeds_2 = self.text_encoder_2.get_input_embeddings().weight.data
        for i, tid in enumerate(token_ids_2):
            # Project to second encoder's dimension if needed
            if embedding.shape[-1] != embeds_2.shape[-1]:
                # Simple projection - in practice might want learned projection
                embeds_2[tid] = torch.nn.functional.pad(
                    embedding[i],
                    (0, embeds_2.shape[-1] - embedding.shape[-1])
                )
            else:
                embeds_2[tid] = embedding[i]

    def freeze_all(self) -> None:
        """Freeze all model parameters including embeddings."""
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

    def freeze_all_except_embeddings(self) -> None:
        """Freeze all except concept embeddings in both encoders."""
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Unfreeze embeddings
        self.text_encoder.get_input_embeddings().requires_grad_(True)
        self.text_encoder_2.get_input_embeddings().requires_grad_(True)

    def __call__(
        self,
        prompt: str,
        image: torch.Tensor,
        mask: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> torch.Tensor:
        """Run SDXL inpainting generation."""
        from torchvision.transforms.functional import to_pil_image

        image_pil = to_pil_image((image[0] + 1) / 2)
        mask_pil = to_pil_image(mask[0])

        result = self.pipeline(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

        return result.images[0]
