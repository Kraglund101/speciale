"""
T2I-Adapter for injecting anomaly mask spatial conditioning into UNet.

Input: 2 channels @ 64×64 (latent resolution)
  - Channel 0: Binary anomaly mask (maxpooled to 64×64)
  - Channel 1: Dilated ring around anomaly (maxpooled to 64×64)

Architecture:
  Block 0: Conv1×1(2→320) + GN+SiLU + Conv3×3(320→320) + GN+SiLU + Conv3×3(320→320, zero-init)
  Block 1: Conv3×3(320→640, s=2) + GN+SiLU + Conv3×3(640→640, zero-init)
  Block 2: Conv3×3(640→1280, s=2) + GN+SiLU + Conv3×3(1280→1280, zero-init)
  Block 3: Conv3×3(1280→1280, s=2) + GN+SiLU + Conv3×3(1280→1280, zero-init)

  Channels match down_block OUTPUTS: [320, 640, 1280, 1280] for SD 1.5.

Injection modes (controlled via `injection_mode` flag):

  "cascade": Features added to the block output BEFORE it continues to the
      next block. The modified output flows into the next encoder block AND
      gets saved as a skip connection. Both encoder and decoder are biased.

      block[0] → output + feat[0] → saved as skip → also flows to block[1]
      block[1] → output + feat[1] → saved as skip → also flows to block[2]
      ...

  "skip_only": Features added to the saved skip connections AFTER the entire
      encoder finishes. The flowing sample between blocks is never modified.
      Encoder runs clean, only decoder sees the adapter signal.

      block[0] → output → flows clean to block[1]
      block[1] → output → flows clean to block[2]
      ...
      then: skip[0] += feat[0], skip[1] += feat[1], ...
      decoder receives modified skips

Each feature is masked to anomaly regions and scaled by a learned parameter.
Zero-init on last conv per block → adapter contributes nothing at step 0.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingBlock(nn.Module):
    """Block 0: per-pixel embedding (1×1) then spatial mixing (3×3s)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.embed(x)))
        x = self.act2(self.norm2(self.conv1(x)))
        x = self.conv_out(x)
        return x


class DownsampleBlock(nn.Module):
    """Blocks 1-3: stride-2 downsample + channel expansion, then zero-init conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.conv_out(x)
        return x


class T2IAdapter(nn.Module):
    """Lightweight spatial adapter injecting mask conditioning into UNet.

    Args:
        in_channels: Input channels (default: 2 for mask + dilated ring).
        block_out_channels: Channel dims matching UNet down_block outputs.
            Default: [320, 640, 1280, 1280] for SD 1.5.
        injection_mode: Where to inject features into UNet.
            "cascade"   — encoder + decoder (features flow through encoder)
            "skip_only" — decoder only (encoder untouched)
    """

    def __init__(
        self,
        in_channels: int = 2,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        injection_mode: str = "cascade",
    ):
        super().__init__()
        assert injection_mode in ("cascade", "skip_only")
        self.block_out_channels = block_out_channels
        self.injection_mode = injection_mode

        # Block 0: embedding + spatial mixing
        self.block0 = EmbeddingBlock(in_channels, block_out_channels[0])

        # Blocks 1-3: stride-2 downsample
        self.down_blocks = nn.ModuleList()
        for i in range(1, len(block_out_channels)):
            self.down_blocks.append(
                DownsampleBlock(block_out_channels[i - 1], block_out_channels[i])
            )

        # Learned scale per block, init to 1.0
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in block_out_channels
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Process adapter input and produce masked, scaled features.

        Args:
            x: Adapter input [B, in_channels, 64, 64].
            mask: Binary anomaly mask [B, 1, 64, 64] at latent resolution.

        Returns:
            List of 4 feature tensors (masked + scaled), matching down_block
            output dims: [B,320,64,64], [B,640,32,32], [B,1280,16,16], [B,1280,8,8]
        """
        # Precompute masks at each resolution via maxpool
        masks = [mask]  # 64×64
        m = mask
        for _ in range(len(self.block_out_channels) - 1):
            m = F.max_pool2d(m, kernel_size=2)
            masks.append(m)

        features = []

        # Block 0 — internal flow is unmasked
        h = self.block0(x)
        features.append(h * masks[0] * self.scales[0])

        # Blocks 1-3 — h flows unmasked, outputs get masked
        for i, block in enumerate(self.down_blocks):
            h = block(h)
            features.append(h * masks[i + 1] * self.scales[i + 1])

        return features

    def prepare_unet_kwargs(
        self,
        features: List[torch.Tensor],
    ) -> dict:
        """Build the kwargs to pass to UNet forward based on injection_mode.

        Args:
            features: Output of self.forward().

        Returns:
            Dict of kwargs to unpack into unet(...).

        Usage:
            features = adapter(x, mask)
            kwargs = adapter.prepare_unet_kwargs(features)
            noise_pred = unet(sample, timestep, encoder_hidden_states, **kwargs)
        """
        if self.injection_mode == "cascade":
            # Added inside each block → flows to next block + saved as skip
            return {"down_intrablock_additional_residuals": list(features)}
        else:
            # Added to saved skips after encoder finishes → decoder only
            return {"down_block_additional_residuals": list(features)}
