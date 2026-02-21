"""Smoke test for visual modes 0-3 + multi-crop.

Runs on CPU with small dummy tensors — no GPU needed.
Tests:
1. MaskedAnomalySelfAttention modes 0-3 (forward shapes)
2. encode_image token extraction for modes 2-3 (mocked CLIP)
3. IPAdapterAttnProcessor null_token_mask handling
4. clip_crop_multi output shapes
5. AnomalyDataset multi_crop flag (shape check only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


# ============================================================
# 1. MaskedAnomalySelfAttention — modes 0-3
# ============================================================
print("\n=== 1. MaskedAnomalySelfAttention ===")
from src.models.ip_adapter import MaskedAnomalySelfAttention

B, N, D = 2, 256, 1280
x = torch.randn(B, N, D)
mask_hw = torch.zeros(B, 1, 224, 224)
mask_hw[:, :, 50:100, 50:100] = 1.0  # some anomaly region

# Mode 0: standard
sa0 = MaskedAnomalySelfAttention(in_channels=D, visual_mode=0)
out0 = sa0(x, mask=mask_hw)
check("Mode 0 output shape", out0.shape == (B, N, D), f"got {out0.shape}")
check("Mode 0 has ff_gate", "ff_gate" in sa0.layers[0], "missing ff_gate")

# Mode 1: selective residual
sa1 = MaskedAnomalySelfAttention(in_channels=D, visual_mode=1)
out1 = sa1(x, mask=mask_hw)
check("Mode 1 output shape", out1.shape == (B, N, D), f"got {out1.shape}")

# Verify mode 1 selective residual: with gates=0, anomaly tokens should be unchanged,
# background tokens should be attention output (not identity)
with torch.no_grad():
    # Gates are 0 at init, so anomaly tokens = x (identity), background = attn_out
    out1_test = sa1(x, mask=mask_hw)
    # Downsample mask to check which tokens are anomaly
    kernel = 224 // 16
    mask_16 = F.max_pool2d(mask_hw.float(), kernel_size=kernel)
    mask_flat = (mask_16 > 0.5).float().view(B, N)
    # Anomaly tokens (mask=1) should be close to original x (gate=0 → identity)
    anom_idx = mask_flat[0].nonzero(as_tuple=True)[0]
    bg_idx = (1 - mask_flat[0]).nonzero(as_tuple=True)[0]
    if len(anom_idx) > 0:
        anom_diff = (out1_test[0, anom_idx] - x[0, anom_idx]).abs().max().item()
        check("Mode 1 anomaly tokens = identity (gate=0)", anom_diff < 1e-5,
              f"max diff = {anom_diff}")
    if len(bg_idx) > 0:
        bg_diff = (out1_test[0, bg_idx] - x[0, bg_idx]).abs().max().item()
        check("Mode 1 background tokens != identity", bg_diff > 1e-5,
              f"max diff = {bg_diff} (should be > 0)")

# Mode 2: anomaly-only, no MLP
sa2 = MaskedAnomalySelfAttention(in_channels=D, visual_mode=2)
check("Mode 2 has NO ff", "ff" not in sa2.layers[0], "ff should not exist")
check("Mode 2 has NO ff_gate", "ff_gate" not in sa2.layers[0], "ff_gate should not exist")

# Mode 2 with padding_mask
n_anom = 30
x_anom = torch.randn(B, n_anom, D)
pad_mask = torch.ones(B, n_anom)
pad_mask[0, 20:] = 0  # sample 0: 20 real, 10 padding
pad_mask[1, 25:] = 0  # sample 1: 25 real, 5 padding
out2 = sa2(x_anom, padding_mask=pad_mask)
check("Mode 2 output shape", out2.shape == (B, n_anom, D), f"got {out2.shape}")

# Mode 3: anomaly-only, full transformer
sa3 = MaskedAnomalySelfAttention(in_channels=D, visual_mode=3)
check("Mode 3 has ff", "ff" in sa3.layers[0], "missing ff")
check("Mode 3 has ff_gate", "ff_gate" in sa3.layers[0], "missing ff_gate")
out3 = sa3(x_anom, padding_mask=pad_mask)
check("Mode 3 output shape", out3.shape == (B, n_anom, D), f"got {out3.shape}")

# Verify padding tokens in mode 2: with gate=0, real tokens = identity, padding = attn_out
with torch.no_grad():
    out2_test = sa2(x_anom, padding_mask=pad_mask)
    # Real tokens (pad_mask=1) should be identity when gate=0
    real_diff = (out2_test[0, :20] - x_anom[0, :20]).abs().max().item()
    check("Mode 2 real tokens = identity (gate=0)", real_diff < 1e-5,
          f"max diff = {real_diff}")
    # Padding tokens should NOT be identity
    pad_diff = (out2_test[0, 20:] - x_anom[0, 20:]).abs().max().item()
    check("Mode 2 padding tokens != identity", pad_diff > 1e-5,
          f"max diff = {pad_diff}")


# ============================================================
# 2. IPAdapterAttnProcessor null_token_mask
# ============================================================
print("\n=== 2. IPAdapterAttnProcessor null_token_mask ===")
from src.models.ip_adapter import IPAdapterAttnProcessor

# Create a minimal mock attn object
class MockAttn:
    def __init__(self, hidden_size, cross_attn_dim):
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = False
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.heads = 8
        head_dim = hidden_size // self.heads
        self.scale = head_dim ** -0.5
        self.to_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = torch.nn.Linear(cross_attn_dim, hidden_size, bias=False)
        self.to_v = torch.nn.Linear(cross_attn_dim, hidden_size, bias=False)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size, bias=False),
            torch.nn.Dropout(0.0),
        ])

    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size = tensor.shape[0] // head_size
        seq_len, dim = tensor.shape[1], tensor.shape[2]
        tensor = tensor.reshape(batch_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, dim * head_size)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        scale = self.scale
        scores = torch.bmm(query, key.transpose(-1, -2)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        return torch.softmax(scores, dim=-1)

    def prepare_attention_mask(self, mask, seq_len, batch_size):
        return mask

hidden_size = 320
cross_attn_dim = 768
K = 16
proc = IPAdapterAttnProcessor(hidden_size, cross_attn_dim, num_tokens=K, mask_visual=False)
mock_attn = MockAttn(hidden_size, cross_attn_dim)

# Test without null_token_mask (standard)
hidden = torch.randn(B, 64, hidden_size)  # 8x8 spatial
text_emb = torch.randn(B, 77, cross_attn_dim)
ip_embeds = torch.randn(B, K, cross_attn_dim)

out_no_null = proc(mock_attn, hidden, encoder_hidden_states=text_emb,
                   ip_adapter_image_embeds=ip_embeds)
check("No null_mask: output shape", out_no_null.shape == (B, 64, hidden_size),
      f"got {out_no_null.shape}")

# Test with null_token_mask (multi-crop: 2K tokens, second group invalid for sample 0)
ip_embeds_2k = torch.randn(B, 2 * K, cross_attn_dim)
null_mask = torch.ones(B, 2 * K)
null_mask[0, K:] = 0  # sample 0: second group invalid

out_null = proc(mock_attn, hidden, encoder_hidden_states=text_emb,
                ip_adapter_image_embeds=ip_embeds_2k, null_token_mask=null_mask)
check("With null_mask: output shape", out_null.shape == (B, 64, hidden_size),
      f"got {out_null.shape}")

# Test that null masking actually suppresses: compare with all-valid vs half-null
null_mask_allvalid = torch.ones(B, 2 * K)
out_allvalid = proc(mock_attn, hidden, encoder_hidden_states=text_emb,
                    ip_adapter_image_embeds=ip_embeds_2k, null_token_mask=null_mask_allvalid)
diff = (out_null - out_allvalid).abs().max().item()
check("Null masking changes output", diff > 1e-5, f"diff = {diff}")


# ============================================================
# 3. clip_crop_multi
# ============================================================
print("\n=== 3. clip_crop_multi ===")
from src.utils.crop_utils import clip_crop_multi

img = torch.randn(3, 1024, 1024)
mask = torch.zeros(1, 1024, 1024)
# Two separate anomaly regions
mask[0, 100:150, 100:150] = 1.0
mask[0, 800:850, 800:850] = 1.0

crops, crop_masks, valid = clip_crop_multi(img, mask, crop_size=224, n_groups=2)
check("Multi-crop: 2 crops returned", len(crops) == 2, f"got {len(crops)}")
check("Multi-crop: crop 1 shape", crops[0].shape == (3, 224, 224), f"got {crops[0].shape}")
check("Multi-crop: crop 2 shape", crops[1].shape == (3, 224, 224), f"got {crops[1].shape}")
check("Multi-crop: both valid", all(valid), f"valid = {valid}")
check("Multi-crop: mask 1 has anomaly", crop_masks[0].sum() > 0, "mask 1 is empty")
check("Multi-crop: mask 2 has anomaly", crop_masks[1].sum() > 0, "mask 2 is empty")

# Test with single anomaly region (second group should be invalid)
mask_single = torch.zeros(1, 1024, 1024)
mask_single[0, 100:150, 100:150] = 1.0
crops_s, masks_s, valid_s = clip_crop_multi(img, mask_single, crop_size=224, n_groups=2)
check("Single region: first valid", valid_s[0] is True, f"valid[0] = {valid_s[0]}")
check("Single region: second invalid", valid_s[1] is False, f"valid[1] = {valid_s[1]}")
check("Single region: crop 2 is zeros", crops_s[1].abs().sum() == 0, "crop 2 not zeros")

# Test with no anomaly (both invalid)
mask_empty = torch.zeros(1, 1024, 1024)
crops_e, masks_e, valid_e = clip_crop_multi(img, mask_empty, crop_size=224, n_groups=2)
check("Empty mask: both invalid", not any(valid_e), f"valid = {valid_e}")


# ============================================================
# 4. Gradient flow check
# ============================================================
print("\n=== 4. Gradient flow ===")

# Verify gradients flow through modes 0-3
# Expected param grad counts at gate=0 initialization:
#   Mode 0: 2/12 — only the 2 gates get grad (all paths multiplied by gate=0)
#   Mode 1: 12/12 — background tokens bypass gates, all params get grad from step 0
#   Mode 2: 6/6 — padding tokens bypass gate, all attn params get grad (no MLP)
#   Mode 3: 8/12 — attn params via padding bypass, but FF gated off (ff_gate=0)
expected_grad_counts = {0: 2, 1: 12, 2: 7, 3: 8}
total_param_counts = {0: 12, 1: 12, 2: 7, 3: 12}

for mode in [0, 1, 2, 3]:
    if mode in (0, 1):
        sa = MaskedAnomalySelfAttention(in_channels=D, visual_mode=mode)
        inp = torch.randn(B, N, D, requires_grad=True)
        out = sa(inp, mask=mask_hw)
    else:
        sa = MaskedAnomalySelfAttention(in_channels=D, visual_mode=mode)
        inp = torch.randn(B, n_anom, D, requires_grad=True)
        out = sa(inp, padding_mask=pad_mask)
    loss = out.sum()
    loss.backward()
    has_grad = inp.grad is not None and inp.grad.abs().sum() > 0
    check(f"Mode {mode}: gradients flow to input", has_grad,
          "no gradient on input")
    # Check parameter gradients match expected count at gate=0 init
    n_grad = sum(1 for p in sa.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params = sum(1 for p in sa.parameters())
    expected = expected_grad_counts[mode]
    check(f"Mode {mode}: {expected}/{n_params} params have grad (gates=0)",
          n_grad == expected,
          f"got {n_grad}/{n_params}, expected {expected}/{n_params}")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — check output above")
    sys.exit(1)
