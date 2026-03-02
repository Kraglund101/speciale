"""Quick benchmark: deterministic vs non-deterministic cuDNN.

Runs 50 UNet forward passes each way and reports timing.
"""
import time
import torch
from diffusers import UNet2DConditionModel

device = "cuda"
N_WARMUP = 5
N_BENCH = 50
BS = 10

# Load once, reuse for both benchmarks
print("Loading UNet...", flush=True)
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-inpainting", subfolder="unet",
).to(device)
unet.eval()


def bench(label: str):
    # Typical inpainting input: [B, 9, 64, 64]
    dummy_input = torch.randn(BS, 9, 64, 64, device=device, dtype=torch.float16)
    dummy_t = torch.tensor([500], device=device).expand(BS)
    dummy_text = torch.randn(BS, 77, 768, device=device, dtype=torch.float16)

    # Warmup
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(N_WARMUP):
            unet(dummy_input, dummy_t, encoder_hidden_states=dummy_text)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(N_BENCH):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            unet(dummy_input, dummy_t, encoder_hidden_states=dummy_text)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    std_ms = (sum((t - avg_ms / 1000) ** 2 for t in times) / len(times)) ** 0.5 * 1000
    print(f"  {label}: {avg_ms:.1f} ms/step  ± {std_ms:.1f} ms")
    return avg_ms


print(f"Benchmarking UNet forward (BS={BS}, {N_BENCH} steps each)...\n")

# Non-deterministic (fast)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
t_fast = bench("NON-DETERMINISTIC (benchmark=True) ")

# Deterministic (reproducible)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
t_det = bench("DETERMINISTIC     (benchmark=False)")

pct = (t_det / t_fast - 1) * 100
extra_min = (t_det - t_fast) * 50000 / 1000 / 60
print(f"\n  Slowdown: {pct:+.1f}%")
print(f"  Per 50k steps (UNet-only): {extra_min:+.0f} min")
