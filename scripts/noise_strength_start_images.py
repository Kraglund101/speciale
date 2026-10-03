#!/usr/bin/env python3
"""What the generator starts from at each noise strength, in image space (user 2026-10-03): the clean canvas latent
noised to the start timestep of ``generate_anomagic_single`` (same formula: start_step = int(steps * (1 - strength))),
decoded with the VAE. Same three anomalies / crops as noise_strength_sweep.py. Runs on the CPU (VAE + scheduler only).
The noise draw is a CPU draw with the image seed, so the pattern differs from the GPU draw of the real run; the amount
is identical. Prints timestep, signal and noise scale per strength.

  python scripts/noise_strength_start_images.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline  # noqa: E402,F401
from diffusers.schedulers.scheduling_utils import SchedulerMixin  # noqa: E402,F401
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import noise_strength_sweep as N  # noqa: E402

S, P = N.S, N.S.P
MODEL, STEPS = "runwayml/stable-diffusion-inpainting", P.GEN["num_steps"]


def main() -> None:
    import diffusers
    pixel = "--pixel" in sys.argv                                        # illustration: noise on the pixels, not the latent
    cfg = diffusers.DiffusionPipeline.load_config(MODEL, local_files_only=True)
    sched = getattr(diffusers, cfg["scheduler"][1]).from_pretrained(MODEL, subfolder="scheduler", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", local_files_only=True).eval()
    sched.set_timesteps(STEPS); ts = sched.timesteps; ab = sched.alphas_cumprod
    print(f"scheduler {type(sched).__name__}, {len(ts)} timesteps for {STEPS} steps")
    print("| noise strength | denoising steps run | start timestep | canvas scale sqrt(abar) | noise scale sqrt(1-abar) | noise / canvas |")
    print("|---|---|---|---|---|---|")
    starts = {}
    for ns in N.STRENGTHS:
        k = max(0, int(len(ts) * (1 - ns))); t = int(ts[k]); a = float(ab[t]); starts[ns] = ts[k]
        print(f"| {ns:.1f} | {len(ts) - k} | {t} | {a ** 0.5:.3f} | {(1 - a) ** 0.5:.3f} | {((1 - a) / a) ** 0.5:.2f} |")
    man = S.manifest(N.SEED); types = ["breakage", "burnt", "colour_diff"]; T, LW = N.TILE, N.LABW
    font = ImageFont.truetype("arial.ttf", 14); small = ImageFont.truetype("arial.ttf", 12)
    sheet = Image.new("RGB", (LW + (1 + len(N.STRENGTHS)) * T, 30 + len(types) * T), (25, 25, 25)); dr = ImageDraw.Draw(sheet)
    for c, h in enumerate(["clean canvas"] + [f"start at {x:.1f}" + (" (pipeline)" if x == 0.7 else "") for x in N.STRENGTHS]):
        dr.text((LW + c * T + 6, 8), h, fill=(230, 230, 230), font=small)
    for r, cls in enumerate(types):
        st = man["sets"]["s2"][cls][0]; prep = S.pdir(N.SEED, "s2", cls, st["i"]); seed = json.loads((prep / "meta.json").read_text())["seed"]
        M = N.mask_resize(np.array(Image.open(prep / "placed_mask.png").convert("L")) > 127, 512); b = N.box(M); y = 30 + r * T
        can = np.array(Image.open(P.NORMAL_DIR / f"{st['canvas']}.JPG").convert("RGB").resize((512, 512), Image.LANCZOS))
        dr.text((6, y + 8), cls, fill=(240, 240, 240), font=font); sheet.paste(N.crop(can, b, M), (LW, y))
        x = torch.from_numpy(can).permute(2, 0, 1)[None].float() / 127.5 - 1
        if pixel:                                                        # same schedule, noise added to the pixels
            noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))
            for c, ns in enumerate(N.STRENGTHS):
                im = sched.add_noise(x, noise, starts[ns].unsqueeze(0))[0].clamp(-1, 1)
                sheet.paste(N.crop(((im.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8), b), (LW + (1 + c) * T, y))
            continue
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
            noise = torch.randn(lat.shape, generator=torch.Generator().manual_seed(seed))
            for c, ns in enumerate(N.STRENGTHS):
                z = sched.add_noise(lat, noise, starts[ns].unsqueeze(0))
                im = vae.decode(z / vae.config.scaling_factor).sample[0].clamp(-1, 1)
                im = ((im.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
                sheet.paste(N.crop(im, b), (LW + (1 + c) * T, y))
        print(f"{cls}: done", flush=True)
    out = S.OUT / "sheets" / ("noise_strength_start_pixel.png" if pixel else "noise_strength_start_images.png"); sheet.save(out); print(out, sheet.size)


if __name__ == "__main__":
    main()
