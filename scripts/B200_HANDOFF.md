# Handoff: open-set experiment on the B200 (UCloud)

Written 2026-10-03 by the Claude Code session on Frederik's Windows PC (RTX 4090), for the Claude Code session on the
server. Read this first, then `results/open_set_v2/PLAN.md` (design; sections 2, 8, 9, 10) and
`results/open_set_v2/GENERATION_PLAN.md` (data volume, cost). Both are in the data zip.

## What this is

Master's-thesis follow-up. A diffusion generator (SD 1.5 inpainting + IP-Adapter Plus + T2I-Adapter, checkpoint
"clean-20k", trained without cashew) paints synthetic defects on normal VisA cashew images; a UniNet detector is trained
on normals + synthetics and tested on real defects. The experiment to run here is **open-set, leave-one-type-out**:
6 cashew defect types, fold k trains on synthetics of the other 5 types and is tested on real defects of type k.
The question for the report: **does open-set synthetic training beat normal-only training (M1)?**

- 10 seeds (42 123 7 99 256 11 22 33 44 55) x 6 folds x 16 arms = 960 training runs, ~119,000 diffusion images.
- Arms: `m1` (normal-only); `diffusion_in`, `diffusion_cross`, `dtd`, `cutmix` at stages 1-3; `diffusion_in`,
  `diffusion_cross` at stage 4; `diffusion_cross` at stage 5. Stages are defined at the top of `scripts/open_set_steps.py`.
- Primary metric: composite image AUROC (mean of test shift levels 0x-4x), mean of epochs 15-20, paired with M1 on the
  same seed and fold. `scripts/open_set_tables.py` writes `results/open_set_v2/TABLES.md`.

## How to run

```bash
cd speciale                      # the folder MUST be named "speciale" (open_set_steps.rebase() relies on it)
unzip -q speciale_b200_data.zip  # data + pretrained models, paths relative to the project root
export HF_HOME=$PWD/_cache/huggingface TORCH_HOME=$PWD/_cache/torch HF_HUB_OFFLINE=1

python scripts/open_set_launch.py --profile b200 --smoke                      # 2 epochs, seed 42, fold holes, all arms
nohup python scripts/open_set_launch.py --profile b200 --seeds 42 123 7 99 256 11 22 33 44 55 > launch_full.out 2>&1 &
```

The launcher does plan -> generate -> refine -> dtd -> cutmix -> check -> train -> tables per seed, is resumable
(existing images and finished 20-epoch runs are skipped), and deletes `checkpoints/` after every run.
Progress: `results/open_set_v2/launch.log`; per-step logs: `results/open_set_v2/logs/`.

GPUs: the launcher reads `nvidia-smi -L`. One whole GPU -> normal. MIG slices or several GPUs -> one process pinned
per device via `CUDA_VISIBLE_DEVICES` (2 processes per slice by default, 8 per whole GPU). Overrides: `--gpus N`,
`--gpu-ids MIG-... MIG-...`, `--gen-procs`, `--train-slots`, `--stagger`. The first log line says what was detected.

## What has and has not been tested (be suspicious here)

- Tested on Windows: plan, DTD, CutMix, check (CPU stages); diffusion generation and mask refinement of ~1,400 pilot
  images; one-epoch M1 and CAAO trainings through `train_uninet.py`.
- **Never run end to end:** the launcher's training stage for the open-set arms, and `open_set_tables.py` on real runs.
- **Never run on Linux.** One hard-coded Windows path was already found and fixed (`generate_cashew.CASHEW_ROOT`).
  Expect more of this kind: backslashes in stored paths, `arial.ttf` (only used by sheet scripts, has a fallback).
- **Package completeness is unverified** beyond the planning stage. The zip contents were chosen from a file-access
  trace of local runs; if something is missing, the error will name the path. The list is `ITEMS` in
  `scripts/build_b200_package.py` - tell Frederik which path is missing rather than working around it.
- GPU detection: parsed from a MIG listing written from memory, not from real output. Check the first log line.
- The process counts of the `b200` profile (8 + 8 per GPU) are guesses. CPU cores for data loading are the likely limit.
- Local package versions: python 3.11, torch 2.5.1+cu121, torchvision 0.20.1, diffusers 0.32.2, transformers 4.44.2,
  timm 1.0.29, safetensors 0.4.5, numpy 1.26.4, scipy 1.11.4, pillow 10.2.0, scikit-learn 1.2.2, opencv 4.9.
  Known local issue: flash / mem-efficient SDPA segfaulted on PyTorch 2.3.0+cu121 (Windows); do not disable the math
  SDP kernel if you see that workaround in the code.

## Where the pieces are

- `scripts/open_set_launch.py` - launcher. `scripts/open_set_steps.py` - plan / generate / refine / dtd / cutmix /
  check / run command. `scripts/pregenerate_synthetic.py` (imported as `P`) - placement, generation settings `GEN`,
  `MASK_FACTOR`, reference and donor pools. `scripts/train_uninet.py` - wrapper with the locked "hero" settings.
- UniNet runner (not in git, in the zip):
  `anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew/experiment_UniNet/temp_uninet_experiment/run_uninet_cashew.py`
- Generator checkpoint: `results/generator_nocashew_20k_constlr/checkpoint_20000`.

## Rules (decided by Frederik; do not reopen)

- **Do not change the experiment design or the model settings.** Locked: `--hero caao` = projection weight 0.2, no
  stop-gradient, `--no-photometric`, fixed test shifts; generator clean-20k, visual CFG 7, noise strength 0.7, 50-step
  schedule, mask factor 0.25 x p90, no caption, no rejection. Fix portability bugs only; if a fix could change results
  (seeds, sampling, masks, which files are used), stop and tell him.
- Masks are always down-sampled with max-pool, never nearest / bilinear; VisA masks are read as `> 0`.
- Never pair canvas / mask / image by filename guesswork; the manifests and `meta.json` files are the source of truth.
- Never hard-code mappings or constants from memory; read them from the code / data.
- Be truthful and brief. He wants: what ran, what failed (with the log line), what you changed. No flattery, no long
  option lists. Say "not verified" when it is not.
- Do not keep training checkpoints (the launcher deletes them). Disk: ~135 GB of images for 10 seeds.
- Only run what he asks; label your own ideas as suggestions.

## What to bring back

`results/open_set_v2/TABLES.md`, `results/open_set_v2/launch.log`, the `results.json` of every run
(`results/open_set_v2/runs/<arm>/seed_<s>/fold_<k>/results.json`), and a list of every code change you made (commit
them on the `b200-open-set` branch and push, so the Windows side can pull them).

## ORDER (decided 2026-10-03 evening; overrides "How to run" above)

The open-set experiment must NOT be started yet: its generator setting (noise strength, now 0.7 / 35 steps) is not
decided. First and only job for now is the **noise-strength test**:

```bash
python scripts/noise_strength_thesis_sets.py --fixed-steps 50 --strengths 0.025 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
python scripts/noise_strength_thesis_sets.py --strengths 1.0
nohup python scripts/noise_strength_train.py > noise_train.out 2>&1 &     # 14 arms x 5 seeds = 70 runs
```

Thesis split (44 fixed training synthetics), final model. Arms: 12 noise strengths with 50 denoising steps, `mix`
(random strength per synthetic and epoch), `base` (the pipeline's own 0.7 / 35-step set). Result:
`results/noise_strength_test/TABLES.md`. After that: stop and wait. The open-set pilot (seed 42, folds holes +
breakage) and the full run come only after Frederik has chosen the noise strength and said go.
