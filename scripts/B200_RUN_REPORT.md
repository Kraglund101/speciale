# B200 run report (server-side Claude session, 2026-10-03 to 2026-10-06)

Written for verification by the Windows-side session. Everything below is taken from files on the server
(`/work/Jobs/data/other/speciale`, branch `b200-open-set`). Where something was not checked, it says **not verified**.

## 0. Deviations from the protocol and judgement calls (read first)

| # | What | Why / effect | Where to check |
|---|---|---|---|
| D1 | Package versions differ from the Windows side: torch 2.7.1+cu128 / torchvision 0.22.1 (not 2.5.1 / 0.20.1); anomalib 2.6.2 (Windows version unknown); imgaug 0.4.0 installed `--no-deps` | torch 2.5.1 has no Blackwell (sm_100) kernels. Everything else pinned to the handoff versions. Numerical differences vs the 4090 are expected; size not measured for open-set | §2 |
| D2 | Seed 42 diffusion images were made by **two sampler implementations** with the same setting (0.4, 50 steps, timesteps 381..1): 7,939 images by commit ffe1fea (`noise_strength_thesis_sets.fixed_scheduler` + `s35_timesteps`, the exact code of the `ns_040_s50` test set), the rest by df5c916 `EvenStepDDIM` (`P.GEN`, even_steps=True). Launched with `OPEN_SET_ACCEPT_EXISTING=1` to get past `generator_stamp()`. | Done to fit one full seed into the job time. Handoff says EvenStepDDIM reproduces ns_040_s50 within max 2/255. **Seed 123 is 100 % EvenStepDDIM.** I did not measure the difference on these images. | `results/open_set_v2/IMAGE_PROVENANCE.md`, `images_made_with_ffe1fea_scheduler.txt` |
| D3 | The planned smoke test and the 2-fold pilot were **not** completed; user asked to go straight to full seeds. | The open-set training stage was first exercised by a 1-epoch manual test (diffusion_cross_s1, seed 42, fold holes, rc 0) and then by the real runs. | launch.log |
| D4 | Seed-42 placements were planned in two launches: holes+breakage at 2026-10-04 09:25 (code ffe1fea), the other 4 folds at 19:33 (code 540aa2d). | `git diff ffe1fea 540aa2d` shows no change to `stage_plan`, `P.place` or anything in the placement path (only generator / stamp / dtd-cutmix sharding). | git |
| D5 | DTD / CutMix were split over 12 processes (`--cpu-procs`, commit 540aa2d). | Each item uses its own `P.method_seed(...)` RNG, so outputs should be identical to one process. **Not verified by a byte comparison.** | `stage_dtd`, `stage_cutmix` |
| D6 | First DTD attempt failed (`ModuleNotFoundError: No module named 'imgaug'`, 2026-10-05 00:45, launch STOP). Fixed by installing imgaug 0.4.0, resumed 07:54. | All 12 DTD shards rc 0 on rerun. DTD images of both seeds were produced only after the fix. | launch.log |

No change was made to: seeds, placement, masks (max-pool / `> 0`), mask factor, generator checkpoint, CFG, hero settings,
arms, epochs, test shifts, metrics, `open_set_tables.py`.

## 1. Code changes made on the server (all on `b200-open-set`, pushed)

| commit | file | change |
|---|---|---|
| 07b749f | `scripts/train_uninet.py` | `--gpu` default None: keep inherited `CUDA_VISIBLE_DEVICES` (launchers pin MIG slices). Before, every run was forced onto device 0 -> OOM / NVML assert on the 3rd concurrent run. Placement only. |
| ffe1fea | `scripts/open_set_steps.py` | 0.4 / 50-step generation via `fixed_scheduler`. **Superseded** by the Windows commits df5c916 / 516fa73 (generator only from `P.GEN`). |
| 540aa2d | `scripts/open_set_steps.py`, `scripts/open_set_launch.py` | `--shard` for dtd / cutmix, `--cpu-procs` (default 12) in the launcher. |

Code state for all open-set training runs: **540aa2d** (= Windows ce323ad + 540aa2d). `git status` clean except untracked helpers.
Untracked helpers in the project root (not in git): `run_env.sh` (venv + HF_HOME/TORCH_HOME/HF_HUB_OFFLINE=1), `os_full.sh`
(the launcher call below), `os_tables_live.py` / `os_tables_loop.sh` (snapshot table of finished runs -> `TABLES_live.md`),
older `ns_*.sh`, `os_pilot.sh`, `os_chain.sh`.

## 2. Environment

- 4 x NVIDIA B200 MIG `1g.23gb` slices (20.5 GiB, 18 SMs each), 24-core CPU quota. Detected by `open_set_launch.detect_gpus()`;
  one process pinned per slice via `CUDA_VISIBLE_DEVICES=MIG-...`; 8 generation processes, 8 training slots (2 per slice; a
  training needs ~7.5 GiB, 3 do not fit).
- Python 3.11.16 (uv), torch 2.7.1+cu128, torchvision 0.22.1+cu128, anomalib 2.6.2, diffusers 0.32.2, transformers 4.44.2,
  timm 1.0.29, numpy 1.26.4, scipy 1.11.4, pillow 10.2.0, scikit-learn 1.2.2, opencv-headless 4.9.0, imgaug 0.4.0.
  Full list: `_cache/venv_freeze.txt`.
- Data: the 16.1 GB zip, unpacked into the project root unchanged. `make_leakfree_thesis_set.py` was run once (2026-10-04).

## 3. What was run

### Noise-strength test (2026-10-03/04) — `results/noise_strength_test/TABLES.md`
`noise_strength_thesis_sets.py --fixed-steps 50 --strengths 0.025 ... 0.9` and `--strengths 1.0` (4 processes, one strength
group per slice, all "SETS DONE"), then `noise_strength_train.py` (70 runs, rc 0). First training attempt crashed (D-fix
07b749f), archived in `results/noise_strength_test/failed_attempt1/`, none of its runs reused. Arm `070` (0.7 / 35 steps
re-rendered on the B200): 5 runs rc 0, within 0.005 of the shipped baseline on every metric.

### Open-set (2026-10-04 to 06) — `results/open_set_v2/`
Command (tmux `osfull`, `os_full.sh`):
```
OPEN_SET_ACCEPT_EXISTING=1 python scripts/open_set_launch.py --profile b200 --seeds 42 123 7 99 256 11 22 33 44 55
```
No `--smoke`, no `OPEN_SET_EPOCH_LIMIT`, no `--sets`, no `--arms` / `--folds` filter. `generator_settings.json`:
`{"ckpt":"clean-20k","mask_factor":0.25,"noise_strength":0.4,"num_steps":50,"even_steps":true,"guidance_scale":7.0,"band_mode":2,"cfg_mode":"visual","raw_background":true}`

Stage log (`launch.log`):

| seed | plan | generate | refine | dtd | cutmix | check | training |
|---|---|---|---|---|---|---|---|
| 42 | rc 0 | rc 0 (8 procs; plus 09:30-13:01 on 10-04 before a job restart) | rc 0, 207.6 min | rc 1 x12 (imgaug), rerun rc 0 | rc 0 | rc 0 | 96 runs rc 0, 10-05 08:04 -> ~11:50 |
| 123 | rc 0 | rc 0, 324.7 min | rc 0, 218.1 min | rc 0 | rc 0 | rc 0 | 96 runs rc 0, 10-05 20:59 -> 10-06 00:17 |
| 7 | rc 0 | interrupted by job end (2,371 images, no masks) | – | – | – | – | none |

Image counts (images = masks): seed 42 11,721 (diffusion_in 4,891, diffusion_cross 6,830); seed 123 11,854
(4,937 / 6,917). The two seeds differ by 133 images; **cause not verified** (expected to follow from the per-seed plan; the
`check` stage passed for both).

Job restarts: the UCloud job restarted 10-04 ~13:02 (tmux died) and ended 10-06 ~01:20 during seed 7 generation. After the first, all 7,939
existing PNGs were opened with PIL (0 broken) before resuming; the launcher skips existing images and finished runs.

## 4. Verification done on the server (2026-10-06)

- 192 `results.json` (2 seeds x 6 folds x 16 arms), every one with 20 epochs and all 5 shift levels; no `checkpoints/` left.
- Run configuration from the logs (seed 123, fold holes; same pattern in the others I looked at):
  - `m1`: `hero m1` = "Mode 1 — normal-only baseline (WRN-50-2, local KD)", photometric OFF, Baseline-extend +45 normals,
    Train 145 / Test 70.
  - every other arm: `hero caao` = "Mode 4 — CAAO hero (ConvNeXt-B, local KD, w_canvas=1, projcons 0.2, no SG)", photometric
    OFF, `--synthetic-epochs` 20 epochs x 45 synthetics, EpochPoolSampler 100 normals + 45 synthetics, Train 1000 / Test 70.
  - all: `LEAK GUARD: replaced [] ... dropped []`.
- Leakage / protocol script over all lists of seeds 42 and 123 (162,000 synthetic entries): no held-out class in any
  synthetic path, no synthetic canvas among the 50 test normals, no train normal or M1 extra normal among the test normals,
  no missing image / mask / canvas file. Test anomalies per fold contain only the held-out type
  (holes: "small holes"; breakage: "corner or edge breakage", "middle breakage"; burnt; colour_diff: "different colour spot";
  colour_same: "same colour spot"; scratches: "small scratches").

## 5. Things that look suspicious and were NOT resolved

1. **M1 is near chance**: composite 0.556, clean (0x) 0.645. Note that M1 and the other arms differ in more than the
   synthetics: `open_set_steps.command()` uses `--hero m1` (WRN-50-2 backbone, Mode 1) for M1 and `--hero caao`
   (ConvNeXt-B, Mode 4, canvas alignment, projection consistency) for every other arm. So "+0.32 vs M1" is synthetics **plus**
   backbone / training mode. Whether that is the intended comparison is a design question for Frederik; I did not change it.
   Not checked: M1 on this test set on the Windows side.
2. Synthetic arms are very high on holes / breakage / burnt / colour_diff (0.90-0.99) and much lower on colour_same /
   scratches (0.59-0.72). Not investigated.
3. Stages 1-5 give nearly identical numbers (within ~0.005 per family). Not investigated.
4. D2 (mixed sampler in seed 42) — can be tested by comparing seed-42 vs seed-123 rows, or by regenerating seed 42.

## 6. Files to look at

- `results/open_set_v2/TABLES.md` (regenerated 2026-10-06 with `scripts/open_set_tables.py`, unchanged code)
- `results/open_set_v2/launch.log`, `launch_full.out`, `results/open_set_v2/logs/` (one log per step / run)
- `results/open_set_v2/lists/seed_<s>/fold_<k>/` (the exact train / test / synthetic lists every run used)
- `results/open_set_v2/runs/<arm>/seed_<s>/fold_<k>/results.json`
- `results/open_set_v2/IMAGE_PROVENANCE.md`, `images_made_with_ffe1fea_scheduler.txt`, `generator_settings.json`
