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

## 7. Addendum 2026-10-06: M1 teacher weights (D7) — M1 rerun

**Cause of the near-chance M1:** anomalib 2.6.2 (installed on the server, D1) builds the UniNet teacher with
`torchvision.models.get_model_weights(backbone).DEFAULT`, which in torchvision 0.22 is Wide-ResNet-50-2 **IMAGENET1K_V2**
(`wide_resnet50_2-9ba9bcbe.pth`, silently downloaded 2026-10-03 22:10 — HF_HUB_OFFLINE does not block torchvision).
anomalib 2.1.0 / 2.2.0 / 2.3.0 (and the Windows side, 2.2.0) use `wide_resnet50_2(pretrained=True)` = **IMAGENET1K_V1**
(`95faca4d`, the file shipped in the zip). Diffing the wheels: in `anomalib/models/image/uninet/` this line is the ONLY
difference between 2.2.0 and 2.6.2.

**Scope:** only M1. The CAAO arms build the same anomalib model and then `_swap_to_convnext()` replaces teachers,
bottleneck, student, DFS and head, so the WRN teacher is discarded. The 180 non-M1 runs are unaffected by this.

**Fix:** server venv now anomalib **2.2.0** (rest of `_cache/venv_freeze.txt` unchanged: numpy 1.26.4, opencv-python-headless
4.9.0; anomalib 2.2.0 pulls opencv-python and numpy 2 as dependencies — both removed / reverted). Teacher call verified:
`getattr(torchvision.models, backbone)(pretrained=True)`.

**Diagnostic (seed 42, fold holes, same command):** old V2 teacher composite-late 0.518 / clean-late 0.671 -> V1 teacher
0.814 / 0.998.

**Rerun:** all 12 M1 runs (seeds 42, 123 x 6 folds), `open_set_launch.py --seeds 42 123 --arms m1 --stages train tables`,
on one whole B200 (6 at a time), 2026-10-06 08:36-09:00. Old M1 runs kept in `results/open_set_v2/runs_superseded_v2teacher/m1`
(logs in `logs_superseded_v2teacher/`, old table in `TABLES_v2teacher_m1.md`). `TABLES.md` and everything in `b200_results/`
now use the V1 M1. The noise-strength test used only CAAO runs (no M1), so it is unaffected.

## 8. Addendum 2026-10-07: seeds 7, 99, 256, 11, 22, 33, 44, 55 (whole B200)

**Hardware:** one whole NVIDIA B200 (no MIG), UCloud job j-12412980, 48-core CPU quota. Same venv as section 7
(anomalib 2.2.0, V1 WRN teacher for M1), same code (no code change since 540aa2d; Windows commits up to e421549 pulled).

**Command** (`os_8seeds.sh` in the project root, tmux `os8`):
```
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log; nvidia-cuda-mps-control -d
OPEN_SET_ACCEPT_EXISTING=1 python scripts/open_set_launch.py --profile b200 --gen-procs 12 --seeds 7 99 256 11 22 33 44 55
```

**Process settings changed during the run (D8, no effect on settings, seeds, sampling or masks):**
- `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`: torch started 192 CPU threads per process; with 8-12 processes on a 48-core quota the
  job was CPU-throttled in 50 % of scheduler periods. After the cap: 0 throttled periods.
- NVIDIA MPS: kernels of the parallel processes run concurrently instead of time-sliced. Same kernels; outputs expected to
  be identical, **not verified by a byte comparison**.
- `--gen-procs 12` (was 8): generation / refine sharding only; every item has its own seed.
- Effect: generation 55 -> 114 images/min; mask refinement 218 min (MIG) -> 7 min per seed (it was dominated by the same
  thread thrashing). Refined masks checked on seed 7: all 11,829 present, 0 empty in a 300-sample, area distribution
  matching seed 123 (median 0.72 % vs 0.77 %).
- Seed 7: 2,371 of its images had been generated on the MIG job before (2026-10-06, EvenStepDDIM, same settings).
- The launcher was restarted twice at the start of seed 7's generation (09:28, 09:35) to apply the above; images written
  in the minutes before each stop were opened with PIL (0 broken) and generation resumed (existing images skipped).

**Result:** 768 runs (8 seeds x 6 folds x 16 arms), all rc 0, 20 epochs each; launcher `=== launch finished`
2026-10-07 16:03; no failed stage or run since 2026-10-06 (`b200_results/ALL_SEEDS_DONE.md`). Per seed on the whole B200:
generate ~105 min, refine ~7 min, dtd + cutmix ~3 min, 96 trainings ~115 min (8 at a time).

**Totals now:** 10 seeds x 6 folds x 16 arms = 960 runs in `b200_results/open_set_v2/` (TABLES.md, per_epoch_aurocs.csv,
results_json.tar.xz, lists.tar.xz). The M1-vs-rest backbone / mode confound from section 5.1 still applies.

## 9. Addendum 2026-10-09: DTD and CutMix at stage 4 (120 runs)

Order: `scripts/B200_HANDOFF.md`, section "ORDER 2026-10-09". Code: commit e34faa3 (Windows side; ARMS += dtd 4,
cutmix 4; stage_dtd / stage_cutmix also process the s4 sets). **No code change on the server.**

Machine: one whole B200 (UCloud job j-12417651), same venv as sections 7-8 (anomalib 2.2.0).
Command (`os_s4.sh`): OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, NVIDIA MPS, then
```
OPEN_SET_ACCEPT_EXISTING=1 python scripts/open_set_launch.py --profile b200 --seeds 42 123 7 99 256 11 22 33 44 55 \
 --arms dtd_s4 cutmix_s4 --train-slots 12 --stagger 10 --stages plan dtd cutmix check train tables >> launch_s4_dtd_cutmix.out 2>&1
```
`--train-slots 12 --stagger 10` are process settings (12 trainings in parallel; 12 new runs per seed).
The launcher was started 14:05 with the default 8 slots and restarted 14:08 with 12, before any training had
started (seed 42 plan / dtd / cutmix had finished; existing images are skipped).

Manifests: the 10 `results/open_set_v2/manifest/seed_*.json` were NOT rebuilt (`stage_plan` only writes a missing
manifest); md5 checked unchanged after the plan stage; all 10 archived in `b200_results/open_set_v2/manifest.tar.xz`
(commit ac67f71). Preps were reused (plan skips existing meta.json).

Result: 120 new runs (dtd_s4, cutmix_s4 x 6 folds x 10 seeds), all 20 epochs; `TABLES.md` rows:
```
| DTD | 4 | 60 | 0.832 ± 0.122 | 0.923 ± 0.102 | 0.829 ± 0.123 | 0.784 ± 0.094 | +0.056 | 40 of 60 |
| CutMix | 4 | 60 | 0.804 ± 0.142 | 0.891 ± 0.111 | 0.804 ± 0.149 | 0.734 ± 0.150 | +0.028 | 36 of 60 |
| DTD | 4 | 0.907 | 0.929 | 0.856 | 0.962 | 0.668 | 0.668 |
| CutMix | 4 | 0.923 | 0.922 | 0.801 | 0.936 | 0.584 | 0.656 |
```
Failed stages / runs on 10-09: none

`b200_results/open_set_v2/results_json.tar.xz` now holds all 1,080 results.json; TABLES.md, per_epoch_aurocs.csv,
lists.tar.xz, launch.log and launch_s4_dtd_cutmix.out updated. Not verified: that the s4 DTD / CutMix images are
byte-identical to what a Windows run would make (only file counts / run lists were checked by the launcher).
