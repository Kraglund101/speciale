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

## 12. Mix-synthetic (configuration 6), 50 epochs (order 2026-10-15 (2))

60 runs (10 seeds x 6 folds), `scripts/open_set_mix.py` (Windows commit 620829d) with three server changes (this commit):
(1) `build` does not call `run_lists` (it rewrote the shared train/test/epoch lists that the Mode 3 job on another UCloud
job was reading); it reads the existing configuration-1 lists and asserts each equals `fold_epochs()` computed in memory
(all 240 equal; shared list md5 unchanged); (2) runs go to `results/open_set_v2/runs_mix_synthetic_s6_e50/` (`runs/` is
read-only); (3) `--epochs 50` (user decision): the 20-epoch mix lists are kept and the runner cycles the pools
(epoch % 20); since 20 % 4 == 0 the rotation METHODS[(j + e) % 4] continues: per slot each method 12-13 of 50 epochs,
562-563 slot-epochs per method per run. CAAO (Mode 4), no photometric, same train normals / test / seeds as the other arms.
Machine: separate whole B200 (UCloud job j-12422456), anomalib 2.2.0, OMP/MKL 4 threads, MPS, 12 runs in
parallel. Smoke test (seed 42 fold holes, 2 epochs, scratch dir) passed (EpochPoolSampler 20 pools) and was deleted.

Result: 60 runs, all 50 epochs. Failed runs: none. Packed: `results_json_mix_s6_e50.tar.xz` (60 results.json +
60 epochs_mix_s6.json), `launch_mix_s6.out`, `launch_mix_s6_lines.log`. No table here (50 epochs; compare with the
50-epoch CAAO runs of section 11 for seeds 42/123/7, or with the 20-epoch table only with that caveat).

## 13. Score dump, configuration 1, all 10 seeds (order 2026-10-15 (3), extended to 10 seeds by Frederik)

300 runs (m1, cutmix_s1, dtd_s1, diffusion_in_s1, diffusion_cross_s1 x 6 folds x 10 seeds), `scripts/open_set_scores.py` +
`score_runner.py` + `open_set_score_hist.py` (Windows commit b844fd3) with two server changes (this commit): (1)
`open_set_scores.command` uses a read-only `run_lists` (asserts every list on disk equals what run_lists would write;
no list is rewritten, because the Mode 3 job on another UCloud job reads them); (2) `score_runner` also stores the test
image paths per entry (`files`, from the shuffle=False test dataset; `files_mismatch` true if count / labels disagree).
Machine: whole B200 (UCloud job
j-12422456), anomalib 2.2.0, OMP/MKL 4 threads, MPS, 12 runs in parallel. Smoke test (dtd_s1, fold holes, 2
epochs) passed: epochs 0/1 x main/0x-4x, 70 scores each, EpochPoolSampler 20 pools; a second smoke test (seed 7,
diffusion_cross_s1, fold scratches) checked the `files` field (65 paths, 0 mismatches); both deleted before the full run.
All 1,260 run-list files byte-identical before and after (md5). `runs/` untouched (read-only).

All 300 runs 'scores ok'; `open_set_score_hist.py` AUROC check passed (it asserts recomputed AUROC == results.json).
Reproduction check (rerun vs original `runs/<arm>/seed_<s>/fold_<k>/results.json`, same seed and command):

Seed 42, best-0x rerun / original:

| arm | holes | breakage | burnt | colour_diff | colour_same | scratches |
|---|---|---|---|---|---|---|
| m1 | 0.999 / 0.999 | 0.998 / 0.998 | 0.962 / 0.962 | 1.000 / 1.000 | 0.724 / 0.724 | 0.916 / 0.916 |
| cutmix_s1 | 0.999 / 1.000 | 0.971 / 0.995 | 0.980 / 0.960 | 1.000 / 1.000 | 0.758 / 0.766 | 0.869 / 0.903 |
| dtd_s1 | 1.000 / 1.000 | 0.998 / 0.997 | 0.990 / 0.978 | 1.000 / 1.000 | 0.834 / 0.822 | 0.893 / 0.869 |
| diffusion_in_s1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.998 | 1.000 / 1.000 | 0.870 / 0.860 | 0.795 / 0.885 |
| diffusion_cross_s1 | 1.000 / 1.000 | 1.000 / 0.996 | 1.000 / 0.996 | 1.000 / 1.000 | 0.844 / 0.928 | 0.899 / 0.887 |

All 10 seeds, best-0x rerun minus original (60 runs per arm): mean / mean |diff| / max |diff|:

- m1: +0.0000 / 0.0000 / 0.0000
- cutmix_s1: +0.0007 / 0.0033 / 0.0920
- dtd_s1: +0.0021 / 0.0022 / 0.0467
- diffusion_in_s1: +0.0000 / 0.0030 / 0.0907
- diffusion_cross_s1: -0.0011 / 0.0019 / 0.0840

Packed: `score_dist_c1.tar.xz` (300 x results.json + image_scores.json), `figures/score_hist_c1/seed_<s>_<type>.png` (60),
`launch_scores_c1.out`.

Note (section 13): the image paths are stored under the key **`files`** (plus `files_mismatch`), NOT `images` as in Windows
commit a696235, which was pushed while these runs were training. The server version of `score_runner.py` (the one that
produced the 300 dumps) is kept in this commit. Checked after the run: all 64,800 entries of the 300 `image_scores.json`
have `files`, 0 mismatches (count and labels agree with the shuffle=False test dataset).

## 10. Mode 3 repeat (ORDER 2026-10-15)

All 17 synthetic arms x 6 folds x 10 seeds = 1,020 runs with `--hero mode3` (`--modes 3 --lkd-mode local --backbone
convnext_b`, code 8bbde60 from the Windows side; no server-side code change), results in `results/open_set_v2/runs_mode3/`.
Same run lists / images / seeds as the CAAO runs; no generation, no manifest rebuild (10 manifests byte-identical to
the 2026-10-09 backup). Runner sha256 0110be259b605cc049748a09fb2d3891aaeff5d056d07dd3d2d7a4e2b9079fa0 (= Windows copy), anomalib 2.2.0, torch 2.7.1+cu128, whole B200.
Process settings as section 9 (OMP/MKL 4 threads, MPS, 12 trainings in parallel, stagger 10 s).

Stop-gradient: Mode 3 uses UniNetLossControllable; it never reads `no_detach_any`, and with the default (no
`--stop-gradient`) the KD target is not detached (`tf_n_kd = tf_n`) - i.e. no detaching; the frozen source teacher runs
under no_grad in Mode 3 and Mode 4 alike. Checked in the runner source, not by a gradient test.

Bootstrap: commit bca9fe9 had mode3 with --backbone wrn50; a 2-epoch smoke test with it was killed and deleted before
finishing; after 8bbde60 the smoke test (seed 42, fold holes, cutmix_s1 + diffusion_in_s1, 2 epochs) passed and was
deleted before the full run.

Result: 1,020 runs, all 20 epochs, results key "Mode 3 ...". Failed stages / runs: none

Packed: `b200_results/open_set_v2/results_json_mode3.tar.xz` (1,020 results.json), `launch_mode3.out`,
`launch_mode3_lines.log`. `open_set_tables.py` reads `runs/` only, so no Mode 3 table was made here.
The automatic push (finish_mode3.sh) failed at 22:01: a script edit had dropped the line that copies `launch_mode3.out`,
so `git add` failed and nothing was committed. Verification had passed; the files were committed and pushed by hand.

## 11. Mode 4 (CAAO) with 50 epochs, seeds 42 / 123 / 7 (order 2026-10-15)

All 18 arms (m1 + 17 CAAO synthetic arms) x 6 folds x seeds 42, 123, 7 = 324 runs, 50 epochs instead of 20, otherwise
the same settings, run lists and images as section 3 / 8 / 9. Results in `results/open_set_v2/runs_e50/` (the 20-epoch
`runs/` and `runs_mode3/` were not touched). Code change (server, this commit): `open_set_launch.py --train-epochs N` /
`OPEN_SET_TRAIN_EPOCHS`: passes `--epochs N` to the runner, writes to `runs[_<hero>]_e<N>/`, logs
`train_<arm>_e<N>_seed<s>_<k>.log`, skips the tables stage. The runner cycles the 20 per-epoch synthetic pools
(`EpochPoolSampler`: pool = epoch % 20), so epochs 21-50 reuse pools 1-20 - no new images. M1 also ran 50 epochs.
LR schedule and other epoch-dependent behaviour follow the runner's handling of --epochs (not changed, not checked).

Protection: before the run all finished 20-epoch results.json (runs/, runs_mode3/) were made read-only (chmod a-w) and
their md5 recorded; after the run all 2100 were byte-identical (checked by finish_e50.sh).

Table: `TABLES_e50.md` (open_set_tables.py code on runs_e50/, 'late' = epochs 45-50, paired with the 50-epoch M1).
```
| arm | stage | runs | composite, late | clean (0x), late | composite, last | strong blur (3x-4x), late | composite vs M1 (paired) | runs better than M1 |
|---|---|---|---|---|---|---|---|---|
| normal-only (M1) | – | 18 | 0.684 ± 0.064 | 0.941 ± 0.075 | 0.687 ± 0.072 | 0.467 ± 0.116 | – | – |
| target-object | 1 | 18 | 0.773 ± 0.165 | 0.829 ± 0.176 | 0.758 ± 0.161 | 0.724 ± 0.148 | +0.088 | 11 of 18 |
| cross-object | 1 | 18 | 0.791 ± 0.155 | 0.827 ± 0.166 | 0.800 ± 0.150 | 0.743 ± 0.132 | +0.107 | 12 of 18 |
| DTD | 1 | 18 | 0.713 ± 0.144 | 0.817 ± 0.193 | 0.712 ± 0.139 | 0.660 ± 0.095 | +0.029 | 10 of 18 |
| CutMix | 1 | 18 | 0.760 ± 0.163 | 0.817 ± 0.166 | 0.740 ± 0.160 | 0.744 ± 0.144 | +0.076 | 13 of 18 |
| target-object | 2 | 18 | 0.803 ± 0.157 | 0.848 ± 0.169 | 0.802 ± 0.150 | 0.741 ± 0.151 | +0.119 | 12 of 18 |
| cross-object | 2 | 18 | 0.824 ± 0.139 | 0.852 ± 0.153 | 0.818 ± 0.137 | 0.765 ± 0.136 | +0.140 | 14 of 18 |
| DTD | 2 | 18 | 0.732 ± 0.147 | 0.845 ± 0.176 | 0.721 ± 0.155 | 0.668 ± 0.114 | +0.048 | 10 of 18 |
| CutMix | 2 | 18 | 0.803 ± 0.159 | 0.842 ± 0.146 | 0.802 ± 0.162 | 0.764 ± 0.152 | +0.118 | 13 of 18 |
| target-object | 3 | 18 | 0.800 ± 0.157 | 0.854 ± 0.169 | 0.801 ± 0.155 | 0.737 ± 0.146 | +0.116 | 12 of 18 |
| cross-object | 3 | 18 | 0.816 ± 0.142 | 0.834 ± 0.165 | 0.809 ± 0.143 | 0.757 ± 0.139 | +0.132 | 14 of 18 |
| DTD | 3 | 18 | 0.714 ± 0.154 | 0.835 ± 0.181 | 0.724 ± 0.142 | 0.646 ± 0.120 | +0.030 | 9 of 18 |
| CutMix | 3 | 18 | 0.803 ± 0.166 | 0.836 ± 0.157 | 0.793 ± 0.172 | 0.762 ± 0.165 | +0.119 | 12 of 18 |
| target-object | 4 | 18 | 0.800 ± 0.158 | 0.843 ± 0.175 | 0.802 ± 0.156 | 0.736 ± 0.148 | +0.116 | 12 of 18 |
| cross-object | 4 | 18 | 0.817 ± 0.145 | 0.843 ± 0.163 | 0.814 ± 0.142 | 0.760 ± 0.138 | +0.133 | 13 of 18 |
| DTD | 4 | 18 | 0.727 ± 0.146 | 0.835 ± 0.198 | 0.709 ± 0.147 | 0.652 ± 0.113 | +0.043 | 10 of 18 |
| CutMix | 4 | 18 | 0.800 ± 0.164 | 0.841 ± 0.137 | 0.797 ± 0.166 | 0.757 ± 0.167 | +0.116 | 14 of 18 |
| cross-object | 5 | 18 | 0.813 ± 0.151 | 0.842 ± 0.169 | 0.806 ± 0.152 | 0.759 ± 0.141 | +0.129 | 12 of 18 |

```
Failed stages / runs: none. Packed: `results_json_e50.tar.xz` (324), `launch_e50.out`, `launch_e50_lines.log`.
