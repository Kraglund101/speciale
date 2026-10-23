# Where the time goes: one full seed on one B200 (measured 2026-10-21/22, seed 66)

Setup: one whole B200, 48-core CPU quota, OMP/MKL 4 threads per process, NVIDIA MPS, 12 generation processes, 12
trainings in parallel - the same as the 10-seed run. Input files pre-read once (a fresh UCloud job reads cold files
at ~3.5 s each from WekaFS; with a warm cache the stages ran at the same speed as in the 10-seed run).

## 1. Pipeline stages (launch.log, wall time)

| stage | wall time | what it does |
|---|---|---|
| plan | 3.5 min | manifest: canvases, budget anomalies, placements, donor pairing |
| generate | **109.9 min** | ~11,900 diffusion images, 12 processes, ~111 images/min |
| refine | 7.3 min | refined label masks (DINOv3) |
| dtd | 2.0 min | DTD texture anomalies (CPU) |
| cutmix | 3.3 min | CutMix anomalies (CPU) |
| check | 1.4 min | consistency checks |
| **prep total** | **~2.1 h** | |
| train | ~1.6 h (estimate) | 108 runs x ~10.5 min, 12 in parallel (= 9 rounds) |
| **seed total** | **~3.7 h** | |

The train-stage profile of this seed was lost (the job ended at ~02:05 before py-spy wrote its file; 77/108 runs done).
Per-run times below come from (a) the clean 12-parallel runs of 2026-10-09/15 (9.7-11.6 min per 20-epoch Mode 4 run)
and (b) an isolated, instrumented 3-epoch run on the same B200 (2026-10-21).

## 2. Inside generation (py-spy, 20 Hz, generation processes, main threads)

| part | share |
|---|---|
| denoising loop: UNet forward (Python side: launching the layers' GPU kernels) | 36 % |
| denoising loop: scheduler `step` = waiting for the GPU (`int(timestep)` forces a CUDA sync) | 47 % |
| saving PNGs (PIL) | 4 % |
| IP-Adapter image encoding | 2 % |
| model loading at start | 3.5 % |
| rest (masks, crops, scheduler setup, ...) | ~7 % |

So ~83 % of generation is the 50-step denoising loop itself (GPU-bound). Note: py-spy sees CPU stacks; with CUDA's
asynchronous execution, GPU time shows up as waiting at the next synchronisation point.

## 3. Inside one training run (Mode 4, 20 epochs)

Isolated instrumented run (timers around train_one_epoch / evaluate; 3 epochs, extrapolated to 20):

| part | per epoch | per run (20 epochs) | share |
|---|---|---|---|
| startup (data + model build) | - | ~10 s | ~2 % |
| training step (100 normals + 45 synthetics) | 8.7 s | ~174 s | ~28 % |
| evaluation: clean + 5 shift levels (6 passes x 2.8 s) | 17 s | ~340 s | **~55 %** |
| end of run: discrepancy-map figures (~84 s) + checkpoint save | - | ~96 s | ~15 % |
| **total** | | **~10.3 min** | |

With 12 runs in parallel each run also took ~10 min, i.e. parallelism costs little per run - the B200 gains by running
12 at once. Evaluation (every epoch, 6 test variants) is the largest single cost; the figures at the end are a fixed
~1.5 min per run.

## 4. GPU / CPU

- Generation: GPU ~700 W, SM ~100 %, memory bandwidth ~20 %: GPU-bound.
- Training (clean, 12 parallel): GPU ~450-520 W, SM 80-100 %: GPU partly idle while runs evaluate on CPU-side code.
- CPU: without the 4-thread cap torch started 192 threads per process and the quota was throttled 50 % of the time
  (2026-10-06); with the cap, ~0 %.

## 5. Profiler overhead (lesson)

py-spy with --subprocesses samples every process: at 20 Hz over the ~230 training processes (12 trainers + their
data-loader workers) it used ~40 of 48 cores and slowed training ~2.5x; at 2 Hz ~12 cores; at 1 Hz runs still took
~13 min instead of ~10. For training, profile one run at a time (as in section 3) rather than all 12.

Files: `PROFILE_SUMMARY.md` (auto summary), `pyspy_prep.txt.xz` (collapsed stacks, open in speedscope.app),
`launch_seed66_lines.log`, `timeline.log`, `gpu_dmon.log`, `cpu_stat.log`.
