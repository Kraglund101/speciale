# Profile of one full seed (66) on the B200

## Timeline
```
2026-10-21 21:18:12 start prep stages (plan generate refine dtd cutmix check)
2026-10-21 23:25:45 prep stages done; start train stage (108 runs)
2026-10-22 01:32:00 first train stage stopped at 53/108 runs: py-spy at 20 Hz over ~230 processes used ~40 of 48 cores (CPU throttled every period, GPU 15-23 %) - timings of those 53 runs are distorted (results valid)
Created autostash: 2184112
Applied autostash.
2026-10-22 01:32:06 PUSHED 7d1bc1e Seed-66 profile, part 1: prep stages (plan..check) profile + stage times (training restarting at 2 Hz)
2026-10-22 01:32:06 restart train stage (remaining runs), py-spy 2 Hz
2026-10-22 01:35:08 first train stage stopped at 53/108 runs: py-spy at 20 Hz over ~230 processes used ~40 of 48 cores (CPU throttled every period, GPU 15-23 %) - timings of those 53 runs are distorted (results valid)
```

## Stage wall times (launch.log)
```
10-21 20:22  seed 66 plan: rc [1] in 0.1 min
10-21 20:22  seed 66: STOP, plan failed (see /work/Jobs/data/other/speciale/results/open_set_v2/logs)
10-21 20:22  seed 66: training 108 runs, 12 at a time
10-21 21:21  seed 66 plan: rc [0] in 3.5 min
10-21 23:11  seed 66 generate: rc [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] in 109.9 min
10-21 23:18  seed 66 refine: rc [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] in 7.3 min
10-21 23:20  seed 66 dtd: rc [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] in 2.0 min
10-21 23:24  seed 66 cutmix: rc [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] in 3.3 min
10-21 23:25  seed 66 check: rc [0] in 1.4 min
10-21 23:25  seed 66: training 108 runs, 12 at a time
10-22 01:32  seed 66: training 108 runs, 12 at a time
```

Trainings: 53 runs rc 0, wall per run mean 25.0 min (min 12.5, max 33.4); 12 in parallel; runs of the 1 Hz training restart only.

## py-spy: prep stages (24.91 h process-time sampled at 20 Hz, idle included, main threads only)

| process type | processes | process-time | share |
|---|---|---|---|
| stage generate | 12 | 20.24 h | 81.2 % |
| launcher (open_set_launch.py) | 2 | 2.14 h | 8.6 % |
| stage refine | 12 | 1.43 h | 5.8 % |
| stage cutmix | 12 | 37.9 min | 2.5 % |
| stage dtd | 12 | 23.5 min | 1.6 % |
| stage plan | 1 | 3.3 min | 0.2 % |
| stage check | 1 | 1.5 min | 0.1 % |

## CPU (cgroup, 48-core quota)

Mean CPU use 29.4 cores; throttled in 74275 of 151851 scheduler periods.

## GPU (nvidia-smi dmon, every 10 s)

Power mean 477 W (cap 1000 W); SM activity mean 64 %; memory-bandwidth activity mean 12 %; 1518 samples.

Raw profiles: `pyspy_prep.txt`, `pyspy_train.txt` (collapsed stacks; open in https://www.speedscope.app or
any flamegraph tool). Process-time = main-thread samples / rate (prep 20 Hz, train 1 Hz) per process (secondary threads excluded), so 12 parallel trainings add up to ~12x wall time.
