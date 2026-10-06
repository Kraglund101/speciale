# Numbers from the B200 runs (2026-10-03 to 06)

Copied here (outside `results/`) so a pull cannot collide with local result folders. Report: `scripts/B200_RUN_REPORT.md`.

- `open_set_v2/`: seeds 42 and 123, 6 folds x 16 arms = 192 runs.
  `TABLES.md` (open_set_tables.py), `per_epoch_aurocs.csv` (every run x shift level x epoch: image and pixel AUROC, extracted
  from results.json), `results_json.tar.xz` (all 192 results.json, unpack at the project root -> results/open_set_v2/runs/...),
  `lists.tar.xz` (exact train / test / synthetic lists every run used; absolute server paths), `manifest.tar.xz`,
  `launch.log`, `launch_full.out`, `generator_settings.json`, `IMAGE_PROVENANCE.md` + `images_made_with_ffe1fea_scheduler.txt`.
- `noise_strength_test/`: 75 runs (14 arms x 5 seeds + arm 070 x 5). `TABLES.md`, `progress.log`, `per_epoch_aurocs.csv`,
  `results_json.tar.xz` (unpacks to results/noise_strength_test/runs/...).

**2026-10-06:** M1 rerun with the V1 Wide-ResNet teacher (anomalib 2.2.0); see `scripts/B200_RUN_REPORT.md` section 7.
`open_set_v2/TABLES.md`, `results_json.tar.xz` and `per_epoch_aurocs.csv` now contain the V1 M1. The superseded V2-teacher
M1 runs: `results_json_m1_superseded_v2teacher.tar.xz`, table `TABLES_v2teacher_m1.md`. M1 rerun log: `launch_m1_v1.out`.
