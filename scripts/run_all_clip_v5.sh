#!/bin/bash
set -e
cd /c/Users/frede/desktop/kandidat/speciale

PY=/c/Users/frede/anaconda3/python.exe
S=scripts/clip_consistency_training.py

run() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "============================================"
    shift
    $PY $S \
      --captions-file anomverse_extension/datasets/full_training_dataset/captions_from_master.json \
      --splits-dir anomverse_extension/datasets/full_training_dataset/splits_by_type \
      --data-root anomverse_extension/datasets/full_training_dataset \
      --batch-size 8 --steps 500 --save-every 500 \
      --augment --band-mode 2 \
      --pilot-subset-size 500 \
      --clip-gamma 1.0 \
      --diag-interval 1 \
      --no-live-viewer \
      "$@"
}

# Baseline (4 runs)
run "1/20  baseline cutoff=0.5 scale=1"   --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_cutoff_scale1
run "2/20  baseline cutoff=0   scale=1"   --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_nocutoff_scale1
run "3/20  baseline cutoff=0.5 scale=2"   --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_cutoff_scale2
run "4/20  baseline cutoff=0   scale=2"   --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_nocutoff_scale2

# LoRA rank=8 mid+up (4 runs)
run "5/20  lora8 cutoff=0.5 scale=1"      --lora-rank 8  --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora8_cutoff_scale1
run "6/20  lora8 cutoff=0   scale=1"      --lora-rank 8  --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora8_nocutoff_scale1
run "7/20  lora8 cutoff=0.5 scale=2"      --lora-rank 8  --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora8_cutoff_scale2
run "8/20  lora8 cutoff=0   scale=2"      --lora-rank 8  --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora8_nocutoff_scale2

# LoRA rank=16 mid+up (4 runs)
run "9/20  lora16 cutoff=0.5 scale=1"     --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora16_cutoff_scale1
run "10/20 lora16 cutoff=0   scale=1"     --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora16_nocutoff_scale1
run "11/20 lora16 cutoff=0.5 scale=2"     --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora16_cutoff_scale2
run "12/20 lora16 cutoff=0   scale=2"     --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora16_nocutoff_scale2

# LoRA rank=32 mid+up (4 runs)
run "13/20 lora32 cutoff=0.5 scale=1"     --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora32_cutoff_scale1
run "14/20 lora32 cutoff=0   scale=1"     --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora32_nocutoff_scale1
run "15/20 lora32 cutoff=0.5 scale=2"     --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora32_cutoff_scale2
run "16/20 lora32 cutoff=0   scale=2"     --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora32_nocutoff_scale2

# Finetune W_Q/W_O mid+up (4 runs)
run "17/20 finetune cutoff=0.5 scale=1"   --unfreeze-qo mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_finetune_cutoff_scale1
run "18/20 finetune cutoff=0   scale=1"   --unfreeze-qo mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_finetune_nocutoff_scale1
run "19/20 finetune cutoff=0.5 scale=2"   --unfreeze-qo mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_finetune_cutoff_scale2
run "20/20 finetune cutoff=0   scale=2"   --unfreeze-qo mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_finetune_nocutoff_scale2

echo ""
echo "=== ALL 20 RUNS COMPLETE ==="
