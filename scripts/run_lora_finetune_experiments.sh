#!/bin/bash
# LoRA + fine-tuning experiments for CLIP consistency training
# Run after clip_v5_* experiments complete

PY=/c/Users/frede/anaconda3/python.exe
SCRIPT=scripts/clip_consistency_training.py
BASE_ARGS="--captions-file anomverse_extension/datasets/full_training_dataset/captions_from_master.json \
  --splits-dir anomverse_extension/datasets/full_training_dataset/splits_by_type \
  --data-root anomverse_extension/datasets/full_training_dataset \
  --batch-size 8 --steps 500 --save-every 500 \
  --augment --band-mode 2 \
  --pilot-subset-size 500 \
  --clip-gamma 1.0 \
  --no-live-viewer"

# ── LoRA rank=8, mid+up ──
echo "=== LoRA rank=8, cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 8 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora8_cutoff_scale1

echo "=== LoRA rank=8, no cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 8 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora8_nocutoff_scale1

echo "=== LoRA rank=8, cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 8 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora8_cutoff_scale2

echo "=== LoRA rank=8, no cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 8 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora8_nocutoff_scale2

# ── LoRA rank=16, mid+up ──
echo "=== LoRA rank=16, cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora16_cutoff_scale1

echo "=== LoRA rank=16, no cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora16_nocutoff_scale1

echo "=== LoRA rank=16, cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora16_cutoff_scale2

echo "=== LoRA rank=16, no cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 16 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora16_nocutoff_scale2

# ── LoRA rank=32, mid+up ──
echo "=== LoRA rank=32, cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora32_cutoff_scale1

echo "=== LoRA rank=32, no cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_lora32_nocutoff_scale1

echo "=== LoRA rank=32, cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora32_cutoff_scale2

echo "=== LoRA rank=32, no cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --lora-rank 32 --lora-mode mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_lora32_nocutoff_scale2

# ── Fine-tune mid+up (unfreeze W_Q/W_O) ──
echo "=== Finetune mid_up, cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --unfreeze-qo mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 1.0 --save-dir results/clip_v5_finetune_cutoff_scale1

echo "=== Finetune mid_up, no cutoff, scale=1 ==="
$PY $SCRIPT $BASE_ARGS --unfreeze-qo mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 1.0 --save-dir results/clip_v5_finetune_nocutoff_scale1

echo "=== Finetune mid_up, cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --unfreeze-qo mid_up --clip-alpha-cutoff 0.5 --ip-adapter-scale 2.0 --save-dir results/clip_v5_finetune_cutoff_scale2

echo "=== Finetune mid_up, no cutoff, scale=2 ==="
$PY $SCRIPT $BASE_ARGS --unfreeze-qo mid_up --clip-alpha-cutoff 0   --ip-adapter-scale 2.0 --save-dir results/clip_v5_finetune_nocutoff_scale2

echo "=== ALL DONE ==="
