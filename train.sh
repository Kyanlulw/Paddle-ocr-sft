#!/bin/bash
# =============================================================================
# PaddleOCR-VL Fine-tuning Script for Manga109 on 2x T4 GPUs with Accelerate
# =============================================================================

# Optional: export WANDB_API_KEY="your-wandb-api-key"
# Optional: export CUDA_VISIBLE_DEVICES=0,1

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    sft_paddleocr_vl.py \
    --run_name "PaddleOCR-VL-Manga109s" \
    --wandb_project "paddleocr-vl-sft" \
    --wandb_tags "manga109,t4x2,bf16" \
    --model_path PaddlePaddle/PaddleOCR-VL \
    --dataset_backend manga109 \
    --split train \
    --max_length 1536 \
    --pad_to_multiple_of 8 \
    --output_dir ./sft_output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --per_device_eval_batch_size 2 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --optim adamw_torch \
    --report_to wandb
