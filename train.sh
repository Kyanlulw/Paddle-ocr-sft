#!/bin/bash
# =============================================================================
# PaddleOCR-VL training launcher with Accelerate.
# Supports both Manga109 and mixed30k backends.
# =============================================================================

set -euo pipefail

# Optional: export WANDB_API_KEY="your-wandb-api-key"
# Optional: export CUDA_VISIBLE_DEVICES=0,1

DATASET_BACKEND="${DATASET_BACKEND:-${1:-manga109}}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
REPORT_TO="${REPORT_TO:-wandb}"

COMMON_ARGS=(
    --model_path PaddlePaddle/PaddleOCR-VL
    --max_length 1536
    --pad_to_multiple_of 8
    --learning_rate 2e-5
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --weight_decay 0.01
    --max_grad_norm 1.0
    --logging_steps 10
    --save_strategy steps
    --save_steps 2000
    --save_total_limit 3
    --dataloader_num_workers 2
    --gradient_checkpointing
    --ddp_find_unused_parameters false
    --optim adamw_torch
    --report_to "${REPORT_TO}"
)

case "${DATASET_BACKEND}" in
    manga109)
        RUN_ARGS=(
            --run_name "PaddleOCR-VL-Manga109s"
            --wandb_project "paddleocr-vl-sft"
            --wandb_tags "manga109,t4x2,bf16"
            --dataset_backend manga109
            --split train
            --output_dir ./sft_output
            --num_train_epochs 1
            --per_device_train_batch_size 2
            --gradient_accumulation_steps 8
            --eval_strategy steps
            --eval_steps 500
            --per_device_eval_batch_size 2
        )
        ;;
    mixed30k)
        TRAIN_ANNOTATION_PATH="${TRAIN_ANNOTATION_PATH:-./mixed30k.json}"
        DATASET1_IMAGE_ROOT="${DATASET1_IMAGE_ROOT:-/path/to/dataset1/images}"
        DATASET2_IMAGE_ROOT="${DATASET2_IMAGE_ROOT:-/path/to/dataset2/images}"
        DATASET2_METADATA_PATH="${DATASET2_METADATA_PATH:-/path/to/openvivqa_train_v2.json}"
        EVAL_ANNOTATION_PATH="${EVAL_ANNOTATION_PATH:-}"

        RUN_ARGS=(
            --run_name "PaddleOCR-VL-mixed30k"
            --wandb_project "paddleocr-vl-sft"
            --wandb_tags "mixed30k,t4x2,bf16"
            --dataset_backend mixed30k
            --train_annotation_path "${TRAIN_ANNOTATION_PATH}"
            --dataset1_image_root "${DATASET1_IMAGE_ROOT}"
            --dataset2_image_root "${DATASET2_IMAGE_ROOT}"
            --dataset2_metadata_path "${DATASET2_METADATA_PATH}"
            --output_dir ./sft_output_mixed30k
            --num_train_epochs 3
            --per_device_train_batch_size 1
            --gradient_accumulation_steps 16
            --eval_strategy no
        )

        if [[ -n "${EVAL_ANNOTATION_PATH}" ]]; then
            RUN_ARGS+=(
                --eval_annotation_path "${EVAL_ANNOTATION_PATH}"
                --eval_strategy steps
                --eval_steps 500
                --per_device_eval_batch_size 1
            )
        fi
        ;;
    *)
        echo "Unsupported DATASET_BACKEND: ${DATASET_BACKEND}" >&2
        echo "Expected 'manga109' or 'mixed30k'." >&2
        exit 1
        ;;
esac

accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    sft_paddleocr_vl.py \
    "${COMMON_ARGS[@]}" \
    "${RUN_ARGS[@]}"
