#!/bin/bash
# =============================================================================
# PaddleOCR-VL Fine-tuning Script for mixed30k on RTX 3060 (12GB VRAM)
#
# Update the three dataset paths below before running.
# Evaluation is disabled by default; add --eval_annotation_path if you have one.
# =============================================================================

python sft_paddleocr_vl.py \
    --run_name "PaddleOCR-VL-mixed30k" \
    --model_path PaddlePaddle/PaddleOCR-VL \
    --dataset_backend mixed30k \
    --train_annotation_path /kaggle/input/datasets/trnlqung/qwen-scored-9300samples/unsorted/mixed30k.json \
    --dataset1_image_root /kaggle/input/datasets/trnlqung/vitext-vqa/ViTextVQA_images/st_images \
    --dataset2_image_root /kaggle/input/datasets/trnlqung/openvivqa/images/images \
    --max_length 1536 \
    --pad_to_multiple_of 8 \
    --output_dir ./sft_output_mixed30k \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --dataloader_num_workers 2 \
    --gradient_checkpointing \
    --optim adamw_torch \
    --report_to none
