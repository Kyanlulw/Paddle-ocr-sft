"""
Supervised Fine-Tuning (SFT) script for PaddleOCR-VL on Manga109s dataset.

This script fine-tunes PaddleOCR-VL for Japanese manga OCR using BF16 precision.
Optimized for RTX 3060 (12GB VRAM) but works on any GPU supporting BF16.

Usage:
    python sft_paddleocr_vl.py --help
    bash train.sh
"""

import gc
from dataclasses import dataclass, field
import os
from typing import Optional

import torch
import transformers as transformers_module

try:
    from accelerate.utils import memory as accelerate_memory

    if not hasattr(accelerate_memory, "clear_device_cache"):
        def _clear_device_cache_compat(garbage_collection: bool = False) -> None:
            if garbage_collection:
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accelerate_memory.clear_device_cache = _clear_device_cache_compat
except ImportError:
    pass

if not hasattr(transformers_module, "EncoderDecoderCache"):
    _dynamic_cache_base = getattr(transformers_module, "DynamicCache", object)

    class EncoderDecoderCache(_dynamic_cache_base):
        """Compatibility shim for older Transformers releases."""

        pass

    transformers_module.EncoderDecoderCache = EncoderDecoderCache

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from custom_collator import CustomDataCollatorForVisionLanguageModeling
from ocr_dataset import MangaDataset, Mixed30kDataset


class BF16Trainer(Trainer):
    """
    Custom Trainer using BF16 autocast without GradScaler.
    
    RTX 3060 and newer GPUs support BF16 which has better numerical stability 
    than FP16 and doesn't require loss scaling. This trainer wraps the training
    and prediction steps with torch.amp.autocast for BF16 computation.
    """
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to use BF16 autocast."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to use BF16 autocast during evaluation."""
        model.eval()
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = self.compute_loss(model, inputs)
                return (loss, None, None)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_path: str = field(
        default="PaddlePaddle/PaddleOCR-VL",
        metadata={"help": "Path to the PaddleOCR-VL model (HuggingFace ID or local path)"}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable Flash Attention 2 for faster training. "
                "Requires flash-attn package and A100/H100 GPU. "
                "Default False for RTX 3060 compatibility."
            )
        },
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""

    dataset_backend: str = field(
        default="manga109",
        metadata={
            "help": (
                "Dataset backend to use: 'manga109' for OCR crops or "
                "'mixed30k' for two-source VQA data."
            )
        },
    )
    split: str = field(
        default="train",
        metadata={"help": "Dataset split for Manga109 training: 'train'."},
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "Dataset split for Manga109 evaluation: 'test'."},
    )
    train_annotation_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the training annotation file. Required when "
                "dataset_backend='mixed30k'."
            )
        },
    )
    eval_annotation_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional path to the evaluation annotation file for mixed30k."
            )
        },
    )
    dataset1_image_root: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Root directory for dataset1 images when "
                "dataset_backend='mixed30k'."
            )
        },
    )
    dataset2_image_root: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Root directory for dataset2 images when "
                "dataset_backend='mixed30k'."
            )
        },
    )
    dataset2_metadata_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional OpenViVQA metadata JSON used to validate or recover "
                "dataset2 filenames."
            )
        },
    )
    max_length: int = field(
        default=1536,
        metadata={
            "help": (
                "Maximum sequence length (image + text tokens). "
                "PaddleOCR-VL images use 400-2000+ tokens depending on size."
            )
        },
    )
    eval_limit_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Limit eval dataset size to reduce memory usage."},
    )
    skip_packages: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of synthetic data package IDs to skip"},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=8,
        metadata={"help": "Pad sequence length to multiple of this value for GPU efficiency."},
    )


@dataclass
class TrackingArguments:
    """Arguments for experiment tracking integrations."""

    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases entity or team name."},
    )
    wandb_mode: str = field(
        default="online",
        metadata={"help": "Weights & Biases mode: online, offline, or disabled."},
    )
    wandb_tags: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated Weights & Biases tags."},
    )
    wandb_notes: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Weights & Biases run notes."},
    )
    wandb_watch: str = field(
        default="false",
        metadata={
            "help": "Weights & Biases model watch mode: false, gradients, or all."
        },
    )
    wandb_log_model: str = field(
        default="false",
        metadata={
            "help": "Weights & Biases model logging mode: false, end, or checkpoint."
        },
    )


def _get_eval_strategy(training_args: TrainingArguments) -> str:
    """Normalize eval strategy across Transformers versions."""
    strategy = getattr(
        training_args,
        "eval_strategy",
        getattr(training_args, "evaluation_strategy", "no"),
    )
    if hasattr(strategy, "value"):
        return strategy.value
    return str(strategy)


def _build_dataset(
    data_args: DataArguments,
    *,
    is_eval: bool,
    skip_packages,
):
    """Instantiate the configured dataset backend."""
    backend = data_args.dataset_backend.lower()

    if backend == "manga109":
        return MangaDataset(
            split=data_args.eval_split if is_eval else data_args.split,
            limit_size=data_args.eval_limit_size if is_eval else None,
            augment=False,
            skip_packages=skip_packages,
        )

    if backend != "mixed30k":
        raise ValueError(
            f"Unsupported dataset_backend={data_args.dataset_backend!r}. "
            "Expected 'manga109' or 'mixed30k'."
        )

    annotation_path = (
        data_args.eval_annotation_path if is_eval else data_args.train_annotation_path
    )
    if not annotation_path:
        if is_eval:
            return None
        raise ValueError(
            "train_annotation_path is required when dataset_backend='mixed30k'."
        )

    return Mixed30kDataset(
        annotation_path=annotation_path,
        dataset1_image_root=data_args.dataset1_image_root,
        dataset2_image_root=data_args.dataset2_image_root,
        dataset2_metadata_path=data_args.dataset2_metadata_path,
        limit_size=data_args.eval_limit_size if is_eval else None,
        validate_source_split=not is_eval,
    )


def _report_to_includes_wandb(training_args: TrainingArguments) -> bool:
    """Return True when Weights & Biases logging is enabled."""
    report_to = getattr(training_args, "report_to", None)

    if report_to is None:
        return False

    if isinstance(report_to, str):
        normalized = report_to.strip().lower()
        if normalized in {"", "none"}:
            return False
        return normalized in {"wandb", "all"}

    if isinstance(report_to, (list, tuple, set)):
        normalized = {str(item).strip().lower() for item in report_to}
        return "wandb" in normalized or "all" in normalized

    return False


def _setup_wandb(
    training_args: TrainingArguments,
    tracking_args: TrackingArguments,
):
    """Initialize Weights & Biases before Trainer callbacks attach."""
    if not _report_to_includes_wandb(training_args):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb tracking was requested via --report_to wandb, but the "
            "package is not installed. Run `pip install -r requirements.txt` "
            "or `pip install wandb`."
        ) from exc

    os.environ["WANDB_WATCH"] = tracking_args.wandb_watch
    os.environ["WANDB_LOG_MODEL"] = tracking_args.wandb_log_model

    if not training_args.should_log:
        return None

    init_kwargs = {
        "project": tracking_args.wandb_project,
        "entity": tracking_args.wandb_entity,
        "name": training_args.run_name,
        "notes": tracking_args.wandb_notes,
        "mode": tracking_args.wandb_mode,
        "tags": (
            [tag.strip() for tag in tracking_args.wandb_tags.split(",") if tag.strip()]
            if tracking_args.wandb_tags
            else None
        ),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}

    if wandb.run is None:
        wandb.init(**init_kwargs)

    print("Weights & Biases tracking enabled.")
    if wandb.run is not None:
        print(f"W&B run: {wandb.run.name} ({wandb.run.id})")

    return wandb


def _get_world_size() -> int:
    """Return the distributed world size from torch or environment."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _patch_rope_compat() -> None:
    """Backfill older Transformers RoPE registries for PaddleOCR-VL."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:
        return

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters_compat(
        config=None,
        device=None,
        seq_len=None,
        **rope_kwargs,
    ):
        del seq_len

        if config is None:
            base = rope_kwargs["base"]
            dim = rope_kwargs["dim"]
        else:
            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                hidden_size = getattr(config, "hidden_size")
                num_attention_heads = getattr(config, "num_attention_heads")
                head_dim = hidden_size // num_attention_heads
            dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters_compat
    print("Patched Transformers RoPE registry with a 'default' handler.")


def train():
    """Main training function."""

    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrackingArguments, TrainingArguments)
    )
    model_args, data_args, tracking_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    # Required for VL models
    training_args.remove_unused_columns = False
    training_args.prediction_loss_only = True  # Avoid OOM during evaluation

    world_size = _get_world_size()
    print(f"Training world size: {world_size} process(es)")

    # Load model in BF16
    print(f"Loading model from {model_args.model_path}...")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if world_size == 1 and DEVICE == "cuda":
        print("Single-process CUDA training detected.")
    elif world_size > 1:
        print("Distributed training detected; letting Accelerate place the model.")
    else:
        print("CUDA is unavailable; training will fall back to CPU placement.")

    if model_args.use_flash_attention_2:
        print("🚀 Flash Attention 2 enabled")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    _patch_rope_compat()
    model = AutoModelForCausalLM.from_pretrained(model_args.model_path, **model_kwargs)
    print(f"✓ Model loaded in {next(model.parameters()).dtype}")

    processor = AutoProcessor.from_pretrained(
        model_args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    # Parse skip_packages
    skip_packages = None
    if data_args.skip_packages:
        skip_packages = [int(x.strip()) for x in data_args.skip_packages.split(",")]

    # Load datasets
    print(f"\nUsing dataset backend: {data_args.dataset_backend}")

    if data_args.dataset_backend.lower() == "mixed30k" and data_args.skip_packages:
        print("skip_packages is ignored for dataset_backend='mixed30k'.")

    if data_args.dataset_backend.lower() == "mixed30k":
        print(
            "split/eval_split are ignored for dataset_backend='mixed30k'; "
            "annotation paths drive dataset selection."
        )

    print("\nLoading training dataset...")
    train_dataset = _build_dataset(
        data_args,
        is_eval=False,
        skip_packages=skip_packages,
    )
    print(f"Training dataset size: {len(train_dataset)}")

    eval_dataset = _build_dataset(
        data_args,
        is_eval=True,
        skip_packages=skip_packages,
    )
    if eval_dataset is None:
        eval_strategy = _get_eval_strategy(training_args)
        if eval_strategy != "no":
            raise ValueError(
                "No evaluation annotation path was provided for mixed30k, so "
                "evaluation must be disabled. Pass --eval_strategy no or set "
                "--eval_annotation_path."
            )
        print("\nSkipping evaluation dataset.")
    else:
        print("\nLoading evaluation dataset...")
        print(f"Evaluation dataset size: {len(eval_dataset)}")

    # Data collator
    collator = CustomDataCollatorForVisionLanguageModeling(
        processor,
        max_length=data_args.max_length,
        pad_to_multiple_of=data_args.pad_to_multiple_of,
    )

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    wandb_module = _setup_wandb(training_args, tracking_args)
    
    # Initialize trainer
    print("Initializing BF16Trainer...")
    trainer = BF16Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    checkpoint = training_args.resume_from_checkpoint
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save model
    print(f"\nSaving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if wandb_module is not None and wandb_module.run is not None:
        wandb_module.finish()
    print("✓ Training complete!")


if __name__ == "__main__":
    train()
