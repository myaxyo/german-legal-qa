"""Fine-tuning pipeline for German Legal QA models.

Supports:
  - LoRA (3B models, fits comfortably in 16GB)
  - QLoRA (7B models, fits in ~8-10GB with 4-bit quantization)
  - Apple Silicon (MPS) and CUDA training

Uses TRL's SFTTrainer for instruction-tuning on legal QA pairs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from glqa.config import get_settings

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------


def load_training_data(max_samples: int | None = None) -> Dataset:
    """Load instruction-tuning data from processed QA files.

    Combines:
      - GerLayQA processed pairs
      - Synthetic QA pairs (if available)
    """
    settings = get_settings().data
    processed_dir = settings.statutes_dir.parent.parent / "processed" / "chunks"

    all_records = []

    # Load QA instruction data
    qa_file = processed_dir / "qa_instruction.jsonl"
    if qa_file.exists():
        with open(qa_file, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record.get("split", "train") == "train":
                    all_records.append(record)

    # Load synthetic QA (if generated)
    synthetic_file = processed_dir / "synthetic_qa.jsonl"
    if synthetic_file.exists():
        with open(synthetic_file, encoding="utf-8") as f:
            for line in f:
                all_records.append(json.loads(line))

    if max_samples and len(all_records) > max_samples:
        import random

        random.shuffle(all_records)
        all_records = all_records[:max_samples]

    logger.info("Loaded %d training examples", len(all_records))
    return Dataset.from_list(all_records)


def format_instruction(example: dict, tokenizer) -> str:
    """Format a single example into chat template for instruction tuning."""
    system_msg = (
        "Du bist ein deutschsprachiger Rechtsassistent. Beantworte rechtliche Fragen "
        "präzise und verständlich. Zitiere relevante Paragraphen wenn möglich. "
        "Dies ist keine Rechtsberatung."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback format
        return (
            f"### System:\n{system_msg}\n\n"
            f"### Frage:\n{example['instruction']}\n\n"
            f"### Antwort:\n{example['output']}"
        )


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

# Model presets for common configurations
MODEL_PRESETS = {
    "3b": {
        "model_id": "unsloth/Llama-3.2-1B-Instruct",
        "quantization": "none",
        "lora_r": 8,
        "lora_alpha": 16,
        "batch_size": 2,
        "gradient_accumulation": 8,
    },
    "3b-qlora": {
        "model_id": "unsloth/Llama-3.2-3B-Instruct",
        "quantization": "4bit",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 8,
        "gradient_accumulation": 2,
    },
    "7b": {
        "model_id": "unsloth/Qwen2.5-7B-Instruct",
        "quantization": "4bit",  # QLoRA for 7B
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 2,
        "gradient_accumulation": 8,
    },
}


def load_base_model(
    model_id: str, quantization: str, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model with optional quantization."""
    console.print(f"[blue]Loading base model: {model_id} (quant={quantization})[/blue]")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs: dict = {}

    if quantization == "4bit" and device != "mps":
        # bitsandbytes 4-bit (CUDA only)
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
    elif quantization == "8bit" and device != "mps":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    else:
        # MPS or full precision: load to CPU first, then move
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Move to MPS if needed
    if device == "mps" and load_kwargs.get("device_map") is None:
        model = model.to("mps")

    # Prepare for k-bit training if quantized
    if quantization in ("4bit", "8bit") and device != "mps":
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(
    model_size: str = "3b",
    resume: bool = False,
    max_samples: int | None = 5000,
) -> Path:
    """Run the full LoRA/QLoRA fine-tuning pipeline.

    Args:
        model_size: One of '3b', '3b-qlora', '7b'.
        resume: Whether to resume from last checkpoint.
        max_samples: Limit training data (for debugging).

    Returns:
        Path to the saved adapter weights.
    """
    settings = get_settings().training
    device = get_settings().device

    # Get preset (or use config defaults)
    preset = MODEL_PRESETS.get(model_size, MODEL_PRESETS["3b"])
    model_id = preset["model_id"]
    quantization = preset["quantization"]

    console.print(f"\n[bold]═══ German Legal QA Fine-Tuning ═══[/bold]")
    console.print(f"  Model:        {model_id}")
    console.print(f"  Quantization: {quantization}")
    console.print(f"  Device:       {device}")
    console.print(f"  LoRA rank:    {preset['lora_r']}")
    console.print()

    # Load model
    model, tokenizer = load_base_model(model_id, quantization, device)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=preset["lora_r"],
        lora_alpha=preset["lora_alpha"],
        lora_dropout=settings.lora_dropout,
        target_modules=settings.target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    console.print(
        f"[green]Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)[/green]\n"
    )

    # Load data
    dataset = load_training_data(max_samples=max_samples)
    if len(dataset) == 0:
        console.print("[red]No training data found. Run `glqa fetch qa_pairs` and `glqa process` first.[/red]")
        return Path()

    console.print(f"[blue]Training on {len(dataset)} examples[/blue]")

    # Output directory
    output_dir = settings.output_dir / f"legal-qa-{model_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=preset["batch_size"],
        gradient_accumulation_steps=preset["gradient_accumulation"],
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=0.01,
        bf16=False,
        fp16=device == "mps" or device == "cuda",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        gradient_checkpointing=device != "mps",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_length=512,
        packing=False,
        dataloader_pin_memory=False,
        use_cpu=False,
        resume_from_checkpoint=resume,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=lambda ex: format_instruction(ex, tokenizer),
    )

    # Train
    console.print("[bold green]Starting training...[/bold green]\n")
    trainer.train(resume_from_checkpoint=resume if resume else None)

    # Save the adapter
    adapter_path = output_dir / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Adapter saved to: {adapter_path}")

    # Save training info
    info = {
        "model_id": model_id,
        "model_size": model_size,
        "quantization": quantization,
        "lora_r": preset["lora_r"],
        "lora_alpha": preset["lora_alpha"],
        "epochs": settings.epochs,
        "training_samples": len(dataset),
        "adapter_path": str(adapter_path),
    }
    info_path = adapter_path / "training_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return adapter_path
