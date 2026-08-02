"""MLX-native LoRA fine-tuning for Apple Silicon.

This is the correct way to train on M1/M2/M3 Macs.
MLX uses Apple's Metal GPU directly — no PyTorch MPS overhead.

Usage:
    uv run python -m glqa.training.train_mlx
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from glqa.config import get_settings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
LORA_RANK = 16
LORA_LAYERS = 16
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
ITERS = 500  # Fewer iters, still enough to learn
MAX_SEQ_LENGTH = 512

SYSTEM_MSG = (
    "Du bist ein deutschsprachiger Rechtsassistent. Beantworte rechtliche Fragen "
    "präzise und verständlich. Zitiere relevante Paragraphen wenn möglich. "
    "Dies ist keine Rechtsberatung."
)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------


def prepare_training_data(max_samples: int = 10000) -> Path:
    """Convert QA data into JSONL format expected by mlx-lm.

    Creates data/mlx_training/{train,valid,test}.jsonl
    mlx-lm expects {"text": "..."} format.
    """
    settings = get_settings().data
    qa_file = settings.qa_pairs_dir / "gerlayqa.jsonl"

    output_dir = Path("data/mlx_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not qa_file.exists():
        print(f"Error: {qa_file} not found. Run `make fetch-qa` first.")
        raise FileNotFoundError(qa_file)

    records = []
    with open(qa_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("question") and r.get("answer"):
                records.append(r)
            if len(records) >= max_samples:
                break

    print(f"Loaded {len(records)} QA pairs")

    # Format as chat conversations
    train_records = []
    valid_records = []
    test_records = []

    for i, r in enumerate(records):
        # Use correct chat format based on model
        if "qwen" in MODEL_ID.lower():
            text = (
                f"<|im_start|>system\n"
                f"{SYSTEM_MSG}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{r['question']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{r['answer']}<|im_end|>"
            )
        else:
            text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{SYSTEM_MSG}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{r['question']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{r['answer']}<|eot_id|>"
            )
        entry = {"text": text}

        if i % 10 == 0:
            valid_records.append(entry)
        elif i % 10 == 1:
            test_records.append(entry)
        else:
            train_records.append(entry)

    # Write files
    for name, recs in [("train", train_records), ("valid", valid_records), ("test", test_records)]:
        with open(output_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  Train: {len(train_records)} | Valid: {len(valid_records)} | Test: {len(test_records)}")
    print(f"  Saved to: {output_dir}")

    return output_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_mlx_training():
    """Run LoRA fine-tuning using mlx-lm CLI."""
    print("\n═══ German Legal QA – MLX LoRA Training ═══\n")
    print(f"  Model:      {MODEL_ID}")
    print(f"  LoRA rank:  {LORA_RANK}")
    print(f"  LoRA layers: {LORA_LAYERS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Iterations: {ITERS}")
    print(f"  Max seq:    {MAX_SEQ_LENGTH}")
    print(f"  Device:     Apple Silicon (MLX Metal)")
    print()

    # Step 1: Prepare data
    print("▶ Preparing training data...")
    data_dir = prepare_training_data()
    print()

    # Step 2: Run training via mlx-lm CLI
    adapter_path = Path("models/mlx_adapters/legal-qa-3b")
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL_ID,
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", "lora",
        "--num-layers", str(LORA_LAYERS),
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(ITERS),
        "--learning-rate", str(LEARNING_RATE),
        "--steps-per-report", "10",
        "--steps-per-eval", "200",
        "--save-every", "100",
        "--adapter-path", str(adapter_path),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
    ]

    print("▶ Starting training...")
    print(f"  Command: {' '.join(cmd[-10:])}")
    print()

    # Run training (streams output to terminal)
    result = subprocess.run(cmd, cwd=Path.cwd())

    if result.returncode != 0:
        print(f"\n✗ Training failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ Training complete!")
    print(f"  Adapter saved to: {adapter_path}")

    # Save training info
    info = {
        "model_id": MODEL_ID,
        "framework": "mlx",
        "lora_rank": LORA_RANK,
        "lora_layers": LORA_LAYERS,
        "iters": ITERS,
        "batch_size": BATCH_SIZE,
        "adapter_path": str(adapter_path),
    }
    with open(adapter_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    run_mlx_training()
