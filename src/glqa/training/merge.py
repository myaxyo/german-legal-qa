"""Merge LoRA adapter weights into the base model for faster inference.

After fine-tuning, the adapter is stored separately. This script merges
the adapter back into the base model so inference doesn't need PEFT loaded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
console = Console()


def merge_adapter(
    adapter_path: str | Path,
    output_path: str | Path | None = None,
    push_to_hub: bool = False,
    hub_id: str = "",
) -> Path:
    """Merge a LoRA adapter into the base model and save.

    Args:
        adapter_path: Path to the saved LoRA adapter.
        output_path: Where to save the merged model (default: adapter_path/../merged).
        push_to_hub: Whether to push to HuggingFace Hub.
        hub_id: HuggingFace model ID for pushing.

    Returns:
        Path to the merged model.
    """
    adapter_path = Path(adapter_path)

    # Load training info to get base model ID
    info_path = adapter_path / "training_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        base_model_id = info["model_id"]
    else:
        # Try to read from adapter config
        from peft import PeftConfig

        config = PeftConfig.from_pretrained(str(adapter_path))
        base_model_id = config.base_model_name_or_path

    output_path = Path(output_path) if output_path else adapter_path.parent / "merged"
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Loading base model: {base_model_id}[/blue]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",  # Merge on CPU to save memory
    )

    console.print(f"[blue]Loading adapter: {adapter_path}[/blue]")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    console.print("[blue]Merging weights...[/blue]")
    model = model.merge_and_unload()

    console.print(f"[blue]Saving merged model to {output_path}[/blue]")
    model.save_pretrained(str(output_path))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(output_path))

    if push_to_hub and hub_id:
        console.print(f"[blue]Pushing to Hub: {hub_id}[/blue]")
        model.push_to_hub(hub_id)
        tokenizer.push_to_hub(hub_id)

    console.print(f"[bold green]✓ Merged model saved to {output_path}[/bold green]")
    return output_path
