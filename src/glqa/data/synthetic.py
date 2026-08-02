"""Synthetic QA generation from statute text (Bashir et al. approach).

Given a statute section, generate Q/A pairs using an instruction-tuned LLM.
This expands the training data beyond existing QA datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tqdm import tqdm

from glqa.config import get_settings

logger = logging.getLogger(__name__)

# Template for prompting a model to generate QA from a statute
QA_GENERATION_PROMPT = """Du bist ein Experte für deutsches Recht. Gegeben ist folgender Gesetzestext:

---
{law} – {section}: {title}

{text}
---

Erstelle genau 3 Frage-Antwort-Paare, die ein Bürger stellen könnte.
Die Fragen sollen in einfacher Sprache formuliert sein.
Die Antworten sollen den Paragraphen zitieren und verständlich erklären.

Format (JSON-Array):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


def generate_synthetic_qa(
    statutes_path: Path | None = None,
    output_path: Path | None = None,
    model_name: str | None = None,
    max_sections: int = 5000,
) -> Path:
    """Generate synthetic QA pairs from processed statute sections.

    Uses a local LLM (the same one used for generation) to create
    training pairs from statute text.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = get_settings()
    statutes_path = statutes_path or (settings.data.statutes_dir.parent.parent / "processed" / "chunks" / "statutes.jsonl")
    output_path = output_path or (settings.data.qa_pairs_dir.parent.parent / "processed" / "chunks" / "synthetic_qa.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = model_name or settings.rag.generator_model

    logger.info("Loading model %s for synthetic QA generation", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Read statute sections
    sections = []
    if statutes_path.exists():
        with open(statutes_path, encoding="utf-8") as f:
            for line in f:
                sections.append(json.loads(line))
                if len(sections) >= max_sections:
                    break

    logger.info("Generating QA pairs from %d statute sections", len(sections))
    total_pairs = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for section in tqdm(sections, desc="Generating synthetic QA"):
            prompt = QA_GENERATION_PROMPT.format(
                law=section["law"],
                section=section["section"],
                title=section.get("title", ""),
                text=section["text"][:1500],  # Truncate very long sections
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Parse the JSON response
            try:
                # Find JSON array in the response
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > start:
                    qa_pairs = json.loads(response[start:end])
                    for pair in qa_pairs:
                        if "question" in pair and "answer" in pair:
                            entry = {
                                "instruction": pair["question"],
                                "input": "",
                                "output": pair["answer"],
                                "source": "synthetic",
                                "reference": section.get("reference", ""),
                                "split": "train",
                            }
                            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            total_pairs += 1
            except json.JSONDecodeError:
                logger.debug("Failed to parse QA from section %s", section.get("reference", ""))
                continue

    logger.info("Generated %d synthetic QA pairs → %s", total_pairs, output_path)
    return output_path
