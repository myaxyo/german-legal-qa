"""Generator – produce answers from retrieved context using an LLM."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from glqa.config import get_settings

logger = logging.getLogger(__name__)


class LegalGenerator:
    """Generates legal answers using an instruction-tuned LLM.

    Supports:
    - Base HuggingFace models
    - Fine-tuned LoRA adapters
    - Quantized (4-bit / 8-bit) inference
    """

    def __init__(
        self,
        model_path: str | None = None,
        adapter_path: str | Path | None = None,
        quantize: bool = True,
    ):
        settings = get_settings().rag
        self.model_name = model_path or settings.generator_model
        self.adapter_path = adapter_path
        self.max_new_tokens = settings.max_new_tokens
        self.temperature = settings.temperature
        self.do_sample = settings.do_sample
        self.system_prompt = settings.system_prompt
        self.device = get_settings().device

        self._model = None
        self._tokenizer = None
        self._quantize = quantize

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load model and tokenizer (lazy, on first use)."""
        logger.info("Loading generator: %s", self.model_name)

        # Quantization config for memory efficiency
        quant_config = None
        if self._quantize and self.device != "mps":
            # bitsandbytes quantization (CUDA only)
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "mps" else torch.bfloat16,
            "device_map": "auto",
        }
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        # Load LoRA adapter if specified
        if self.adapter_path:
            from peft import PeftModel

            logger.info("Loading LoRA adapter from %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(self._model, str(self.adapter_path))

        self._model.eval()
        logger.info("Generator loaded successfully on %s", self.device)

    def generate(self, question: str, context: str) -> str:
        """Generate an answer given a question and retrieved context.

        Args:
            question: The user's legal question in German.
            context: Formatted context from the retriever.

        Returns:
            The model's answer as a string.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Kontext (relevante Rechtstexte):\n\n{context}\n\n"
                    f"---\n\nFrage: {question}\n\n"
                    "Beantworte die Frage basierend auf dem obigen Kontext. "
                    "Zitiere die relevanten Paragraphen."
                ),
            },
        ]

        # Use chat template if available
        if self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            input_text = f"{self.system_prompt}\n\n{messages[1]['content']}\n\nAntwort:"

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        return answer

    def generate_without_context(self, question: str) -> str:
        """Generate an answer without RAG context (for comparison/evaluation)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Frage: {question}\n\n"
                    "Beantworte die Frage so gut du kannst basierend auf deinem Wissen "
                    "über deutsches Recht. Zitiere relevante Paragraphen wenn möglich."
                ),
            },
        ]

        if self.tokenizer.chat_template:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = f"{self.system_prompt}\n\n{messages[1]['content']}\n\nAntwort:"

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
