"""Central configuration for the German Legal QA project.

Uses pydantic-settings so values can come from environment variables,
a .env file, or be set programmatically.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Settings classes
# ---------------------------------------------------------------------------


class DataSettings(BaseSettings):
    """Settings for data acquisition and processing."""

    model_config = SettingsConfigDict(env_prefix="GLQA_DATA_")

    # Raw data paths
    statutes_dir: Path = RAW_DIR / "statutes"
    decisions_dir: Path = RAW_DIR / "decisions"
    qa_pairs_dir: Path = RAW_DIR / "qa_pairs"

    # Processing
    chunk_size: int = Field(default=512, description="Tokens per chunk for statute splitting")
    chunk_overlap: int = Field(default=64, description="Overlap tokens between chunks")
    max_decisions: int = Field(default=100_000, description="Max court decisions to process")

    # Sources
    gesetze_im_internet_base_url: str = "https://www.gesetze-im-internet.de"
    neuris_api_base_url: str = "https://testphase.rechtsinformationen.bund.de/v1"
    gerlayqa_hf_dataset: str = "fhswf/GerLayQA"


class EmbeddingSettings(BaseSettings):
    """Settings for embedding and vector store."""

    model_config = SettingsConfigDict(env_prefix="GLQA_EMB_")

    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence-transformer model for German legal embeddings",
    )
    index_path: Path = PROCESSED_DIR / "embeddings" / "faiss_index"
    embedding_dim: int = 384
    device: str = "mps"  # Apple Silicon; change to "cuda" on GPU machines
    batch_size: int = 64
    normalize: bool = True


class RAGSettings(BaseSettings):
    """Settings for retrieval-augmented generation."""

    model_config = SettingsConfigDict(env_prefix="GLQA_RAG_")

    # Retrieval
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    rerank: bool = Field(default=True, description="Whether to rerank retrieved chunks")
    score_threshold: float = Field(default=0.3, description="Minimum similarity score")

    # Generation
    generator_model: str = Field(
        default="unsloth/Llama-3.2-3B-Instruct",
        description="HuggingFace model ID for generation",
    )
    max_new_tokens: int = 1024
    temperature: float = 0.3
    do_sample: bool = True
    system_prompt: str = (
        "Du bist ein deutschsprachiger Rechtsassistent. Beantworte die Frage "
        "ausschließlich anhand der bereitgestellten Rechtstexte. Gib die genaue "
        "Paragraphen-Referenz an (z.B. §123 BGB). Wenn die Antwort nicht aus dem "
        "Kontext ableitbar ist, sage dies ehrlich. Dies ist keine Rechtsberatung."
    )


class TrainingSettings(BaseSettings):
    """Settings for LoRA / QLoRA fine-tuning."""

    model_config = SettingsConfigDict(env_prefix="GLQA_TRAIN_")

    # Model selection
    base_model: str = "unsloth/Llama-3.2-3B-Instruct"
    quantization: Literal["none", "4bit", "8bit"] = "4bit"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    bf16: bool = True  # M1/M2/M3 supports bfloat16 via MPS

    # Output
    output_dir: Path = MODELS_DIR / "checkpoints"
    hub_model_id: str = ""  # Optional: push to HuggingFace Hub


class EvalSettings(BaseSettings):
    """Settings for evaluation."""

    model_config = SettingsConfigDict(env_prefix="GLQA_EVAL_")

    eval_dataset_path: Path = EVAL_DIR / "test_set.jsonl"
    results_dir: Path = PROJECT_ROOT / "results"
    metrics: list[str] = Field(
        default=["exact_match", "f1", "rouge_l", "bert_score", "citation_accuracy"]
    )
    num_samples: int = Field(default=50, description="Number of eval samples to run")


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="GLQA_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name: str = "German Legal QA"
    device: str = "mps"  # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" fallback
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    data: DataSettings = Field(default_factory=DataSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    evaluation: EvalSettings = Field(default_factory=EvalSettings)


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_settings: AppSettings | None = None


def get_settings() -> AppSettings:
    """Return cached application settings (lazy-loaded)."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings
