# German Legal QA – Project Makefile
# ==================================
# All common tasks accessible via `make <target>`

.PHONY: help install dev fetch process index train evaluate demo lint test clean

UV := uv
RUN := $(UV) run

# Load .env if it exists
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install project dependencies
	$(UV) sync

dev: ## Install with dev dependencies
	$(UV) sync --extra dev --extra mlx

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

fetch: ## Download all raw data (statutes, decisions, QA pairs)
	$(RUN) python -m glqa.cli fetch all

fetch-statutes: ## Download federal statutes only
	$(RUN) python -m glqa.cli fetch statutes

fetch-decisions: ## Download court decisions only
	$(RUN) python -m glqa.cli fetch decisions

fetch-qa: ## Download GerLayQA from HuggingFace
	$(RUN) python -m glqa.cli fetch qa_pairs

process: ## Process raw data into chunks
	$(RUN) python -m glqa.cli process

index: ## Build FAISS vector index
	$(RUN) python -m glqa.cli index

data-pipeline: fetch process index ## Run full data pipeline (fetch → process → index)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train: ## Fine-tune 3B model with LoRA (MLX – Apple Silicon)
	$(RUN) python -m glqa.training.train_mlx

train-pytorch: ## Fine-tune with PyTorch (for CUDA/Colab)
	$(RUN) python -m glqa.cli train --model-size 3b

train-7b: ## Fine-tune 7B model with QLoRA
	$(RUN) python -m glqa.cli train --model-size 7b

train-resume: ## Resume training from checkpoint
	$(RUN) python -m glqa.cli train --resume

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

evaluate: ## Run evaluation on test set
	$(RUN) python -m glqa.cli evaluate

evaluate-compare: ## Run comparative evaluation (all configs)
	$(RUN) python -m glqa.evaluation.run_eval --compare

# ---------------------------------------------------------------------------
# Demo & Interactive
# ---------------------------------------------------------------------------

demo: ## Launch Gradio demo
	$(RUN) python -m glqa.cli demo

demo-share: ## Launch demo with public URL
	$(RUN) python -m glqa.cli demo --share

ask: ## Ask a question (usage: make ask Q="Was besagt §823 BGB?")
	$(RUN) python -m glqa.cli ask "$(Q)"

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

lint: ## Run linter (ruff)
	$(RUN) ruff check src/ demo/ tests/
	$(RUN) ruff format --check src/ demo/ tests/

format: ## Auto-format code
	$(RUN) ruff format src/ demo/ tests/
	$(RUN) ruff check --fix src/ demo/ tests/

test: ## Run tests
	$(RUN) pytest tests/ -v

typecheck: ## Run type checker
	$(RUN) mypy src/

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build: ## Build Docker image
	docker build -t german-legal-qa .

docker-run: ## Run demo in Docker
	docker run -p 7860:7860 --rm german-legal-qa

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove generated files (keep raw data)
	rm -rf data/processed/
	rm -rf models/
	rm -rf results/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Remove everything including raw data
	rm -rf data/
