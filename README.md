# German Legal QA Assistant

**"Ask German Law"** – A retrieval-augmented, fine-tuned LLM chatbot for German legal questions.

This system answers questions about German civil, criminal, and labor law by combining:
- A **vector-indexed knowledge base** of federal statutes and court decisions
- A **fine-tuned 3B/7B language model** (LoRA/QLoRA) specialized in German legal text
- **Source citations** pointing to exact paragraphs (§) and court decisions

> **Disclaimer:** This is a research prototype. It does not constitute legal advice.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/YOUR_USER/german-legal-qa.git
cd german-legal-qa
uv sync

# Download data
make fetch

# Process and index
make process
make index

# Ask a question
make ask Q="Was besagt §823 BGB?"

# Launch the demo
make demo
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  User Query │────▶│  Embedder    │────▶│  FAISS Index     │
└─────────────┘     │  (MiniLM)   │     │  (Statutes +     │
                    └──────────────┘     │   Decisions)     │
                                         └────────┬─────────┘
                                                  │ Top-K
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Answer    │◀────│  Generator   │◀────│  Retrieved       │
│  + Sources  │     │  (Llama 3.2  │     │  Context         │
└─────────────┘     │   + LoRA)    │     └──────────────────┘
                    └──────────────┘
```

## Project Structure

```
german-legal-qa/
├── src/glqa/              # Main package
│   ├── config.py          # Pydantic settings
│   ├── cli.py             # Typer CLI
│   ├── data/              # Download, process, synthetic QA
│   ├── rag/               # Embedder, indexer, retriever, generator, pipeline
│   ├── training/          # LoRA/QLoRA fine-tuning, adapter merge
│   └── evaluation/        # Metrics, comparison, plots
├── demo/                  # Gradio web interface
├── configs/               # YAML configuration
├── data/                  # Raw + processed data (gitignored)
├── models/                # Trained adapters (gitignored)
├── results/               # Evaluation outputs
├── tests/                 # Unit tests
└── notebooks/             # Exploration notebooks
```

## Data Sources

| Source | Description | License |
|--------|-------------|---------|
| [Gesetze im Internet](https://www.gesetze-im-internet.de) | All federal German statutes (XML) | Public domain |
| [Open Legal Data](https://de.openlegaldata.io) | 100K+ anonymized court decisions | CC-BY |
| [GerLayQA](https://huggingface.co/datasets/fhswf/GerLayQA) | German legal forum Q&A pairs | Research |

## Training

Fine-tune with LoRA (3B, fits in 16GB) or QLoRA (7B, fits in ~8GB):

```bash
# 3B LoRA – ~2-4h on M1 Pro
make train

# 7B QLoRA – ~4-8h on M1 Pro
make train-7b
```

Supported base models:
- `unsloth/Llama-3.2-3B-Instruct`

## Evaluation

Compare configurations (base vs fine-tuned, with/without RAG):

```bash
make evaluate
```

Metrics: Exact Match, Token F1, ROUGE-L, BERTScore, Citation Accuracy.

## Configuration

All settings configurable via:
- Environment variables (`GLQA_*`)
- `.env` file in project root
- `configs/default.yaml`

Key settings:
```bash
GLQA_DEVICE=mps              # mps | cuda | cpu
GLQA_RAG_TOP_K=5             # retrieval depth
GLQA_TRAIN_EPOCHS=3          # training epochs
GLQA_TRAIN_BASE_MODEL=unsloth/Llama-3.2-3B-Instruct
```

## Hardware Requirements

| Task | RAM | Time |
|------|-----|------|
| Data processing | 8GB+ | ~30min |
| FAISS indexing | 8GB+ | ~15min |
| 3B LoRA training | 16GB+ | 2-4h |
| 7B QLoRA training | 16GB+ | 4-8h |
| Inference (3B) | 8GB+ | ~2s/query |
| Inference (7B, 4-bit) | 10GB+ | ~4s/query |

Tested on: MacBook Pro M1 Pro (32GB).

## CLI Reference

```
glqa fetch [statutes|decisions|qa_pairs|all]  # Download data
glqa process                                   # Process raw → chunks
glqa index                                     # Build FAISS index
glqa train [--model-size 3b|7b]               # Fine-tune
glqa evaluate [--use-rag/--no-use-rag]        # Evaluate
glqa ask "question"                            # Single query
glqa demo [--port 7860] [--share]             # Launch UI
```

## License

MIT

## Citation

```bibtex
@software{german_legal_qa_2026,
  title={German Legal QA Assistant},
  author={Jon},
  year={2026},
  url={https://github.com/YOUR_USER/german-legal-qa}
}
```
