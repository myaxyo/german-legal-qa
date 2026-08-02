# Architecture

## System Overview

The German Legal QA system has four main components that can be used independently or together:

### 1. Data Pipeline (`glqa.data`)

Acquires and preprocesses German legal text from three public sources:

- **Statutes** (gesetze-im-internet.de): XML format, parsed into individual §-sections with structured metadata
- **Court Decisions** (Open Legal Data): JSONL from paginated API, cleaned and chunked
- **QA Pairs** (GerLayQA): HuggingFace dataset of citizen questions with lawyer answers

Processing steps:
1. Download raw data with resume support
2. Parse/clean into structured JSON records
3. Chunk long documents with paragraph-boundary-aware splitting
4. Format QA pairs into instruction-tuning format
5. Optionally generate synthetic QA from statutes

### 2. RAG Pipeline (`glqa.rag`)

Retrieval-Augmented Generation ensures answers are grounded in actual legal text:

```
Query → Embed → FAISS Search → Rerank → Format Context → Generate → Answer
```

Components:
- **Embedder**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, German-aware)
- **Indexer**: FAISS IndexFlatIP (cosine sim) or IndexIVFFlat for >50K docs
- **Retriever**: Score filtering + deduplication + statute boosting
- **Generator**: Instruction-tuned LLM with chat template prompting

### 3. Training (`glqa.training`)

LoRA/QLoRA fine-tuning to adapt general LLMs to German legal domain:

| Configuration | Model | Quantization | VRAM | Time (M1 Pro) |
|---------------|-------|-------------|------|---------------|
| 3B LoRA | Llama-3.2-3B | FP16 | ~8GB | 2-4h |
| 3B QLoRA | Llama-3.2-3B | 4-bit | ~4GB | 2-3h |
| 7B QLoRA | Qwen2.5-7B | 4-bit | ~10GB | 4-8h |

Training uses:
- SFTTrainer from TRL with sequence packing
- Cosine learning rate schedule
- Gradient checkpointing for memory efficiency
- Chat template formatting matching inference

### 4. Evaluation (`glqa.evaluation`)

Systematic comparison across configurations:

| Metric | What it measures |
|--------|-----------------|
| Exact Match | Full answer correctness |
| Token F1 | Word overlap with reference |
| ROUGE-L | Longest common subsequence |
| BERTScore | Semantic similarity (multilingual BERT) |
| Citation Accuracy | Correct §-reference extraction |
| Latency | Retrieval + generation time |

## Design Decisions

1. **Sentence-transformers over OpenAI**: Runs locally, no API costs, reproducible
2. **FAISS over Chroma/Pinecone**: No server dependency, fast on CPU, simple persistence
3. **pydantic-settings**: Type-safe config, env var support, IDE autocomplete
4. **TRL SFTTrainer**: Handles chat templates, packing, gradient checkpointing out of the box
5. **Gradio over Streamlit**: Better for chat UIs, built-in sharing, simpler deployment
