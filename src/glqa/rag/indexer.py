"""FAISS index builder and manager for legal document retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from rich.console import Console
from tqdm import tqdm

from glqa.config import get_settings
from glqa.rag.embedder import LegalEmbedder

logger = logging.getLogger(__name__)
console = Console()


class LegalIndex:
    """FAISS-backed vector index with metadata store for legal documents."""

    def __init__(self, index_path: Path | None = None):
        settings = get_settings().embedding
        self.index_path = index_path or settings.index_path
        self.metadata_path = self.index_path.parent / "metadata.jsonl"
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build_from_chunks(
        self,
        chunk_files: list[Path] | None = None,
        embedder: LegalEmbedder | None = None,
    ) -> None:
        """Build a FAISS index from processed chunk files.

        Each line in a chunk file is a JSON object with at least a 'text' field.
        The metadata (everything except embeddings) is stored separately.
        """
        settings = get_settings()
        if chunk_files is None:
            chunks_dir = settings.data.statutes_dir.parent.parent / "processed" / "chunks"
            chunk_files = list(chunks_dir.glob("*.jsonl"))

        if not chunk_files:
            console.print("[red]No chunk files found. Run `glqa process` first.[/red]")
            return

        if embedder is None:
            embedder = LegalEmbedder()

        # Load all chunks
        console.print(f"[blue]Loading chunks from {len(chunk_files)} files...[/blue]")
        all_texts: list[str] = []
        all_metadata: list[dict] = []

        for chunk_file in chunk_files:
            with open(chunk_file, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text or len(text) < 20:
                        continue

                    # Prepend reference and title for better embedding quality
                    prefix_parts = []
                    if record.get("reference"):
                        prefix_parts.append(record["reference"])
                    if record.get("title"):
                        prefix_parts.append(record["title"])
                    if prefix_parts:
                        embed_text = " – ".join(prefix_parts) + ": " + text
                    else:
                        embed_text = text

                    all_texts.append(embed_text)
                    # Store full record in metadata for retrieval context
                    all_metadata.append(record)

        console.print(f"[blue]Loaded {len(all_texts)} chunks. Encoding...[/blue]")

        if not all_texts:
            console.print("[red]No chunks to index. Run `glqa process` with data first.[/red]")
            return

        # Encode in batches
        embeddings = embedder.encode(all_texts, show_progress=True)

        # Build FAISS index
        dim = embeddings.shape[1]
        console.print(f"[blue]Building FAISS index (dim={dim}, n={len(embeddings)})...[/blue]")

        # Use IndexFlatIP for cosine similarity (vectors are normalized)
        if len(embeddings) > 50_000:
            # For large indices, use IVF for faster search
            nlist = min(int(np.sqrt(len(embeddings))), 1024)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings.astype(np.float32))
            self.index.nprobe = min(nlist // 4, 64)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings.astype(np.float32))
        self.metadata = all_metadata

        # Save
        self._save()
        console.print(f"[green]✓ Index built: {len(embeddings)} vectors → {self.index_path}[/green]")

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for meta in self.metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        logger.info("Index saved to %s", self.index_path)

    def load(self) -> None:
        """Load existing index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"No index found at {self.index_path}. Run `glqa index` first.")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = []
        with open(self.metadata_path, encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        logger.info("Loaded index: %d vectors, %d metadata entries", self.index.ntotal, len(self.metadata))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search the index and return top-k results with metadata.

        Returns list of dicts with 'score' and all metadata fields.
        """
        if self.index is None:
            self.load()

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            result = {**self.metadata[idx], "score": float(score)}
            results.append(result)

        return results


def build_index() -> None:
    """CLI entry point: build the FAISS index from processed chunks."""
    index = LegalIndex()
    index.build_from_chunks()
