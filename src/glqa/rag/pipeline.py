"""End-to-end RAG pipeline – ties retriever and generator together."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from glqa.config import get_settings
from glqa.rag.generator import LegalGenerator
from glqa.rag.retriever import LegalRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Result of a legal QA query."""

    question: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    used_rag: bool = True


class LegalQAPipeline:
    """Full RAG pipeline: question → retrieve → generate → answer with citations.

    This is the main interface for the German Legal QA system.
    """

    def __init__(
        self,
        retriever: LegalRetriever | None = None,
        generator: LegalGenerator | None = None,
        adapter_path: str | Path | None = None,
    ):
        self._retriever = retriever
        self._generator = generator
        self._adapter_path = adapter_path
        self._initialized = False

    @property
    def retriever(self) -> LegalRetriever:
        if self._retriever is None:
            self._retriever = LegalRetriever()
        return self._retriever

    @property
    def generator(self) -> LegalGenerator:
        if self._generator is None:
            self._generator = LegalGenerator(adapter_path=self._adapter_path)
        return self._generator

    def ask(self, question: str, use_rag: bool = True, top_k: int | None = None) -> QAResult:
        """Ask a legal question and get an answer with sources.

        Args:
            question: Legal question in German.
            use_rag: Whether to use retrieval (if False, generates from model knowledge only).
            top_k: Number of documents to retrieve.

        Returns:
            QAResult with answer, sources, and timing info.
        """
        result = QAResult(question=question, used_rag=use_rag)

        if use_rag:
            # Retrieve relevant documents
            t0 = time.perf_counter()
            retrieval_results = self.retriever.retrieve(question, top_k=top_k)
            result.retrieval_time_ms = (time.perf_counter() - t0) * 1000
            result.retrieval_results = retrieval_results

            # Format context
            context = self.retriever.format_context(retrieval_results)

            # Extract source references
            result.sources = [r.reference for r in retrieval_results if r.reference]

            # Generate answer with context
            t0 = time.perf_counter()
            result.answer = self.generator.generate(question, context)
            result.generation_time_ms = (time.perf_counter() - t0) * 1000
        else:
            # Generate without retrieval
            t0 = time.perf_counter()
            result.answer = self.generator.generate_without_context(question)
            result.generation_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "QA complete – retrieval: %.0fms, generation: %.0fms, sources: %d",
            result.retrieval_time_ms,
            result.generation_time_ms,
            len(result.sources),
        )

        return result


# ---------------------------------------------------------------------------
# Convenience function for CLI
# ---------------------------------------------------------------------------

_pipeline: LegalQAPipeline | None = None


def ask_question(question: str, use_rag: bool = True) -> dict:
    """Convenience function for CLI and demo usage.

    Returns a dict with 'answer' and 'sources' keys.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = LegalQAPipeline()

    result = _pipeline.ask(question, use_rag=use_rag)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "retrieval_time_ms": result.retrieval_time_ms,
        "generation_time_ms": result.generation_time_ms,
    }
