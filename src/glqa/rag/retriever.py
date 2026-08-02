"""Retriever – find relevant legal text chunks for a given question."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from glqa.config import get_settings
from glqa.rag.embedder import LegalEmbedder
from glqa.rag.indexer import LegalIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved document chunk with metadata."""

    text_preview: str
    reference: str
    source_type: str
    score: float
    metadata: dict = field(default_factory=dict)


class LegalRetriever:
    """Retrieves relevant legal text chunks for a query using FAISS."""

    def __init__(
        self,
        embedder: LegalEmbedder | None = None,
        index: LegalIndex | None = None,
    ):
        self.settings = get_settings().rag
        self.embedder = embedder or LegalEmbedder()
        self.index = index or LegalIndex()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the index on first query."""
        if not self._loaded:
            self.index.load()
            self._loaded = True

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a legal question.

        Uses a hybrid approach:
          1. If the query mentions specific §/Art. references, do keyword lookup first
          2. Then fill remaining slots with semantic search

        Args:
            query: The user's question in German.
            top_k: Number of results (defaults to config).

        Returns:
            List of RetrievalResult sorted by relevance.
        """
        self._ensure_loaded()
        top_k = top_k or self.settings.top_k

        # Step 1: Try keyword-based lookup for specific paragraph references
        keyword_results = self._keyword_lookup(query)

        # Step 2: Semantic search for remaining slots
        semantic_top_k = (top_k * 2) - len(keyword_results)
        query_embedding = self.embedder.encode_query(query)
        raw_results = self.index.search(query_embedding, top_k=max(semantic_top_k, top_k))

        # Combine: keyword results first, then semantic (deduplicated)
        seen_refs: set[str] = set()
        results: list[RetrievalResult] = []

        # Add keyword matches (highest priority)
        for r in keyword_results:
            seen_refs.add(r.reference)
            results.append(r)

        # Add semantic matches
        for r in raw_results:
            if r["score"] < self.settings.score_threshold:
                continue
            ref = r.get("reference", "")
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            results.append(
                RetrievalResult(
                    text_preview=r.get("text", r.get("_text_preview", "")),
                    reference=ref,
                    source_type=r.get("source_type", "unknown"),
                    score=r["score"],
                    metadata={k: v for k, v in r.items() if k not in ("score", "text")},
                )
            )

        # Rerank
        if self.settings.rerank and results:
            results = self._rerank(results)

        return results[:top_k]

    def _keyword_lookup(self, query: str) -> list[RetrievalResult]:
        """Find chunks by exact paragraph reference matching.

        Extracts §/Art. references from the query and scans metadata for matches.
        """
        import re

        # Extract paragraph references from query
        # Matches: §823, § 823, §§ 823, Art. 5
        patterns = [
            r"§+\s*(\d+[a-z]?)\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöü]*)",  # §823 BGB
            r"§+\s*(\d+[a-z]?)",  # §823 (without law name)
        ]

        refs_to_find: list[tuple[str, str]] = []  # (section_num, law_abbrev)
        for pattern in patterns:
            for match in re.finditer(pattern, query):
                groups = match.groups()
                section_num = groups[0]
                law_abbrev = groups[1] if len(groups) > 1 else ""
                refs_to_find.append((section_num, law_abbrev))

        if not refs_to_find:
            return []

        # Scan metadata for matching sections
        results = []
        for i, meta in enumerate(self.index.metadata):
            section = meta.get("section", "")
            law = meta.get("law", "")

            for section_num, law_abbrev in refs_to_find:
                # Check if section number matches
                if section_num not in section:
                    continue
                # If law abbreviation specified, check it matches
                if law_abbrev and law_abbrev.upper() != law.upper():
                    continue
                # Exact section match (avoid §82 matching §823)
                if f"§ {section_num}" in section or f"§{section_num}" in section:
                    text = meta.get("text", "")
                    results.append(
                        RetrievalResult(
                            text_preview=text,
                            reference=meta.get("reference", f"{section} {law}"),
                            source_type=meta.get("source_type", "statute"),
                            score=1.0,  # Perfect match
                            metadata={k: v for k, v in meta.items() if k != "text"},
                        )
                    )
                    break  # Found match for this ref

        return results

    def _rerank(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Simple reranking: deduplicate similar chunks and boost statutes.

        A full reranker (cross-encoder) can be added later for higher quality.
        """
        seen_refs: set[str] = set()
        reranked: list[RetrievalResult] = []

        for r in results:
            # Deduplicate by reference
            if r.reference and r.reference in seen_refs:
                continue
            seen_refs.add(r.reference)

            # Boost statute results slightly (more authoritative)
            if r.source_type == "statute":
                r.score *= 1.1

            reranked.append(r)

        # Re-sort by adjusted score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieved results into a context string for the generator."""
        if not results:
            return "Keine relevanten Rechtstexte gefunden."

        parts = []
        for i, r in enumerate(results, 1):
            source_label = "Gesetz" if r.source_type == "statute" else "Gerichtsentscheidung"
            parts.append(
                f"[Quelle {i} – {source_label}] {r.reference}\n{r.text_preview}"
            )

        return "\n\n---\n\n".join(parts)
