"""Process raw legal data into chunks ready for embedding and training.

Pipeline:
  1. Parse XML statutes → structured sections with §-references
  2. Clean court decisions → plain text with metadata
  3. Chunk all texts with overlap for RAG indexing
  4. Format QA pairs into instruction-tuning format
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path

from tqdm import tqdm

from glqa.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statute Processing
# ---------------------------------------------------------------------------


def parse_statute_xml(xml_content: str, slug: str) -> list[dict]:
    """Parse a German statute XML into sections with paragraph references.

    Each section becomes a document chunk with metadata:
    - law_abbrev: e.g. "BGB", "StGB"
    - section: §-number
    - title: section title
    - text: the legal text
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(xml_content, "lxml-xml")
    sections = []

    # Find the law abbreviation
    law_abbrev = slug.upper()
    abbrev_tag = soup.find("jurabk")
    if abbrev_tag:
        law_abbrev = abbrev_tag.text.strip()

    # Find all norms (sections)
    for norm in soup.find_all("norm"):
        # Get paragraph number
        enbez = norm.find("enbez")
        if not enbez:
            continue
        section_id = enbez.text.strip()  # e.g. "§ 123"

        # Get title
        title_tag = norm.find("titel")
        title = title_tag.text.strip() if title_tag else ""

        # Get text content
        text_parts = []
        for textdaten in norm.find_all("textdaten"):
            for p in textdaten.find_all("P"):
                text_parts.append(p.get_text(separator=" ", strip=True))

        if not text_parts:
            continue

        full_text = "\n".join(text_parts)
        full_text = re.sub(r"\s+", " ", full_text).strip()

        sections.append(
            {
                "law": law_abbrev,
                "section": section_id,
                "title": title,
                "text": full_text,
                "source_type": "statute",
                "reference": f"{section_id} {law_abbrev}",
            }
        )

    return sections


def process_statutes(output_path: Path | None = None) -> Path:
    """Process all downloaded statute ZIPs into structured JSONL."""
    settings = get_settings().data
    xml_dir = settings.statutes_dir / "xml"
    output_path = output_path or (settings.statutes_dir.parent.parent / "processed" / "chunks" / "statutes.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not xml_dir.exists():
        logger.warning("No statute XMLs found at %s. Run `glqa fetch statutes` first.", xml_dir)
        return output_path

    total_sections = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for zip_path in tqdm(list(xml_dir.glob("*.zip")), desc="Processing statutes"):
            slug = zip_path.stem
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
                    for xml_name in xml_files:
                        xml_content = zf.read(xml_name).decode("utf-8", errors="replace")
                        sections = parse_statute_xml(xml_content, slug)
                        for section in sections:
                            f.write(json.dumps(section, ensure_ascii=False) + "\n")
                            total_sections += 1
            except (zipfile.BadZipFile, Exception) as e:
                logger.warning("Failed to process %s: %s", zip_path.name, e)

    logger.info("Processed %d statute sections → %s", total_sections, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Court Decision Processing
# ---------------------------------------------------------------------------


def clean_decision_text(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace from court decision text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator="\n")
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def process_decisions(output_path: Path | None = None) -> Path:
    """Process raw court decisions JSONL into cleaned chunks."""
    settings = get_settings().data
    raw_file = settings.decisions_dir / "decisions.jsonl"
    output_path = output_path or (settings.decisions_dir.parent.parent / "processed" / "chunks" / "decisions.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        logger.warning("No decisions found at %s. Run `glqa fetch decisions` first.", raw_file)
        return output_path

    chunk_size = settings.chunk_size * 4  # Characters (rough: 4 chars ≈ 1 token for German)
    chunk_overlap = settings.chunk_overlap * 4
    total_chunks = 0

    with open(raw_file, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Processing decisions"):
            record = json.loads(line)
            content = record.get("content", "")
            if not content or len(content) < 100:
                continue

            text = clean_decision_text(content)
            if len(text) < 100:
                continue

            # Chunk the decision
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                entry = {
                    "id": f"{record['id']}_chunk{i}",
                    "court": record.get("court", ""),
                    "date": record.get("date", ""),
                    "file_number": record.get("file_number", ""),
                    "text": chunk,
                    "source_type": "decision",
                    "reference": f"{record.get('court', '')} {record.get('file_number', '')} ({record.get('date', '')})",
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_chunks += 1

    logger.info("Processed %d decision chunks → %s", total_chunks, output_path)
    return output_path


# ---------------------------------------------------------------------------
# QA Pairs → Instruction Tuning Format
# ---------------------------------------------------------------------------


def process_qa_pairs(output_path: Path | None = None) -> Path:
    """Convert GerLayQA into instruction-tuning JSONL format.

    Format: {"instruction": question, "input": "", "output": answer}
    Also creates a train/test split for evaluation.
    """
    settings = get_settings().data
    raw_file = settings.qa_pairs_dir / "gerlayqa.jsonl"
    output_path = output_path or (settings.qa_pairs_dir.parent.parent / "processed" / "chunks" / "qa_instruction.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        logger.warning("No QA pairs found at %s. Run `glqa fetch qa_pairs` first.", raw_file)
        return output_path

    total = 0
    with open(raw_file, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            question = record.get("question", "").strip()
            answer = record.get("answer", "").strip()

            if not question or not answer:
                continue

            entry = {
                "instruction": question,
                "input": "",
                "output": answer,
                "category": record.get("category", ""),
                "split": record.get("split", "train"),
                "source": "gerlayqa",
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1

    # Create eval split
    eval_path = output_path.parent.parent.parent / "eval" / "test_set.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_count = 0

    with open(output_path, encoding="utf-8") as fin, open(eval_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            if record.get("split") == "test":
                fout.write(line)
                eval_count += 1

    logger.info("Processed %d QA pairs → %s (eval: %d)", total, output_path, eval_count)
    return output_path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = 2048, overlap: int = 256) -> list[str]:
    """Split text into overlapping chunks by character count.

    Tries to split on paragraph boundaries when possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to find a paragraph break near the end
        if end < len(text):
            # Look for paragraph break in the last 20% of the chunk
            search_start = end - (chunk_size // 5)
            para_break = text.rfind("\n\n", search_start, end)
            if para_break > start:
                end = para_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_processing() -> None:
    """Run the full data processing pipeline."""
    from rich.console import Console

    console = Console()

    steps = [
        ("Statutes", process_statutes),
        ("Decisions", process_decisions),
        ("QA Pairs", process_qa_pairs),
    ]

    for name, func in steps:
        console.print(f"\n[bold blue]▶ Processing {name}...[/bold blue]")
        try:
            path = func()
            console.print(f"[green]  ✓ {name} → {path}[/green]")
        except Exception as e:
            console.print(f"[red]  ✗ {name} failed: {e}[/red]")
            logger.exception("Processing failed for %s", name)

    console.print("\n[bold green]✓ All processing complete.[/bold green]")
