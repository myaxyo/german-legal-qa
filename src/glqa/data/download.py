"""Download raw German legal data from public sources.

Sources:
  1. Gesetze im Internet (Federal statutes in XML)
  2. Open Legal Data API (anonymized court decisions)
  3. GerLayQA (HuggingFace dataset – forum-based legal Q/A)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

from glqa.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Federal Statutes (Gesetze im Internet)
# ---------------------------------------------------------------------------


def download_statutes(output_dir: Path | None = None) -> Path:
    """Download German federal statutes from gesetze-im-internet.de.

    The site provides a table-of-contents XML that lists all laws,
    each linking to a zip containing the law as XML.
    """
    settings = get_settings().data
    output_dir = output_dir or settings.statutes_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = settings.gesetze_im_internet_base_url
    toc_url = f"{base_url}/gii-toc.xml"

    logger.info("Fetching statute table of contents from %s", toc_url)
    resp = requests.get(toc_url, timeout=30)
    resp.raise_for_status()

    # Save the raw ToC
    toc_path = output_dir / "gii-toc.xml"
    toc_path.write_bytes(resp.content)

    # Parse law links from ToC
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(resp.content, "lxml-xml")
    items = soup.find_all("item")
    logger.info("Found %d statutes in ToC", len(items))

    laws_dir = output_dir / "xml"
    laws_dir.mkdir(exist_ok=True)

    downloaded = 0
    for item in tqdm(items, desc="Downloading statutes"):
        link = item.find("link")
        if link is None:
            continue
        zip_url = link.text.strip()
        # Extract slug from URL for the filename (e.g. "bgb" from ".../bgb/xml.zip")
        parts = zip_url.rstrip("/").split("/")
        slug = parts[-2] if len(parts) >= 2 else parts[-1]
        zip_path = laws_dir / f"{slug}.zip"

        if zip_path.exists():
            continue

        try:
            r = requests.get(zip_url, timeout=15)
            if r.status_code == 200:
                zip_path.write_bytes(r.content)
                downloaded += 1
            # Politeness: don't hammer the server
            time.sleep(0.2)
        except requests.RequestException as e:
            logger.warning("Failed to download %s: %s", slug, e)

    logger.info("Downloaded %d new statute files to %s", downloaded, laws_dir)
    return laws_dir


# ---------------------------------------------------------------------------
# 2. Court Decisions (NeuRIS – rechtsinformationen.bund.de)
# ---------------------------------------------------------------------------


def download_decisions(output_dir: Path | None = None, max_records: int | None = None) -> Path:
    """Download court decisions from the NeuRIS federal legal information portal.

    Uses the official API at testphase.rechtsinformationen.bund.de/v1/
    which provides BGH, BVerfG, BAG, BFH, BSG, BVerwG decisions as open data.
    """
    settings = get_settings().data
    output_dir = output_dir or settings.decisions_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    max_records = max_records or settings.max_decisions

    base_url = settings.neuris_api_base_url
    output_file = output_dir / "decisions.jsonl"

    # Resume support: count existing lines
    existing = 0
    if output_file.exists():
        with open(output_file) as f:
            existing = sum(1 for _ in f)
        logger.info("Resuming from %d existing records", existing)

    total_fetched = existing
    page_size = 100
    offset = existing

    with open(output_file, "a", encoding="utf-8") as f:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Downloading decisions...", total=max_records)
            progress.update(task, completed=existing)

            while total_fetched < max_records:
                url = f"{base_url}/case-law?size={page_size}&offset={offset}"
                try:
                    resp = requests.get(url, timeout=30, headers={"Accept": "application/json"})
                    if resp.status_code == 404:
                        # Try alternative endpoint format
                        url = f"{base_url}/rechtsprechung?size={page_size}&offset={offset}"
                        resp = requests.get(url, timeout=30, headers={"Accept": "application/json"})
                    resp.raise_for_status()
                    data = resp.json()
                except requests.RequestException as e:
                    logger.error("API error at offset %d: %s", offset, e)
                    # Try once more, then break
                    time.sleep(5)
                    try:
                        resp = requests.get(url, timeout=30, headers={"Accept": "application/json"})
                        resp.raise_for_status()
                        data = resp.json()
                    except requests.RequestException:
                        logger.error("Persistent API error – stopping at %d records", total_fetched)
                        break

                # Handle both list and paginated response formats
                if isinstance(data, list):
                    results = data
                else:
                    results = data.get("results", data.get("content", data.get("data", [])))

                if not results:
                    logger.info("No more results at offset %d", offset)
                    break

                for record in results:
                    if total_fetched >= max_records:
                        break
                    entry = {
                        "id": record.get("id", record.get("eli", "")),
                        "court": record.get("court", record.get("gericht", "")),
                        "date": record.get("date", record.get("entscheidungsdatum", "")),
                        "file_number": record.get("fileNumber", record.get("aktenzeichen", "")),
                        "type": record.get("type", record.get("dokumenttyp", "")),
                        "content": record.get("content", record.get("inhalt", record.get("text", ""))),
                        "ecli": record.get("ecli", ""),
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_fetched += 1

                progress.update(task, completed=total_fetched)
                offset += page_size
                time.sleep(0.3)  # Rate limiting

    logger.info("Total decisions: %d stored at %s", total_fetched, output_file)
    return output_file


# ---------------------------------------------------------------------------
# 3. GerLayQA (HuggingFace)
# ---------------------------------------------------------------------------


def download_qa_pairs(output_dir: Path | None = None) -> Path:
    """Download GerLayQA dataset from the official GitHub repository.

    This contains German legal questions asked by laypersons
    with expert answers from lawyers, covering BGB, StGB, and ZPO.

    Source: trusthlt/eacl24-german-legal-questions (EACL 2024 paper)
    """
    settings = get_settings().data
    output_dir = output_dir or settings.qa_pairs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading GerLayQA from GitHub (trusthlt/eacl24-german-legal-questions)")

    # Direct raw file URLs from the GitHub repository
    base_raw_url = (
        "https://raw.githubusercontent.com/trusthlt/eacl24-german-legal-questions/main/data"
    )
    files_to_download = [
        "GerLayQA.json",
        "bgb_train.json",
        "bgb_dev.json",
        "bgb_eval.json",
        "stgb_QA.json",
        "zpo_QA.json",
    ]

    output_file = output_dir / "gerlayqa.jsonl"
    total = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for filename in files_to_download:
            url = f"{base_raw_url}/{filename}"
            logger.info("Fetching %s", url)

            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.warning("Failed to download %s: %s", filename, e)
                continue

            # Determine split from filename
            if "train" in filename:
                split = "train"
            elif "eval" in filename:
                split = "test"
            elif "dev" in filename:
                split = "validation"
            else:
                split = "train"

            # Determine category from filename
            if "stgb" in filename.lower():
                category = "StGB"
            elif "zpo" in filename.lower():
                category = "ZPO"
            else:
                category = "BGB"

            # Handle both list and dict formats
            records = data if isinstance(data, list) else data.get("data", [data])

            for item in records:
                # The dataset uses varying field names across files
                question = (
                    item.get("Question_text", "")
                    or item.get("question", "")
                    or item.get("Question", "")
                )
                answer = (
                    item.get("Answer_text", "")
                    or item.get("answer", "")
                    or item.get("Answer", "")
                    or item.get("lawyer_answer", "")
                )

                if not question:
                    continue

                entry = {
                    "split": split,
                    "question": question,
                    "answer": answer,
                    "category": item.get("category", category),
                    "source": "gerlayqa",
                    "paragraphs": item.get("Paragraphs", item.get("paragraphs", [])),
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total += 1

    logger.info("Saved %d QA pairs to %s", total, output_file)
    return output_file


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_download(source: str = "all") -> None:
    """Run data download pipeline for the specified source(s)."""
    from rich.console import Console

    console = Console()

    sources = {
        "statutes": download_statutes,
        "decisions": download_decisions,
        "qa_pairs": download_qa_pairs,
    }

    if source == "all":
        targets = list(sources.items())
    elif source in sources:
        targets = [(source, sources[source])]
    else:
        console.print(f"[red]Unknown source: {source}[/red]")
        console.print(f"Available: {', '.join(sources.keys())}, all")
        return

    for name, func in targets:
        console.print(f"\n[bold blue]▶ Downloading {name}...[/bold blue]")
        try:
            path = func()
            console.print(f"[green]  ✓ {name} → {path}[/green]")
        except Exception as e:
            console.print(f"[red]  ✗ {name} failed: {e}[/red]")
            logger.exception("Download failed for %s", name)
