"""Run evaluation experiments and generate comparison reports.

Evaluates models in multiple configurations:
  1. Base model (no fine-tuning, no RAG)
  2. Base model + RAG
  3. Fine-tuned model (no RAG)
  4. Fine-tuned model + RAG

Generates a JSON report and optional plots.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from glqa.config import get_settings
from glqa.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Test Data Loading
# ---------------------------------------------------------------------------


def load_eval_data(path: Path | None = None, max_samples: int | None = None) -> list[dict]:
    """Load the evaluation dataset (JSONL with instruction/output fields)."""
    settings = get_settings().evaluation
    path = path or settings.eval_dataset_path
    max_samples = max_samples or settings.num_samples

    if not path.exists():
        console.print(f"[red]Eval dataset not found at {path}[/red]")
        console.print("[yellow]Run `glqa fetch qa_pairs` and `glqa process` to create it.[/yellow]")
        return []

    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= max_samples:
                break

    return data


# ---------------------------------------------------------------------------
# Single Configuration Evaluation
# ---------------------------------------------------------------------------


def evaluate_config(
    eval_data: list[dict],
    model_path: str = "",
    use_rag: bool = True,
    config_name: str = "default",
) -> dict:
    """Evaluate a single model configuration.

    Args:
        eval_data: List of dicts with 'instruction' (question) and 'output' (reference answer).
        model_path: Path to fine-tuned adapter (empty = base model).
        use_rag: Whether to use RAG retrieval.
        config_name: Name for this configuration in the report.

    Returns:
        Dict with config info, metrics, and per-sample results.
    """
    from glqa.rag.pipeline import LegalQAPipeline

    console.print(f"\n[bold blue]Evaluating: {config_name}[/bold blue]")
    console.print(f"  Model: {'base' if not model_path else model_path}")
    console.print(f"  RAG: {use_rag}")

    pipeline = LegalQAPipeline(adapter_path=model_path if model_path else None)

    predictions = []
    references = []
    per_sample_results = []
    total_retrieval_ms = 0.0
    total_generation_ms = 0.0

    for sample in tqdm(eval_data, desc=f"Evaluating {config_name}"):
        question = sample["instruction"]
        reference = sample["output"]

        try:
            result = pipeline.ask(question, use_rag=use_rag)
            prediction = result.answer
            total_retrieval_ms += result.retrieval_time_ms
            total_generation_ms += result.generation_time_ms
        except Exception as e:
            logger.warning("Error evaluating question: %s", e)
            prediction = ""

        predictions.append(prediction)
        references.append(reference)

        per_sample_results.append(
            {
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "sources": result.sources if "result" in dir() else [],
            }
        )

    # Compute metrics
    metrics = compute_all_metrics(predictions, references, compute_bertscore=len(eval_data) <= 200)

    # Timing
    n = len(eval_data)
    metrics["avg_retrieval_ms"] = total_retrieval_ms / n if n > 0 else 0
    metrics["avg_generation_ms"] = total_generation_ms / n if n > 0 else 0
    metrics["avg_total_ms"] = (total_retrieval_ms + total_generation_ms) / n if n > 0 else 0

    return {
        "config_name": config_name,
        "model_path": model_path,
        "use_rag": use_rag,
        "num_samples": n,
        "metrics": metrics,
        "samples": per_sample_results[:20],  # Keep first 20 for inspection
    }


# ---------------------------------------------------------------------------
# Full Evaluation Run
# ---------------------------------------------------------------------------


def run_evaluation(
    model_path: str = "",
    use_rag: bool = True,
    compare: bool = False,
) -> None:
    """Run evaluation and optionally compare configurations.

    If compare=True, runs all 4 configurations (base/finetuned × RAG/no-RAG).
    Otherwise, runs only the specified configuration.
    """
    settings = get_settings().evaluation
    eval_data = load_eval_data()

    if not eval_data:
        return

    console.print(f"\n[bold]═══ German Legal QA Evaluation ═══[/bold]")
    console.print(f"  Samples: {len(eval_data)}")

    results = []

    if compare:
        # Run all configurations
        configs = [
            {"model_path": "", "use_rag": False, "config_name": "base_no_rag"},
            {"model_path": "", "use_rag": True, "config_name": "base_with_rag"},
            {"model_path": model_path, "use_rag": False, "config_name": "finetuned_no_rag"},
            {"model_path": model_path, "use_rag": True, "config_name": "finetuned_with_rag"},
        ]
        for cfg in configs:
            if cfg["model_path"] == "" and "finetuned" in cfg["config_name"]:
                # Skip fine-tuned configs if no model path
                if not model_path:
                    console.print(f"[yellow]Skipping {cfg['config_name']} (no model_path)[/yellow]")
                    continue
            result = evaluate_config(eval_data, **cfg)
            results.append(result)
    else:
        result = evaluate_config(
            eval_data,
            model_path=model_path,
            use_rag=use_rag,
            config_name="evaluation",
        )
        results.append(result)

    # Display results table
    _display_results(results)

    # Save results
    results_dir = settings.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/green]")


def _display_results(results: list[dict]) -> None:
    """Display evaluation results as a rich table."""
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Configuration", style="cyan")
    table.add_column("Exact Match", justify="right")
    table.add_column("Token F1", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("Citation Acc.", justify="right")
    table.add_column("BERTScore F1", justify="right")
    table.add_column("Avg. Latency", justify="right")

    for r in results:
        m = r["metrics"]
        table.add_row(
            r["config_name"],
            f"{m.get('exact_match', 0):.3f}",
            f"{m.get('token_f1', 0):.3f}",
            f"{m.get('rouge_l', 0):.3f}",
            f"{m.get('citation_accuracy', 0):.3f}",
            f"{m.get('bert_score_f1', 0):.3f}" if "bert_score_f1" in m else "—",
            f"{m.get('avg_total_ms', 0):.0f}ms",
        )

    console.print()
    console.print(table)
