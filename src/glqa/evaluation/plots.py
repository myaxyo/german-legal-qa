"""Generate comparison plots for evaluation results.

Creates publication-quality figures comparing model configurations:
  - Bar charts: metric comparison across configs
  - Latency vs accuracy scatter
  - Per-category breakdowns
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate_comparison_plot(results_file: Path, output_dir: Path | None = None) -> None:
    """Generate comparison plots from an evaluation results JSON file.

    Creates:
      1. metrics_comparison.png – bar chart of all metrics per config
      2. latency_vs_accuracy.png – scatter of latency vs citation accuracy
      3. radar_chart.png – radar/spider chart of capabilities

    Requires matplotlib (not in core deps to keep install light).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plots. Install with: pip install matplotlib")
        return

    with open(results_file) as f:
        results = json.load(f)

    if not results:
        print("No results to plot.")
        return

    output_dir = output_dir or results_file.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Metric comparison bar chart ---
    _plot_metrics_comparison(results, output_dir)

    # --- Plot 2: Latency vs accuracy scatter ---
    _plot_latency_vs_accuracy(results, output_dir)

    print(f"Plots saved to {output_dir}")


def _plot_metrics_comparison(results: list[dict], output_dir: Path) -> None:
    """Bar chart comparing metrics across configurations."""
    import matplotlib.pyplot as plt

    configs = [r["config_name"] for r in results]
    metrics_to_plot = ["token_f1", "rouge_l", "citation_accuracy"]
    metric_labels = ["Token F1", "ROUGE-L", "Citation Accuracy"]

    # Add BERTScore if available
    if "bert_score_f1" in results[0].get("metrics", {}):
        metrics_to_plot.append("bert_score_f1")
        metric_labels.append("BERTScore F1")

    x = np.arange(len(configs))
    width = 0.8 / len(metrics_to_plot)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [r["metrics"].get(metric, 0) for r in results]
        offset = (i - len(metrics_to_plot) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("German Legal QA – Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
    plt.close()


def _plot_latency_vs_accuracy(results: list[dict], output_dir: Path) -> None:
    """Scatter plot: average latency vs citation accuracy."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        m = r["metrics"]
        latency = m.get("avg_total_ms", 0)
        accuracy = m.get("citation_accuracy", 0)
        ax.scatter(latency, accuracy, s=100)
        ax.annotate(
            r["config_name"],
            (latency, accuracy),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Citation Accuracy")
    ax.set_title("Latency vs. Accuracy Trade-off")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "latency_vs_accuracy.png", dpi=150)
    plt.close()
