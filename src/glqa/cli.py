"""CLI entry point for the German Legal QA project."""

import typer
from rich.console import Console

app = typer.Typer(
    name="glqa",
    help="German Legal QA Assistant – Ask German Law",
    no_args_is_help=True,
)
console = Console()


@app.command()
def fetch(
    source: str = typer.Argument(
        help="Data source to fetch: 'statutes', 'decisions', 'qa_pairs', or 'all'"
    ),
):
    """Download raw legal data from configured sources."""
    from glqa.data.download import run_download

    run_download(source)


@app.command()
def process():
    """Process raw data into chunks ready for indexing and training."""
    from glqa.data.process import run_processing

    run_processing()


@app.command()
def index():
    """Build FAISS vector index from processed chunks."""
    from glqa.rag.indexer import build_index

    build_index()


@app.command()
def train(
    model_size: str = typer.Option("3b", help="Model size: '3b' or '7b'"),
    resume: bool = typer.Option(False, help="Resume from last checkpoint"),
):
    """Fine-tune a model with LoRA/QLoRA on legal German data."""
    from glqa.training.finetune import run_training

    run_training(model_size=model_size, resume=resume)


@app.command()
def evaluate(
    model_path: str = typer.Option("", help="Path to fine-tuned model (empty = base model)"),
    use_rag: bool = typer.Option(True, help="Whether to use RAG pipeline"),
):
    """Run evaluation on the test set."""
    from glqa.evaluation.run_eval import run_evaluation

    run_evaluation(model_path=model_path, use_rag=use_rag)


@app.command()
def ask(
    question: str = typer.Argument(help="Legal question in German"),
    use_rag: bool = typer.Option(True, help="Use RAG retrieval"),
):
    """Ask a single legal question (CLI interface)."""
    from glqa.rag.pipeline import ask_question

    result = ask_question(question, use_rag=use_rag)
    console.print(f"\n[bold green]Antwort:[/bold green]\n{result['answer']}\n")
    if result.get("sources"):
        console.print("[bold blue]Quellen:[/bold blue]")
        for src in result["sources"]:
            console.print(f"  • {src}")


@app.command()
def demo(
    port: int = typer.Option(7860, help="Port for the Gradio demo"),
    share: bool = typer.Option(False, help="Create a public Gradio link"),
):
    """Launch the interactive Gradio demo."""
    from demo.app import launch_demo

    launch_demo(port=port, share=share)


if __name__ == "__main__":
    app()
