"""TuneForge CLI -- click-based command-line interface."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from tuneforge import __version__
from tuneforge.config import EvalConfig, FineTuneConfig, ServeConfig
from tuneforge.core import Evaluator, FineTuner, ModelServer

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="tuneforge")
def main() -> None:
    """TuneForge -- fine-tune, evaluate, and serve open-source LLMs."""


# ---------------------------------------------------------------------------
# finetune
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="HuggingFace model name or local path.")
@click.option("--data", required=True, help="Path to JSONL training data.")
@click.option("--output-dir", default="./tuneforge-output", help="Output directory.")
@click.option("--lora-r", default=16, type=int, help="LoRA rank.")
@click.option("--lora-alpha", default=32, type=int, help="LoRA alpha.")
@click.option("--lr", default=2e-4, type=float, help="Learning rate.")
@click.option("--epochs", default=3, type=int, help="Number of epochs.")
@click.option("--batch-size", default=4, type=int, help="Batch size.")
@click.option("--max-seq-length", default=512, type=int, help="Max sequence length.")
@click.option("--no-quantize", is_flag=True, help="Disable 4-bit quantization.")
@click.option("--dry-run", is_flag=True, help="Show training plan without running.")
def finetune(
    model: str,
    data: str,
    output_dir: str,
    lora_r: int,
    lora_alpha: int,
    lr: float,
    epochs: int,
    batch_size: int,
    max_seq_length: int,
    no_quantize: bool,
    dry_run: bool,
) -> None:
    """Fine-tune a model with LoRA (or show a dry-run plan)."""
    config = FineTuneConfig(
        model_name=model,
        dataset_path=data,
        output_dir=output_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        quantize_4bit=not no_quantize,
    )
    tuner = FineTuner(config)

    if dry_run:
        tuner.dry_run()
    else:
        try:
            tuner.run()
        except RuntimeError:
            sys.exit(1)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace name.")
@click.option("--port", default=8000, type=int, help="Port for the server.")
@click.option("--host", default="0.0.0.0", help="Host to bind to.")
@click.option("--adapter", default=None, help="Optional LoRA adapter path.")
@click.option(
    "--output",
    default=None,
    help="Output path for the generated serve script.",
)
def serve(
    model: str,
    port: int,
    host: str,
    adapter: str | None,
    output: str | None,
) -> None:
    """Generate a FastAPI serve script for a model."""
    config = ServeConfig(
        model_path=model,
        host=host,
        port=port,
        adapter_path=adapter,
    )
    server = ModelServer(config)
    script_path = server.generate_serve_script(output_path=output)

    # Also show the endpoint spec
    spec = server.endpoint_spec()
    console.print(f"[dim]Endpoint spec:[/dim] {json.dumps(spec, indent=2)}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@main.command(name="eval")
@click.option("--model", required=True, help="Model path (used for labelling only in MVP).")
@click.option("--dataset", required=True, help="Path to JSONL eval dataset.")
@click.option(
    "--metrics",
    default="exact_match,contains,length_ratio",
    help="Comma-separated list of metrics.",
)
def eval_cmd(model: str, dataset: str, metrics: str) -> None:
    """Evaluate model outputs against expected references."""
    metric_list = [m.strip() for m in metrics.split(",")]

    config = EvalConfig(
        model_path=model,
        eval_dataset=dataset,
        metrics=metric_list,
    )
    evaluator = Evaluator(config)

    try:
        results = evaluator.evaluate_from_file()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# quickstart
# ---------------------------------------------------------------------------

_SAMPLE_DATA = [
    {
        "instruction": "Summarize the following text in one sentence.",
        "input": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve over time without being explicitly programmed.",
        "output": "Machine learning is an AI subset that allows systems to learn and improve from data automatically.",
    },
    {
        "instruction": "Translate the following English sentence to French.",
        "input": "The weather is beautiful today.",
        "output": "Le temps est magnifique aujourd'hui.",
    },
    {
        "instruction": "Write a Python function that returns the factorial of a number.",
        "input": "",
        "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    },
    {
        "instruction": "Classify the sentiment of this review as positive, negative, or neutral.",
        "input": "The product arrived on time and works exactly as described. Very happy with my purchase!",
        "output": "positive",
    },
    {
        "instruction": "Explain what LoRA fine-tuning is in simple terms.",
        "input": "",
        "output": "LoRA (Low-Rank Adaptation) is a technique that fine-tunes a large language model by training only a small set of additional parameters instead of updating the entire model, making it much faster and more memory-efficient.",
    },
]


@main.command()
@click.option(
    "--output-dir",
    default="./tuneforge-quickstart",
    help="Directory to write quickstart files into.",
)
def quickstart(output_dir: str) -> None:
    """Generate sample data and run a dry-run to demonstrate the workflow."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_file = out / "sample_data.jsonl"

    # Write sample data
    with sample_file.open("w", encoding="utf-8") as f:
        for entry in _SAMPLE_DATA:
            f.write(json.dumps(entry) + "\n")

    console.print(
        Panel(
            "[bold]Welcome to TuneForge![/bold]\n\n"
            "This quickstart will:\n"
            "  1. Generate a sample instruction-following dataset\n"
            "  2. Show a dry-run fine-tuning plan\n"
            "  3. Demonstrate evaluation with sample data\n"
            "  4. Generate a model-serving script\n\n"
            f"Working directory: [cyan]{out.resolve()}[/cyan]",
            title="TuneForge Quickstart",
            border_style="blue",
        )
    )

    # Step 1: Show sample data
    console.print(f"\n[bold cyan]Step 1:[/bold cyan] Sample data written to {sample_file}")
    console.print(f"  Contains {len(_SAMPLE_DATA)} instruction-following examples.\n")

    # Step 2: Dry-run finetune
    console.print("[bold cyan]Step 2:[/bold cyan] Running fine-tune dry-run...\n")
    ft_config = FineTuneConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        dataset_path=str(sample_file),
        output_dir=str(out / "finetune-output"),
    )
    tuner = FineTuner(ft_config)
    tuner.dry_run()

    # Step 3: Evaluation demo
    console.print("[bold cyan]Step 3:[/bold cyan] Running evaluation demo...\n")
    eval_file = out / "eval_data.jsonl"
    eval_entries = [
        {
            "input": "Summarize AI",
            "expected": "AI is a branch of computer science.",
            "predicted": "AI is a branch of computer science.",
        },
        {
            "input": "Translate hello",
            "expected": "bonjour",
            "predicted": "Bonjour! The French word for hello is bonjour.",
        },
        {
            "input": "Sentiment?",
            "expected": "positive",
            "predicted": "positive",
        },
    ]
    with eval_file.open("w", encoding="utf-8") as f:
        for entry in eval_entries:
            f.write(json.dumps(entry) + "\n")

    eval_config = EvalConfig(
        model_path="meta-llama/Llama-2-7b-hf",
        eval_dataset=str(eval_file),
    )
    evaluator = Evaluator(eval_config)
    evaluator.evaluate_from_file()

    # Step 4: Generate serve script
    console.print("[bold cyan]Step 4:[/bold cyan] Generating serve script...\n")
    serve_config = ServeConfig(
        model_path="meta-llama/Llama-2-7b-hf",
        port=8000,
    )
    server = ModelServer(serve_config)
    server.generate_serve_script(output_path=out / "serve.py")

    # Final summary
    console.print(
        Panel(
            textwrap.dedent(f"""\
            [bold green]Quickstart complete![/bold green]

            Generated files:
              {sample_file}      -- training data
              {eval_file}        -- evaluation data
              {out / 'serve.py'} -- serve script

            Next steps:

              [cyan]# Install GPU extras for actual training[/cyan]
              pip install tuneforge[gpu]

              [cyan]# Fine-tune a model[/cyan]
              tuneforge finetune --model meta-llama/Llama-2-7b-hf \\
                --data {sample_file}

              [cyan]# Serve your fine-tuned model[/cyan]
              pip install fastapi uvicorn
              python {out / 'serve.py'}
            """),
            title="What's Next?",
            border_style="green",
        )
    )
