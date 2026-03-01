"""Core logic for TuneForge: fine-tuning, serving, and evaluation."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tuneforge.config import EvalConfig, FineTuneConfig, ServeConfig

logger = structlog.get_logger(__name__)
console = Console()


def _gpu_packages_available() -> bool:
    """Check whether the GPU extras (transformers, peft, etc.) are installed."""
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import datasets  # noqa: F401
        import accelerate  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# FineTuner
# ---------------------------------------------------------------------------


class FineTuner:
    """Wraps the LoRA fine-tuning workflow.

    Works in two modes:
    * **dry_run()** -- validates the config and prints a human-readable plan.
      No GPU packages required.
    * **run()** -- performs actual fine-tuning using transformers + peft.
      Requires the ``gpu`` extras to be installed
      (``pip install tuneforge[gpu]``).
    """

    def __init__(self, config: FineTuneConfig) -> None:
        self.config = config

    # -- dry run (always works) -------------------------------------------

    def dry_run(self) -> dict[str, Any]:
        """Validate config and display a training plan without running anything.

        Returns a summary dict suitable for serialisation / testing.
        """
        cfg = self.config

        # Validate that dataset file looks reasonable (may not exist yet)
        dataset_path = Path(cfg.dataset_path)
        dataset_info = self._inspect_dataset(dataset_path)

        plan: dict[str, Any] = {
            "mode": "dry_run",
            "model_name": cfg.model_name,
            "dataset_path": str(cfg.dataset_path),
            "dataset_info": dataset_info,
            "output_dir": cfg.output_dir,
            "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha},
            "training": {
                "learning_rate": cfg.learning_rate,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "max_seq_length": cfg.max_seq_length,
            },
            "quantize_4bit": cfg.quantize_4bit,
            "gpu_packages_installed": _gpu_packages_available(),
        }

        self._print_plan(plan)
        return plan

    # -- actual training ---------------------------------------------------

    def run(self) -> Path:
        """Run the full fine-tuning loop.

        Raises
        ------
        RuntimeError
            If GPU extras are not installed.

        Returns
        -------
        Path
            The output directory containing the adapter weights.
        """
        if not _gpu_packages_available():
            console.print(
                Panel(
                    "[bold red]GPU packages are not installed.[/bold red]\n\n"
                    "Fine-tuning requires the GPU extras.  Install them with:\n\n"
                    "  [cyan]pip install tuneforge\\[gpu][/cyan]\n\n"
                    "This installs transformers, peft, datasets, accelerate, "
                    "and bitsandbytes.\n\n"
                    "To preview the training plan without GPU packages, use:\n\n"
                    "  [cyan]tuneforge finetune --model MODEL --data PATH --dry-run[/cyan]",
                    title="Missing GPU Dependencies",
                    border_style="red",
                )
            )
            raise RuntimeError(
                "GPU extras not installed. Run: pip install tuneforge[gpu]"
            )

        return self._train()

    # -- private helpers ---------------------------------------------------

    def _train(self) -> Path:
        """Execute the training loop using transformers + peft."""
        # These imports are guarded -- we already verified availability above.
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
        )
        from peft import (  # type: ignore[import-untyped]
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            TaskType,
        )
        from datasets import load_dataset  # type: ignore[import-untyped]

        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"[bold green]Loading model:[/bold green] {cfg.model_name}")

        quantization_config = None
        if cfg.quantize_4bit:
            import torch  # type: ignore[import-untyped]

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )

        if cfg.quantize_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        console.print(f"[bold green]Loading dataset:[/bold green] {cfg.dataset_path}")
        dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")

        def _tokenize(example: dict[str, str]) -> dict[str, Any]:
            instruction = example.get("instruction", "")
            inp = example.get("input", "")
            output = example.get("output", "")
            prompt = f"### Instruction:\n{instruction}\n"
            if inp:
                prompt += f"### Input:\n{inp}\n"
            prompt += f"### Response:\n{output}"
            tokenized = tokenizer(
                prompt,
                truncation=True,
                max_length=cfg.max_seq_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = dataset.map(_tokenize, remove_columns=dataset.column_names)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        console.print("[bold green]Starting training...[/bold green]")
        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        console.print(
            f"[bold green]Training complete![/bold green] Adapter saved to {output_dir}"
        )
        return output_dir

    @staticmethod
    def _inspect_dataset(path: Path) -> dict[str, Any]:
        """Return basic info about a dataset file (without loading GPU libs)."""
        info: dict[str, Any] = {"exists": path.exists()}
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").strip().splitlines()
                info["num_samples"] = len(lines)
                if lines:
                    first = json.loads(lines[0])
                    info["fields"] = list(first.keys())
                    info["first_sample_preview"] = {
                        k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                        for k, v in first.items()
                    }
            except Exception as exc:
                info["parse_error"] = str(exc)
        return info

    def _print_plan(self, plan: dict[str, Any]) -> None:
        """Pretty-print the dry-run plan to the console."""
        table = Table(title="TuneForge Fine-Tuning Plan", show_header=True)
        table.add_column("Parameter", style="cyan", min_width=20)
        table.add_column("Value", style="white")

        table.add_row("Model", plan["model_name"])
        table.add_row("Dataset", plan["dataset_path"])

        ds_info = plan["dataset_info"]
        if ds_info.get("exists"):
            table.add_row("Dataset samples", str(ds_info.get("num_samples", "?")))
            table.add_row("Dataset fields", ", ".join(ds_info.get("fields", [])))
        else:
            table.add_row("Dataset status", "[yellow]File not found (will be needed for training)[/yellow]")

        table.add_row("Output directory", plan["output_dir"])
        table.add_row("LoRA r / alpha", f'{plan["lora"]["r"]} / {plan["lora"]["alpha"]}')
        table.add_row("Learning rate", str(plan["training"]["learning_rate"]))
        table.add_row("Epochs", str(plan["training"]["epochs"]))
        table.add_row("Batch size", str(plan["training"]["batch_size"]))
        table.add_row("Max sequence length", str(plan["training"]["max_seq_length"]))
        table.add_row("4-bit quantization", str(plan["quantize_4bit"]))

        gpu_status = (
            "[green]Installed[/green]"
            if plan["gpu_packages_installed"]
            else "[yellow]Not installed[/yellow] (install with: pip install tuneforge\\[gpu])"
        )
        table.add_row("GPU packages", gpu_status)

        console.print()
        console.print(table)

        if not plan["gpu_packages_installed"]:
            console.print()
            console.print(
                Panel(
                    "This is a [bold]dry run[/bold]. To perform actual training, "
                    "install GPU extras:\n\n"
                    "  [cyan]pip install tuneforge\\[gpu][/cyan]\n\n"
                    "Then run without --dry-run:\n\n"
                    "  [cyan]tuneforge finetune --model MODEL --data PATH[/cyan]",
                    title="Next Steps",
                    border_style="blue",
                )
            )
        console.print()


# ---------------------------------------------------------------------------
# ModelServer
# ---------------------------------------------------------------------------


class ModelServer:
    """Generates and optionally runs a FastAPI serve script for a model.

    The MVP generates a standalone Python script that can be run independently.
    No GPU packages are needed to *generate* the script.
    """

    def __init__(self, config: ServeConfig) -> None:
        self.config = config

    def generate_serve_script(self, output_path: str | Path | None = None) -> Path:
        """Write a standalone FastAPI serve script.

        Parameters
        ----------
        output_path
            Where to write the script.  Defaults to ``./serve_<model>.py``.

        Returns
        -------
        Path
            The path to the generated script.
        """
        cfg = self.config
        if output_path is None:
            safe_name = cfg.model_path.replace("/", "_").replace("\\", "_")
            output_path = Path(f"serve_{safe_name}.py")
        else:
            output_path = Path(output_path)

        adapter_loading = ""
        if cfg.adapter_path:
            adapter_loading = textwrap.dedent(f"""\

                # Load LoRA adapter
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, "{cfg.adapter_path}")
                model = model.merge_and_unload()
                print("LoRA adapter merged successfully.")
            """)

        script = textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """TuneForge auto-generated serve script.

            Generated for model: {cfg.model_path}
            Run with:  python {output_path.name}
            Requires:  pip install tuneforge[gpu] fastapi uvicorn
            """

            from fastapi import FastAPI
            from pydantic import BaseModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import uvicorn

            app = FastAPI(title="TuneForge Model Server", version="0.1.0")

            # ---- Model loading --------------------------------------------------------

            MODEL_PATH = "{cfg.model_path}"
            print(f"Loading model: {{MODEL_PATH}}")

            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
            {adapter_loading}
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("Model loaded successfully.")

            # ---- API -----------------------------------------------------------------

            class GenerateRequest(BaseModel):
                prompt: str
                max_new_tokens: int = 256
                temperature: float = 0.7

            class GenerateResponse(BaseModel):
                generated_text: str
                model: str = MODEL_PATH

            @app.get("/health")
            def health():
                return {{"status": "ok", "model": MODEL_PATH}}

            @app.post("/generate", response_model=GenerateResponse)
            def generate(req: GenerateRequest):
                inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    do_sample=req.temperature > 0,
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return GenerateResponse(generated_text=text)

            # ---- Entry point ---------------------------------------------------------

            if __name__ == "__main__":
                uvicorn.run(app, host="{cfg.host}", port={cfg.port})
        ''')

        output_path.write_text(script, encoding="utf-8")
        logger.info("serve_script_generated", path=str(output_path))

        console.print()
        console.print(
            Panel(
                f"[bold green]Serve script generated:[/bold green] {output_path}\n\n"
                f"Run it with:\n\n"
                f"  [cyan]python {output_path}[/cyan]\n\n"
                f"Endpoints:\n"
                f"  GET  http://{cfg.host}:{cfg.port}/health\n"
                f"  POST http://{cfg.host}:{cfg.port}/generate\n\n"
                f"Requires: [yellow]pip install tuneforge\\[gpu] fastapi uvicorn[/yellow]",
                title="TuneForge Serve",
                border_style="green",
            )
        )
        console.print()
        return output_path

    def endpoint_spec(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the API endpoints."""
        cfg = self.config
        base = f"http://{cfg.host}:{cfg.port}"
        return {
            "base_url": base,
            "model_path": cfg.model_path,
            "adapter_path": cfg.adapter_path,
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/health",
                    "description": "Health check",
                },
                {
                    "method": "POST",
                    "path": "/generate",
                    "description": "Generate text from a prompt",
                    "request_body": {
                        "prompt": "string (required)",
                        "max_new_tokens": "int (default 256)",
                        "temperature": "float (default 0.7)",
                    },
                    "response_body": {
                        "generated_text": "string",
                        "model": "string",
                    },
                },
            ],
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Simple evaluation of model outputs against expected references.

    Computes three metrics without requiring an LLM-as-judge:

    * **exact_match** -- fraction of outputs that exactly equal the expected text.
    * **contains** -- fraction of outputs that contain the expected text as a
      substring (case-insensitive).
    * **length_ratio** -- average ratio ``len(output) / len(expected)``.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """Compute metrics over parallel lists of predictions and references.

        Parameters
        ----------
        predictions
            Model-generated outputs.
        references
            Gold / expected outputs.

        Returns
        -------
        dict[str, float]
            Metric name -> score.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions ({len(predictions)}) and references ({len(references)}) "
                "must have the same length."
            )

        n = len(predictions)
        if n == 0:
            return {m: 0.0 for m in self.config.metrics}

        results: dict[str, float] = {}

        if "exact_match" in self.config.metrics:
            matches = sum(
                1 for p, r in zip(predictions, references) if p.strip() == r.strip()
            )
            results["exact_match"] = matches / n

        if "contains" in self.config.metrics:
            matches = sum(
                1
                for p, r in zip(predictions, references)
                if r.strip().lower() in p.strip().lower()
            )
            results["contains"] = matches / n

        if "length_ratio" in self.config.metrics:
            ratios: list[float] = []
            for p, r in zip(predictions, references):
                ref_len = len(r.strip())
                pred_len = len(p.strip())
                if ref_len == 0:
                    ratios.append(1.0 if pred_len == 0 else float(pred_len))
                else:
                    ratios.append(pred_len / ref_len)
            results["length_ratio"] = sum(ratios) / n

        return results

    def evaluate_from_file(self) -> dict[str, float]:
        """Load predictions and references from the eval dataset JSONL.

        Each line must have ``"input"``, ``"expected"``, and ``"predicted"``
        fields.  If ``"predicted"`` is missing the entry is skipped.
        """
        path = Path(self.config.eval_dataset)
        if not path.exists():
            raise FileNotFoundError(f"Eval dataset not found: {path}")

        predictions: list[str] = []
        references: list[str] = []

        for line in path.read_text(encoding="utf-8").strip().splitlines():
            entry = json.loads(line)
            if "predicted" in entry and "expected" in entry:
                predictions.append(entry["predicted"])
                references.append(entry["expected"])

        results = self.evaluate(predictions, references)
        self._print_results(results)
        return results

    def _print_results(self, results: dict[str, float]) -> None:
        """Pretty-print evaluation results."""
        table = Table(title="TuneForge Evaluation Results", show_header=True)
        table.add_column("Metric", style="cyan", min_width=16)
        table.add_column("Score", style="white", justify="right")

        for metric, score in results.items():
            table.add_row(metric, f"{score:.4f}")

        console.print()
        console.print(table)
        console.print()
