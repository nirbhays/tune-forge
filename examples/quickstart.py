#!/usr/bin/env python3
"""TuneForge quickstart example -- demonstrates the library API.

This script shows how to use TuneForge programmatically (without the CLI).
It works without GPU packages installed by using dry-run mode and the
CPU-based evaluator.

Usage:
    python quickstart.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tuneforge.config import EvalConfig, FineTuneConfig, ServeConfig
from tuneforge.core import Evaluator, FineTuner, ModelServer


def main() -> None:
    # Create a temporary working directory
    with tempfile.TemporaryDirectory(prefix="tuneforge_example_") as tmpdir:
        work_dir = Path(tmpdir)
        print(f"Working directory: {work_dir}\n")

        # ---------------------------------------------------------------------
        # 1. Prepare sample data
        # ---------------------------------------------------------------------
        print("=" * 60)
        print("STEP 1: Prepare sample training data")
        print("=" * 60)

        sample_data = [
            {
                "instruction": "Summarize this text.",
                "input": "Python is a high-level programming language known for readability.",
                "output": "Python is a readable high-level programming language.",
            },
            {
                "instruction": "Translate to Spanish.",
                "input": "Good morning!",
                "output": "Buenos dias!",
            },
            {
                "instruction": "What is 2 + 2?",
                "input": "",
                "output": "4",
            },
        ]

        data_file = work_dir / "train.jsonl"
        with data_file.open("w", encoding="utf-8") as f:
            for entry in sample_data:
                f.write(json.dumps(entry) + "\n")

        print(f"Wrote {len(sample_data)} samples to {data_file}\n")

        # ---------------------------------------------------------------------
        # 2. Fine-tuning dry run
        # ---------------------------------------------------------------------
        print("=" * 60)
        print("STEP 2: Fine-tune (dry run)")
        print("=" * 60)

        ft_config = FineTuneConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            dataset_path=str(data_file),
            output_dir=str(work_dir / "output"),
            lora_r=8,
            lora_alpha=16,
            learning_rate=1e-4,
            epochs=2,
            batch_size=2,
            max_seq_length=256,
        )

        tuner = FineTuner(ft_config)
        plan = tuner.dry_run()
        print(f"\nDry-run plan keys: {list(plan.keys())}\n")

        # ---------------------------------------------------------------------
        # 3. Evaluation
        # ---------------------------------------------------------------------
        print("=" * 60)
        print("STEP 3: Evaluate outputs")
        print("=" * 60)

        # Simulate predictions vs. references
        predictions = [
            "Python is a readable high-level programming language.",
            "Buenos dias!",
            "The answer is 4.",
        ]
        references = [
            "Python is a readable high-level programming language.",
            "Buenos dias!",
            "4",
        ]

        eval_config = EvalConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            eval_dataset="dummy.jsonl",  # not used in direct evaluate()
        )
        evaluator = Evaluator(eval_config)
        results = evaluator.evaluate(predictions, references)

        print(f"\nEvaluation results: {results}")
        print(f"  exact_match:  {results['exact_match']:.2%}")
        print(f"  contains:     {results['contains']:.2%}")
        print(f"  length_ratio: {results['length_ratio']:.2f}\n")

        # ---------------------------------------------------------------------
        # 4. Generate serve script
        # ---------------------------------------------------------------------
        print("=" * 60)
        print("STEP 4: Generate serve script")
        print("=" * 60)

        serve_config = ServeConfig(
            model_path="meta-llama/Llama-2-7b-hf",
            port=8000,
            adapter_path=str(work_dir / "output"),
        )
        server = ModelServer(serve_config)
        script_path = server.generate_serve_script(output_path=work_dir / "serve.py")
        print(f"\nServe script written to: {script_path}")

        # Show endpoint spec
        spec = server.endpoint_spec()
        print(f"\nEndpoint spec:")
        print(json.dumps(spec, indent=2))

        print("\n" + "=" * 60)
        print("DONE! All steps completed successfully.")
        print("=" * 60)


if __name__ == "__main__":
    main()
