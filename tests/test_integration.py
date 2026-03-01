"""Integration tests for TuneForge CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from tuneforge.cli import main


class TestQuickstart:
    """Test that the quickstart command produces visible output and files."""

    def test_quickstart_creates_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "qs")

        result = runner.invoke(main, ["quickstart", "--output-dir", output_dir])

        assert result.exit_code == 0, f"CLI failed:\n{result.output}"

        # Check that expected files were created
        qs_dir = tmp_path / "qs"
        assert (qs_dir / "sample_data.jsonl").exists()
        assert (qs_dir / "eval_data.jsonl").exists()
        assert (qs_dir / "serve.py").exists()

    def test_quickstart_sample_data_valid_jsonl(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "qs")

        runner.invoke(main, ["quickstart", "--output-dir", output_dir])

        sample_file = tmp_path / "qs" / "sample_data.jsonl"
        lines = sample_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

        for line in lines:
            entry = json.loads(line)
            assert "instruction" in entry
            assert "output" in entry

    def test_quickstart_output_contains_plan(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "qs")

        result = runner.invoke(main, ["quickstart", "--output-dir", output_dir])

        assert "Fine-Tuning Plan" in result.output
        assert "meta-llama/Llama-2-7b-hf" in result.output

    def test_quickstart_output_contains_eval(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "qs")

        result = runner.invoke(main, ["quickstart", "--output-dir", output_dir])

        assert "Evaluation Results" in result.output
        assert "exact_match" in result.output

    def test_quickstart_output_contains_serve(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "qs")

        result = runner.invoke(main, ["quickstart", "--output-dir", output_dir])

        assert "Serve script generated" in result.output
        assert "serve.py" in result.output

    def test_quickstart_serve_script_content(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_dir = str(tmp_path / "qs")

        runner.invoke(main, ["quickstart", "--output-dir", output_dir])

        serve_script = (tmp_path / "qs" / "serve.py").read_text(encoding="utf-8")
        assert "FastAPI" in serve_script or "fastapi" in serve_script
        assert "/generate" in serve_script
        assert "/health" in serve_script


class TestFinetuneCLI:
    """Test the finetune command."""

    def test_finetune_dry_run(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            json.dumps({"instruction": "test", "input": "", "output": "ok"}) + "\n",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["finetune", "--model", "test/model", "--data", str(data_file), "--dry-run"],
        )

        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert "Fine-Tuning Plan" in result.output
        assert "test/model" in result.output

    def test_finetune_missing_data_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["finetune", "--model", "m"])
        assert result.exit_code != 0  # missing --data


class TestServeCLI:
    """Test the serve command."""

    def test_serve_generates_script(self, tmp_path: Path) -> None:
        runner = CliRunner()
        output_file = str(tmp_path / "my_serve.py")

        result = runner.invoke(
            main,
            ["serve", "--model", "test/model", "--port", "9000", "--output", output_file],
        )

        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert Path(output_file).exists()

        content = Path(output_file).read_text(encoding="utf-8")
        assert "9000" in content
        assert "test/model" in content


class TestEvalCLI:
    """Test the eval command."""

    def test_eval_runs(self, tmp_path: Path) -> None:
        eval_file = tmp_path / "eval.jsonl"
        entries = [
            {"input": "q", "expected": "a", "predicted": "a"},
            {"input": "q2", "expected": "b", "predicted": "c"},
        ]
        eval_file.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["eval", "--model", "dummy", "--dataset", str(eval_file)],
        )

        assert result.exit_code == 0, f"CLI failed:\n{result.output}"
        assert "exact_match" in result.output

    def test_eval_missing_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["eval", "--model", "m", "--dataset", "/nonexistent.jsonl"],
        )
        assert result.exit_code != 0
