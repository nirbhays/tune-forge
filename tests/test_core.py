"""Tests for TuneForge config validation, dry_run, and evaluator metrics."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from tuneforge.config import EvalConfig, FineTuneConfig, ServeConfig
from tuneforge.core import Evaluator, FineTuner


# ---------------------------------------------------------------------------
# FineTuneConfig validation
# ---------------------------------------------------------------------------


class TestFineTuneConfig:
    def test_defaults(self) -> None:
        cfg = FineTuneConfig(
            model_name="test-model",
            dataset_path="data.jsonl",
        )
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.learning_rate == 2e-4
        assert cfg.epochs == 3
        assert cfg.batch_size == 4
        assert cfg.max_seq_length == 512
        assert cfg.quantize_4bit is True
        assert cfg.output_dir == "./tuneforge-output"

    def test_custom_values(self) -> None:
        cfg = FineTuneConfig(
            model_name="my-model",
            dataset_path="train.jsonl",
            output_dir="/tmp/out",
            lora_r=8,
            lora_alpha=16,
            learning_rate=1e-5,
            epochs=5,
            batch_size=2,
            max_seq_length=1024,
            quantize_4bit=False,
        )
        assert cfg.lora_r == 8
        assert cfg.epochs == 5
        assert cfg.quantize_4bit is False

    def test_dataset_path_must_have_extension(self) -> None:
        with pytest.raises(ValidationError, match="dataset_path must point to a file"):
            FineTuneConfig(model_name="m", dataset_path="just_a_directory")

    def test_learning_rate_positive(self) -> None:
        with pytest.raises(ValidationError):
            FineTuneConfig(
                model_name="m",
                dataset_path="d.jsonl",
                learning_rate=-0.001,
            )

    def test_lora_r_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            FineTuneConfig(
                model_name="m",
                dataset_path="d.jsonl",
                lora_r=0,
            )


# ---------------------------------------------------------------------------
# ServeConfig validation
# ---------------------------------------------------------------------------


class TestServeConfig:
    def test_defaults(self) -> None:
        cfg = ServeConfig(model_path="my-model")
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.adapter_path is None

    def test_port_range(self) -> None:
        with pytest.raises(ValidationError):
            ServeConfig(model_path="m", port=99999)


# ---------------------------------------------------------------------------
# EvalConfig validation
# ---------------------------------------------------------------------------


class TestEvalConfig:
    def test_defaults(self) -> None:
        cfg = EvalConfig(model_path="m", eval_dataset="eval.jsonl")
        assert cfg.metrics == ["exact_match", "contains", "length_ratio"]

    def test_unknown_metric(self) -> None:
        with pytest.raises(ValidationError, match="Unknown metric"):
            EvalConfig(model_path="m", eval_dataset="e.jsonl", metrics=["bleu"])


# ---------------------------------------------------------------------------
# FineTuner.dry_run
# ---------------------------------------------------------------------------


class TestFineTunerDryRun:
    def test_dry_run_returns_plan(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            json.dumps({"instruction": "hi", "input": "", "output": "hello"}) + "\n",
            encoding="utf-8",
        )

        cfg = FineTuneConfig(
            model_name="test/model",
            dataset_path=str(data_file),
        )
        tuner = FineTuner(cfg)
        plan = tuner.dry_run()

        assert plan["mode"] == "dry_run"
        assert plan["model_name"] == "test/model"
        assert plan["dataset_info"]["exists"] is True
        assert plan["dataset_info"]["num_samples"] == 1
        assert "instruction" in plan["dataset_info"]["fields"]

    def test_dry_run_missing_dataset(self, tmp_path: Path) -> None:
        cfg = FineTuneConfig(
            model_name="test/model",
            dataset_path=str(tmp_path / "nonexistent.jsonl"),
        )
        tuner = FineTuner(cfg)
        plan = tuner.dry_run()

        assert plan["dataset_info"]["exists"] is False

    def test_dry_run_multi_sample(self, tmp_path: Path) -> None:
        data_file = tmp_path / "multi.jsonl"
        lines = [
            json.dumps({"instruction": f"task {i}", "input": "", "output": f"out {i}"})
            for i in range(5)
        ]
        data_file.write_text("\n".join(lines), encoding="utf-8")

        cfg = FineTuneConfig(
            model_name="test/model",
            dataset_path=str(data_file),
        )
        plan = FineTuner(cfg).dry_run()
        assert plan["dataset_info"]["num_samples"] == 5


# ---------------------------------------------------------------------------
# Evaluator metrics
# ---------------------------------------------------------------------------


class TestEvaluator:
    def _make_evaluator(self, metrics: list[str] | None = None) -> Evaluator:
        cfg = EvalConfig(
            model_path="dummy",
            eval_dataset="dummy.jsonl",
            metrics=metrics or ["exact_match", "contains", "length_ratio"],
        )
        return Evaluator(cfg)

    def test_exact_match_all(self) -> None:
        ev = self._make_evaluator(["exact_match"])
        result = ev.evaluate(["hello", "world"], ["hello", "world"])
        assert result["exact_match"] == 1.0

    def test_exact_match_none(self) -> None:
        ev = self._make_evaluator(["exact_match"])
        result = ev.evaluate(["hi", "earth"], ["hello", "world"])
        assert result["exact_match"] == 0.0

    def test_exact_match_partial(self) -> None:
        ev = self._make_evaluator(["exact_match"])
        result = ev.evaluate(["hello", "earth"], ["hello", "world"])
        assert result["exact_match"] == 0.5

    def test_exact_match_strips_whitespace(self) -> None:
        ev = self._make_evaluator(["exact_match"])
        result = ev.evaluate(["  hello  "], ["hello"])
        assert result["exact_match"] == 1.0

    def test_contains_all(self) -> None:
        ev = self._make_evaluator(["contains"])
        result = ev.evaluate(
            ["The answer is bonjour, of course.", "positive sentiment"],
            ["bonjour", "positive"],
        )
        assert result["contains"] == 1.0

    def test_contains_case_insensitive(self) -> None:
        ev = self._make_evaluator(["contains"])
        result = ev.evaluate(["HELLO WORLD"], ["hello"])
        assert result["contains"] == 1.0

    def test_contains_none(self) -> None:
        ev = self._make_evaluator(["contains"])
        result = ev.evaluate(["xyz"], ["abc"])
        assert result["contains"] == 0.0

    def test_length_ratio_exact(self) -> None:
        ev = self._make_evaluator(["length_ratio"])
        result = ev.evaluate(["hello"], ["hello"])
        assert result["length_ratio"] == pytest.approx(1.0)

    def test_length_ratio_double(self) -> None:
        ev = self._make_evaluator(["length_ratio"])
        result = ev.evaluate(["helloworld"], ["hello"])
        assert result["length_ratio"] == pytest.approx(2.0)

    def test_length_ratio_empty_reference(self) -> None:
        ev = self._make_evaluator(["length_ratio"])
        result = ev.evaluate([""], [""])
        assert result["length_ratio"] == pytest.approx(1.0)

    def test_all_metrics(self) -> None:
        ev = self._make_evaluator()
        result = ev.evaluate(["hello", "world"], ["hello", "world"])
        assert "exact_match" in result
        assert "contains" in result
        assert "length_ratio" in result
        assert result["exact_match"] == 1.0
        assert result["contains"] == 1.0
        assert result["length_ratio"] == pytest.approx(1.0)

    def test_empty_lists(self) -> None:
        ev = self._make_evaluator()
        result = ev.evaluate([], [])
        assert result["exact_match"] == 0.0
        assert result["contains"] == 0.0
        assert result["length_ratio"] == 0.0

    def test_mismatched_lengths(self) -> None:
        ev = self._make_evaluator()
        with pytest.raises(ValueError, match="same length"):
            ev.evaluate(["a"], ["a", "b"])

    def test_evaluate_from_file(self, tmp_path: Path) -> None:
        eval_file = tmp_path / "eval.jsonl"
        entries = [
            {"input": "q1", "expected": "hello", "predicted": "hello"},
            {"input": "q2", "expected": "world", "predicted": "World"},
        ]
        eval_file.write_text(
            "\n".join(json.dumps(e) for e in entries),
            encoding="utf-8",
        )

        cfg = EvalConfig(model_path="dummy", eval_dataset=str(eval_file))
        ev = Evaluator(cfg)
        result = ev.evaluate_from_file()

        assert result["exact_match"] == 0.5  # only first matches exactly
        assert result["contains"] == 1.0  # both contain (case-insensitive)

    def test_evaluate_from_file_missing(self) -> None:
        cfg = EvalConfig(model_path="dummy", eval_dataset="/nonexistent.jsonl")
        ev = Evaluator(cfg)
        with pytest.raises(FileNotFoundError):
            ev.evaluate_from_file()
