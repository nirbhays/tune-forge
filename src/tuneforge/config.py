"""Configuration models for TuneForge workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FineTuneConfig(BaseModel):
    """Configuration for the fine-tuning workflow."""

    model_name: str = Field(
        description="HuggingFace model name or local path (e.g. 'meta-llama/Llama-2-7b-hf')."
    )
    dataset_path: str = Field(
        description="Path to a JSONL dataset file with instruction-following entries."
    )
    output_dir: str = Field(
        default="./tuneforge-output",
        description="Directory where the fine-tuned adapter weights will be saved.",
    )
    lora_r: int = Field(default=16, ge=1, description="LoRA rank.")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha scaling factor.")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate for training.")
    epochs: int = Field(default=3, ge=1, description="Number of training epochs.")
    batch_size: int = Field(default=4, ge=1, description="Per-device training batch size.")
    max_seq_length: int = Field(default=512, ge=1, description="Maximum sequence length.")
    quantize_4bit: bool = Field(
        default=True, description="Whether to load the base model in 4-bit quantization."
    )

    @field_validator("dataset_path")
    @classmethod
    def _validate_dataset_path(cls, v: str) -> str:
        p = Path(v)
        if not p.suffix:
            raise ValueError(f"dataset_path must point to a file, got: {v}")
        return v


class ServeConfig(BaseModel):
    """Configuration for model serving."""

    model_path: str = Field(
        description="Path to the base model or HuggingFace model name."
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to.")
    port: int = Field(default=8000, ge=1, le=65535, description="Port to bind the server to.")
    adapter_path: Optional[str] = Field(
        default=None, description="Optional path to LoRA adapter weights to merge."
    )


class EvalConfig(BaseModel):
    """Configuration for model evaluation."""

    model_path: str = Field(
        description="Path to the model (or adapter) to evaluate."
    )
    eval_dataset: str = Field(
        description="Path to a JSONL evaluation dataset with 'input' and 'expected' fields."
    )
    metrics: list[str] = Field(
        default=["exact_match", "contains", "length_ratio"],
        description="List of metric names to compute.",
    )

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, v: list[str]) -> list[str]:
        allowed = {"exact_match", "contains", "length_ratio"}
        for m in v:
            if m not in allowed:
                raise ValueError(
                    f"Unknown metric '{m}'. Allowed metrics: {', '.join(sorted(allowed))}"
                )
        return v
