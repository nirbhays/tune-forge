# TuneForge

**Fine-tune, evaluate, and serve open-source LLMs in three commands. No GPU required to start.**

[![CI](https://github.com/YOUR_ORG/tuneforge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/tuneforge/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/tuneforge.svg)](https://pypi.org/project/tuneforge/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen)
![Works Without GPU](https://img.shields.io/badge/GPU-optional-orange)

> Validate your fine-tuning setup in 30 seconds. Train when you're ready.

---

## The Problem

Fine-tuning LLMs involves too many moving parts:
- Setting up LoRA configs, quantization, and tokenizers
- Writing training scripts from scratch every time
- Evaluating outputs with no built-in tooling
- Deploying the model requires yet another script

## The Fix

TuneForge wraps the entire fine-tuning lifecycle into three CLI commands:

```bash
tuneforge finetune --model meta-llama/Llama-2-7b-hf --data train.jsonl --dry-run
tuneforge eval --model my-model --dataset eval.jsonl
tuneforge serve --model my-model
```

**No GPU?** Every command works in preview/dry-run mode. Validate your setup locally, then run on GPU when ready.

## Quickstart

```bash
pip install -e .
tuneforge quickstart
```

This generates sample data, shows a dry-run training plan, runs an evaluation demo, and creates a serve script -- all without GPU packages.

## Why TuneForge?

| Approach | Setup | GPU Required | Eval Built-in | Serve Built-in |
|----------|-------|:------------:|:--------------:|:--------------:|
| Raw transformers + peft | Heavy | Yes | No | No |
| Axolotl | YAML config | Yes | Limited | No |
| LLaMA-Factory | Web UI | Yes | Limited | Yes |
| **TuneForge** | **3 commands** | **Optional** | **Yes** | **Yes** |

## CLI Reference

### `tuneforge finetune`

```bash
tuneforge finetune --model MODEL --data DATASET [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | HuggingFace model name or path |
| `--data` | (required) | Path to JSONL training data |
| `--output-dir` | `./tuneforge-output` | Output directory for adapter |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha scaling factor |
| `--lr` | `2e-4` | Learning rate |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--max-seq-length` | `512` | Maximum sequence length |
| `--no-quantize` | `False` | Disable 4-bit quantization |
| `--dry-run` | `False` | Show plan without training |

### `tuneforge eval`

```bash
tuneforge eval --model MODEL --dataset eval.jsonl [--metrics exact_match,contains]
```

**Metrics:** `exact_match`, `contains`, `length_ratio`

**Dataset format** (JSONL):
```json
{"input": "What is 2+2?", "expected": "4", "predicted": "4"}
```

### `tuneforge serve`

```bash
tuneforge serve --model MODEL [--adapter ADAPTER_PATH] [--port 8000]
```

Generates a standalone FastAPI script with `/health` and `/generate` endpoints.

### `tuneforge quickstart`

```bash
tuneforge quickstart [--output-dir ./my-demo]
```

Full demo pipeline -- no GPU needed.

## Training Data Format

```json
{"instruction": "Summarize this text.", "input": "Long text here...", "output": "Summary."}
{"instruction": "Translate to French.", "input": "Hello!", "output": "Bonjour!"}
```

## GPU Requirements

| Feature | Base Install | GPU Extras (`tuneforge[gpu]`) |
|---------|:------------:|:-----------------------------:|
| CLI and config | Yes | Yes |
| Dry-run planning | Yes | Yes |
| Evaluation | Yes | Yes |
| Serve script generation | Yes | Yes |
| Actual fine-tuning | No | Yes |
| Running serve script | No | Yes |

```bash
# Install GPU extras when ready
pip install -e ".[gpu]"
```

## Library API

```python
from tuneforge.config import FineTuneConfig, EvalConfig, ServeConfig
from tuneforge.core import FineTuner, Evaluator, ModelServer

# Dry run
config = FineTuneConfig(model_name="meta-llama/Llama-2-7b-hf", dataset_path="train.jsonl")
plan = FineTuner(config).dry_run()

# Evaluation
eval_config = EvalConfig(model_path="my-model", eval_dataset="eval.jsonl")
results = Evaluator(eval_config).evaluate(predictions=["4"], references=["4"])

# Serve script
serve_config = ServeConfig(model_path="my-model", port=8000)
ModelServer(serve_config).generate_serve_script()
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ tests/
mypy src/
```

## License

MIT. See [LICENSE](LICENSE).
