# TuneForge — Claude Code Context

## What This Project Is

TuneForge is a CLI tool that wraps the entire LLM fine-tuning lifecycle — from data prep through training, evaluation, and serving — into three commands. It works on any machine, including laptops with no GPU, via a dry-run mode that validates the entire pipeline without loading any model weights.

This project belongs to Nirbhay Singh. Hands-on LLM fine-tuning experience is a hard differentiator for Staff+ AI roles at Anthropic, OpenAI, Databricks, and similar companies. TuneForge demonstrates that Nirbhay can productionize the fine-tuning workflow, not just use it. Related projects:

- **model-ledger** — tracks the compliance and lineage of models produced by TuneForge
- **data-mint** — generates the synthetic JSONL training data consumed by `tuneforge finetune`
- **agent-loom** — orchestrates agents that use models fine-tuned here

## Project Layout

```
src/tuneforge/
  __init__.py     # package version
  config.py       # Pydantic config models (FineTuneConfig, EvalConfig, ServeConfig)
  core.py         # FineTuner, Evaluator, ModelServer — all core logic
  cli.py          # Click CLI (finetune, eval, serve, quickstart)
examples/
  quickstart.py   # Library API usage
  sample_data.jsonl
tests/
  test_core.py         # 38 unit tests
  test_integration.py
docs/
  architecture.md
```

## The Three Core Commands

### 1. `tuneforge finetune`

Fine-tunes a model using LoRA. Add `--dry-run` to validate setup without a GPU.

```bash
# Preview the training plan (no GPU, no downloads)
tuneforge finetune \
  --model meta-llama/Llama-2-7b-hf \
  --data train.jsonl \
  --dry-run

# Actual fine-tuning (requires pip install tuneforge[gpu])
tuneforge finetune \
  --model meta-llama/Llama-2-7b-hf \
  --data train.jsonl \
  --output-dir ./my-adapter \
  --lora-r 16 \
  --lora-alpha 32 \
  --lr 2e-4 \
  --epochs 3 \
  --batch-size 4 \
  --max-seq-length 512
```

Key flags: `--no-quantize` disables 4-bit QLoRA (use if you have plenty of VRAM). `--dry-run` always works without GPU packages installed.

### 2. `tuneforge eval`

Evaluates model outputs against expected references. Does not require GPU — it reads pre-generated predictions from a JSONL file.

```bash
tuneforge eval \
  --model my-adapter \
  --dataset eval.jsonl \
  --metrics exact_match,contains,length_ratio
```

**Eval JSONL format** (one JSON object per line):
```json
{"input": "What is 2+2?", "expected": "4", "predicted": "4"}
```

**Built-in metrics:**
| Metric | What it measures |
|---|---|
| `exact_match` | Predicted == expected (case-sensitive) |
| `contains` | Expected string appears anywhere in predicted |
| `length_ratio` | len(predicted) / len(expected), scored 0-1 |

### 3. `tuneforge serve`

Generates a standalone FastAPI script with `/health` and `/generate` endpoints. Does not require GPU — it writes the script file; running the script does.

```bash
tuneforge serve \
  --model my-adapter \
  --adapter ./my-adapter/lora_weights \
  --port 8000 \
  --output serve.py

# Then when you have GPU:
pip install fastapi uvicorn
python serve.py
```

### Bonus: `tuneforge quickstart`

```bash
tuneforge quickstart --output-dir ./demo
```

Runs the full pipeline demo (sample data, dry-run plan, eval demo, serve script generation) with zero GPU or configuration. Good for CI and recruiter demos.

## Supported Base Models and Quantization

Any HuggingFace model ID works as `--model`. Tested base models:

- `meta-llama/Llama-2-7b-hf` (default in examples)
- `mistralai/Mistral-7B-v0.1`
- `tiiuae/falcon-7b`
- Any causal LM available on HuggingFace Hub

**Quantization options:**

| Mode | Flag | VRAM requirement |
|---|---|---|
| 4-bit QLoRA (default) | _(none, on by default)_ | ~6 GB |
| Full precision LoRA | `--no-quantize` | ~14 GB for 7B models |

4-bit quantization is handled by `bitsandbytes` via the `tuneforge[gpu]` extras.

## How LoRA and QLoRA Work in TuneForge

**LoRA (Low-Rank Adaptation):** Instead of updating all model weights, LoRA freezes the base model and inserts small trainable rank-decomposition matrices (A and B) at each attention layer. The output is `W + A*B` where `rank(A*B) = r`. TuneForge exposes `--lora-r` (rank, default 16) and `--lora-alpha` (scaling factor, default 32). Lower rank = fewer parameters = faster training but less capacity.

**QLoRA:** Combines LoRA with 4-bit NF4 quantization of the frozen base weights. The base model loads in 4-bit via `bitsandbytes`, the LoRA adapters train in 16-bit. This is what makes 7B model fine-tuning feasible on a single consumer GPU. TuneForge enables this by default (`quantize_4bit=True` in `FineTuneConfig`).

The `FineTuner.run()` method in `core.py` wires together `transformers.AutoModelForCausalLM`, `peft.LoraConfig`, `peft.get_peft_model()`, and a standard `Trainer` loop. `FineTuner.dry_run()` validates all config and prints the plan without importing any of those packages.

## Training Data Format

JSONL file, one instruction-following example per line:

```json
{"instruction": "Summarize this text.", "input": "Long text here...", "output": "Short summary."}
{"instruction": "Translate to French.", "input": "Hello!", "output": "Bonjour!"}
{"instruction": "Classify sentiment.", "input": "Great product!", "output": "positive"}
```

The `input` field is optional and can be an empty string for instruction-only tasks.

## Dry-Run Mode (CPU Testing Without GPU)

The dry-run mode is a first-class feature designed for:
- Validating config before paying for GPU compute
- CI pipelines that run on CPU-only runners
- Local development and demos

When `--dry-run` is passed (or `FineTuner.dry_run()` is called), TuneForge:
1. Reads and validates `FineTuneConfig` with Pydantic
2. Inspects the dataset file (line count, first example shape)
3. Computes estimated training steps, VRAM requirements, and output size
4. Prints a rich-formatted plan table to the terminal
5. Returns a summary dict — no imports of `transformers`, `peft`, or `torch` happen

`_gpu_packages_available()` in `core.py` checks for `transformers`, `peft`, `datasets`, and `accelerate` at runtime. If any are missing, `FineTuner.run()` prints a helpful error and raises `RuntimeError`.

## Development Setup

```bash
# Base install (everything except GPU training)
pip install -e ".[dev]"

# GPU training extras (when you have CUDA)
pip install -e ".[gpu]"

# Run tests
pytest                   # 38 tests, all pass without GPU
ruff check src/ tests/
mypy src/
```

## How to Add a New Base Model

TuneForge does not maintain a model allowlist — any HuggingFace causal LM works. To add special handling for a new model architecture:

1. In `src/tuneforge/core.py`, find `FineTuner._train()`. If the new model needs custom LoRA target modules (e.g., a model that uses `c_attn` instead of `q_proj`/`v_proj`), add a conditional branch that sets `target_modules` in `LoraConfig`.
2. Add a note in `src/tuneforge/config.py` in the `model_name` field description.
3. Add a dry-run test in `tests/test_core.py` using the new model name string.

## How to Add a New Evaluation Metric

1. Open `src/tuneforge/config.py`. In `EvalConfig._validate_metrics`, add the new metric name to the `allowed` set.
2. Open `src/tuneforge/core.py`. In `Evaluator._compute_metrics()`, add a new `elif metric == "my_metric":` branch that computes and returns a float score.
3. Add tests in `tests/test_core.py` verifying the metric on edge cases (empty string, exact match, partial match).
4. Update the `--metrics` help string in `src/tuneforge/cli.py`.

## Python API

```python
from tuneforge.config import FineTuneConfig, EvalConfig, ServeConfig
from tuneforge.core import FineTuner, Evaluator, ModelServer

# Validate setup (works without GPU)
config = FineTuneConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    dataset_path="train.jsonl",
    lora_r=8,
    epochs=5,
)
plan = FineTuner(config).dry_run()

# Evaluate predictions
eval_config = EvalConfig(
    model_path="my-adapter",
    eval_dataset="eval.jsonl",
    metrics=["exact_match", "contains"],
)
results = Evaluator(eval_config).evaluate_from_file()

# Generate serve script
serve_config = ServeConfig(model_path="my-adapter", port=8080)
ModelServer(serve_config).generate_serve_script(output_path="serve.py")
```
