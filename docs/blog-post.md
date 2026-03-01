# I Spent Six Months Wrestling With Fine-Tuning Scripts. Then I Wrote Three CLI Commands to Replace Them All.

**How TuneForge lets you plan, evaluate, and serve fine-tuned LLMs -- even without a GPU.**

---

I want to tell you about the moment I almost quit fine-tuning open-source models entirely.

It was 11 PM on a Tuesday. I had a messy Jupyter notebook open -- 47 cells deep, half of them broken. I was trying to LoRA fine-tune a Mistral-7B model on a custom instruction dataset. The PEFT config was wrong. The quantization flags conflicted with my CUDA version. I had copy-pasted a training loop from three different blog posts, and none of them agreed on how to handle the tokenizer's padding side. My Colab session had timed out twice, eating six hours of GPU time.

The kicker? I was not even doing anything novel. I was doing the exact same thing thousands of engineers do every week: take a base model, fine-tune it on domain-specific data, evaluate it, and serve it behind an API. This should not require 200 lines of boilerplate and a PhD in debugging CUDA memory errors.

So I built TuneForge. And the design decision that changed everything was one I almost did not make: every single command works without a GPU.

> **TL;DR**
> - TuneForge replaces the entire fine-tuning-to-serving pipeline with three CLI commands: `finetune`, `eval`, and `serve`.
> - Every command has a `--dry-run` mode that works without a GPU -- plan on a laptop, execute on a cloud GPU later.
> - The base install (`pip install tuneforge`) has zero ML dependencies. It installs in about five seconds.
> - The `serve` command *generates* a standalone FastAPI script you own, inspect, and deploy anywhere.

---

## By the Numbers

| | |
|---|---|
| **3** commands | The entire workflow: `finetune`, `eval`, `serve` |
| **38** tests passing | Core workflow is solid and tested |
| **0** config files | Everything is CLI flags. Nothing to edit, nothing to lose |
| **5-second** base install | No PyTorch, no CUDA, no heavy downloads |
| **Works without a GPU** | Every command supports `--dry-run` on any machine |

---

## The Problem Nobody Talks About

Fine-tuning open-source LLMs in 2025 is not hard because the math is hard. LoRA solved the math problem years ago. It is hard because the *workflow* is broken.

**The average misconfigured training run wastes 4-8 GPU hours before anyone notices the mistake.**

Here is what a typical fine-tuning project looks like in practice:

1. You write a data preprocessing script to get your JSONL into the right format.
2. You copy-paste a training script from some GitHub repo, then spend two hours adapting it to your model and dataset.
3. You realize your training data has formatting issues, but only after your first training run fails at step 4,000.
4. You write a separate evaluation script because the training script does not include one.
5. You write yet another script to wrap your model in a FastAPI server.
6. You repeat this entire cycle for every new model or dataset.

Each of these steps involves its own set of imports, its own configuration format, and its own failure modes. The actual fine-tuning -- the forward pass, the backward pass, the adapter merging -- that part works great. Everything *around* it is where projects go to die.

TuneForge replaces this entire pipeline with three commands.

---

## Three Commands. That Is the Whole Idea.

```bash
tuneforge finetune --model mistralai/Mistral-7B-v0.1 \
                   --dataset data/instructions.jsonl \
                   --output ./my-adapter \
                   --epochs 3 --lr 2e-4 --batch-size 4

tuneforge eval --model mistralai/Mistral-7B-v0.1 \
               --adapter ./my-adapter \
               --dataset data/test.jsonl \
               --metrics exact_match contains length_ratio

tuneforge serve --model mistralai/Mistral-7B-v0.1 \
                --adapter ./my-adapter \
                --port 8000
```

`finetune` runs a LoRA training loop with QLoRA 4-bit quantization. It handles the PEFT config, the BitsAndBytes setup, the training loop, and saves the adapter weights. You configure it with flags, not by editing Python files.

`eval` runs your model against a JSONL test set and reports exact_match, contains, and length_ratio metrics. No separate evaluation framework needed.

`serve` does not start a server -- it *generates* a standalone FastAPI script with `/health` and `/generate` endpoints. You get a production-ready file you can deploy anywhere, inspect, and modify. No runtime dependency on TuneForge itself.

**Fine-tuning should be a build step, not a research expedition.**

But the real design insight is not in these commands. It is in what happens when you run them *without a GPU*.

---

## The Graceful Degradation Pattern (And Why It Matters More Than You Think)

Here is the key architectural decision behind TuneForge: the tool is split into two installation tiers. The base install -- `pip install tuneforge` -- gives you Click, Pydantic, Rich, and structlog. No PyTorch, no transformers, no CUDA. It installs in seconds on any machine. The GPU install -- `pip install tuneforge[gpu]` -- adds transformers, peft, datasets, accelerate, and bitsandbytes. This heavy tier only matters when you actually run training.

Why does this matter? Because every command has a `--dry-run` mode that works with the base install alone.

```bash
tuneforge finetune --model mistralai/Mistral-7B-v0.1 \
                   --dataset data/instructions.jsonl \
                   --output ./my-adapter \
                   --epochs 3 --lr 2e-4 --batch-size 4 \
                   --dry-run
```

This prints a detailed execution plan: the LoRA rank and alpha values it will use, the target modules, the quantization config, the estimated memory footprint, the dataset statistics, and the training schedule. All of this without downloading a single model weight or touching a GPU.

This is not a gimmick. This is the pattern that makes TuneForge actually useful in real teams. The person choosing the hyperparameters is not always the person with the GPU. The person reviewing the training config is not always the person running the training job.

Here is a concrete scenario. Your ML lead is on a six-hour flight with nothing but a MacBook Air. She tweaks the hyperparameters and runs `tuneforge finetune --dry-run` to validate the training plan -- LoRA rank 16, alpha 32, 3 epochs, 4-bit quantization, dataset parsed with 12,000 samples. She commits the shell script and closes the laptop. The next morning, the GPU engineer pulls the branch and runs the exact same command without `--dry-run` on an A100. Same config. Same result. No hand-off document. No "here is my notebook, good luck figuring out which cells to run."

Now consider the cost angle. A single A100 hour costs $2-$4. A typical 7B fine-tune takes 2-6 hours. If your config has a typo or your dataset has a formatting issue that only surfaces at step 4,000, you just burned $8-$24 on something that `--dry-run` would have caught in two seconds for free. Multiply that by every engineer on your team who is experimenting with different configs.

I call this "graceful degradation," and I think more ML tools should adopt it. The idea is simple: if a tool cannot do the expensive thing right now, it should still do everything it can. Validate the config. Parse the dataset. Show the plan. Do not just crash with an ImportError about CUDA.

---

## What the Output Looks Like

When you run a dry-run, TuneForge prints a structured Rich table showing every parameter -- model name, dataset path and sample count, LoRA r/alpha, learning rate, epochs, batch size, max sequence length, quantization setting, and GPU package status. Everything you need to review a training plan, in a format you can screenshot and paste into a Slack thread or a PR comment.

When you run `eval`, the output is a clean metrics table:

```
      TuneForge Evaluation Results
┌──────────────────┬──────────┐
│ Metric           │    Score │
├──────────────────┼──────────┤
│ exact_match      │   0.6667 │
│ contains         │   1.0000 │
│ length_ratio     │   1.4833 │
└──────────────────┴──────────┘
```

At a glance: two-thirds of outputs matched exactly, all contained the expected text, and the model generates outputs about 48% longer than expected -- a signal you might want to tune max tokens or add a length penalty.

---

## The Eval Command: More Than an Afterthought

Most fine-tuning tools treat evaluation as someone else's problem. You finish training, export your adapter, and then figure out how to measure whether it actually works.

**If you cannot measure whether your fine-tuned model is better than the base model, you do not have a fine-tuning pipeline. You have a GPU heater.**

TuneForge's `eval` command takes a JSONL file where each line has `input`, `expected`, and `predicted` fields, and computes three metrics:

- **exact_match** -- fraction of outputs that are character-for-character identical to expected. Your strictest signal.
- **contains** -- fraction of outputs that contain the expected text as a substring (case-insensitive). Catches correct answers with extra formatting.
- **length_ratio** -- average `len(output) / len(expected)`. A ratio of 1.0 is ideal. A ratio of 3.0 means rambling. A ratio of 0.2 means truncating.

These three metrics together tell you more than any single accuracy number. A model with 40% exact_match but 95% contains and a length_ratio of 1.8 knows the right answer but over-explains. A model with 90% exact_match and a length_ratio of 0.3 might be truncating. You get a multi-dimensional view of quality without needing an LLM-as-a-judge.

The eval data format is dead simple:

```json
{"input": "Summarize AI", "expected": "AI is a branch of computer science.", "predicted": "AI is a branch of computer science."}
{"input": "Translate hello", "expected": "bonjour", "predicted": "Bonjour! The French word for hello is bonjour."}
```

No special setup. No evaluation server. No API keys. Just a JSONL file and one command.

---

## A Real-World Workflow: From Raw Data to Served Model

Here is what a complete TuneForge project looks like end to end.

**Step 1: Prepare your data.** You have customer support transcripts and want a 7B model that generates helpful responses. Format as instruction-following JSONL:

```json
{"instruction": "Respond to this support ticket.", "input": "Order #4521 arrived damaged.", "output": "I'm sorry about the damage. I've initiated a replacement that will arrive within 2-3 business days."}
```

You build 8,000 training examples in `data/train.jsonl` and 500 test examples in `data/test.jsonl`.

**Step 2: Plan the run (no GPU needed).** Run `tuneforge finetune --data data/train.jsonl --model mistralai/Mistral-7B-v0.1 --output-dir ./support-adapter --epochs 3 --lr 2e-4 --batch-size 4 --dry-run`. The output confirms: 8,000 samples parsed, all fields present, LoRA rank 16, alpha 32, 4-bit quantization enabled. Commit the command to your Makefile.

**Step 3: Train (GPU required).** Same command, without `--dry-run`, on an A100. TuneForge loads the model in 4-bit quantization, applies LoRA targeting q_proj and v_proj, runs the training loop via Hugging Face Trainer, and saves adapter weights to `./support-adapter/`.

**Step 4: Evaluate.** Run `tuneforge eval --model mistralai/Mistral-7B-v0.1 --dataset data/test.jsonl --metrics exact_match,contains,length_ratio`. If exact_match is low but contains is high, your model knows the answer but phrases it differently. Adjust and re-train.

**Step 5: Serve.** Run `tuneforge serve --model mistralai/Mistral-7B-v0.1 --adapter ./support-adapter --port 8000`. This generates a standalone `serve.py` with `/health` and `/generate` endpoints that loads the base model, merges the adapter, and runs a FastAPI server. Zero runtime dependency on TuneForge.

Five steps. Three commands. No notebooks. No YAML. Every step reproducible in CI.

---

## The Quickstart: A Complete Workflow With Zero GPU

You can experience the full TuneForge pipeline right now, on whatever machine you are reading this on.

```bash
pip install tuneforge
tuneforge quickstart
```

The `quickstart` command generates a sample JSONL dataset, runs a dry-run fine-tune to show the training plan, executes an evaluation pass with mock predictions, and generates a FastAPI serving script. About ten seconds, Python 3.9+ required, nothing else.

This is not a toy demo. The serving script, the evaluation metrics, and the training plan are all identical to what you get with a real model. You are looking at the real tool, just without the expensive parts.

---

## Why CLI, Not Notebooks

I have an opinion that will upset some people: Jupyter notebooks are a terrible interface for fine-tuning workflows.

Notebooks are great for exploration and visualization. But fine-tuning is not an exploratory task. Once you know your hyperparameters, fine-tuning is a *build pipeline*. It should be reproducible, version-controlled, automatable, and reviewable. Notebooks fail at all four.

You cannot meaningfully diff a notebook in a pull request. You cannot run a notebook in CI without additional tooling. You cannot easily parameterize a notebook from the outside. And every time you restart the kernel and "Run All," you are praying that cell 23 does not depend on a variable defined in a cell you deleted two hours ago.

A CLI command goes in a Makefile, a shell script, a CI pipeline, or a Docker entrypoint. It has explicit inputs and outputs. It either works or it fails, and when it fails, you get a stack trace, not a dead kernel. TuneForge is a CLI tool because fine-tuning is a build step. Treat it like one.

---

## Honest Comparison: TuneForge vs. Axolotl vs. LLaMA-Factory

I want to be direct about where TuneForge sits relative to the established tools.

**Axolotl** is the current heavyweight. It supports dozens of model architectures, multiple training strategies (full fine-tune, LoRA, QLoRA, RLHF, DPO), and has a rich YAML-based configuration system. If you need cutting-edge training research or exotic model support, Axolotl is probably the right choice. The tradeoff is complexity: config files can be hundreds of lines, installation is non-trivial, and the learning curve is steep.

**LLaMA-Factory** offers a beautiful web UI and an impressive range of training methods. Excellent for teams that want a visual interface. The tradeoff is harder integration into automated pipelines and a scope that makes debugging challenging.

**TuneForge** is deliberately smaller. It does LoRA/QLoRA fine-tuning, metric-based evaluation, and serving script generation. That is it. No full fine-tuning, no RLHF, no DPO, no multi-node distributed training. What it does, it does in three commands with zero configuration files.

If you need a Swiss Army knife, use Axolotl. If you want a visual workshop, use LLaMA-Factory. If you want the shortest path from "I have a JSONL file" to "I have a served model," and you want every step reproducible and CI-friendly, TuneForge is what I built for you.

---

## What TuneForge Can't Do Yet (And Why I Shipped It Anyway)

I believe in shipping with a known-issues list, so here is mine.

**Single-GPU only.** No multi-GPU or distributed training. If your model does not fit on one GPU even with QLoRA 4-bit quantization, you need a different tool.

**LoRA/QLoRA only.** No full fine-tuning, no RLHF, no DPO, no preference optimization. The roadmap includes some of these, but today, you get LoRA.

**Limited model architecture support.** Works with models that Hugging Face transformers and PEFT support for LoRA -- most popular architectures -- but no architecture-specific optimizations or custom attention implementations.

**Young project.** 38 tests passing and the core workflow is solid, but not battle-tested across thousands of production deployments yet.

**No experiment tracking built in.** No W&B, no MLflow integration yet. Training logs go to stdout via structlog, which is structured and parseable, but it is not a dashboard.

These are real limitations, not future features spun as positives. If any are dealbreakers, I would rather you know now.

But here is why I shipped anyway: the alternative is that people keep copy-pasting training loops from blog posts, keep burning GPU hours on misconfigured runs, and keep writing bespoke evaluation scripts for every project. A tool that does three things well and is honest about what it cannot do is more useful than a tool that promises everything and delivers a stack trace.

---

## Try It Right Now

Every misconfigured training run that crashes at step 4,000 wastes hours of GPU time you will never get back. Every hand-off that relies on "just run my notebook" is a deployment risk. Every fine-tuning project without structured evaluation is guesswork dressed up as engineering.

The base install takes about five seconds:

```bash
pip install tuneforge
tuneforge quickstart
```

That gives you the full dry-run pipeline. When you are ready for real training:

```bash
pip install tuneforge[gpu]
```

Your training data is JSONL with three fields:

```json
{"instruction": "Summarize the following text.", "input": "The cat sat on the mat...", "output": "A cat sat on a mat."}
{"instruction": "Translate to French.", "input": "Hello, world.", "output": "Bonjour, le monde."}
```

From there, the three commands in the second section of this post are your entire workflow.

The project is open source. If you try it and something breaks, open an issue. If you try it and something works well, I would genuinely love to hear about it. And if you have ever spent a late Tuesday night debugging a PEFT config in a 47-cell notebook, I think you will appreciate what three CLI commands and a `--dry-run` flag can do for your workflow.

**Stop burning GPU hours on configs you could have validated for free.**

Build the plan on your laptop. Run the training on your GPU. Serve the model anywhere. That is TuneForge.

---

*TuneForge is open source and available on PyPI. Star the repo, try the quickstart, and let me know what you think.*
