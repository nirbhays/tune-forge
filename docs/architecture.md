# TuneForge Architecture

## Overview

TuneForge is a CLI that simplifies the fine-tune → evaluate → serve workflow for open-source LLMs. It provides graceful degradation: full functionality with GPU packages, dry-run/planning mode without them.

## C4 Diagrams

### Level 1: System Context

```mermaid
graph TB
    User["ML Engineer"]
    TuneForge["TuneForge<br/>Fine-Tuning CLI"]
    HF["HuggingFace Hub<br/>(model downloads)"]
    Data["Training Data<br/>(JSONL)"]
    Model["Fine-tuned Model<br/>(LoRA adapter)"]
    API["Inference API<br/>(FastAPI)"]

    User -->|"CLI commands"| TuneForge
    TuneForge -->|"download base model"| HF
    Data -->|"training data"| TuneForge
    TuneForge -->|"LoRA adapter"| Model
    TuneForge -->|"serve"| API

    style TuneForge fill:#f59e0b,stroke:#d97706,color:#fff
```

### Level 2: Container Diagram

```mermaid
graph TB
    subgraph TuneForge["TuneForge"]
        CLI["CLI<br/>(click)"]
        FT["FineTuner<br/>(transformers + peft)"]
        Eval["Evaluator<br/>(metrics)"]
        Serve["ModelServer<br/>(script generator)"]
        Config["Config<br/>(pydantic)"]
    end

    CLI --> FT
    CLI --> Eval
    CLI --> Serve
    Config --> FT
    Config --> Eval
    Config --> Serve

    style TuneForge fill:#fffbeb,stroke:#f59e0b
    style FT fill:#f59e0b,color:#fff
    style Eval fill:#10b981,color:#fff
    style Serve fill:#3b82f6,color:#fff
```

### Sequence Diagram: Quickstart Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Config
    participant FT as FineTuner

    User->>CLI: tuneforge quickstart
    CLI->>CLI: generate sample_data.jsonl
    CLI->>Config: FineTuneConfig(model="gpt2", ...)
    CLI->>FT: dry_run(config)
    FT->>FT: validate config
    FT->>FT: estimate resource requirements
    FT-->>CLI: DryRunResult(steps, estimates)
    CLI-->>User: display plan + sample data
```

## Design Decisions

### Graceful Degradation Without GPU Packages

**Chose:** Core CLI works without `transformers`/`peft`/`torch`. These are optional extras (`pip install tuneforge[gpu]`).

**Why:** Many users want to explore the workflow, plan fine-tuning, or generate scripts without a GPU machine. The `--dry-run` flag and `quickstart` command work on any machine.

### Script Generation vs. Direct Serving

**Chose:** Generate standalone serve scripts rather than embedding a full inference server.

**Why:** Inference servers (vLLM, TGI) have complex dependencies. Generating a script lets users customize and run it in their own environment.

## Extension Points

1. Additional model families (Llama, Phi, Gemma)
2. Custom evaluation metrics
3. Integration with eval frameworks (DeepEval, Promptfoo)
4. Direct vLLM/TGI serve mode
