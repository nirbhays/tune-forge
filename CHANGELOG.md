# Changelog

All notable changes to TuneForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-15

### Added

- LoRA fine-tuning with 4-bit quantization support
- Dry-run mode for validating training plans without GPU packages
- Evaluation framework with exact_match, contains, and length_ratio metrics
- FastAPI serve script generation for model deployment
- CLI commands: `finetune`, `eval`, `serve`, `quickstart`
- Optional GPU extras for actual training (transformers, peft, accelerate)
- Rich terminal output for plans and results
