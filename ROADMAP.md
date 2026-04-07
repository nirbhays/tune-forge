# TuneForge Roadmap

## Vision
Fine-tuning an open-source LLM should be as simple as training a scikit-learn model — three commands, any hardware.

## ✅ Shipped
- LoRA and QLoRA fine-tuning (4-bit quantization)
- Built-in evaluation metrics
- FastAPI model serving
- GPU optional — validate config locally, train on cloud
- 38+ tests

## 🔨 In Progress
- [ ] Mistral / Mixtral base model support
- [ ] W&B (Weights & Biases) training logging
- [ ] GGUF export for llama.cpp deployment

## 📋 Planned — Q2 2025
- [ ] Phi-3 and Gemma model support
- [ ] Dataset formatting wizard (convert raw data → JSONL)
- [ ] DPO (Direct Preference Optimisation) training support
- [ ] One-command cloud training (Modal, Runpod, Lambda Labs)

## 📋 Planned — Q3 2025
- [ ] Model merging (merge LoRA adapters)
- [ ] Automated hyperparameter search
- [ ] vLLM serving integration
- [ ] Benchmark suite (MMLU, HellaSwag) post-training eval

## 💡 Under Consideration
- Web UI for no-code fine-tuning
- HuggingFace Hub push integration
- Axolotl config compatibility layer

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). Model support PRs are very welcome!
