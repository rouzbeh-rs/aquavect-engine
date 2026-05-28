# Fine-tuning Scripts

This directory will contain fine-tuning scripts for Phase 4.

These scripts are separate from the core engine because they depend on
heavy ML dependencies (transformers, peft, bitsandbytes, etc.) that
should not be required to use the simulation engine.

## Planned contents (Phase 4):

- `finetune_qwen.py` — Fine-tune Qwen 3.5 4B on aquavect training data
- `evaluate_model.py` — Evaluate a fine-tuned model on the aquavect benchmark
- `requirements_finetune.txt` — ML-specific dependencies

## Usage (planned):

```bash
# Generate training data first
python -m aquavect generate -n 10000

# Fine-tune
python scripts/finetune_qwen.py \
    --data aquavect_data/training_data.jsonl \
    --model Qwen/Qwen3.5-4B \
    --output models/aquavect-qwen-v1

# Evaluate
python scripts/evaluate_model.py \
    --model models/aquavect-qwen-v1 \
    --benchmark benchmark.json
```
