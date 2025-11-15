# Quick Start Guide

This guide will help you get started with the Machine Translation Project in minutes.

## Prerequisites Check

Before starting, ensure you have:
- Python 3.8 or higher
- At least 8GB RAM
- GPU (optional but recommended for training)

## Step 1: Installation (5 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd Machine-Translation-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

## Step 2: Prepare Data (2 minutes)

1. Download or prepare your translation dataset
2. Format: JSONL file with one JSON object per line:
   ```json
   {"chinese": "你好，世界！", "english": "Hello, world!"}
   {"chinese": "今天天气真好", "english": "The weather is nice today"}
   ```
3. Place files in `data/` directory:
   - `data/translation2019zh_train.json`
   - `data/translation2019zh_valid.json`

## Step 3: Quick Test with Pre-trained Model (1 minute)

Test the inference without training:

```bash
python scripts/inference.py \
    --config config.yaml \
    --text "你好，世界！"
```

Expected output:
```
Source: 你好，世界！
Translation: Hello, world!
```

## Step 4: Train a Model (30+ minutes)

### Option A: Use Pre-trained Model (Fastest)

The config is already set to use pre-trained model. Just run:

```bash
python scripts/train.py --config config.yaml
```

### Option B: Train Transformer Model

1. Edit `config.yaml`:
   ```yaml
   model:
     type: "transformer"
   ```

2. Run training:
   ```bash
   python scripts/train.py --config config.yaml
   ```

### Option C: Train RNN Model

1. Edit `config.yaml`:
   ```yaml
   model:
     type: "rnn"
   ```

2. Run training:
   ```bash
   python scripts/train.py --config config.yaml
   ```

## Step 5: Evaluate Your Model (2 minutes)

After training, evaluate on test set:

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output results.json
```

## Step 6: Use Your Model (1 minute)

### Interactive Translation

```bash
python scripts/inference.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

Then type Chinese text and see translations!

## Common Issues

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: "spaCy model not found"
**Solution**: Download models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

### Issue: "Data file not found"
**Solution**: Ensure data files are in `data/` directory with correct names.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Experiment with different hyperparameters in `config.yaml`
- Try different model architectures
- Add your own evaluation metrics

## Getting Help

- Check logs in `logs/training.log`
- Review configuration in `config.yaml`
- Open an issue on GitHub

Happy translating! 🚀

