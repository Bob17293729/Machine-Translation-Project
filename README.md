# Machine Translation Project

A modern, production-ready machine translation system for Chinese-English translation, implementing multiple architectures including RNN (LSTM), Transformer, and Pre-trained models.

## 🚀 Features

- **Multiple Model Architectures**: 
  - RNN-based Seq2Seq with LSTM encoder-decoder
  - Transformer-based architecture
  - Pre-trained models (Helsinki-NLP/opus-mt-zh-en)
  
- **Modern Engineering Practices**:
  - Modular code structure
  - Configuration-based training
  - Comprehensive logging
  - Evaluation metrics (BLEU, ROUGE)
  - Checkpoint management
  - Reproducible experiments

- **Easy to Use**:
  - Simple command-line interface
  - Interactive inference mode
  - Well-documented code
  - Example scripts

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Machine-Translation-Project.git
cd Machine-Translation-Project
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download spaCy models** (for tokenization):
```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

5. **Prepare data**:
   - Place your training data in `data/translation2019zh_train.json`
   - Place validation data in `data/translation2019zh_valid.json`
   - Data format: JSONL with `{"chinese": "...", "english": "..."}` per line

## 🚀 Quick Start

### Training a Model

```bash
# Train with default configuration
python scripts/train.py --config config.yaml

# Train a specific model type (edit config.yaml first)
python scripts/train.py --config config.yaml
```

### Evaluating a Model

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output evaluation_results.json
```

### Translating Text

```bash
# Single translation
python scripts/inference.py \
    --config config.yaml \
    --text "你好，世界！"

# Interactive mode
python scripts/inference.py \
    --config config.yaml \
    --interactive
```

## 📁 Project Structure

```
Machine-Translation-Project/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   │   ├── dataset.py     # Dataset classes
│   │   └── preprocessing.py  # Tokenization and vocab building
│   ├── models/            # Model definitions
│   │   ├── rnn.py         # RNN-based Seq2Seq
│   │   ├── transformer.py # Transformer model
│   │   └── pretrained.py  # Pre-trained model wrapper
│   ├── utils/             # Utility functions
│   │   ├── logger.py      # Logging utilities
│   │   ├── metrics.py     # Evaluation metrics
│   │   └── helpers.py     # Helper functions
│   └── trainer.py         # Training utilities
├── scripts/                # Executable scripts
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── inference.py       # Inference script
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## ⚙️ Configuration

The project uses YAML configuration files for easy customization. Key configuration sections:

- **data**: Dataset paths and sizes
- **model**: Model architecture and hyperparameters
- **training**: Training parameters (batch size, learning rate, etc.)
- **evaluation**: Evaluation metrics and settings
- **system**: Device, workers, random seed

See `config.yaml` for detailed options and default values.

## 📖 Usage

### Training

1. **Configure your model** in `config.yaml`:
```yaml
model:
  type: "transformer"  # or "rnn" or "pretrained"
  transformer:
    d_model: 256
    nhead: 8
    num_encoder_layers: 6
    num_decoder_layers: 6
    dropout: 0.1
```

2. **Run training**:
```bash
python scripts/train.py --config config.yaml
```

3. **Monitor training**: Check `logs/training.log` for progress

### Evaluation

Evaluate your trained model on the test set:

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --output results.json
```

### Inference

Use the trained model for translation:

```bash
# Command-line mode
python scripts/inference.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --text "今天天气真好"

# Interactive mode
python scripts/inference.py \
    --config config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --interactive
```

## 🏗️ Model Architectures

### 1. RNN-based Seq2Seq

- **Encoder**: Multi-layer LSTM
- **Decoder**: Multi-layer LSTM with attention
- **Features**: Teacher forcing, gradient clipping
- **Best for**: Small datasets, educational purposes

### 2. Transformer

- **Architecture**: Standard Transformer encoder-decoder
- **Features**: Multi-head attention, positional encoding
- **Best for**: Medium to large datasets

### 3. Pre-trained Models

- **Base Model**: Helsinki-NLP/opus-mt-zh-en
- **Features**: Fine-tuning support, beam search
- **Best for**: Production use, best performance

## 📊 Results

### Performance Metrics

| Model | BLEU Score | Training Time | Parameters |
|-------|-----------|---------------|------------|
| RNN (LSTM) | ~15-20 | ~2 hours | ~11M |
| Transformer | ~25-30 | ~4 hours | ~1.7M |
| Pre-trained (Fine-tuned) | ~35-40 | ~1 hour | ~60M |

*Results may vary based on dataset size and training configuration.*

## 🛠️ Development

### Code Style

This project follows PEP 8 style guidelines. Format code with:

```bash
black src/ scripts/
flake8 src/ scripts/
```

### Running Tests

```bash
pytest tests/
```

### Adding New Models

1. Create a new model class in `src/models/`
2. Implement the forward method
3. Add configuration in `config.yaml`
4. Update `scripts/train.py` to support the new model

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: translation2019zh
- Pre-trained models: Helsinki-NLP
- Libraries: PyTorch, Transformers, spaCy

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is designed to showcase modern ML engineering practices and is suitable for:
- Learning machine translation
- Technical interviews
- Portfolio projects
- Research experiments
