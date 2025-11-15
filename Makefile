.PHONY: help install train eval infer clean format lint test

help:
	@echo "Machine Translation Project - Makefile Commands"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Train model with default config"
	@echo "  make eval        - Evaluate model"
	@echo "  make infer       - Run inference (interactive)"
	@echo "  make format      - Format code with black"
	@echo "  make lint        - Lint code with flake8"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean generated files"

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -m spacy download zh_core_web_sm

train:
	python scripts/train.py --config config.yaml

eval:
	python scripts/evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pt

infer:
	python scripts/inference.py --config config.yaml --interactive

format:
	black src/ scripts/ --line-length 100
	black *.py --line-length 100

lint:
	flake8 src/ scripts/ --max-line-length=100 --extend-ignore=E203

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache

