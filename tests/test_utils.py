"""Tests for utility functions."""

import torch
import pytest
from src.utils.helpers import set_seed, count_parameters, get_device
from src.utils.metrics import compute_bleu


def test_set_seed():
    """Test random seed setting."""
    set_seed(42)
    a = torch.randn(1)
    set_seed(42)
    b = torch.randn(1)
    assert torch.allclose(a, b)


def test_count_parameters():
    """Test parameter counting."""
    model = torch.nn.Linear(10, 5)
    params = count_parameters(model)
    assert params == 10 * 5 + 5  # weights + bias


def test_get_device():
    """Test device selection."""
    device = get_device()
    assert device in [torch.device("cpu"), torch.device("cuda")]


def test_compute_bleu():
    """Test BLEU score computation."""
    predictions = ["the cat is on the mat", "there is a cat on the mat"]
    references = [["the cat is on the mat"], ["a cat is on the mat"]]
    
    bleu_score = compute_bleu(predictions, references)
    assert 0 <= bleu_score <= 100

