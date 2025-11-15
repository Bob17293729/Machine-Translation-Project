"""Tests for model implementations."""

import torch
import pytest
from src.models.rnn import Seq2SeqRNN
from src.models.transformer import TransformerModel
from src.models.pretrained import PretrainedModel


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_rnn_model(device):
    """Test RNN model forward pass."""
    model = Seq2SeqRNN(
        input_dim=1000,
        output_dim=1000,
        emb_dim=128,
        hid_dim=256,
        n_layers=2,
        encoder_dropout=0.1,
        decoder_dropout=0.1,
        device=device,
    )
    model.to(device)

    src = torch.randint(0, 1000, (10, 4)).to(device)  # (src_len, batch_size)
    trg = torch.randint(0, 1000, (12, 4)).to(device)  # (trg_len, batch_size)

    output = model(src, trg)
    assert output.shape == (12, 4, 1000)


def test_transformer_model(device):
    """Test Transformer model forward pass."""
    model = TransformerModel(
        input_dim=1000,
        output_dim=1000,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_length=128,
    )
    model.to(device)

    src = torch.randint(0, 1000, (10, 4)).to(device)  # (src_len, batch_size)
    tgt = torch.randint(0, 1000, (12, 4)).to(device)  # (tgt_len, batch_size)

    output = model(src, tgt)
    assert output.shape == (12, 4, 1000)


def test_pretrained_model():
    """Test pretrained model initialization."""
    model = PretrainedModel(
        model_name="Helsinki-NLP/opus-mt-zh-en",
        fine_tune=False,
    )
    assert model.model is not None
    assert model.tokenizer is not None

