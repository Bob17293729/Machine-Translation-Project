"""RNN-based Seq2Seq model for machine translation."""

import torch
import torch.nn as nn
import random
from typing import Tuple


class Encoder(nn.Module):
    """Encoder module using LSTM."""

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int,
        dropout: float,
    ):
        """
        Initialize encoder.

        Args:
            input_dim: Input vocabulary size
            emb_dim: Embedding dimension
            hid_dim: Hidden dimension
            n_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            src: Source tensor of shape (src_len, batch_size)

        Returns:
            Tuple of (hidden, cell) states
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """Decoder module using LSTM."""

    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int,
        dropout: float,
    ):
        """
        Initialize decoder.

        Args:
            output_dim: Output vocabulary size
            emb_dim: Embedding dimension
            hid_dim: Hidden dimension
            n_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input: Input tensor of shape (batch_size,)
            hidden: Hidden state
            cell: Cell state

        Returns:
            Tuple of (prediction, hidden, cell)
        """
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2SeqRNN(nn.Module):
    """Seq2Seq model with RNN encoder-decoder architecture."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int,
        encoder_dropout: float,
        decoder_dropout: float,
        device: torch.device,
    ):
        """
        Initialize Seq2Seq model.

        Args:
            input_dim: Input vocabulary size
            output_dim: Output vocabulary size
            emb_dim: Embedding dimension
            hid_dim: Hidden dimension
            n_layers: Number of LSTM layers
            encoder_dropout: Encoder dropout probability
            decoder_dropout: Decoder dropout probability
            device: Device to run on
        """
        super().__init__()
        self.encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, encoder_dropout)
        self.decoder = Decoder(
            output_dim, emb_dim, hid_dim, n_layers, decoder_dropout
        )
        self.device = device

    def forward(
        self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Source tensor of shape (src_len, batch_size)
            trg: Target tensor of shape (trg_len, batch_size)
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            Output tensor of shape (trg_len, batch_size, output_dim)
        """
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            input = trg[t] if teacher_force else top1

        return outputs

