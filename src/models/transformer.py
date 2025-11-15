"""Transformer model for machine translation."""

import torch
import torch.nn as nn
import math
from typing import Optional
from einops import rearrange


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer model for sequence-to-sequence translation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 128,
    ):
        """
        Initialize Transformer model.

        Args:
            input_dim: Input vocabulary size
            output_dim: Output vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.linear = nn.Linear(d_model, output_dim)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_length)
        self.d_model = d_model

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.7,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src: Source tensor of shape (src_len, batch_size)
            tgt: Target tensor of shape (tgt_len, batch_size)
            teacher_forcing_ratio: Teacher forcing ratio (not used in this implementation)
            src_key_padding_mask: Source padding mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            tgt_mask: Target mask for causal attention

        Returns:
            Output tensor of shape (tgt_len, batch_size, output_dim)
        """
        src_emb = self.pos_enc(
            self.src_embedding(src) * math.sqrt(self.d_model)
        )
        tgt_emb = self.pos_enc(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        out = self.linear(out)
        return out

    @staticmethod
    def generate_nopeek_mask(length: int, device: torch.device) -> torch.Tensor:
        """
        Generate no-peek mask for causal attention.

        Args:
            length: Sequence length
            device: Device to create mask on

        Returns:
            Mask tensor
        """
        mask = rearrange(torch.triu(torch.ones(length, length)) == 1, "h w -> w h")
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.to(device)

