"""Wrapper for pretrained translation models."""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional


class PretrainedModel(nn.Module):
    """Wrapper for pretrained HuggingFace translation models."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-zh-en",
        fine_tune: bool = True,
    ):
        """
        Initialize pretrained model.

        Args:
            model_name: HuggingFace model identifier
            fine_tune: Whether to fine-tune the model (trainable parameters)
        """
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            **kwargs: Model inputs (input_ids, attention_mask, labels, etc.)

        Returns:
            Model outputs
        """
        return self.model(**kwargs)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 5,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate translations.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Beam search size
            **kwargs: Additional generation parameters

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Prepare decoder input IDs from labels (shifted for teacher forcing).

        Args:
            labels: Label token IDs

        Returns:
            Decoder input IDs
        """
        return self.model.prepare_decoder_input_ids_from_labels(labels)

