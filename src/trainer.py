"""Training utilities for machine translation models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
from torch.optim import AdamW
import math
import numpy as np
from typing import Dict, Optional, Any, Tuple, List
import os

from src.utils.logger import setup_logger
from src.utils.metrics import compute_bleu
from src.utils.helpers import save_checkpoint, get_device


class Trainer:
    """Trainer class for machine translation models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None,
        logger=None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            valid_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            criterion: Loss function
            device: Device to train on
            config: Training configuration
            logger: Logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device or get_device()
        self.config = config or {}
        self.logger = logger or setup_logger()

        self.model.to(self.device)
        self.best_bleu = 0.0
        self.train_losses = []
        self.valid_losses = []
        self.valid_bleus = []

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False
        )

        for batch_idx, batch_data in enumerate(progress_bar):
            batch_data = self._move_to_device(batch_data)
            loss = self._compute_loss(batch_data)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.get("gradient_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )

            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        predictions = []
        references = []

        with torch.no_grad():
            progress_bar = tqdm(
                self.valid_loader, desc=f"Epoch {epoch+1} [Valid]", leave=False
            )

            for batch_data in progress_bar:
                batch_data = self._move_to_device(batch_data)
                loss = self._compute_loss(batch_data, training=False)
                total_loss += loss.item()

                # Generate predictions for BLEU (only for pretrained models)
                if isinstance(batch_data, dict) and "input_ids" in batch_data:
                    try:
                        preds, refs = self._generate_predictions(batch_data)
                        predictions.extend(preds)
                        references.extend(refs)
                    except Exception as e:
                        self.logger.warning(f"Could not generate predictions: {e}")

        avg_loss = total_loss / len(self.valid_loader)
        self.valid_losses.append(avg_loss)

        metrics = {"loss": avg_loss, "ppl": math.exp(avg_loss)}

        if predictions and references:
            bleu_score = compute_bleu(predictions, [[r] for r in references])
            metrics["bleu"] = bleu_score
            self.valid_bleus.append(bleu_score)

        return metrics

    def _move_to_device(self, data: Any) -> Any:
        """Move data to device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(item) for item in data)
        return data

    def _compute_loss(
        self, batch_data: Dict[str, torch.Tensor], training: bool = True
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        if isinstance(batch_data, dict) and "labels" in batch_data:
            # Pretrained model format
            outputs = self.model(**batch_data)
            return outputs.loss
        else:
            # Custom model format
            if self.criterion is None:
                raise ValueError("Criterion must be provided for custom models")
            # This needs to be implemented based on model type
            raise NotImplementedError("Custom model loss computation")

    def _generate_predictions(
        self, batch_data: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Generate predictions for a batch (for pretrained models).

        Args:
            batch_data: Batch data dictionary

        Returns:
            Tuple of (predictions, references) as lists of strings
        """
        if not isinstance(batch_data, dict) or "input_ids" not in batch_data:
            return [], []

        # Generate tokens
        generated_tokens = self.model.generate(
            batch_data["input_ids"],
            attention_mask=batch_data.get("attention_mask"),
            max_length=self.config.get("max_target_length", 128),
        ).cpu().numpy()

        # Decode predictions
        if hasattr(self.model, "tokenizer"):
            tokenizer = self.model.tokenizer
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

            # Decode references
            label_tokens = batch_data["labels"].cpu().numpy()
            label_tokens = np.where(
                label_tokens != -100,
                label_tokens,
                tokenizer.pad_token_id,
            )
            decoded_labels = tokenizer.batch_decode(
                label_tokens, skip_special_tokens=True
            )

            predictions = [pred.strip() for pred in decoded_preds]
            references = [label.strip() for label in decoded_labels]

            return predictions, references

        return [], []

    def train(self, num_epochs: int, save_dir: Optional[str] = None) -> None:
        """
        Train model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            valid_metrics = self.validate(epoch)

            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Valid Loss: {valid_metrics['loss']:.4f} - "
                f"Valid PPL: {valid_metrics['ppl']:.2f}"
            )

            if "bleu" in valid_metrics:
                self.logger.info(f"Valid BLEU: {valid_metrics['bleu']:.2f}")

            # Save checkpoint
            if save_dir:
                is_best = valid_metrics.get("bleu", 0) > self.best_bleu
                if is_best:
                    self.best_bleu = valid_metrics["bleu"]

                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_metrics["loss"],
                    "valid_bleu": valid_metrics.get("bleu", 0),
                }

                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                best_path = os.path.join(save_dir, "best_model.pt")

                save_checkpoint(checkpoint, checkpoint_path, is_best, best_path)

                if is_best:
                    self.logger.info(f"Saved best model with BLEU: {self.best_bleu:.2f}")

