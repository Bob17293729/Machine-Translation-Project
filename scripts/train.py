"""Training script for machine translation models."""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW

from src.data.dataset import TranslationDataset
from src.data.preprocessing import collate_fn_pretrained
from src.models.rnn import Seq2SeqRNN
from src.models.transformer import TransformerModel
from src.models.pretrained import PretrainedModel
from src.utils.logger import setup_logger
from src.utils.helpers import set_seed, count_parameters, get_device
from src.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model(config: dict, device: torch.device, logger):
    """Create model based on configuration."""
    model_type = config["model"]["type"].lower()

    if model_type == "rnn":
        model_config = config["model"]["rnn"]
        model = Seq2SeqRNN(
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            emb_dim=model_config["emb_dim"],
            hid_dim=model_config["hid_dim"],
            n_layers=model_config["n_layers"],
            encoder_dropout=model_config["dropout"],
            decoder_dropout=model_config["dropout"],
            device=device,
        )
    elif model_type == "transformer":
        model_config = config["model"]["transformer"]
        model = TransformerModel(
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_encoder_layers=model_config["num_encoder_layers"],
            num_decoder_layers=model_config["num_decoder_layers"],
            dim_feedforward=model_config.get("dim_feedforward", 2048),
            dropout=model_config["dropout"],
            max_seq_length=model_config["max_seq_length"],
        )
    elif model_type == "pretrained":
        model_config = config["model"]["pretrained"]
        model = PretrainedModel(
            model_name=model_config["model_name"],
            fine_tune=model_config["fine_tune"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Created {model_type} model with {count_parameters(model):,} parameters")
    return model


def create_data_loaders(config: dict, model, logger):
    """Create data loaders."""
    data_config = config["data"]
    train_config = config["training"]

    # Load datasets
    train_dataset = TranslationDataset(
        data_config["train_file"],
        max_size=data_config["max_dataset_size"],
    )
    train_data, valid_data = train_dataset.split(
        data_config["train_set_size"], data_config["valid_set_size"]
    )
    test_dataset = TranslationDataset(data_config["test_file"])

    logger.info(
        f"Dataset sizes - Train: {len(train_data)}, "
        f"Valid: {len(valid_data)}, Test: {len(test_dataset)}"
    )

    # Create collate function
    if config["model"]["type"] == "pretrained":
        collate_fn = lambda batch: collate_fn_pretrained(
            batch,
            model.tokenizer,
            model,
            max_length=data_config["max_input_length"],
        )
    else:
        # Custom collate function for RNN/Transformer
        collate_fn = None  # TODO: Implement custom collate function

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["system"].get("num_workers", 0),
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["system"].get("num_workers", 0),
    )

    return train_loader, valid_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train machine translation model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup
    set_seed(config["system"]["seed"])
    device = get_device()
    logger = setup_logger(
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
        format_string=config["logging"]["format"],
    )

    logger.info("Starting training...")
    logger.info(f"Using device: {device}")

    # Create model
    model = create_model(config, device, logger)

    # Create data loaders
    train_loader, valid_loader = create_data_loaders(config, model, logger)

    # Create optimizer and scheduler
    train_config = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 0.01),
    )

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=train_config.get("warmup_steps", 0),
        num_training_steps=train_config["num_epochs"] * len(train_loader),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config,
        logger=logger,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Train
    trainer.train(
        num_epochs=train_config["num_epochs"],
        save_dir=train_config["save_dir"],
    )

    logger.info("Training completed!")


if __name__ == "__main__":
    main()

