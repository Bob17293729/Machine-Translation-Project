"""Evaluation script for machine translation models."""

import argparse
import yaml
import torch
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import TranslationDataset
from src.data.preprocessing import collate_fn_pretrained
from src.models.rnn import Seq2SeqRNN
from src.models.transformer import TransformerModel
from src.models.pretrained import PretrainedModel
from src.utils.logger import setup_logger
from src.utils.helpers import load_checkpoint, get_device
from src.utils.metrics import compute_metrics


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

    return model


def evaluate_model(model, test_loader, config, device, logger):
    """Evaluate model on test set."""
    model.eval()
    predictions = []
    references = []
    sources = []

    eval_config = config["evaluation"]
    max_length = config["data"]["max_target_length"]

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}

            # Generate predictions
            if config["model"]["type"] == "pretrained":
                generated_tokens = model.generate(
                    batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    max_length=max_length,
                    num_beams=eval_config.get("beam_size", 5),
                ).cpu().numpy()

                # Decode
                decoded_preds = model.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_sources = model.tokenizer.batch_decode(
                    batch_data["input_ids"].cpu().numpy(),
                    skip_special_tokens=True,
                )

                # Decode labels
                label_tokens = batch_data["labels"].cpu().numpy()
                label_tokens = torch.where(
                    torch.tensor(label_tokens) != -100,
                    torch.tensor(label_tokens),
                    torch.tensor(model.tokenizer.pad_token_id),
                )
                decoded_labels = model.tokenizer.batch_decode(
                    label_tokens.numpy(), skip_special_tokens=True
                )

                sources.extend([s.strip() for s in decoded_sources])
                predictions.extend([p.strip() for p in decoded_preds])
                references.extend([l.strip() for l in decoded_labels])
            else:
                # TODO: Implement evaluation for RNN/Transformer models
                logger.warning("Evaluation for RNN/Transformer models not yet implemented")
                break

    # Compute metrics
    metrics = compute_metrics(
        predictions,
        references,
        metrics=eval_config.get("metrics", ["bleu"]),
        tokenize="zh" if config["data"].get("source_lang") == "zh" else None,
    )

    return metrics, sources, predictions, references


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate machine translation model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup
    device = get_device()
    logger = setup_logger(
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
        format_string=config["logging"]["format"],
    )

    logger.info("Starting evaluation...")
    logger.info(f"Using device: {device}")

    # Create model
    model = create_model(config, device, logger)
    model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create test data loader
    test_dataset = TranslationDataset(config["data"]["test_file"])
    
    if config["model"]["type"] == "pretrained":
        collate_fn = lambda batch: collate_fn_pretrained(
            batch,
            model.tokenizer,
            model,
            max_length=config["data"]["max_input_length"],
        )
    else:
        collate_fn = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["system"].get("num_workers", 0),
    )

    # Evaluate
    metrics, sources, predictions, references = evaluate_model(
        model, test_loader, config, device, logger
    )

    # Print results
    logger.info("Evaluation Results:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name.upper()}: {metric_value:.4f}")

    # Save results
    results = {
        "metrics": metrics,
        "examples": [
            {
                "source": src,
                "prediction": pred,
                "reference": ref,
            }
            for src, pred, ref in zip(sources[:100], predictions[:100], references[:100])
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

