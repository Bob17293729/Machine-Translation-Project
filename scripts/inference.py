"""Inference script for machine translation models."""

import argparse
import yaml
import torch

from src.models.pretrained import PretrainedModel
from src.utils.logger import setup_logger
from src.utils.helpers import load_checkpoint, get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def translate_text(model, text: str, max_length: int = 128, device=None):
    """Translate a single text."""
    model.eval()
    device = device or get_device()

    # Tokenize input
    inputs = model.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Generate translation
    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
        )

    # Decode output
    translation = model.tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )[0]

    return translation


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Translate text using trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional for pretrained models)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to translate",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup
    device = get_device()
    logger = setup_logger(level=config["logging"]["level"])

    logger.info("Initializing model...")

    # Create model
    if config["model"]["type"] == "pretrained":
        model = PretrainedModel(
            model_name=config["model"]["pretrained"]["model_name"],
            fine_tune=config["model"]["pretrained"]["fine_tune"],
        )
    else:
        raise NotImplementedError("Inference only supported for pretrained models currently")

    model.to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Translate
    if args.interactive:
        print("Enter Chinese text to translate (type 'quit' to exit):")
        while True:
            text = input("\n> ")
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text.strip():
                continue

            translation = translate_text(
                model,
                text,
                max_length=config["data"]["max_target_length"],
                device=device,
            )
            print(f"Translation: {translation}")
    else:
        translation = translate_text(
            model,
            args.text,
            max_length=config["data"]["max_target_length"],
            device=device,
        )
        print(f"Source: {args.text}")
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()

