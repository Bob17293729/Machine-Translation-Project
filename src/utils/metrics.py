"""Evaluation metrics for machine translation."""

from typing import List, Dict, Optional
import evaluate
from sacrebleu.metrics import BLEU
import numpy as np


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    tokenize: Optional[str] = None,
) -> float:
    """
    Compute BLEU score.

    Args:
        predictions: List of predicted translations
        references: List of reference translations (can be multiple per prediction)
        tokenize: Tokenization method for BLEU (e.g., "zh" for Chinese)

    Returns:
        BLEU score
    """
    if tokenize:
        bleu = BLEU(tokenize=tokenize)
        return bleu.corpus_score(predictions, references).score
    else:
        bleu_metric = evaluate.load("bleu")
        # Convert references to list of lists if needed
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]
        return bleu_metric.compute(predictions=predictions, references=references)["bleu"]


def compute_rouge(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores.

    Args:
        predictions: List of predicted translations
        references: List of reference translations

    Returns:
        Dictionary with ROUGE scores
    """
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def compute_metrics(
    predictions: List[str],
    references: List[str],
    metrics: Optional[List[str]] = None,
    tokenize: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute multiple metrics.

    Args:
        predictions: List of predicted translations
        references: List of reference translations
        metrics: List of metric names to compute (default: ["bleu"])
        tokenize: Tokenization method for BLEU

    Returns:
        Dictionary with metric scores
    """
    if metrics is None:
        metrics = ["bleu"]

    results = {}

    if "bleu" in metrics:
        # BLEU expects references as list of lists
        refs = [[ref] for ref in references]
        results["bleu"] = compute_bleu(predictions, refs, tokenize=tokenize)

    if "rouge" in metrics:
        rouge_scores = compute_rouge(predictions, references)
        results.update(rouge_scores)

    return results

