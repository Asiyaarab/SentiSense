"""Model evaluation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sentisense.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    """Compute a flat dict of headline metrics for easy reporting."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
    }
    return metrics


def report_evaluation(y_true, y_pred) -> dict[str, float]:
    """Print a full report and return headline metrics."""
    logger.info("=" * 60)
    logger.info("EVALUATION ON HELD-OUT TEST SET")
    logger.info("=" * 60)
    metrics = evaluate_model(y_true, y_pred)
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("F1 (macro): %.4f", metrics["f1_macro"])
    logger.info("Precision (macro): %.4f", metrics["precision_macro"])
    logger.info("Recall (macro): %.4f", metrics["recall_macro"])
    logger.info("\n%s", classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("=" * 60)
    return metrics
