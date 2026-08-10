"""Prediction helpers for serving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """Single prediction result — easy to serialize / log / display."""

    label: str
    confidence: float  # 0..1
    probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


def predict_with_confidence(model, vectorizer, texts: list[str]) -> list[Prediction]:
    """Run the model on raw texts and return one Prediction per input."""
    if not texts:
        return []

    X = vectorizer.transform(texts)
    labels = model.predict(X)

    # Some models (e.g. raw LinearSVC) don't have predict_proba.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
    else:
        # Fall back to decision_function + softmax.
        from scipy.special import softmax  # local import — optional dep
        decision = model.decision_function(X)
        proba = softmax(decision, axis=1)
        classes = list(getattr(model, "classes_", np.unique(labels)))

    out: list[Prediction] = []
    for label, probs in zip(labels, proba):
        probs_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        out.append(
            Prediction(
                label=str(label),
                confidence=float(max(probs)),
                probabilities=probs_dict,
            )
        )
    return out
