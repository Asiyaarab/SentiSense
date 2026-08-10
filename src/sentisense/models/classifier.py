"""Classifier factory.

Two interchangeable models:
    - ``logreg`` — Logistic Regression (fast, interpretable baseline)
    - ``svm``    — LinearSVC + calibrated probabilities (usually +1-2% F1)
"""

from __future__ import annotations

from typing import Literal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sentisense.config import ModelConfig

ModelName = Literal["logreg", "svm"]


def build_classifier(
    name: ModelName = "logreg",
    config: ModelConfig | None = None,
):
    """Build a classifier by name.

    LinearSVC doesn't expose ``predict_proba`` natively, so we wrap it in
    CalibratedClassifierCV to give the app a confidence score.
    """
    cfg = config or ModelConfig(name=name)

    if name == "logreg":
        return LogisticRegression(
            max_iter=cfg.max_iter,
            C=cfg.C,
            solver="lbfgs",
        )
    if name == "svm":
        base = LinearSVC(max_iter=cfg.max_iter, C=cfg.C)
        return CalibratedClassifierCV(base, cv=3, n_jobs=cfg.n_jobs)
    raise ValueError(f"Unknown model name: {name!r}. Use 'logreg' or 'svm'.")
