"""Tests for sentisense.models.classifier + end-to-end training on sample data."""

import numpy as np
import pytest

from sentisense.data import load_sample_dataset, preprocess_series
from sentisense.features import build_vectorizer
from sentisense.models.classifier import build_classifier


def test_build_logreg() -> None:
    from sklearn.linear_model import LogisticRegression

    clf = build_classifier("logreg")
    assert isinstance(clf, LogisticRegression)


def test_build_svm_is_calibrated() -> None:
    from sklearn.calibration import CalibratedClassifierCV

    clf = build_classifier("svm")
    assert isinstance(clf, CalibratedClassifierCV)


def test_build_classifier_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        build_classifier("random-forest")  # type: ignore[arg-type]


def test_end_to_end_logreg_on_sample_data() -> None:
    """Full pipeline on the tiny built-in sample — should easily hit >70%."""
    df = load_sample_dataset()
    df["cleaned"] = preprocess_series(df["text"])
    df = df[df["cleaned"].str.len() > 0]
    x_train_text = df["cleaned"].tolist()
    y_train = df["label"].tolist()

    vec = build_vectorizer()
    x_train = vec.fit_transform(x_train_text)
    clf = build_classifier("logreg")
    clf.fit(x_train, y_train)

    preds = clf.predict(x_train)
    acc = (np.array(preds) == np.array(y_train)).mean()
    # On a balanced sample the linear baseline should learn the obvious pattern.
    assert acc > 0.7


def test_svm_with_proba() -> None:
    """LinearSVC wrapped in CalibratedClassifierCV should expose predict_proba."""
    df = load_sample_dataset()
    df["cleaned"] = preprocess_series(df["text"])
    df = df[df["cleaned"].str.len() > 0]
    vec = build_vectorizer()
    x_mat = vec.fit_transform(df["cleaned"].tolist())
    clf = build_classifier("svm")
    clf.fit(x_mat, df["label"].tolist())
    assert hasattr(clf, "predict_proba")
    proba = clf.predict_proba(x_mat[:2])
    assert proba.shape[1] == 2
    assert np.allclose(proba.sum(axis=1), 1.0)
