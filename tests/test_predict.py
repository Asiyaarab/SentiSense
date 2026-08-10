"""Tests for sentisense.models.predict."""

from sentisense.data import load_sample_dataset, preprocess_series
from sentisense.features import build_vectorizer
from sentisense.models.classifier import build_classifier
from sentisense.models.predict import predict_with_confidence


def _trained_pair():
    df = load_sample_dataset()
    df["cleaned"] = preprocess_series(df["text"])
    df = df[df["cleaned"].str.len() > 0]
    vec = build_vectorizer()
    X = vec.fit_transform(df["cleaned"].tolist())
    clf = build_classifier("logreg")
    clf.fit(X, df["label"].tolist())
    return clf, vec


def test_predict_returns_one_per_input() -> None:
    clf, vec = _trained_pair()
    preds = predict_with_confidence(clf, vec, ["great movie", "awful terrible boring"])
    assert len(preds) == 2


def test_prediction_confidence_between_zero_and_one() -> None:
    clf, vec = _trained_pair()
    preds = predict_with_confidence(clf, vec, ["amazing!", "horrible"])
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0


def test_prediction_has_probabilities_dict() -> None:
    clf, vec = _trained_pair()
    preds = predict_with_confidence(clf, vec, ["amazing!"])
    assert len(preds[0].probabilities) == 2
    assert set(preds[0].probabilities.keys()) == {"positive", "negative"}


def test_prediction_to_dict_serializable() -> None:
    clf, vec = _trained_pair()
    preds = predict_with_confidence(clf, vec, ["amazing!"])
    d = preds[0].to_dict()
    assert "label" in d
    assert "confidence" in d
    assert "probabilities" in d
    assert isinstance(d["confidence"], float)


def test_predict_empty_input_returns_empty() -> None:
    clf, vec = _trained_pair()
    assert predict_with_confidence(clf, vec, []) == []
