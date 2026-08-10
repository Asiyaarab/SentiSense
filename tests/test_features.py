"""Tests for sentisense.features.vectorizer."""

from sentisense.config import VectorizerConfig
from sentisense.features.vectorizer import build_vectorizer


def test_build_vectorizer_default_returns_tfidf() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = build_vectorizer()
    assert isinstance(v, TfidfVectorizer)


def test_build_vectorizer_override() -> None:
    cfg = VectorizerConfig(max_features=100, ngram_range=(1, 1))
    v = build_vectorizer(cfg)
    assert v.max_features == 100
    assert v.ngram_range == (1, 1)


def test_vectorizer_fit_transform_shape() -> None:
    # Use a small-corpus-friendly config (default max_df=0.95 needs more docs).
    from sentisense.config import VectorizerConfig

    cfg = VectorizerConfig(min_df=1, max_df=1.0, ngram_range=(1, 1))
    v = build_vectorizer(cfg)
    corpus = [
        "great movie loved it",
        "terrible waste of time",
        "amazing performance",
        "boring slow plot",
    ]
    X = v.fit_transform(corpus)
    assert X.shape[0] == 4
    assert X.shape[1] > 0
    # Sparse matrix
    assert hasattr(X, "toarray")


def test_vectorizer_consistent_vocabulary() -> None:
    from sentisense.config import VectorizerConfig

    cfg = VectorizerConfig(min_df=1, max_df=1.0, ngram_range=(1, 1))
    v = build_vectorizer(cfg)
    v.fit(["hello world", "world peace"])
    X_train = v.transform(["hello peace"])
    assert X_train.shape[1] == len(v.vocabulary_)
