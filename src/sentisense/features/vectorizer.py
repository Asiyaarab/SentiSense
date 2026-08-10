"""TF-IDF feature builder."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from sentisense.config import VectorizerConfig


def build_vectorizer(config: VectorizerConfig | None = None) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer with our project defaults.

    Args:
        config: Optional override. Defaults to ``settings.vectorizer``.
    """
    cfg = config or VectorizerConfig()
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents=cfg.strip_accents,
    )
