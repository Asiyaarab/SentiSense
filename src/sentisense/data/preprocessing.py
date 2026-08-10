"""Text preprocessing pipeline.

Same logic is reused by both training (``train.py``) and serving
(``app.py`` / ``api.py``) so model and app stay in lock-step.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sentisense.utils import ensure_nltk_resources, get_logger

logger = get_logger(__name__)

# Lazy singletons — instantiated on first use to keep module import fast.
_lemma: WordNetLemmatizer | None = None
_stop: set[str] | None = None

# Contractions expand list — small but impactful for IMDB reviews.
_CONTRACTIONS: Final[dict[str, str]] = {
    "won't": "will not",
    "can't": "cannot",
    "shan't": "shall not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
    "y'all": "you all",
}

_HTML_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_URL_RE: Final[re.Pattern[str]] = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z\s]")
_MULTISPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_REPEAT_CHAR_RE: Final[re.Pattern[str]] = re.compile(r"(.)\1{2,}")  # sooooo -> soo


def _expand_contractions(text: str) -> str:
    """Expand English contractions to their full form."""
    for k, v in _CONTRACTIONS.items():
        # word-boundary safe replacement
        text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text


def _normalize_repeats(text: str) -> str:
    """Collapse 3+ repeated characters to 2 (sooooo -> soo)."""
    return _MULTISPACE_RE.sub(" ", _REPEAT_CHAR_RE.sub(r"\1\1", text))


def _get_tools() -> tuple[WordNetLemmatizer, set[str]]:
    """Lazy-load NLTK resources exactly once per process.

    Negation words (``not``, ``no``, ``nor``, ``neither``, ``never``) are
    removed from the standard NLTK stopword set — they carry critical
    sentiment signal (``not good`` vs ``good``).
    """
    global _lemma, _stop
    if _lemma is None or _stop is None:
        ensure_nltk_resources()
        _lemma = WordNetLemmatizer()
        _stop = set(stopwords.words("english")) - {
            "not",
            "no",
            "nor",
            "neither",
            "never",
            "none",
            "cannot",
        }
    return _lemma, _stop


@lru_cache(maxsize=10_000)
def clean_text(text: str) -> str:
    """Clean and normalize a single review string.

    Steps:
        1. Expand contractions
        2. Lowercase
        3. Strip HTML tags
        4. Strip URLs
        5. Collapse repeated characters
        6. Strip non-alpha
        7. Tokenize, remove stopwords, lemmatize
        8. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = _expand_contractions(text)
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _normalize_repeats(text)
    text = _NON_ALPHA_RE.sub(" ", text)

    lemma, stop = _get_tools()
    tokens = word_tokenize(text)
    tokens = [lemma.lemmatize(w) for w in tokens if w not in stop and len(w) > 1]
    return " ".join(tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply ``clean_text`` to a pandas Series, with progress logging.

    Rows that clean to an empty string are dropped. The original index is
    preserved on the rows that survive.

    BUGFIX: this used to end with ``.reset_index(drop=True)``. Every caller
    in this repo does ``df["cleaned"] = preprocess_series(df["text"])`` and
    then filters on ``df["cleaned"].str.len() > 0`` (train.py,
    tests/test_classifier.py, tests/test_predict.py). Because pandas
    assignment aligns by index label, a reset index silently paired each
    surviving row with the *wrong* text as soon as any row upstream had been
    dropped — every row after the first drop shifted by one, scrambling
    text/label pairs without raising an error. Keeping the original index
    means the assignment lines up correctly and the caller's length-filter
    drops exactly (and only) the rows that were actually empty.
    """
    logger.info("Preprocessing %d rows...", len(series))
    cleaned = series.astype(str).map(clean_text)
    mask = cleaned.str.len() > 0
    if (~mask).sum() > 0:
        logger.warning("Dropped %d empty rows after cleaning", int((~mask).sum()))
    return cleaned[mask]
