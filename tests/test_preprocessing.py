"""Tests for sentisense.data.preprocessing."""

import pandas as pd

from sentisense.data.preprocessing import clean_text, preprocess_series


def test_clean_text_lowercases() -> None:
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_strips_html() -> None:
    assert "<br>" not in clean_text("Great <br> movie <b>amazing</b>")
    assert "<b>" not in clean_text("Great <br> movie <b>amazing</b>")


def test_clean_text_strips_urls() -> None:
    out = clean_text("Check this out https://example.com it is great")
    assert "http" not in out and "example.com" not in out
    assert "check" in out


def test_clean_text_strips_punctuation() -> None:
    out = clean_text("Wow!!! Really?? The movie was... amazing.")
    # Punctuation removed; words preserved.
    assert "!" not in out and "?" not in out and "." not in out
    assert "amazing" in out


def test_clean_text_expands_contractions() -> None:
    out = clean_text("I can't believe it's not butter")
    # "not" must be preserved (critical for sentiment) — the WordNet lemmatizer
    # turns "cannot" into the stopword "can", but "not" itself is kept.
    assert "not" in out
    # Stopwords stripped: "I", "it", "is" should be gone.
    padded = f" {out} "
    assert " i " not in padded
    assert " is " not in padded
    assert " it " not in padded


def test_clean_text_collapses_repeats() -> None:
    # "Sooooo" = 6 o's. Our regex collapses 3+ repeats to 2.
    out = clean_text("Sooooo good")
    assert "soo" in out  # 3+ -> 2


def test_clean_text_handles_empty() -> None:
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_clean_text_handles_non_string() -> None:
    assert clean_text(123) == ""  # type: ignore[arg-type]


def test_clean_text_removes_stopwords() -> None:
    out = clean_text("the and a is it of for to")
    # All of these are stopwords; output should be empty (or just whitespace).
    assert out.strip() == ""


def test_preprocess_series_drops_empty() -> None:
    s = pd.Series(["Great movie!", "", "   ", "Terrible, awful!"])
    out = preprocess_series(s)
    # 2 valid rows after dropping blanks.
    assert len(out) == 2
    assert all(isinstance(v, str) for v in out)


def test_preprocess_series_preserves_index_for_correct_realignment() -> None:
    """Regression test for a real bug: preprocess_series used to
    `.reset_index(drop=True)`, so `df["cleaned"] = preprocess_series(df["text"])`
    silently paired surviving rows with the wrong original row (and thus the
    wrong label) as soon as any row upstream was dropped as empty.
    """
    df = pd.DataFrame({
        "text": ["good", "", "bad", "", "great"],
        "label": ["pos", "neg", "neg", "neg", "pos"],
    })
    df["cleaned"] = preprocess_series(df["text"])
    df = df[df["cleaned"].str.len() > 0]

    # Every surviving row's cleaned text must still match its own original text.
    assert set(df["cleaned"]) == {"good", "bad", "great"}
    assert dict(zip(df["text"], df["cleaned"])) == {
        "good": "good", "bad": "bad", "great": "great",
    }
    # The two empty rows (and only those) were dropped — labels intact.
    assert len(df) == 3
    assert df.loc[df["text"] == "bad", "label"].iloc[0] == "neg"
