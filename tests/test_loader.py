"""Tests for sentisense.data.loader."""


from sentisense.data.loader import load_sample_dataset


def test_load_sample_dataset_has_expected_columns() -> None:
    df = load_sample_dataset()
    assert "text" in df.columns
    assert "label" in df.columns


def test_load_sample_dataset_has_both_classes() -> None:
    df = load_sample_dataset()
    labels = set(df["label"].unique())
    assert "positive" in labels
    assert "negative" in labels


def test_load_sample_dataset_is_nonempty() -> None:
    df = load_sample_dataset()
    assert len(df) > 20


def test_load_sample_dataset_no_nulls_in_required_columns() -> None:
    df = load_sample_dataset()
    assert df["text"].notna().all()
    assert df["label"].notna().all()
