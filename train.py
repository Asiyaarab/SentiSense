"""SentiSense — root training entry point.

Usage:
    python train.py --sample                  # 30-sec smoke test (built-in tiny CSV)
    python train.py                            # full IMDB 50K (real ~91-92% numbers)
    python train.py --model svm                # use LinearSVC instead of LogReg
    python train.py --data path/to/file.csv    # custom dataset
    python train.py --sample-size 1000         # random subset of N rows

Outputs:
    models/model.pkl
    models/vectorizer.pkl
    models/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

# Allow `python train.py` from project root without a prior `pip install -e .`.
# BUGFIX: this used to insert the repo root itself, but `sentisense` lives
# under `<repo root>/src/sentisense`, so `from sentisense... import ...`
# below only ever worked because CI runs `pip install -e .` first — anyone
# just running `python train.py` after `pip install -r requirements.txt`
# (as the README's quickstart says) hit `ModuleNotFoundError: No module
# named 'sentisense'`.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentisense.config import (  # noqa: E402
    MODELS_DIR,
    settings,
)
from sentisense.data import load_dataset, load_sample_dataset, preprocess_series  # noqa: E402
from sentisense.features import build_vectorizer  # noqa: E402
from sentisense.models import build_classifier  # noqa: E402
from sentisense.models.evaluate import report_evaluation  # noqa: E402
from sentisense.utils import ensure_dir, get_logger  # noqa: E402

logger = get_logger("sentisense.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SentiSense sentiment classifier.")
    p.add_argument(
        "--sample", action="store_true", help="Use the tiny built-in sample dataset (smoke test)."
    )
    p.add_argument(
        "--sample-size", type=int, default=None, help="Use N random rows from the full dataset."
    )
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a custom CSV (must have 'review' and 'sentiment' columns).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "svm"],
        help="Classifier to use. 'svm' (LinearSVC) usually wins by ~1-2 F1.",
    )
    p.add_argument("--C", type=float, default=1.0, help="Regularization strength (default 1.0).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(MODELS_DIR)

    # ---- 1. Load ----
    if args.sample:
        df = load_sample_dataset()
    elif args.data:
        df = load_dataset(args.data)
    else:
        df = load_dataset()

    if args.sample_size and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        logger.info("Subsampled to %d rows", len(df))

    # ---- 2. Preprocess ----
    t0 = time.time()
    df["cleaned"] = preprocess_series(df["text"])
    df = df[df["cleaned"].str.len() > 0]
    logger.info("Preprocessing done in %.1fs", time.time() - t0)

    # ---- 3. Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"],
        df["label"],
        test_size=settings.train.test_size,
        random_state=settings.train.random_state,
        stratify=df["label"] if settings.train.stratify else None,
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # ---- 4. Vectorize ----
    logger.info(
        "Vectorizing (TF-IDF, %d features, ngrams=%s)...",
        settings.vectorizer.max_features,
        settings.vectorizer.ngram_range,
    )
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    logger.info("Vocab size: %d", len(vectorizer.vocabulary_))

    # ---- 5. Train ----
    logger.info("Training %s...", args.model)
    t0 = time.time()
    model = build_classifier(
        name=args.model, config=settings.model.__class__(name=args.model, C=args.C)
    )
    model.fit(X_train_tfidf, y_train)
    logger.info("Training done in %.1fs", time.time() - t0)

    # ---- 6. Evaluate ----
    y_pred = model.predict(X_test_tfidf)
    metrics = report_evaluation(y_test, y_pred)

    # ---- 7. Save ----
    joblib.dump(model, MODELS_DIR / "model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    with (MODELS_DIR / "metrics.json").open("w") as f:
        json.dump({"model": args.model, **metrics}, f, indent=2)
    logger.info("Saved model + vectorizer + metrics to %s", MODELS_DIR)

    # ---- 8. Persist a small eval set for the app to demo with ----
    sample_out = MODELS_DIR / "sample_predictions.json"
    sample = []
    for text, true, pred in zip(X_test.iloc[:5], y_test.iloc[:5], y_pred[:5], strict=False):
        sample.append({"text": str(text)[:200], "true": str(true), "pred": str(pred)})
    sample_out.write_text(json.dumps(sample, indent=2))
    logger.info("Saved demo predictions to %s", sample_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
