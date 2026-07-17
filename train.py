"""
train.py — Trains the SentiSense sentiment classifier.

Run this once (or whenever the dataset changes) to:
  1. Load and clean the IMDB dataset
  2. Train a TF-IDF + Logistic Regression pipeline
  3. Evaluate it on a held-out test split
  4. Print a classification report (copy these numbers into README.md)
  5. Save the trained model + vectorizer to disk (model.pkl, vectorizer.pkl)

app.py then just LOADS these saved files instead of retraining on every run.

Usage:
    python train.py
"""

import os
import re
import time

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = "IMDB Dataset.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def setup_nltk_resources():
    """Downloads required NLTK resources if not already present."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK resource: {name} ...")
            nltk.download(name, quiet=True)


def preprocess_text(text, lemmatizer, stop_words):
    """Cleans and normalizes a single review string."""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # strip HTML tags
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)  # strip URLs/symbols
    tokens = word_tokenize(text)
    filtered = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1
    ]
    return " ".join(filtered)


def main():
    print("Setting up NLTK resources...")
    setup_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Place the IMDB dataset CSV in this folder."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")
    df.rename(columns={"review": "text", "sentiment": "sentiment"}, inplace=True)
    df.dropna(subset=["text", "sentiment"], inplace=True)

    print(f"Preprocessing {len(df)} reviews (this can take a few minutes)...")
    start = time.time()
    df["cleaned_text"] = df["text"].apply(
        lambda t: preprocess_text(t, lemmatizer, stop_words)
    )
    df = df[df["cleaned_text"].str.len() > 0]
    print(f"Preprocessing done in {time.time() - start:.1f}s")

    X = df["cleaned_text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing (TF-IDF, 1-2 grams)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("\n" + "=" * 60)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=4))
    print("=" * 60)
    print("Copy the numbers above into the Results section of README.md")
    print("=" * 60 + "\n")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved trained model to '{MODEL_PATH}'")
    print(f"Saved fitted vectorizer to '{VECTORIZER_PATH}'")


if __name__ == "__main__":
    main()