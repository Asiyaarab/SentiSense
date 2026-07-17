# SentiSense - Real-Time Movie Review Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Asiyaarab/SentiSense/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](https://github.com/Asiyaarab/SentiSense/blob/main)

> Classifies **50,000+ IMDb movie reviews** as positive or negative using a TF-IDF + Logistic Regression NLP pipeline, deployed as a real-time Streamlit web app.
---

## What it does

Type or paste any movie review -> SentiSense predicts whether the sentiment is **positive** or **negative** in real time, with a confidence score.

**Why it matters:** Sentiment classification is the foundation of brand monitoring, review aggregation, customer feedback triage, and product analytics. This project builds that pipeline end-to-end.

---

## Key features

- Trained on the full IMDb dataset (50,000 labeled reviews)
- Production-style text preprocessing - HTML stripping, lowercasing, stopword removal, lemmatization
- TF-IDF vectorization with n-grams (1, 2) for richer feature capture
- Logistic Regression classifier - fast, interpretable, strong baseline
- Real-time predictions in the browser via Streamlit
- Confidence scores - not just label, but how sure the model is
- Model is trained once (`train.py`) and loaded by the app — no retraining on every run
- Reproducible pipeline - same train/test split, same preprocessing, same numbers every run

---

## Tech stack

| Layer    | Tools                                                                    |
| -------- | ------------------------------------------------------------------------ |
| Language | Python 3.10+                                                             |
| NLP      | NLTK (stopwords, lemmatizer, tokenize)                                   |
| ML       | Scikit-learn (TF-IDF, Logistic Regression)                               |
| Data     | Pandas, NumPy                                                            |
| App      | Streamlit                                                                |
| Dataset  | [IMDb 50K Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) |

---

## Architecture

```
Raw review text
      |
      v
[ 1. Clean ]  ->  strip HTML, lowercase, remove punctuation
      |
      v
[ 2. Tokenize ]  ->  NLTK word_tokenize
      |
      v
[ 3. Normalize ]  ->  remove stopwords, lemmatize
      |
      v
[ 4. Vectorize ]  ->  TF-IDF (1-2 grams)
      |
      v
[ 5. Classify ]  ->  Logistic Regression -> {positive, negative} + probability
      |
      v
[ 6. Display ]  ->  Streamlit UI with confidence bar
```

---

## Results

Evaluation on a held-out 20% test split (`random_state=42` for reproducibility).
Run `python train.py` to regenerate these numbers on your machine.

| Metric              | Class    | Score   |
| -------------------- | -------- | ------- |
| **Accuracy**         | overall  | 89.05 % |
| **F1 (positive)**    | positive | 0.8917  |
| **F1 (negative)**    | negative | 0.8893  |
| **Precision (avg)**  | macro    | 0.8907  |
| **Recall (avg)**     | macro    | 0.8905  |

**Full classification report (held-out 10,000 reviews):**

| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| negative     | 0.8994    | 0.8794 | 0.8893   | 5,000   |
| positive     | 0.8820    | 0.9016 | 0.8917   | 5,000   |
| **accuracy** |           |        | 0.8905   | 10,000  |

**Sample predictions:**

| Review                                                             | Predicted | Confidence |
| -------------------------------------------------------------------- | --------- | ---------- |
| `"This movie was absolutely brilliant. The acting was top-notch."` | Positive  | 83.8 %     |
| `"Boring, predictable, and a complete waste of two hours."`        | Negative  | 99.9 %     |
| `"It was okay - not great, not terrible."`                         | Negative  | 92.6 %     |

> Note: the third review ("okay - not great, not terrible") is genuinely ambiguous/neutral phrasing, but the model still has to pick positive or negative (it wasn't trained on a neutral class) — here it leaned negative with 92.6% confidence, likely because "not great" and "terrible" carry more negative-associated words than "okay" carries positive ones after stopword removal.

---

## Project structure

```
SentiSense/
|-- app.py                  # Streamlit web app (loads trained model, no retraining)
|-- train.py                # Trains model, evaluates it, saves model.pkl + vectorizer.pkl
|-- requirements.txt        # Python dependencies
|-- .gitignore
|-- LICENSE
|-- README.md
|-- WHATS_NEW.md            # Notes on the train.py / app.py update (safe to delete)
```

---

## How to run

### 1. Clone

```
git clone https://github.com/Asiyaarab/SentiSense.git
cd SentiSense
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Install NLTK data (one-time)

```
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 4. Add the dataset

Download `IMDB Dataset.csv` (IMDb 50K Movie Reviews) and place it in the project root.

### 5. Train the model (one-time, or whenever you retrain)

```
python train.py
```

This prints Accuracy / Precision / Recall / F1 and saves `model.pkl` + `vectorizer.pkl`.

### 6. Run the app

```
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Requirements

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
nltk>=3.8
joblib>=1.3
```

---

## What I learned

- **Text preprocessing is 80% of NLP work** - clean text in, clean predictions out. Stripping HTML, normalizing case, and lemmatizing gave a bigger accuracy jump than swapping classifiers.
- **Why TF-IDF still matters** - before reaching for transformers, classical bag-of-words with the right preprocessing often beats them on small/medium datasets and is 100x faster to train.
- **Confidence scores beat hard labels** - a model saying "60% positive" is more useful than a binary yes/no, especially for borderline reviews.
- **Separating training from serving** - retraining a model on every app run doesn't scale. Splitting into `train.py` (train once, save) and `app.py` (load and serve) is the standard production pattern.

---

## Future work

- [ ] Replace Logistic Regression with a fine-tuned DistilBERT and compare
- [ ] Add aspect-based sentiment (positive acting, negative plot)
- [ ] Multi-language support via multilingual embeddings
- [ ] REST API wrapper for production use

---

## Author

**Asiya Arab** - BCA, Shreyarth University - ML Intern @ Webify.ai

- Email: aashiyaarab39@gmail.com
- LinkedIn: linkedin.com/in/asiya-arab
- GitHub: github.com/Asiyaarab

---

## License

MIT - free to use, modify, and learn from.
