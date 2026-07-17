# What's new in this update

This folder contains an updated, more "production-style" version of the app.
Two changes from before:

## 1. `train.py` (new file)

A separate script that:
- Loads `IMDB Dataset.csv`
- Cleans/preprocesses the text
- Trains the TF-IDF + Logistic Regression model
- **Evaluates it on a held-out test split and prints Accuracy, Precision, Recall, F1**
- Saves the trained model to `model.pkl` and the fitted vectorizer to `vectorizer.pkl`

Run this once (or whenever you retrain):
```
python train.py
```

Copy the printed numbers into the **Results** section of your `README.md`
(replace the `[FILL_IN]` placeholders with real Accuracy / F1 / Precision / Recall).

## 2. `app.py` (updated)

- No longer retrains the model on every run — it just **loads** `model.pkl` and
  `vectorizer.pkl` (much faster startup).
- Now shows a **confidence score** (e.g. "92.3% confident this review is POSITIVE")
  with a progress bar, matching what the README already promises.

## Updated run order

```
pip install -r requirements.txt
python train.py          # train once, saves model.pkl + vectorizer.pkl
streamlit run app.py      # loads the saved model, no retraining
```

## Note on `.gitignore`

`model.pkl`, `vectorizer.pkl`, and `IMDB Dataset.csv` are excluded from git
(they're either large or regenerable). Each person who clones the repo should
run `train.py` locally once to generate them. If you'd rather ship the trained
model with the repo instead, remove those two lines from `.gitignore`.

You can delete this file (`WHATS_NEW.md`) once you've read it — it's just a
note for this update, not part of the permanent project docs.
