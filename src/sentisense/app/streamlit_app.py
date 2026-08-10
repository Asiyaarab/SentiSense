"""Streamlit front-end for SentiSense.

Run with:
    streamlit run src/sentisense/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly via `streamlit run src/sentisense/app/streamlit_app.py`
# BUGFIX: this was `parents[3]`, which resolves to the *repo root*
# (…/SentiSense), not the `src/` directory the `sentisense` package actually
# lives under (…/SentiSense/src/sentisense). Adding the repo root to
# sys.path does not make `sentisense` importable — only `src.sentisense`
# would be — so this raised `ModuleNotFoundError: No module named 'sentisense'`
# for anyone who followed the README's plain `pip install -r requirements.txt`
# quickstart instead of `pip install -e .`. `parents[2]` is `src/`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from sentisense.config import MODELS_DIR  # noqa: E402
from sentisense.data.preprocessing import clean_text  # noqa: E402
from sentisense.models.predict import predict_with_confidence  # noqa: E402
from sentisense.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"


@st.cache_resource(show_spinner="Loading model...")
def _load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        st.error(
            f"Trained model files not found at:\n  {MODEL_PATH}\n  {VECTORIZER_PATH}\n\n"
            "Run `python train.py` first to train and save the model."
        )
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def main() -> None:
    st.set_page_config(page_title="SentiSense", page_icon="🎬", layout="centered")
    st.title("🎬 SentiSense — Movie Review Sentiment")
    st.caption("TF-IDF + Linear classifier trained on the IMDb 50K corpus.")

    model, vectorizer = _load_artifacts()

    st.write("---")
    st.header("✍️ Enter Your Review")
    user_input = st.text_area(
        "Paste a movie review below:",
        "This was the most brilliant movie I've seen all year. The director's vision was flawless and the acting was phenomenal!",
        height=160,
    )

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a review to analyze.")
            return
        cleaned = clean_text(user_input)
        if not cleaned:
            st.error("Review was empty after cleaning (only stopwords / symbols).")
            return

        prediction = predict_with_confidence(model, vectorizer, [cleaned])[0]
        st.write("---")
        st.header("✨ Prediction Result")
        col1, col2 = st.columns([1, 4])
        is_pos = prediction.label.lower() == "positive"
        col1.metric(
            "Sentiment",
            prediction.label.upper(),
            delta="👍" if is_pos else "👎",
        )
        message = f"**{prediction.confidence * 100:.1f}% confident this review is {prediction.label.upper()}**"
        (col2.success if is_pos else col2.error)(message)
        st.progress(min(max(prediction.confidence, 0.0), 1.0))
        with st.expander("Model Insights", expanded=False):
            st.caption("Cleaned review:")
            st.code(cleaned, language="text")
            st.caption("Class probabilities:")
            st.json(prediction.to_dict()["probabilities"])

    st.write("---")
    with st.expander("How it works"):
        st.markdown(
            """
            **Pipeline:**
            1. **Clean** — strip HTML / URLs, expand contractions, lowercase, normalize repeats
            2. **Tokenize** — NLTK word tokenizer
            3. **Lemmatize** — WordNet lemmatizer + stopword removal
            4. **Vectorize** — TF-IDF (1-2 grams, 50K features, sublinear TF)
            5. **Classify** — Logistic Regression or LinearSVC (calibrated)
            6. **Display** — label + confidence + probabilities

            The model is trained once via `python train.py` and loaded here.
            """
        )


if __name__ == "__main__":
    main()
else:
    # Streamlit runs the file as a script, so we still call main().
    main()
