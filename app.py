import os
import re
import time

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# --- 1. Resource setup (cached so it only runs once per session) ---


@st.cache_resource(show_spinner=False)
def setup_nltk_resources():
    """Checks and downloads necessary NLTK resources robustly."""
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
    }
    with st.spinner("Checking NLTK resources..."):
        for name, path in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                st.info(f"Downloading required NLTK resource: '{name}'...")
                try:
                    nltk.download(name, quiet=True)
                except Exception as e:
                    st.error(f"Failed to download '{name}': {e}")


setup_nltk_resources()


@st.cache_resource
def get_nlp_tools():
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    return lemmatizer, stop_words


lemmatizer, stop_words = get_nlp_tools()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    filtered = [
        lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1
    ]
    return " ".join(filtered)


# --- 2. Load the pre-trained model + vectorizer (no retraining here) ---


@st.cache_resource
def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error(
            f"Trained model files not found ('{MODEL_PATH}', '{VECTORIZER_PATH}'). "
            "Run `python train.py` first to train and save the model."
        )
        st.stop()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, tfidf_vectorizer = load_model_and_vectorizer()

# --- 3. Streamlit layout ---

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("A Machine Learning app built with Python, scikit-learn, and **Streamlit**.")
st.write("---")

st.header("✍️ Enter Your Review")
user_input = st.text_area(
    "Paste a movie review below:",
    "This was the most brilliant movie I've seen all year. The director's vision was flawless and the acting was phenomenal!",
    height=150,
)

if st.button("Analyze Sentiment", use_container_width=True, type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            cleaned_input = preprocess_text(user_input)

            if not cleaned_input:
                st.error(
                    "The review was too short or contained only stop words/symbols "
                    "after cleaning. Cannot predict."
                )
            else:
                input_tfidf = tfidf_vectorizer.transform([cleaned_input])

                final_prediction = model.predict(input_tfidf)[0]
                probabilities = model.predict_proba(input_tfidf)[0]
                confidence = max(probabilities) * 100

                st.write("---")
                st.header("✨ Prediction Result")

                col1, col2 = st.columns([1, 4])

                if final_prediction == "positive":
                    col1.metric(label="Sentiment", value="POSITIVE", delta="👍")
                    col2.success(
                        f"**{confidence:.1f}% confident this review is POSITIVE**"
                    )
                else:
                    col1.metric(label="Sentiment", value="NEGATIVE", delta="👎")
                    col2.error(
                        f"**{confidence:.1f}% confident this review is NEGATIVE**"
                    )

                st.progress(confidence / 100)

                st.subheader("Model Insights:")
                st.caption("Review after preprocessing:")
                st.code(cleaned_input, language="text")

st.write("---")
st.markdown(
    """
    **How it works:**
    The application uses a **Logistic Regression** classifier trained on 50,000 IMDB movie reviews.
    1. **Preprocessing** cleans the text (removes HTML tags, symbols, stop words, and lemmatizes words).
    2. **TF-IDF Vectorizer** converts the cleaned text into a numerical format, giving importance scores to different words.
    3. The **Model** uses these scores to determine if the review leans positive or negative, along with a confidence score.

    The model is trained once via `train.py` and loaded here — it is not retrained on every run.
    """
)