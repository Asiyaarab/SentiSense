FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system deps (only what's needed for most scientific Python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements.txt && pip install -r requirements-api.txt

# Copy the project
COPY pyproject.toml ./
COPY src/ ./src/
COPY train.py ./
COPY data/raw/sample_reviews.csv ./data/raw/sample_reviews.csv

# Install the package itself so `import sentisense` works
RUN pip install -e .

# Pre-download NLTK data so the runtime doesn't need network access
RUN python -c "import nltk; \
    [nltk.download(r, quiet=True) for r in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"

# Streamlit defaults
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501 8000

# Default: run the Streamlit app
CMD ["streamlit", "run", "src/sentisense/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
