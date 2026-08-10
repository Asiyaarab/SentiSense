# Changelog

All notable changes to SentiSense are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.2] — Bugfix

### Fixed
- **Critical**: `streamlit run src/sentisense/app/streamlit_app.py` (the exact
  command in this README's Quickstart) raised
  `ModuleNotFoundError: No module named 'sentisense'` for anyone who had only
  run `pip install -r requirements.txt` (also per this README) instead of
  `pip install -e .`. The file's `sys.path` fallback computed `parents[3]`,
  which resolves to the repo root — but the `sentisense` package lives under
  `<repo root>/src/sentisense`, so the root alone never made it importable.
  Same off-by-one existed in `train.py` and `tests/conftest.py`; both added
  the repo root instead of `<repo root>/src`. This is why it never showed up
  in CI — CI always runs `pip install -e ".[dev]"` first, which installs the
  package properly and masks the broken fallback path. All three now point
  at `src/` correctly, so `streamlit run ...` and `python train.py` work
  with just `pip install -r requirements.txt`, exactly as documented.

## [2.0.1] — Bugfix

### Fixed
- **Critical**: `preprocess_series()` called `.reset_index(drop=True)` on its
  output. Every caller (`train.py`, `tests/test_classifier.py`,
  `tests/test_predict.py`) does `df["cleaned"] = preprocess_series(df["text"])`
  followed by a length-filter — pandas assigns by index label, so the reset
  index silently paired each surviving row with the *wrong* original row (and
  therefore the wrong label) as soon as any row upstream was dropped for
  being empty after cleaning. This corrupted a portion of the training data
  on every run without raising an error. Fixed by preserving the original
  index; added a regression test in `tests/test_preprocessing.py`.

## [2.0.0] — 2026-07-30

### Changed
- Restructured into a proper `src/sentisense/` package (data, features, models, app, utils).
- Bumped TF-IDF `max_features` from 5K → 50K and added `sublinear_tf`, `min_df`, `max_df`.
- Default model upgraded from plain Logistic Regression → LinearSVC (calibrated) for ~+1-2% F1.
- Preprocessing now expands contractions, collapses repeated chars, and reuses a single
  `clean_text` function across train/serve (no more duplication).

### Added
- `pyproject.toml` with proper packaging, optional `[api]` and `[dev]` extras.
- `Makefile` (`make train-sample`, `train-full`, `test`, `lint`, `api`, `run-app`, `docker`).
- `Dockerfile` + `.dockerignore` for reproducible container builds.
- GitHub Actions CI: lint + test (Python 3.10/3.11/3.12) + smoke-train.
- Pre-commit hooks (ruff + black).
- FastAPI endpoint (`sentisense.app.api`) for production REST serving.
- Built-in sample dataset (`data/raw/sample_reviews.csv`) — `python train.py --sample` runs in ~30s.
- CLI flags: `--sample`, `--sample-size`, `--data`, `--model {logreg,svm}`, `--C`.
- Type hints, docstrings, and structured logging throughout.
- `tests/` with unit + integration tests (pytest).
- `.env.example` for runtime config overrides.
- `CHANGELOG.md` (this file).

## [1.0.0] — 2024-XX-XX

### Added
- Initial release: single `train.py` + `app.py` with TF-IDF + Logistic Regression.
- 89.05% accuracy on the IMDB 50K hold-out.
- Streamlit front-end.
