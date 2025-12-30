import os
import glob
import re
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from textblob import TextBlob

try:
    import nltk
    from nltk.corpus import stopwords

    try:
        _ = stopwords.words("english")
    except LookupError:  # download at runtime if not present
        nltk.download("stopwords")
except Exception:  # keep optional to avoid hard failures
    nltk = None
    stopwords = None


# ------------------------------------------------------------
# Utility: stopwords and basic text cleaning
# ------------------------------------------------------------

def _get_stopwords() -> set:
    """Return a set of English stopwords (fallback to small built-in list)."""
    if stopwords is not None:
        try:
            return set(stopwords.words("english"))
        except LookupError:
            pass

    return {
        "a",
        "an",
        "the",
        "and",
        "or",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "for",
        "it",
        "this",
        "that",
    }


ENGLISH_STOPWORDS = _get_stopwords()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(text: Optional[str]) -> str:
    """Clean review text: lowercase, remove punctuation/digits/stopwords.

    Returns a cleaned string (no tokenization yet).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    return " ".join(tokens)


def tokenize(text: Optional[str]) -> List[str]:
    """Simple whitespace-based tokenization after basic cleaning."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def sentiment_score(text: Optional[str]) -> float:
    """Compute TextBlob sentiment polarity in [-1, 1]."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


# ------------------------------------------------------------
# Data loading & inspection
# ------------------------------------------------------------

def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a CSV trying multiple encodings to avoid common encoding errors."""
    encodings_to_try = ["utf-8", "utf-8-sig", "latin1"]
    last_error: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # pragma: no cover - diagnostic
            last_error = e
    raise RuntimeError(f"Failed to read {path} with common encodings: {last_error}")


def load_all_csvs(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """Load all CSV files in a directory into a dict: {name: DataFrame}."""
    abs_dir = os.path.abspath(data_dir)
    csv_paths = sorted(glob.glob(os.path.join(abs_dir, "*.csv")))

    if not csv_paths:
        print(f"No CSV files found in {abs_dir}")
        return {}

    print(f"Using data directory: {abs_dir}")
    print("Found CSV files:")
    for p in csv_paths:
        print(f" - {os.path.basename(p)}")

    dataframes: Dict[str, pd.DataFrame] = {}
    for path in csv_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print("\n" + "=" * 80)
        print(f"Loading: {path}")
        print("=" * 80)
        try:
            df = read_csv_robust(path)
        except Exception as e:
            print(f"  Error reading {path}: {e}")
            continue

        dataframes[name] = df

        print(f"Shape: {df.shape}")
        print("Columns:", list(df.columns))
        print("\nFirst 5 rows:")
        print(df.head())

    return dataframes


# ------------------------------------------------------------
# Preprocessing pipeline for reviews
# ------------------------------------------------------------

def preprocess_reviews_df(
    df: pd.DataFrame,
    review_col: str,
    new_clean_col: str = "review_clean",
    new_tokens_col: str = "review_tokens",
    new_sentiment_col: str = "sentiment_score",
) -> pd.DataFrame:
    """Apply text cleaning, tokenization, and sentiment to a reviews DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input data containing a review text column.
    review_col : str
        Name of the column with raw review text.
    new_clean_col : str
        Column name for cleaned text.
    new_tokens_col : str
        Column name for tokenized text (list of tokens).
    new_sentiment_col : str
        Column name for TextBlob sentiment polarity.
    """
    if review_col not in df.columns:
        raise KeyError(f"Column '{review_col}' not found in DataFrame.")

    out = df.copy()
    out[new_clean_col] = out[review_col].apply(clean_text)
    out[new_tokens_col] = out[review_col].apply(tokenize)
    out[new_sentiment_col] = out[review_col].apply(sentiment_score)

    return out


def run_preprocessing(
    data_dir: str = "data",
    review_file_hint: Optional[str] = None,
    review_col_candidates: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """High-level helper to load data, inspect, and preprocess review text.

    Parameters
    ----------
    data_dir : str
        Directory where CSV files live.
    review_file_hint : str, optional
        Substring hint to choose which CSV likely contains reviews (e.g. 'review').
    review_col_candidates : list of str, optional
        Candidate column names for review text (e.g. ['review', 'reviews', 'text']).

    Returns
    -------
    all_data : dict
        Mapping from CSV base name to raw DataFrame.
    processed_reviews : DataFrame or None
        Preprocessed reviews DataFrame if a suitable file/column was found.
    """
    all_data = load_all_csvs(data_dir=data_dir)
    processed_reviews: Optional[pd.DataFrame] = None

    if not all_data:
        return all_data, None

    # Auto-detect a likely review file
    review_df_name = None
    if review_file_hint:
        for name in all_data.keys():
            if review_file_hint.lower() in name.lower():
                review_df_name = name
                break

    if review_df_name is None:
        # Fallback: just pick the main steam file if present, else first one
        if "steam" in all_data:
            review_df_name = "steam"
        else:
            review_df_name = next(iter(all_data.keys()))

    df_reviews = all_data[review_df_name]
    print("\nSelected review DataFrame:", review_df_name)
    print("Columns:", list(df_reviews.columns))

    # Detect review column
    if review_col_candidates is None:
        review_col_candidates = ["review", "reviews", "review_text", "text"]

    review_col = None
    for cand in review_col_candidates:
        if cand in df_reviews.columns:
            review_col = cand
            break

    if review_col is None:
        print("Could not find a review text column in", review_df_name)
        print("Tried candidates:", review_col_candidates)
        print("Please inspect the columns above and choose the correct review column.")
        return all_data, None

    print(f"Using review column: '{review_col}'")

    processed_reviews = preprocess_reviews_df(df_reviews, review_col=review_col)
    print("\nPreview of processed reviews (first 5 rows):")
    cols_to_show = [review_col, "review_clean", "review_tokens", "sentiment_score"]
    existing_cols = [c for c in cols_to_show if c in processed_reviews.columns]
    print(processed_reviews[existing_cols].head())

    return all_data, processed_reviews


if __name__ == "__main__":
    # When run directly, perform loading, inspection, and basic preprocessing.
    run_preprocessing(data_dir="data")
