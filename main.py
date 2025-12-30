"""Main entry point for running data loading and preprocessing.

Usage (from project root):

    python main.py

This will:
- Load all CSVs from the `data/` directory (with robust encoding handling).
- Print basic information (shape, columns, head) for each file.
- Attempt to identify a review-containing file and column.
- Clean and tokenize reviews and compute a sentiment score using TextBlob.
"""

from src import preprocess


def main() -> None:
    # You can adjust these if you know the specific file/column names.
    preprocess.run_preprocessing(
        data_dir="data",
        review_file_hint=None,  # e.g. "review" or "comments" if you have such a file
        review_col_candidates=["review", "reviews", "review_text", "text"],
    )


if __name__ == "__main__":
    main()
