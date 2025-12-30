import os
import glob
import pandas as pd


def load_and_inspect_data(data_dir: str = "data") -> None:
    """Load all CSV files in data_dir and print basic info.

    - Lists CSV files found.
    - For each file, prints its name, shape, head(), and info().
    """
    abs_data_dir = os.path.abspath(data_dir)
    print(f"Using data directory: {abs_data_dir}")

    csv_paths = sorted(glob.glob(os.path.join(abs_data_dir, "*.csv")))
    if not csv_paths:
        print("No CSV files found in data directory.")
        return

    print("\nFound CSV files:")
    for path in csv_paths:
        print(f" - {os.path.basename(path)}")

    for path in csv_paths:
        print("\n" + "=" * 80)
        print(f"File: {os.path.basename(path)}")
        print("=" * 80)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        print(f"Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())

        print("\nColumn info:")
        df.info()


if __name__ == "__main__":
    load_and_inspect_data()
