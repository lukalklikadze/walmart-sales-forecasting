"""
train_val_split.py

Splits the Walmart Recruiting - Store Sales Forecasting `train.csv`
into a training set and a validation set using a TIME-BASED split.

Why time-based (and not random)?
---------------------------------
This is a forecasting problem: the real test set (test.csv) is entirely
in the future relative to train.csv. If you split randomly, your model
gets to "see" weeks that come after the weeks it's being validated on,
which leaks information and gives you an overly optimistic (and
misleading) validation score. Instead, we hold out the most recent
N weeks per the whole dataset as validation -- mirroring how test.csv
relates to train.csv.

Usage
-----
    python train_val_split.py --input train.csv --output-dir ./splits --val-weeks 8

This will produce:
    ./splits/train_split.csv
    ./splits/val_split.csv
"""

import argparse
import os
import pandas as pd


def time_based_split(df: pd.DataFrame, date_col: str = "Date", val_weeks: int = 8):
    """
    Split a dataframe into train/val by holding out the most recent
    `val_weeks` distinct weeks (by date_col) as validation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a date column parseable to datetime.
    date_col : str
        Name of the date column (default 'Date', as in Walmart train.csv).
    val_weeks : int
        Number of most-recent distinct weeks to hold out for validation.

    Returns
    -------
    train_df, val_df : pd.DataFrame, pd.DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    unique_dates = sorted(df[date_col].unique())
    if val_weeks >= len(unique_dates):
        raise ValueError(
            f"val_weeks ({val_weeks}) must be smaller than the number of "
            f"distinct weeks in the data ({len(unique_dates)})."
        )

    cutoff_date = unique_dates[-val_weeks]  # first date that belongs to validation

    train_df = df[df[date_col] < cutoff_date].reset_index(drop=True)
    val_df = df[df[date_col] >= cutoff_date].reset_index(drop=True)

    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Time-based train/val split for Walmart sales forecasting.")
    parser.add_argument("--input", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to write train_split.csv / val_split.csv")
    parser.add_argument("--val-weeks", type=int, default=8, help="Number of most recent weeks to hold out for validation")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows.")

    train_df, val_df = time_based_split(df, val_weeks=args.val_weeks)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train_split.csv")
    val_path = os.path.join(args.output_dir, "val_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train split: {len(train_df):,} rows -> {train_path}")
    print(f"Val split:   {len(val_df):,} rows -> {val_path}")
    if len(train_df) > 0:
        print(f"Train date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    if len(val_df) > 0:
        print(f"Val date range:   {val_df['Date'].min()} to {val_df['Date'].max()}")


if __name__ == "__main__":
    main()
