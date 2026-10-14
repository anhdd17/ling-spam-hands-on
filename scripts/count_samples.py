"""Đếm số lượng mẫu trong Ling-Spam dataset (tổng + theo nhãn)."""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "lingSpam.csv"
LABEL_NAMES = {0: "mẫu 0 (ham)", 1: " mẫu 1 (spam)"}


def count_samples(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    counts = df["Label"].value_counts().sort_index()
    summary = pd.DataFrame({
        "label": counts.index.map(LABEL_NAMES),
        "count": counts.values,
        "percent": (counts.values / counts.sum() * 100).round(2),
    })
    print(f"File: {csv_path}")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số mẫu thiếu Body: {df['Body'].isna().sum()}")
    print(f"Số mẫu trùng Body: {df['Body'].duplicated().sum()}\n")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_PATH)
    count_samples(parser.parse_args().csv_path)
