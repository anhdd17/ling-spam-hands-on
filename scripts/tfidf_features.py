"""Chuyển Ling-Spam sang ma trận TF-IDF (fit trên train, transform trên test)."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "raw" / "lingSpam.csv"
OUT_DIR = ROOT / "data" / "processed"
LABEL_NAMES = {0: "ham", 1: "spam"}


def top_terms(X: sparse.csr_matrix, y: np.ndarray, vocab: np.ndarray, label: int, k: int) -> list[str]:
    """Các từ có TF-IDF trung bình cao nhất trong lớp `label`."""
    mean_scores = np.asarray(X[y == label].mean(axis=0)).ravel()
    return vocab[mean_scores.argsort()[::-1][:k]].tolist()


def main(csv_path: Path, max_features: int, test_size: float, top_k: int, save: bool) -> None:
    df = pd.read_csv(csv_path, index_col=0)
    n_before = len(df)
    df = df.drop_duplicates("Body")
    print(f"Đọc {n_before} mẫu, bỏ {n_before - len(df)} mẫu trùng -> còn {len(df)} mẫu")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Body"], df["Label"], test_size=test_size, stratify=df["Label"], random_state=42
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # chỉ giữ từ chữ cái, dài >= 2
    )
    X_train = vectorizer.fit_transform(X_train_text)  # học từ vựng + IDF chỉ trên train
    X_test = vectorizer.transform(X_test_text)

    density = X_train.nnz / (X_train.shape[0] * X_train.shape[1]) * 100
    print(f"\nX_train: {X_train.shape}  (train: {np.bincount(y_train)} ham/spam)")
    print(f"X_test:  {X_test.shape}  (test:  {np.bincount(y_test)} ham/spam)")
    print(f"Số ô khác 0 trong X_train: {X_train.nnz} ({density:.2f}% — ma trận sparse)")

    vocab = vectorizer.get_feature_names_out()
    for label, name in LABEL_NAMES.items():
        words = top_terms(X_train, y_train.to_numpy(), vocab, label, top_k)
        print(f"\nTop {top_k} từ đặc trưng cho {name}: {', '.join(words)}")

    idf = vectorizer.idf_
    print(f"\n5 từ IDF thấp nhất (phổ biến): {', '.join(vocab[idf.argsort()[:5]])}")
    print(f"5 từ IDF cao nhất (hiếm):       {', '.join(vocab[idf.argsort()[::-1][:5]])}")

    first = X_train[0]
    order = first.data.argsort()[::-1][:5]
    print(f"\nVí dụ mẫu train[0] ({LABEL_NAMES[y_train.iloc[0]]}), 5 từ trọng số cao nhất:")
    for i in order:
        print(f"  {vocab[first.indices[i]]:<15} {first.data[i]:.4f}")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(OUT_DIR / "X_train_tfidf.npz", X_train)
        sparse.save_npz(OUT_DIR / "X_test_tfidf.npz", X_test)
        np.save(OUT_DIR / "y_train.npy", y_train.to_numpy())
        np.save(OUT_DIR / "y_test.npy", y_test.to_numpy())
        joblib.dump(vectorizer, OUT_DIR / "tfidf_vectorizer.joblib")
        print(f"\nĐã lưu ma trận + vectorizer vào {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--save", action="store_true", help="lưu kết quả vào data/processed/")
    args = parser.parse_args()
    main(args.csv_path, args.max_features, args.test_size, args.top_k, args.save)
