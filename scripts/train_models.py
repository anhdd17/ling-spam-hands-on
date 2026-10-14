"""Train + đánh giá các mô hình phân loại spam trên ma trận TF-IDF.

Cần chạy trước: python scripts/tfidf_features.py --save
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

MODELS = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LinearSVC": LinearSVC(class_weight="balanced"),
}


def load_data():
    if not (DATA_DIR / "X_train_tfidf.npz").exists():
        raise SystemExit("Chưa có dữ liệu TF-IDF. Chạy: python scripts/tfidf_features.py --save")
    return (
        sparse.load_npz(DATA_DIR / "X_train_tfidf.npz"),
        sparse.load_npz(DATA_DIR / "X_test_tfidf.npz"),
        np.load(DATA_DIR / "y_train.npy"),
        np.load(DATA_DIR / "y_test.npy"),
    )


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    rows, predictions = [], {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_spam": precision_score(y_test, y_pred),
            "recall_spam": recall_score(y_test, y_pred),
            "f1_spam": f1_score(y_test, y_pred),
        })

    results = pd.DataFrame(rows).sort_values("f1_spam", ascending=False)
    print(results.to_string(index=False, float_format="%.4f"))

    for name, y_pred in predictions.items():
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"\n{name} — confusion matrix:")
        print("              dự đoán ham  dự đoán spam")
        print(f"  thật ham    {tn:>11}  {fp:>12}   <- {fp} email thật bị đánh nhầm là spam")
        print(f"  thật spam   {fn:>11}  {tp:>12}   <- {fn} spam lọt qua")

    best = results.iloc[0]["model"]
    print(f"\nMô hình tốt nhất theo F1 (spam): {best}\n")
    print(classification_report(y_test, predictions[best], target_names=["ham", "spam"], digits=4))


if __name__ == "__main__":
    main()
