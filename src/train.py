# ============================================================
# train.py
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "processed.csv"
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
FEATURE_COLUMNS_PATH = BASE_DIR / "data" / "processed" / "feature_columns.json"

TARGET_COL = "mortality_label"
ID_COLS = ["subject_id", "hadm_id"]


def evaluate_model(y_true: pd.Series, y_proba: pd.Series) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def get_feature_columns() -> list[str]:
    if FEATURE_COLUMNS_PATH.exists():
        with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    df = pd.read_csv(DATA_PATH)
    return [col for col in df.columns if col not in {TARGET_COL, *ID_COLS}]


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    feature_columns = get_feature_columns()
    if not feature_columns:
        raise ValueError("No feature columns found in processed data.")

    X = df[feature_columns]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_probs)

    scale_pos = len(y_train) / y_train.sum()
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate_model(y_test, xgb_probs)

    metrics = {
        "dataset_shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "train_test_split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": 0.2,
            "random_state": 42,
        },
        "feature_columns": feature_columns,
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(xgb, MODEL_PATH)

    print("Logistic Regression metrics:")
    print(json.dumps(lr_metrics, indent=2))
    print("\nXGBoost metrics:")
    print(json.dumps(xgb_metrics, indent=2))
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
