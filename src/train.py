# ============================================================
# train.py
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

DATA_PATH = "data/processed/processed.csv"
MODEL_PATH = "models/xgb_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(["mortality_label", "subject_id", "hadm_id"], axis=1)
    y = df["mortality_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # Baseline model
    # -----------------------------
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    print("Logistic Regression ROC-AUC:", round(lr_auc, 3))

    # -----------------------------
    # XGBoost model
    # -----------------------------
    scale_pos = len(y_train) / y_train.sum()

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print("XGBoost ROC-AUC:", round(xgb_auc, 3))

    # -----------------------------
    # Save model
    # -----------------------------
    joblib.dump(xgb, MODEL_PATH)
    print("✅ Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
