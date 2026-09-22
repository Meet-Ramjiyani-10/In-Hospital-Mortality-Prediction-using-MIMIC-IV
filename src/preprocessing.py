# ============================================================
# preprocessing.py
# ============================================================

import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "mimic_mortality.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "processed.csv"
FEATURE_COLUMNS_PATH = BASE_DIR / "data" / "processed" / "feature_columns.json"

LAB_COLS = [
    "hgb_mean",
    "wbc_mean",
    "platelets_mean",
    "creatinine_mean",
    "sodium_mean",
    "potassium_mean",
    "bicarbonate_mean",
    "glucose_mean",
]
TARGET_COL = "mortality_label"
ID_COLS = ["subject_id", "hadm_id"]


def ensure_directories() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    missing_columns = [col for col in LAB_COLS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required lab columns: {missing_columns}")

    for col in LAB_COLS:
        df[f"{col}_present"] = df[col].notna().astype(int)

    imputer = SimpleImputer(strategy="median")
    df[LAB_COLS] = imputer.fit_transform(df[LAB_COLS])

    df.to_csv(PROCESSED_PATH, index=False)

    feature_columns = [
        column for column in df.columns if column not in {TARGET_COL, *ID_COLS}
    ]
    with FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    print(f"Processed dataset saved to: {PROCESSED_PATH}")
    print(f"Feature metadata saved to: {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
