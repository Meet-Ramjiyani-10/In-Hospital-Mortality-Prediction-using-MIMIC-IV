# ============================================================
# preprocessing.py
# ============================================================

import pandas as pd
from sklearn.impute import SimpleImputer

RAW_PATH = "data/raw/mimic_mortality.csv"
PROCESSED_PATH = "data/processed/processed.csv"

LAB_COLS = [
    'hgb_mean',
    'wbc_mean',
    'platelets_mean',
    'creatinine_mean',
    'sodium_mean',
    'potassium_mean',
    'bicarbonate_mean',
    'glucose_mean'
]


def main():
    df = pd.read_csv(RAW_PATH)

    # -----------------------------
    # Missingness indicators
    # -----------------------------
    for col in LAB_COLS:
        df[col + "_present"] = df[col].notnull().astype(int)

    # -----------------------------
    # Median imputation
    # -----------------------------
    imputer = SimpleImputer(strategy="median")
    df[LAB_COLS] = imputer.fit_transform(df[LAB_COLS])

    # -----------------------------
    # Save processed data
    # -----------------------------
    df.to_csv(PROCESSED_PATH, index=False)
    print(" Processed dataset saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    main()
