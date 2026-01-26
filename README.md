# In-Hospital Mortality Prediction using MIMIC-IV

## Overview

This project focuses on building an interpretable machine learning pipeline to predict **in-hospital mortality** using early clinical data from the MIMIC-IV dataset.

The objective is to estimate patient risk using information available within the **first 24 hours of hospital admission**, following realistic clinical deployment constraints.

---

## Motivation

Early identification of high-risk patients is a critical challenge in hospital care. Delayed recognition of clinical deterioration can lead to poorer outcomes and increased strain on intensive care resources.

Machine learning models, when designed carefully, can assist clinicians by providing early risk estimates. However, healthcare data introduces challenges such as missing values, temporal leakage, and ethical responsibility.

This project was developed to address these challenges in a structured and transparent manner.

---

## Dataset

**Source:**  
MIMIC-IV (v3.1), a large de-identified clinical database published by the MIT Laboratory for Computational Physiology.

**Tables used:**
- admissions
- labevents

Each row in the final dataset corresponds to one hospital admission.

Patient-level data is not included in this repository in accordance with the MIMIC-IV data usage agreement.

---

## Prediction Task

The goal is to predict whether a patient will die during their hospital stay.

**Target variable:**
- `hospital_expire_flag`
  - `1` — patient died during hospitalization
  - `0` — patient survived

This outcome is objective and directly recorded in the dataset.

---

## Feature Engineering

Only information available within the **first 24 hours after admission** is used to prevent temporal leakage.

**Laboratory features included:**
- Hemoglobin
- White blood cell count
- Platelet count
- Creatinine
- Sodium
- Potassium
- Bicarbonate
- Glucose

For each laboratory test:
- the mean value within the first 24 hours is calculated
- a binary indicator is added to represent whether the test was ordered

This approach captures both physiological measurements and clinician decision-making behavior.

---

## Handling Missing Data

Missing values in clinical datasets are rarely random. Laboratory tests are often ordered selectively based on patient condition.

To reflect this reality:
- explicit lab presence indicators are created
- median imputation is applied after indicator creation

This preserves meaningful clinical signal while ensuring model stability.

---

## Modeling Approach

Two models are trained:

**Baseline model**
- Logistic Regression with class weighting

**Final model**
- XGBoost classifier

The baseline model establishes interpretability and a performance reference point, while XGBoost captures nonlinear feature interactions.

---

## Evaluation Strategy

In-hospital mortality is an imbalanced outcome; therefore, accuracy is not used as the primary metric.

Evaluation focuses on:
- ROC-AUC
- precision and recall for the mortality class

These metrics better reflect clinical priorities.

---

## Explainability

Model predictions are interpreted using SHAP values to identify feature contributions.

Explainability helps verify that learned patterns align with established clinical understanding and improves trust in model behavior.

---

## Project Structure

ML_PROJECT/
│
├── data/
│ ├── raw/ # raw extracted dataset (not version controlled)
│ └── processed/ # cleaned dataset for modeling
│
├── notebooks/
│ └── 01_exploration.ipynb # exploratory data analysis
│
├── src/
│ ├── preprocessing.py # feature engineering pipeline
│ └── train.py # model training and evaluation
│
├── models/
│ └── xgb_model.pkl
│
├── requirements.txt
└── README.md



---

## Key Design Principles

- One row per hospital admission  
- Strict temporal separation between features and outcomes  
- No use of ICU-only data prior to outcome occurrence  
- Explicit handling of clinically driven missingness  
- Clear separation between exploration and modeling code  

These principles mirror best practices in applied healthcare machine learning.

---

## Limitations

- Data originates from a single hospital system  
- Only laboratory data is used  
- The model is not intended for clinical deployment  

These limitations are acknowledged to avoid overinterpretation.

---

## Disclaimer

This project is intended strictly for educational and research purposes.

It does not provide medical advice and must not be used in clinical decision-making.

---

## Author

Meet Ramjiyani
Saish Walavalkar
