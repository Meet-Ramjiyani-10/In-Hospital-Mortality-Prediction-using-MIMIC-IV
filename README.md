In-Hospital Mortality Prediction using MIMIC-IV
Overview

This project focuses on building an interpretable machine learning pipeline to predict in-hospital mortality using early clinical data from the MIMIC-IV dataset.

The motivation behind this work is to understand how routinely collected hospital information, available early during admission, can be used to support risk stratification. Rather than aiming for maximum predictive performance, the project emphasizes correct problem formulation, realistic data handling, and explainability — all of which are essential when working with healthcare data.

Motivation

Hospitals manage large volumes of patients with varying levels of clinical risk. While some patients deteriorate rapidly, early warning signs may not always be obvious during initial assessment.

Predictive models trained on historical data can help identify patterns associated with poor outcomes and assist clinicians in prioritizing care. However, building such models requires careful attention to data leakage, outcome definition, and missing data mechanisms.

This project explores these challenges using a real-world clinical dataset.

Dataset

The dataset used in this project is MIMIC-IV (version 3.1), a large, publicly available, de-identified database containing hospital and ICU data from Beth Israel Deaconess Medical Center.

Only hospital-level tables are used to ensure that all features are available before the outcome occurs.

Tables utilized

admissions

labevents

Each row in the final modeling table corresponds to a single hospital admission.

Prediction Task

The objective is to predict whether a patient will die during their hospital stay.

Target variable

hospital_expire_flag

1 — patient died during hospitalization

0 — patient survived

This outcome is objective, clearly defined, and directly recorded in the dataset, making it suitable for supervised learning.

Feature Engineering

To reflect realistic clinical deployment, only data available within the first 24 hours after hospital admission is used.

Laboratory features

Early laboratory values were extracted and aggregated for the following measurements:

Hemoglobin

White blood cell count

Platelet count

Creatinine

Sodium

Potassium

Bicarbonate

Glucose

For each laboratory test:

the mean value within the first 24 hours was calculated

an indicator variable was added to represent whether the test was ordered

This approach captures both physiological information and clinician decision-making behavior.

Handling Missing Data

Missing values in clinical datasets are rarely random. In many cases, laboratory tests are not ordered because the patient appears clinically stable.

Instead of discarding such records, this project treats missingness as informative by:

creating explicit presence indicators for each laboratory feature

applying median imputation only after indicator creation

This strategy preserves useful signal while maintaining model robustness.

Modeling Approach

Two models were developed:

Baseline model

Logistic Regression with class weighting

This model serves as a transparent baseline and helps establish whether meaningful signal exists in the data.

Final model

XGBoost classifier

The tree-based model captures nonlinear interactions between laboratory features and improves predictive performance while remaining interpretable through post-hoc explanation methods.

Evaluation Strategy

In-hospital mortality is an imbalanced outcome, so accuracy is not used as the primary metric.

Model performance is evaluated using:

ROC-AUC

precision and recall for the mortality class

These metrics better reflect the clinical importance of identifying high-risk patients.

Model Explainability

To understand the behavior of the trained model, SHAP values are used to analyze feature contributions.

This step helps verify whether learned patterns align with known clinical intuition, such as increased risk associated with renal dysfunction or metabolic abnormalities.

Explainability is treated as a core component of the modeling process rather than an optional add-on.

Project Structure
ML_PROJECT/
│
├── data/
│   ├── raw/                 # raw extracted dataset
│   └── processed/           # cleaned and imputed data
│
├── notebooks/
│   └── 01_exploration.ipynb  # exploratory data analysis
│
├── src/
│   ├── preprocessing.py     # feature engineering pipeline
│   └── train.py              # model training and evaluation
│
├── models/
│   └── xgb_model.pkl
│
├── requirements.txt
└── README.md

Key Design Decisions

One row per hospital admission

Strict temporal separation between features and outcomes

No use of ICU-only data prior to ICU admission

Explicit handling of missing clinical information

Separation of exploration, preprocessing, and modeling code

These decisions mirror practices used in applied healthcare machine learning.

Limitations

The model is trained on data from a single hospital system

Only laboratory data is used; vital signs and clinical notes are not included

The model is not intended for real-world deployment

These limitations are acknowledged to avoid overinterpretation of results.

Disclaimer

This project is intended strictly for academic and educational purposes.

It does not provide medical advice and must not be used in clinical decision-making.

Summary

This project demonstrates an end-to-end machine learning workflow using real-world healthcare data, with a focus on:

careful outcome definition

realistic data constraints

meaningful feature engineering

transparent evaluation

model interpretability

