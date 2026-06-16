# Fraud-detection
# Fraud Detection — Adey Innovations Inc.

**10 Academy KAIM Week 5 & 6 Challenge**
Improved Detection of Fraud Cases for E-commerce and Bank Transactions

---

## Overview

This project builds end-to-end fraud detection pipelines for two very different transaction streams:

- **E-commerce transactions** (`Fraud_Data.csv`) — rich behavioral features: signup time, IP address, device, browser, purchase history.
- **Bank credit card transactions** (`creditcard.csv`) — anonymized PCA features (V1–V28) for privacy.

Each stream is treated as a **separate modelling problem** with its own preprocessing, feature engineering, model training, and SHAP explainability pipeline.

---

## Project Structure

```
fraud-detection/
├── data/                        # Add to .gitignore — not committed
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   ├── creditcard.csv
│   └── processed_fraud.csv      # Output of Task 1
├── notebooks/
│   ├── eda-fraud-data.ipynb     # Task 1 — EDA & preprocessing (Fraud_Data)
│   ├── eda-creditcard.ipynb     # Task 1 — EDA (creditcard)
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb           # Task 2 — Model building & evaluation
│   └── shap-explainability.ipynb # Task 3 — SHAP analysis
├── models/                      # Saved model artifacts & plots
│   ├── rf_fraud.pkl
│   ├── rf_creditcard.pkl
│   └── *.png                    # Confusion matrices, PR curves, SHAP plots
├── scripts/
│   └── README.md
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

**Clone the repository**
```bash
git clone https://github.com/Wave-eer/Fraud-detection.git
cd Fraud-detection
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Download the datasets** and place them in `data/`:
- [Fraud_Data.csv](https://drive.google.com/file/d/1BCP10YNCMxzJzNjRoUYcxqBk3EHSz7wf/view)
- [IpAddress_to_Country.csv](https://drive.google.com/file/d/1OagS1rraLIX5wtq8yilXt8gXBcb8fKoC/view)
- [creditcard.csv](https://drive.google.com/file/d/1Nd6bmQuFq_-RQGVXOgNDQBNiPaZl7KH4/view)

**Run notebooks in order:**
```
1. notebooks/eda-fraud-data.ipynb
2. notebooks/eda-creditcard.ipynb
3. notebooks/feature-engineering.ipynb
4. notebooks/modeling.ipynb
5. notebooks/shap-explainability.ipynb
```

---

## Task 1 — Data Analysis & Preprocessing

### Data Cleaning
- Removed duplicate rows from `Fraud_Data.csv`.
- Converted `signup_time` and `purchase_time` to `datetime`.
- Cast `ip_address` to `int64` for range-based IP lookup.
- No missing values were found in either dataset.

### Exploratory Data Analysis
- **Fraud rate:** ~9.4% in `Fraud_Data.csv`; ~0.17% in `creditcard.csv`.
- Both datasets are **severely imbalanced** — overall accuracy is therefore a misleading metric.
- Purchase value and age are roughly normally distributed; no extreme outliers removed.
- Fraud transactions show slightly higher purchase values and are more common at certain hours.

### Geolocation Integration
IP addresses were converted to integers and merged with `IpAddress_to_Country.csv` using `pd.merge_asof` for efficient range-based lookup. Countries with elevated fraud rates were identified and used as features.

### Feature Engineering (Fraud_Data.csv)

| Feature | Description |
|---|---|
| `time_since_signup` | Hours between account creation and purchase |
| `hour_of_day` | Hour extracted from `purchase_time` |
| `day_of_week` | Day name extracted from `purchase_time` |
| `transaction_count` | Total transactions per `user_id` |

### Data Transformation
- **Scaling:** `StandardScaler` applied to `purchase_value`, `age`, `time_since_signup`, `transaction_count`.
- **Encoding:** One-hot encoding for `sex`, `browser`, `source`, `day_of_week`, `country`.

### Class Imbalance Handling
**SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training set only.

**Why SMOTE?**
- Simple duplication of minority samples overfits on the exact same points.
- SMOTE synthesises new fraud examples by interpolating between existing ones, giving the model richer minority-class signal.
- Applied **only to training data** — the test set is never touched, preserving a realistic evaluation.

| Dataset | Before SMOTE (train) | After SMOTE (train) |
|---|---|---|
| Fraud_Data.csv | ~9% fraud | 50% fraud |
| creditcard.csv | ~0.17% fraud | 50% fraud |

---

## Task 2 — Model Building & Training

### Models Trained

Two models were trained on each dataset:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble model — selected as best |

### Why Random Forest?
- Captures non-linear interactions (e.g., high purchase value + new account + unusual hour).
- More robust than a single decision tree; less prone to overfitting than deep boosting methods.
- Built-in feature importances feed directly into SHAP explainability (Task 3).
- Logistic Regression provides a linear boundary that cannot model the complex interaction patterns in fraud data.

### Hyperparameters (Random Forest)

| Parameter | Value | Justification |
|---|---|---|
| `n_estimators` | 200 | More trees → lower variance; diminishing returns beyond ~200 |
| `max_depth` | 15 | Deep enough for complex patterns; capped to avoid overfitting |
| `min_samples_leaf` | 10 | Prevents tiny leaves that memorise noise |
| `class_weight` | balanced | Additional guard against imbalance on top of SMOTE |

### Evaluation Metrics

**Why not accuracy?**
A model that predicts "legitimate" for every transaction achieves >99% accuracy on `creditcard.csv` while catching zero fraud. We use:

- **AUC-PR (Average Precision):** Summarises precision-recall trade-off at all thresholds. Best for imbalanced data.
- **F1-Score:** Harmonic mean of precision and recall — penalises both false positives and false negatives.
- **Confusion Matrix:** Shows exact counts of TP, TN, FP, FN.

### Results — Fraud_Data.csv

| Model | AUC-PR | F1-Score |
|---|---|---|
| Logistic Regression | see notebook | see notebook |
| **Random Forest** | **best** | **best** |

### Results — creditcard.csv

| Model | AUC-PR | F1-Score |
|---|---|---|
| Logistic Regression | see notebook | see notebook |
| **Random Forest** | **best** | **best** |

### Cross-Validation
Stratified 5-Fold CV was used to confirm results are stable across folds and not due to a lucky train-test split.

---

## Task 3 — Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to explain the Random Forest predictions.

### Global Feature Importance

SHAP summary plots show both the magnitude and direction of each feature's influence on the fraud probability. Key findings:

- **`time_since_signup`** — the single strongest predictor. Very short time between signup and purchase is a major fraud signal.
- **`transaction_count`** — high-velocity accounts push fraud probability up significantly.
- **`purchase_value`** — high amounts increase fraud probability, but the relationship is non-linear.
- **`hour_of_day`** — certain late-night hours are disproportionately associated with fraud.
- **Country features** — several country one-hot columns appear in the top predictors, confirming geolocation is a meaningful signal.

### Individual Force Plots

Three transaction types were examined:

- **True Positive** (fraud correctly caught): Short time since signup + high transaction count + high value — all pushed in the same direction.
- **False Positive** (legitimate flagged): New account with high value, but country and device context pulled probability back down — model was borderline.
- **False Negative** (fraud missed): Older account with normal purchase value — fraud was disguised as normal behaviour.

### Business Recommendations

1. **Flag new accounts making large purchases quickly.**
   Transactions within a few hours of account creation with high purchase value should trigger step-up authentication (SMS OTP or manual review).

2. **Implement transaction velocity rules.**
   Accounts with >5 purchases in 1 hour should be rate-limited or held for review. `transaction_count` is consistently a top SHAP driver.

3. **Apply geo-risk scoring.**
   Build a country-level risk tier (High / Medium / Low) based on fraud rates identified in EDA and SHAP country feature importance. Apply stricter verification for high-risk geographies.

4. **Increase monitoring intensity during peak fraud hours.**
   `hour_of_day` features show fraud concentrates at specific times. Increase automated review capacity during those windows instead of uniform 24/7 coverage.

5. **Use composite device + account age rules for borderline cases.**
   False negatives tend to be older accounts with normal-looking values. A secondary rule combining unfamiliar device + sudden high value change would catch some of these.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
shap
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Author

**Arsema** — 10 Academy KAIM Cohort 9
GitHub: [Wave-eer](https://github.com/Wave-eer)
