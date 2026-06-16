import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)

from imblearn.over_sampling import SMOTE

# ====================================
# LOAD DATA
# ====================================

df = pd.read_csv("../data/Fraud_Data.csv")

print("Dataset Shape:", df.shape)

# ====================================
# FEATURE ENGINEERING
# ====================================

df["signup_time"] = pd.to_datetime(df["signup_time"])
df["purchase_time"] = pd.to_datetime(df["purchase_time"])

df["time_since_signup"] = (
    df["purchase_time"] - df["signup_time"]
).dt.total_seconds()

df["hour_of_day"] = df["purchase_time"].dt.hour

df["day_of_week"] = df["purchase_time"].dt.dayofweek

# ====================================
# DROP UNUSED COLUMNS
# ====================================

columns_to_drop = [
    "signup_time",
    "purchase_time",
    "device_id",
    "ip_address"
]

df.drop(columns=columns_to_drop, inplace=True)

# ====================================
# ONE-HOT ENCODING
# ====================================

df = pd.get_dummies(
    df,
    columns=["source", "browser", "sex"],
    drop_first=True
)

# ====================================
# FEATURES & TARGET
# ====================================

X = df.drop("class", axis=1)
y = df["class"]

# ====================================
# TRAIN TEST SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nClass Distribution Before SMOTE")
print(y_train.value_counts())

# ====================================
# SMOTE
# ====================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nClass Distribution After SMOTE")
print(y_train_smote.value_counts())

# ====================================
# LOGISTIC REGRESSION
# ====================================

print("\n========== LOGISTIC REGRESSION ==========")

lr = LogisticRegression(
    max_iter=2000,
    random_state=42
)

lr.fit(X_train_smote, y_train_smote)

lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]

print("F1 Score:",
      f1_score(y_test, lr_pred))

print("AUC-PR:",
      average_precision_score(y_test, lr_prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

# ====================================
# RANDOM FOREST
# ====================================

print("\n========== RANDOM FOREST ==========")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_smote, y_train_smote)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

rf_f1 = f1_score(y_test, rf_pred)
rf_aucpr = average_precision_score(y_test, rf_prob)

print("F1 Score:", rf_f1)
print("AUC-PR:", rf_aucpr)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# ====================================
# SAVE MODEL
# ====================================

joblib.dump(
    rf,
    "random_forest_model.pkl"
)

print("\nModel saved as random_forest_model.pkl")