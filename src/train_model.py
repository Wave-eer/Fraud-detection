import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add root directory to sys.path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    average_precision_score, precision_recall_curve,
    classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_preprocessing import (
    load_data,
    clean_fraud_data,
    engineer_features,
    clean_creditcard_data,
    scale_and_encode
)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def evaluate_model(name, model, X_train, y_train, X_test, y_test, dataset_name):
    """
    Fits model on the training set, prints classification metrics, and saves the confusion matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'-'*50}")
    print(f"  Model: {name} ({dataset_name})")
    print(f"{'-'*50}")
    print(f"  Test AUC-PR  : {auc_pr:.4f}")
    print(f"  Test F1-Score: {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name} ({dataset_name})")
    plt.tight_layout()
    plot_path = f"models/{name.replace(' ', '_')}_({dataset_name})_cm.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {plot_path}")

    return {
        "model_name": name,
        "auc_pr": auc_pr,
        "f1_score": f1,
        "y_proba": y_proba,
        "fitted_model": model
    }

def plot_pr_curves(results, y_test, dataset_name):
    """
    Plots PR curves for all evaluated models on the same plot and saves the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
    for i, res in enumerate(results):
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(rec, prec, color=colors[i % len(colors)], lw=2,
                label=f"{res['model_name']} (AUC-PR = {res['auc_pr']:.4f})")
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve Comparison - {dataset_name}")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = f"models/pr_curve_{dataset_name.lower()}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Precision-Recall curves saved to {plot_path}")

def run_cross_validation(X_train, y_train, dataset_name):
    """
    Runs Stratified 5-Fold Cross-Validation for Random Forest with SMOTE inside the loop (leakage-free).
    """
    print(f"\nRunning Stratified 5-Fold CV for Random Forest on {dataset_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    cv_results = cross_validate(
        cv_pipeline, X_train, y_train, cv=skf,
        scoring=["average_precision", "f1"]
    )
    
    mean_aucpr = cv_results["test_average_precision"].mean()
    std_aucpr = cv_results["test_average_precision"].std()
    mean_f1 = cv_results["test_f1"].mean()
    std_f1 = cv_results["test_f1"].std()
    
    print(f"  CV Mean AUC-PR: {mean_aucpr:.4f} ± {std_aucpr:.4f}")
    print(f"  CV Mean F1    : {mean_f1:.4f} ± {std_f1:.4f}")
    
    return mean_aucpr, std_aucpr, mean_f1, std_f1

def main():
    # 1. Load data
    print("Loading datasets...")
    fraud_df, ip_df, cc_df = load_data("data")
    if fraud_df is None or ip_df is None or cc_df is None:
        raise ValueError("Raw data files could not be loaded. Please ensure they exist in data/raw/")
    
    # 2. Preprocess data using modular functions
    print("Preprocessing E-commerce Fraud data...")
    cleaned_fraud = clean_fraud_data(fraud_df, ip_df)
    engineered_fraud = engineer_features(cleaned_fraud)
    
    print("Preprocessing Credit Card data...")
    cleaned_cc = clean_creditcard_data(cc_df)
    
    print("Scaling and encoding datasets...")
    encoded_fraud, encoded_cc = scale_and_encode(engineered_fraud, cleaned_cc)
    
    # =========================================================================
    # PIPELINE A: E-commerce Fraud (Fraud_Data.csv)
    # =========================================================================
    print("\n" + "="*80)
    print(" PIPELINE A: E-commerce Fraud (Fraud_Data.csv)")
    print("="*80)
    
    COLS_TO_DROP = [
        "class", "user_id", "device_id", "signup_time", "purchase_time",
        "ip_address", "lower_bound_ip_address", "upper_bound_ip_address", "country"
    ]
    y_fraud = encoded_fraud["class"]
    X_fraud = encoded_fraud.drop(columns=[c for c in COLS_TO_DROP if c in encoded_fraud.columns], errors="ignore")
    X_fraud = X_fraud.select_dtypes(include=[np.number, "bool"])
    # Convert boolean columns to integer (0/1) for model compatibility
    for col in X_fraud.select_dtypes(include=["bool"]).columns:
        X_fraud[col] = X_fraud[col].astype(int)
    
    print(f"Feature matrix shape: {X_fraud.shape}")
    print(f"Class distribution: {y_fraud.value_counts().to_dict()}")
    print(f"Fraud rate: {y_fraud.mean()*100:.3f}%")
    
    # Stratified Train/Test Split
    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
        X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42
    )
    
    # Apply SMOTE to the training set only (for model fitting)
    smote = SMOTE(random_state=42)
    X_tr_f_bal, y_tr_f_bal = smote.fit_resample(X_tr_f, y_tr_f)
    print(f"After SMOTE training set class distribution: {pd.Series(y_tr_f_bal).value_counts().to_dict()}")
    
    # Baseline: Logistic Regression
    lr_fraud = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr_f_res = evaluate_model("Logistic Regression", lr_fraud, X_tr_f_bal, y_tr_f_bal, X_te_f, y_te_f, "Fraud")
    
    # Ensemble: Random Forest
    rf_fraud = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    rf_f_res = evaluate_model("Random Forest", rf_fraud, X_tr_f_bal, y_tr_f_bal, X_te_f, y_te_f, "Fraud")
    
    # Cross Validation
    run_cross_validation(X_tr_f, y_tr_f, "Fraud")
    
    # Plot PR Curves
    plot_pr_curves([lr_f_res, rf_f_res], y_te_f, "Fraud")
    
    # Save Best Model
    joblib.dump(rf_f_res["fitted_model"], "models/rf_fraud.pkl")
    print("[OK] Saved Best Model (Random Forest) to models/rf_fraud.pkl")
    
    # =========================================================================
    # PIPELINE B: Credit Card Fraud (creditcard.csv)
    # =========================================================================
    print("\n" + "="*80)
    print(" PIPELINE B: Credit Card Fraud (creditcard.csv)")
    print("="*80)
    
    y_cc = encoded_cc["Class"]
    X_cc = encoded_cc.drop(columns=["Class"])
    X_cc = X_cc.select_dtypes(include=[np.number])
    
    print(f"Feature matrix shape: {X_cc.shape}")
    print(f"Class distribution: {y_cc.value_counts().to_dict()}")
    print(f"Fraud rate: {y_cc.mean()*100:.3f}%")
    
    # Stratified Train/Test Split
    X_tr_cc, X_te_cc, y_tr_cc, y_te_cc = train_test_split(
        X_cc, y_cc, test_size=0.2, stratify=y_cc, random_state=42
    )
    
    # Apply SMOTE to training set only
    X_tr_cc_bal, y_tr_cc_bal = smote.fit_resample(X_tr_cc, y_tr_cc)
    print(f"After SMOTE training set class distribution: {pd.Series(y_tr_cc_bal).value_counts().to_dict()}")
    
    # Baseline: Logistic Regression
    lr_cc = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr_cc_res = evaluate_model("Logistic Regression", lr_cc, X_tr_cc_bal, y_tr_cc_bal, X_te_cc, y_te_cc, "CreditCard")
    
    # Ensemble: Random Forest
    rf_cc = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    rf_cc_res = evaluate_model("Random Forest", rf_cc, X_tr_cc_bal, y_tr_cc_bal, X_te_cc, y_te_cc, "CreditCard")
    
    # Cross Validation
    run_cross_validation(X_tr_cc, y_tr_cc, "CreditCard")
    
    # Plot PR Curves
    plot_pr_curves([lr_cc_res, rf_cc_res], y_te_cc, "CreditCard")
    
    # Save Best Model
    joblib.dump(rf_cc_res["fitted_model"], "models/rf_creditcard.pkl")
    print("[OK] Saved Best Model (Random Forest) to models/rf_creditcard.pkl")

if __name__ == "__main__":
    main()