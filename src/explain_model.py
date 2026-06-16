import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add root directory to sys.path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from src.data_preprocessing import (
    load_data,
    clean_fraud_data,
    engineer_features,
    clean_creditcard_data,
    scale_and_encode
)

sns.set_theme(style="whitegrid")
os.makedirs("models", exist_ok=True)

def plot_builtin_importance(model, feature_names, title, save_path):
    """
    Plots and saves the top 10 built-in feature importances from a model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    plt.title(f"Top 10 Built-In Feature Importances - {title}", fontsize=14, fontweight="bold")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved built-in feature importance plot to {save_path}")

def explain_pipeline(model_path, X_test, y_test, dataset_name):
    """
    Computes SHAP values, generates a global summary plot, and isolates TP, FP, and FN force/waterfall plots.
    """
    print(f"\nExplaining {dataset_name} model...")
    model = joblib.load(model_path)
    feature_names = X_test.columns.tolist()
    
    # 1. Built-in feature importance
    plot_builtin_importance(model, feature_names, dataset_name, f"models/builtin_importance_{dataset_name.lower()}.png")
    
    # 2. SHAP Analysis
    print("Initializing TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Sample a representative subset of test data (e.g., 500 rows) for faster global SHAP calculation
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    print("Computing SHAP values for summary plot...")
    # Get explanation object
    explanation = explainer(X_sample)
    
    # For classification, check if the explanation values are 3D (i.e. [samples, features, classes])
    # Random Forest in shap explainer(X) returns a 3D explanation array: class 0 is index 0, class 1 is index 1.
    # We want class 1 (Fraud probability impact)
    if len(explanation.shape) == 3:
        explanation_class1 = explanation[:, :, 1]
    else:
        explanation_class1 = explanation
        
    # Global SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(explanation_class1, X_sample, show=False)
    plt.title(f"SHAP Summary Plot (Global) - {dataset_name}", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    summary_path = f"models/shap_summary_{dataset_name.lower()}.png"
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"[OK] Saved SHAP summary plot to {summary_path}")
    
    # 3. Predict on the whole test set to locate individual cases
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    
    # Locate indices
    tp_indices = np.where((y_test == 1) & (preds == 1))[0]
    fp_indices = np.where((y_test == 0) & (preds == 1))[0]
    fn_indices = np.where((y_test == 1) & (preds == 0))[0]
    
    # Generate individual waterfall plots
    cases = {"TP": tp_indices, "FP": fp_indices, "FN": fn_indices}
    for case_name, idx_list in cases.items():
        if len(idx_list) == 0:
            print(f"[WARNING] No {case_name} cases found in test set.")
            continue
            
        # Select the first instance
        case_idx = idx_list[0]
        # We need the single row as DataFrame
        single_row = X_test.iloc[[case_idx]]
        
        # Calculate SHAP values for this single instance
        single_explanation = explainer(single_row)
        if len(single_explanation.shape) == 3:
            single_explanation_class1 = single_explanation[0, :, 1]
        else:
            single_explanation_class1 = single_explanation[0]
            
        # Draw Waterfall Plot
        plt.figure(figsize=(8, 5))
        shap.plots.waterfall(single_explanation_class1, show=False)
        plt.title(f"SHAP Waterfall Plot ({case_name}) - {dataset_name}\n"
                  f"Actual: {y_test.iloc[case_idx]} | Pred Prob: {probas[case_idx]:.4f}", 
                  fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        case_path = f"models/shap_{case_name.lower()}_{dataset_name.lower()}.png"
        plt.savefig(case_path, dpi=150)
        plt.close()
        print(f"[OK] Saved {case_name} SHAP plot to {case_path}")

def main():
    # Load and preprocess
    print("Loading datasets...")
    fraud_df, ip_df, cc_df = load_data("data")
    if fraud_df is None or ip_df is None or cc_df is None:
        raise ValueError("Raw data files could not be loaded.")
        
    print("Preprocessing...")
    cleaned_fraud = clean_fraud_data(fraud_df, ip_df)
    engineered_fraud = engineer_features(cleaned_fraud)
    cleaned_cc = clean_creditcard_data(cc_df)
    encoded_fraud, encoded_cc = scale_and_encode(engineered_fraud, cleaned_cc)
    
    # E-commerce setup
    COLS_TO_DROP = [
        "class", "user_id", "device_id", "signup_time", "purchase_time",
        "ip_address", "lower_bound_ip_address", "upper_bound_ip_address", "country"
    ]
    y_fraud = encoded_fraud["class"]
    X_fraud = encoded_fraud.drop(columns=[c for c in COLS_TO_DROP if c in encoded_fraud.columns], errors="ignore")
    X_fraud = X_fraud.select_dtypes(include=[np.number, "bool"])
    for col in X_fraud.select_dtypes(include=["bool"]).columns:
        X_fraud[col] = X_fraud[col].astype(int)
        
    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
        X_fraud, y_fraud, test_size=0.2, stratify=y_fraud, random_state=42
    )
    
    # Credit Card setup
    y_cc = encoded_cc["Class"]
    X_cc = encoded_cc.drop(columns=["Class"])
    X_cc = X_cc.select_dtypes(include=[np.number])
    
    X_tr_cc, X_te_cc, y_tr_cc, y_te_cc = train_test_split(
        X_cc, y_cc, test_size=0.2, stratify=y_cc, random_state=42
    )
    
    # Run explanations
    explain_pipeline("models/rf_fraud.pkl", X_te_f, y_te_f, "Fraud")
    explain_pipeline("models/rf_creditcard.pkl", X_te_cc, y_te_cc, "CreditCard")

if __name__ == "__main__":
    main()
