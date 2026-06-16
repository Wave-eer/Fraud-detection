import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(data_dir):
    """
    Loads raw CSV files from the data directory.
    """
    fraud_path = os.path.join(data_dir, "raw", "Fraud_Data.csv")
    ip_path = os.path.join(data_dir, "raw", "IpAddress_to_Country.csv")
    cc_path = os.path.join(data_dir, "raw", "creditcard.csv")

    fraud_df = pd.read_csv(fraud_path) if os.path.exists(fraud_path) else None
    ip_df = pd.read_csv(ip_path) if os.path.exists(ip_path) else None
    cc_df = pd.read_csv(cc_path) if os.path.exists(cc_path) else None

    return fraud_df, ip_df, cc_df

def clean_fraud_data(fraud_df, ip_df):
    """
    Cleans e-commerce fraud data and merges it with the IP-to-country mapping.
    Unmatched IP addresses are filled with 'Unknown'.
    """
    # Create copy to avoid modifying original
    df = fraud_df.copy()
    
    # Correct data types
    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["ip_address"] = df["ip_address"].astype(np.int64)

    if ip_df is not None:
        ip = ip_df.copy()
        ip["lower_bound_ip_address"] = ip["lower_bound_ip_address"].astype(np.int64)
        ip["upper_bound_ip_address"] = ip["upper_bound_ip_address"].astype(np.int64)

        # Sort values for pd.merge_asof
        df = df.sort_values("ip_address")
        ip = ip.sort_values("lower_bound_ip_address")

        # Perform backward range-based merge
        df = pd.merge_asof(
            df,
            ip,
            left_on="ip_address",
            right_on="lower_bound_ip_address",
            direction="backward"
        )

        # Filter out country names where the IP falls outside the upper bound
        invalid_ip_mask = (df["ip_address"] > df["upper_bound_ip_address"]) | df["country"].isnull()
        df.loc[invalid_ip_mask, "country"] = "Unknown"
        
        # Fill remaining missing country bounds with default values
        df["country"] = df["country"].fillna("Unknown")
    else:
        df["country"] = "Unknown"

    return df

def engineer_features(fraud_df):
    """
    Engineers transaction velocity, account age, and time features,
    and groups low-frequency countries into 'Other' to limit dimensionality.
    """
    df = fraud_df.copy()

    # Time differences
    df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600.0

    # Date/Time features
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.day_name()

    # Device sharing counts (frequency)
    df["device_sharing_count"] = df.groupby("device_id")["device_id"].transform("count")

    # Device rolling velocity (transactions in last 24h)
    df = df.sort_values(["device_id", "purchase_time"])
    device_rolling = df.groupby("device_id").rolling("1D", on="purchase_time")["user_id"].count()
    df["device_tx_count_1d"] = device_rolling.values

    # IP sharing counts (frequency)
    df["ip_sharing_count"] = df.groupby("ip_address")["ip_address"].transform("count")

    # IP rolling velocity (transactions in last 24h)
    df = df.sort_values(["ip_address", "purchase_time"])
    ip_rolling = df.groupby("ip_address").rolling("1D", on="purchase_time")["user_id"].count()
    df["ip_tx_count_1d"] = ip_rolling.values

    # Group rare countries (<50 transactions) to prevent high dimensional sparsity
    country_counts = df["country"].value_counts()
    rare_countries = country_counts[country_counts < 50].index
    df["country"] = df["country"].replace(rare_countries, "Other")

    # Sort back to original index or keep sorted by time (sorting by purchase_time is standard)
    df = df.sort_values("purchase_time").reset_index(drop=True)

    return df

def clean_creditcard_data(cc_df):
    """
    Cleans credit card transactions by dropping duplicate entries.
    """
    if cc_df is None:
        return None
    df = cc_df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def scale_and_encode(fraud_df, cc_df):
    """
    Scales numerical features for both datasets and one-hot encodes e-commerce categoricals.
    """
    # 1. Scale e-commerce data
    scaled_fraud = fraud_df.copy()
    num_cols_fraud = [
        "purchase_value",
        "age",
        "time_since_signup",
        "device_sharing_count",
        "device_tx_count_1d",
        "ip_sharing_count",
        "ip_tx_count_1d"
    ]
    
    scaler_fraud = StandardScaler()
    scaled_fraud[num_cols_fraud] = scaler_fraud.fit_transform(scaled_fraud[num_cols_fraud])
    
    # One-hot encode categoricals for e-commerce
    cat_cols = ["sex", "browser", "source", "day_of_week", "country"]
    encoded_fraud = pd.get_dummies(scaled_fraud, columns=cat_cols, drop_first=True)

    # 2. Scale credit card data
    encoded_cc = None
    if cc_df is not None:
        scaled_cc = cc_df.copy()
        num_cols_cc = ["Amount", "Time"]
        scaler_cc = StandardScaler()
        # Scale and add scaled columns
        scaled_cc[["Amount_scaled", "Time_scaled"]] = scaler_cc.fit_transform(scaled_cc[num_cols_cc])
        
        # Select V features and scaled amount/time as final features
        v_cols = [f"V{i}" for i in range(1, 29)]
        feature_cols = v_cols + ["Amount_scaled", "Time_scaled", "Class"]
        encoded_cc = scaled_cc[feature_cols]

    return encoded_fraud, encoded_cc
