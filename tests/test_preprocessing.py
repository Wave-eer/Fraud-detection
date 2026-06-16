import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    clean_fraud_data,
    clean_creditcard_data,
    engineer_features,
    scale_and_encode
)

@pytest.fixture
def mock_fraud_data():
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "signup_time": [
            "2015-01-01 00:00:00",
            "2015-01-01 01:00:00",
            "2015-01-01 02:00:00",
            "2015-01-01 03:00:00"
        ],
        "purchase_time": [
            "2015-01-01 02:00:00",  # diff = 2h
            "2015-01-01 04:00:00",  # diff = 3h
            "2015-01-01 02:30:00",  # diff = 0.5h
            "2015-01-02 03:00:00"   # diff = 24h
        ],
        "purchase_value": [10.0, 20.0, 30.0, 40.0],
        "device_id": ["DEV1", "DEV1", "DEV2", "DEV1"],  # DEV1 shared 3 times, DEV2 1 time
        "source": ["SEO", "Ads", "SEO", "Direct"],
        "browser": ["Chrome", "Safari", "Chrome", "IE"],
        "sex": ["M", "F", "M", "F"],
        "age": [30, 40, 50, 25],
        "ip_address": [100.0, 200.0, 300.0, 400.0],
        "class": [0, 1, 0, 0]
    })

@pytest.fixture
def mock_ip_data():
    return pd.DataFrame({
        "lower_bound_ip_address": [50.0, 150.0, 250.0],
        "upper_bound_ip_address": [120.0, 220.0, 280.0],  # 300 falls outside (gap), 400 falls outside
        "country": ["USA", "Canada", "Germany"]
    })

def test_clean_fraud_data(mock_fraud_data, mock_ip_data):
    cleaned = clean_fraud_data(mock_fraud_data, mock_ip_data)
    
    # Check shape is preserved (sorting can change order but not shape)
    assert len(cleaned) == 4
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(cleaned["signup_time"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["purchase_time"])
    assert cleaned["ip_address"].dtype == np.int64
    
    # Check geolocation range mapping
    # 100 is between 50 and 120 -> USA
    # 200 is between 150 and 220 -> Canada
    # 300 is > 280 (upper bound of Germany) -> Unknown
    # 400 is not in range -> Unknown
    assert cleaned.loc[cleaned["ip_address"] == 100, "country"].values[0] == "USA"
    assert cleaned.loc[cleaned["ip_address"] == 200, "country"].values[0] == "Canada"
    assert cleaned.loc[cleaned["ip_address"] == 300, "country"].values[0] == "Unknown"
    assert cleaned.loc[cleaned["ip_address"] == 400, "country"].values[0] == "Unknown"

def test_engineer_features(mock_fraud_data, mock_ip_data):
    cleaned = clean_fraud_data(mock_fraud_data, mock_ip_data)
    engineered = engineer_features(cleaned)
    
    # Check time_since_signup
    # user_id 1: signup 00:00, purchase 02:00 -> 2 hours
    assert np.isclose(engineered.loc[engineered["user_id"] == 1, "time_since_signup"].values[0], 2.0)
    
    # Check hour_of_day & day_of_week
    assert "hour_of_day" in engineered.columns
    assert "day_of_week" in engineered.columns
    
    # Check device sharing count
    # DEV1 shared 3 times
    assert engineered.loc[engineered["device_id"] == "DEV1", "device_sharing_count"].values[0] == 3
    # DEV2 shared 1 time
    assert engineered.loc[engineered["device_id"] == "DEV2", "device_sharing_count"].values[0] == 1
    
    # Check rolling velocity count (transactions in last 24h)
    # DEV1 transactions:
    # 1. 2015-01-01 02:00:00 (user 1) -> 1st -> count = 1
    # 2. 2015-01-01 04:00:00 (user 2) -> 2h later -> count = 2
    # 3. 2015-01-02 03:00:00 (user 4) -> 23h after user 2, 25h after user 1.
    #    Within 24h window of user 4's purchase (2015-01-01 03:00 to 2015-01-02 03:00):
    #    Only user 2's purchase (04:00) and user 4's purchase (03:00 next day) fall in range -> count = 2
    user_4_row = engineered[engineered["user_id"] == 4]
    assert user_4_row["device_tx_count_1d"].values[0] == 2
    
    # Check country grouping
    # Since USA, Canada, Germany/Unknown all have low counts (fewer than 50), they should all become "Other" or "Unknown"
    # Actually, country count is very small in mock, so they all group to 'Other' except 'Unknown' (if count is small too)
    # In our implementation: counts < 50 replace with "Other"
    # So all mock countries should be replaced with "Other"
    assert (engineered["country"] == "Other").all()

def test_clean_creditcard_data():
    cc_mock = pd.DataFrame({
        "Time": [0.0, 1.0, 0.0],
        "V1": [1.0, 2.0, 1.0],
        "Amount": [10.0, 20.0, 10.0],
        "Class": [0, 0, 0]
    })
    
    cleaned = clean_creditcard_data(cc_mock)
    assert len(cleaned) == 2  # duplicate dropped

def test_scale_and_encode(mock_fraud_data, mock_ip_data):
    cleaned = clean_fraud_data(mock_fraud_data, mock_ip_data)
    engineered = engineer_features(cleaned)
    
    cc_mock = pd.DataFrame({
        "Time": [0.0, 1.0],
        "V1": [1.0, 2.0],
        "V2": [3.0, 4.0],
        "V3": [1.0, 2.0],
        "V4": [3.0, 4.0],
        "V5": [1.0, 2.0],
        "V6": [3.0, 4.0],
        "V7": [1.0, 2.0],
        "V8": [3.0, 4.0],
        "V9": [1.0, 2.0],
        "V10": [3.0, 4.0],
        "V11": [1.0, 2.0],
        "V12": [3.0, 4.0],
        "V13": [1.0, 2.0],
        "V14": [3.0, 4.0],
        "V15": [1.0, 2.0],
        "V16": [3.0, 4.0],
        "V17": [1.0, 2.0],
        "V18": [3.0, 4.0],
        "V19": [1.0, 2.0],
        "V20": [3.0, 4.0],
        "V21": [1.0, 2.0],
        "V22": [3.0, 4.0],
        "V23": [1.0, 2.0],
        "V24": [3.0, 4.0],
        "V25": [1.0, 2.0],
        "V26": [3.0, 4.0],
        "V27": [1.0, 2.0],
        "V28": [3.0, 4.0],
        "Amount": [10.0, 20.0],
        "Class": [0, 0]
    })
    
    fraud_enc, cc_enc = scale_and_encode(engineered, cc_mock)
    
    # Check scaling (mean should be close to 0 and std close to 1, or scaled columns present)
    assert "Amount_scaled" in cc_enc.columns
    assert "Time_scaled" in cc_enc.columns
    assert "V1" in cc_enc.columns
    assert "Class" in cc_enc.columns
    
    # Check e-commerce dummies
    assert "sex_M" in fraud_enc.columns or "sex_F" in fraud_enc.columns
    assert "source_SEO" in fraud_enc.columns or "source_Direct" in fraud_enc.columns
