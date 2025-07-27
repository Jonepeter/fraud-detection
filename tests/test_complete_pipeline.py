import pytest
import pandas as pd
import numpy as np
from src.complete_pipeline import CompleteFraudPipeline

@pytest.fixture
def sample_fraud_data():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'user_id': np.random.randint(1, 20, n_samples),
        'signup_time': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'purchase_time': pd.date_range('2023-01-01 01:00:00', periods=n_samples, freq='H'),
        'purchase_value': np.random.uniform(10, 1000, n_samples),
        'ip_address': [f"192.168.1.{i%255}" for i in range(n_samples)],
        'class': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_ip_data():
    data = {
        'lower_bound_ip_address': ['192.168.1.0', '10.0.0.0'],
        'upper_bound_ip_address': ['192.168.1.255', '10.255.255.255'],
        'country': ['US', 'CA']
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_credit_data():
    np.random.seed(42)
    n_samples = 100
    
    # Create sample features similar to credit card dataset
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),  # 48 hours in seconds
        'Amount': np.random.uniform(1, 1000, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    # Add V1-V5 features (simplified)
    for i in range(1, 6):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)

def test_pipeline_initialization():
    pipeline = CompleteFraudPipeline()
    assert pipeline.fraud_pipeline is None
    assert pipeline.credit_pipeline is None
    assert pipeline.fraud_data is None
    assert pipeline.credit_data is None
    assert pipeline.best_models == {}

def test_ip_to_int():
    pipeline = CompleteFraudPipeline()
    
    # Test valid IP
    assert pipeline._ip_to_int('192.168.1.1') == 3232235777
    
    # Test invalid IP
    assert pipeline._ip_to_int('invalid.ip') == 0

def test_process_fraud_data(sample_fraud_data, sample_ip_data):
    pipeline = CompleteFraudPipeline()
    
    processed = pipeline._process_fraud_data(sample_fraud_data, sample_ip_data)
    
    # Check if new features are created
    assert 'time_since_signup' in processed.columns
    assert 'hour_of_day' in processed.columns
    assert 'day_of_week' in processed.columns
    assert 'tx_frequency' in processed.columns
    assert 'country' in processed.columns
    
    # Check if timestamp columns are dropped
    assert 'signup_time' not in processed.columns
    assert 'purchase_time' not in processed.columns

def test_process_credit_data(sample_credit_data):
    pipeline = CompleteFraudPipeline()
    
    processed = pipeline._process_credit_data(sample_credit_data)
    
    # Check if new features are created
    assert 'hour_of_day' in processed.columns
    assert 'day_of_week' in processed.columns
    assert 'amount_log' in processed.columns
    
    # Check original columns are preserved
    assert 'Time' in processed.columns
    assert 'Amount' in processed.columns
    assert 'Class' in processed.columns