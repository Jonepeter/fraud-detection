import pytest
import pandas as pd
import numpy as np
from src.pipeline import FraudDetectionPipeline
from src.data_pipeline import DataPipeline

@pytest.fixture
def sample_config():
    return {
        'models': {
            'random_forest': {'n_estimators': 10, 'random_state': 42},
            'xgboost': {'n_estimators': 10, 'random_state': 42},
            'logistic_regression': {'random_state': 42}
        }
    }

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def test_fraud_detection_pipeline(sample_config, sample_data):
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    
    pipeline = FraudDetectionPipeline(sample_config)
    pipeline.fit(X, y, 'random_forest')
    
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    
    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), 2)
    assert all(pred in [0, 1] for pred in predictions)

def test_data_pipeline():
    data_pipeline = DataPipeline()
    
    # Test with sample fraud data
    fraud_data = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2],
        'signup_time': ['2023-01-01 10:00:00'] * 5,
        'purchase_time': ['2023-01-01 11:00:00'] * 5,
        'purchase_value': [100, 200, 150, 120, 180],
        'ip_address': ['192.168.1.1'] * 5,
        'class': [0, 1, 0, 0, 1]
    })
    
    processed = data_pipeline.process_fraud_data(fraud_data)
    
    assert 'time_diff_hours' in processed.columns
    assert 'purchase_hour' in processed.columns
    assert 'user_tx_count' in processed.columns