import pytest
import os
import pandas as pd
from src.load_data import load_data

@pytest.fixture
def sample_csv_path():
    # Create a path to a sample CSV file in the test directory
    return os.path.join(os.path.dirname(__file__), 'test_data', 'sample.csv')

@pytest.fixture
def create_sample_csv(sample_csv_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)
    
    # Create a simple CSV file for testing
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    df.to_csv(sample_csv_path, index=False)
    
    yield sample_csv_path
    
    # Clean up after test
    if os.path.exists(sample_csv_path):
        os.remove(sample_csv_path)

def test_load_data_csv(create_sample_csv):
    # Test loading a CSV file
    df = load_data(create_sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == ['A', 'B']

def test_load_data_invalid_path():
    # Test loading a non-existent file
    with pytest.raises(ValueError):
        load_data('non_existent_file.csv')

def test_load_data_unsupported_extension():
    # Test loading a file with unsupported extension
    with pytest.raises(ValueError):
        load_data('file.unsupported')

def test_load_data_no_filepath():
    # Test loading with no filepath
    with pytest.raises(ValueError):
        load_data(None)