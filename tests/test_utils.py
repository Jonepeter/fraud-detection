import os
import pytest
import tempfile
import yaml
from src.utils import load_config, get_project_root, get_absolute_path, create_directory_if_not_exists

@pytest.fixture
def sample_config():
    return {
        'data': {
            'raw': {
                'ecommerce': 'data/raw/Fraud_Data.csv'
            }
        },
        'models': {
            'random_forest': {
                'n_estimators': 100
            }
        }
    }

@pytest.fixture
def temp_config_file(sample_config):
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
        yaml.dump(sample_config, temp_file)
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Clean up after test
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

def test_load_config(temp_config_file, sample_config):
    # Test loading a valid config file
    config = load_config(temp_config_file)
    assert config == sample_config
    assert config['data']['raw']['ecommerce'] == 'data/raw/Fraud_Data.csv'
    assert config['models']['random_forest']['n_estimators'] == 100

def test_load_config_file_not_found():
    # Test loading a non-existent config file
    with pytest.raises(FileNotFoundError):
        load_config('non_existent_config.yaml')

def test_get_project_root():
    # Test getting the project root directory
    project_root = get_project_root()
    assert os.path.isdir(project_root)
    assert os.path.basename(os.path.dirname(os.path.dirname(__file__))) in project_root

def test_get_absolute_path():
    # Test converting a relative path to an absolute path
    relative_path = 'data/raw/Fraud_Data.csv'
    absolute_path = get_absolute_path(relative_path)
    assert os.path.isabs(absolute_path)
    assert relative_path.replace('/', os.sep) in absolute_path

def test_create_directory_if_not_exists():
    # Test creating a directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, 'test_dir')
        assert not os.path.exists(test_dir)
        
        create_directory_if_not_exists(test_dir)
        assert os.path.exists(test_dir)
        assert os.path.isdir(test_dir)
        
        # Test creating a directory that already exists
        create_directory_if_not_exists(test_dir)
        assert os.path.exists(test_dir)