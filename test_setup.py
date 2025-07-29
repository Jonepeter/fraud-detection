"""
Test script to verify project setup and dependencies.
Run this to check if everything is properly configured.
"""

import sys
import os
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'imblearn', 'xgboost', 'lightgbm', 'shap', 'joblib'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nMissing packages: {failed_imports}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = ['src', 'data', 'models', 'notebooks', 'results']
    required_files = [
        'src/config.py',
        'src/data_preprocessing.py', 
        'src/models.py',
        'src/explainability.py',
        'src/pipeline.py',
        'notebooks/exploratory_analysis.py',
        'requirements.txt',
        'README.md'
    ]
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory: {dir_name}")
        else:
            print(f"✗ Directory: {dir_name} - MISSING")
    
    # Check files
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ File: {file_name}")
        else:
            print(f"✗ File: {file_name} - MISSING")

def test_data_files():
    """Test if data files are present."""
    print("\nTesting data files...")
    
    data_files = [
        'data/Fraud_Data.csv',
        'data/IpAddress_to_Country.csv', 
        'data/creditcard.csv'
    ]
    
    missing_files = []
    for file_name in data_files:
        if os.path.exists(file_name):
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print("\nData files are missing. Please add them to the 'data' directory.")
        print("The project will not run without these files.")
        return False
    else:
        print("\nAll data files found!")
        return True

def test_module_imports():
    """Test if custom modules can be imported."""
    print("\nTesting custom module imports...")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    modules = [
        'config',
        'data_preprocessing', 
        'models',
        'explainability'
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - ERROR: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("FRAUD DETECTION PROJECT - SETUP TEST")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test structure
    test_project_structure()
    
    # Test data files
    data_ok = test_data_files()
    
    # Test custom modules
    test_module_imports()
    
    print("\n" + "=" * 60)
    if imports_ok and data_ok:
        print("✓ SETUP COMPLETE - Ready to run the project!")
        print("Run: python run_project.py")
    else:
        print("✗ SETUP INCOMPLETE - Please fix the issues above")
    print("=" * 60)

if __name__ == "__main__":
    main()