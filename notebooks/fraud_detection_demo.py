"""
Demo notebook for fraud detection project.
This script demonstrates key functionality in a step-by-step manner.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_data_loading():
    """Demonstrate data loading and basic info."""
    print("=" * 50)
    print("DEMO: DATA LOADING")
    print("=" * 50)
    
    try:
        from data_preprocessing import DataPreprocessor
        import config
        
        preprocessor = DataPreprocessor()
        fraud_data, ip_data, credit_data = preprocessor.load_data(
            config.FRAUD_DATA_PATH,
            config.IP_COUNTRY_PATH, 
            config.CREDITCARD_PATH
        )
        
        print("Fraud Data Shape:", fraud_data.shape)
        print("Fraud Data Columns:", list(fraud_data.columns))
        print("\nCredit Data Shape:", credit_data.shape)
        print("Credit Data Columns:", list(credit_data.columns))
        print("\nIP Data Shape:", ip_data.shape)
        print("IP Data Columns:", list(ip_data.columns))
        
        return fraud_data, ip_data, credit_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def demo_preprocessing():
    """Demonstrate preprocessing steps."""
    print("\n" + "=" * 50)
    print("DEMO: DATA PREPROCESSING")
    print("=" * 50)
    
    try:
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Load data first
        fraud_data, ip_data, credit_data = preprocessor.load_data(
            '../data/Fraud_Data.csv',
            '../data/IpAddress_to_Country.csv',
            '../data/creditcard.csv'
        )
        
        # Preprocess fraud data
        print("Processing fraud data...")
        X_fraud, y_fraud = preprocessor.preprocess_fraud_data()
        print(f"Fraud features shape: {X_fraud.shape}")
        print(f"Fraud target distribution: {y_fraud.value_counts().to_dict()}")
        
        # Preprocess credit data
        print("\nProcessing credit data...")
        X_credit, y_credit = preprocessor.preprocess_credit_data()
        print(f"Credit features shape: {X_credit.shape}")
        print(f"Credit target distribution: {y_credit.value_counts().to_dict()}")
        
        return X_fraud, y_fraud, X_credit, y_credit
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None, None, None

def demo_model_training():
    """Demonstrate model training."""
    print("\n" + "=" * 50)
    print("DEMO: MODEL TRAINING")
    print("=" * 50)
    
    try:
        from models import FraudModels
        from sklearn.model_selection import train_test_split
        
        # Use sample data for demo
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=10, n_classes=2,
            n_informative=5, n_redundant=2, n_clusters_per_class=1,
            weights=[0.9, 0.1], random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train models
        model_trainer = FraudModels(random_state=42)
        results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Show results
        print("\nModel Performance:")
        for model_name, result in results.items():
            auc_roc = result['evaluation']['auc_roc']
            auc_pr = result['evaluation']['auc_pr']
            print(f"{model_name}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")
        
        # Get best model
        best_model_name, best_model = model_trainer.get_best_model(results)
        print(f"\nBest model: {best_model_name}")
        
        return results, best_model
        
    except Exception as e:
        print(f"Error in model training: {e}")
        return None, None

def demo_explainability():
    """Demonstrate model explainability."""
    print("\n" + "=" * 50)
    print("DEMO: MODEL EXPLAINABILITY")
    print("=" * 50)
    
    try:
        from explainability import ModelExplainer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        # Create sample data
        X, y = make_classification(
            n_samples=500, n_features=8, n_classes=2,
            n_informative=4, n_redundant=2, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Create explainer
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        explainer = ModelExplainer(model, X_train, X_test, feature_names)
        
        # Get feature importance
        importance_df = explainer.get_feature_importance()
        print("Feature Importance:")
        print(importance_df.head())
        
        # Generate plots (commented out to avoid display issues in script)
        # explainer.plot_summary()
        # explainer.plot_feature_importance()
        
        print("\nExplainability demo completed!")
        return explainer
        
    except Exception as e:
        print(f"Error in explainability demo: {e}")
        return None

def main():
    """Run all demos."""
    print("FRAUD DETECTION PROJECT - DEMO")
    print("This demo shows key functionality of the project")
    
    # Demo 1: Data Loading
    fraud_data, ip_data, credit_data = demo_data_loading()
    
    # Demo 2: Preprocessing (only if data is available)
    if fraud_data is not None:
        X_fraud, y_fraud, X_credit, y_credit = demo_preprocessing()
    
    # Demo 3: Model Training
    results, best_model = demo_model_training()
    
    # Demo 4: Explainability
    explainer = demo_explainability()
    
    print("\n" + "=" * 50)
    print("DEMO COMPLETED!")
    print("=" * 50)
    print("To run the full project:")
    print("1. Add your data files to the 'data' directory")
    print("2. Run: python run_project.py")

if __name__ == "__main__":
    main()