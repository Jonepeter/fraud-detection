"""
Main script to run the complete fraud detection project.
This script provides a simple interface to execute different parts of the pipeline.
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_eda():
    """Run exploratory data analysis."""
    print("Running Exploratory Data Analysis...")
    from notebooks.exploratory_analysis import EDAAnalyzer
    
    analyzer = EDAAnalyzer()
    analyzer.run_complete_eda()
    print("EDA completed!")

def run_pipeline():
    """Run the complete ML pipeline."""
    print("Running Complete ML Pipeline...")
    from src.pipeline import main
    
    main()
    print("Pipeline completed!")

def run_preprocessing_only():
    """Run only data preprocessing."""
    print("Running Data Preprocessing...")
    from src.data_preprocessing import DataPreprocessor
    import src.config as config
    
    preprocessor = DataPreprocessor()
    
    # Load data
    fraud_data, ip_data, credit_data = preprocessor.load_data(
        config.FRAUD_DATA_PATH, 
        config.IP_COUNTRY_PATH, 
        config.CREDITCARD_PATH
    )
    
    # Preprocess fraud data
    X_fraud, y_fraud = preprocessor.preprocess_fraud_data()
    print(f"Fraud data processed: {X_fraud.shape}")
    
    # Preprocess credit data
    X_credit, y_credit = preprocessor.preprocess_credit_data()
    print(f"Credit data processed: {X_credit.shape}")
    
    print("Preprocessing completed!")

def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Project Runner')
    parser.add_argument('--mode', choices=['eda', 'pipeline', 'preprocess', 'all'], 
                       default='all', help='Mode to run')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FRAUD DETECTION PROJECT")
    print("=" * 60)
    
    if args.mode == 'eda':
        run_eda()
    elif args.mode == 'pipeline':
        run_pipeline()
    elif args.mode == 'preprocess':
        run_preprocessing_only()
    elif args.mode == 'all':
        print("Running complete project...")
        try:
            run_eda()
            print("\n" + "="*60 + "\n")
            run_pipeline()
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure your data files are in the 'data' directory:")
            print("- data/Fraud_Data.csv")
            print("- data/IpAddress_to_Country.csv") 
            print("- data/creditcard.csv")

if __name__ == "__main__":
    main()