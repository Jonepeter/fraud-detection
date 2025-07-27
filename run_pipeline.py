#!/usr/bin/env python3
"""
Quick runner script for the complete fraud detection pipeline
"""

from src.complete_pipeline import CompleteFraudPipeline
import os

def main():
    """Run the complete pipeline"""
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("🚀 Starting Complete Fraud Detection Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = CompleteFraudPipeline()
    
    # Run complete pipeline
    try:
        results = pipeline.run_complete_pipeline(
            fraud_path='data/raw/Fraud_Data.csv',
            ip_path='data/raw/IpAddress_to_Country.csv',
            credit_path='data/raw/creditcard.csv'
        )
        
        print("\n✅ Pipeline completed successfully!")
        print("\n📊 Results Summary:")
        print("-" * 30)
        
        for dataset, model_info in pipeline.best_models.items():
            print(f"{dataset} Dataset:")
            print(f"  AUC-ROC: {model_info['auc_roc']:.4f}")
            print(f"  AUC-PR:  {model_info['auc_pr']:.4f}")
            print()
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()