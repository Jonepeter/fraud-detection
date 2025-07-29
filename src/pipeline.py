"""Main pipeline for fraud detection project.

This module defines the FraudDetectionPipeline class, which orchestrates the end-to-end
fraud detection workflow, including data loading, preprocessing, model training, evaluation,
explainability, and report generation. It also provides a main() function for execution.

All major steps are wrapped in try-except blocks to handle and report errors gracefully.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import DataPreprocessor
from models import FraudModels
from explainability import ModelExplainer
import config

class FraudDetectionPipeline:
    """
    Pipeline class for running fraud detection experiments.

    Methods
    -------
    run_fraud_data_pipeline():
        Runs the pipeline for the fraud data set.
    run_credit_data_pipeline():
        Runs the pipeline for the credit card data set.
    generate_final_report():
        Generates a final report comparing results.
    """
    def __init__(self):
        """Initialize the pipeline with preprocessor, model trainer, and results dict."""
        self.preprocessor = DataPreprocessor()
        self.model_trainer = FraudModels(random_state=config.RANDOM_STATE)
        self.results = {}
        
    def run_fraud_data_pipeline(self):
        """
        Run complete pipeline for fraud data.

        Returns
        -------
        tuple
            (results, best_model_name, importance_df)
        """
        print("=== FRAUD DATA PIPELINE ===")
        try:
            # Load data
            print("Loading data...")
            fraud_data, ip_data, _ = self.preprocessor.load_data(
                config.FRAUD_DATA_PATH, 
                config.IP_COUNTRY_PATH, 
                config.CREDITCARD_PATH
            )
            
            # Preprocess data
            print("Preprocessing data...")
            X, y = self.preprocessor.preprocess_fraud_data()
            
            print(f"Dataset shape: {X.shape}")
            print(f"Class distribution: {y.value_counts().to_dict()}")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE, 
                stratify=y
            )
            
            # Handle class imbalance
            print("Handling class imbalance...")
            X_train_balanced, y_train_balanced = self.preprocessor.handle_imbalance(
                X_train, y_train, method='smote'
            )
            
            print(f"Balanced training set shape: {X_train_balanced.shape}")
            print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
            
            # Scale features
            X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
                X_train_balanced, X_test
            )
            
            # Train models
            print("Training models...")
            results = self.model_trainer.train_all_models(
                X_train_scaled, y_train_balanced, X_test_scaled, y_test
            )
            
            # Get best model
            best_model_name, best_model = self.model_trainer.get_best_model(results)
            print(f"Best model: {best_model_name}")
            
            # Save best model
            model_path = self.model_trainer.save_model(
                best_model, f"fraud_data_{best_model_name}", config.MODELS_DIR
            )
            print(f"Best model saved to: {model_path}")
            
            # Model explainability
            print("Generating model explanations...")
            feature_names = X.columns.tolist()
            explainer = ModelExplainer(
                best_model, X_train_scaled, X_test_scaled, feature_names
            )
            
            explanation_dir = os.path.join(config.RESULTS_DIR, 'fraud_data_explanations')
            importance_df = explainer.generate_explanation_report(explanation_dir)
            
            self.results['fraud_data'] = {
                'results': results,
                'best_model': best_model_name,
                'feature_importance': importance_df
            }
            
            return results, best_model_name, importance_df
        except Exception as e:
            print(f"Error in run_fraud_data_pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def run_credit_data_pipeline(self):
        """
        Run complete pipeline for credit card data.

        Returns
        -------
        tuple
            (results, best_model_name, importance_df)
        """
        print("\n=== CREDIT CARD DATA PIPELINE ===")
        try:
            # Preprocess data
            print("Preprocessing data...")
            X, y = self.preprocessor.preprocess_credit_data()
            
            print(f"Dataset shape: {X.shape}")
            print(f"Class distribution: {y.value_counts().to_dict()}")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE, 
                stratify=y
            )
            
            # Handle class imbalance
            print("Handling class imbalance...")
            X_train_balanced, y_train_balanced = self.preprocessor.handle_imbalance(
                X_train, y_train, method='smote'
            )
            
            print(f"Balanced training set shape: {X_train_balanced.shape}")
            print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
            
            # Scale features
            X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
                X_train_balanced, X_test
            )
            
            # Train models
            print("Training models...")
            results = self.model_trainer.train_all_models(
                X_train_scaled, y_train_balanced, X_test_scaled, y_test
            )
            
            # Get best model
            best_model_name, best_model = self.model_trainer.get_best_model(results)
            print(f"Best model: {best_model_name}")
            
            # Save best model
            model_path = self.model_trainer.save_model(
                best_model, f"credit_data_{best_model_name}", config.MODELS_DIR
            )
            print(f"Best model saved to: {model_path}")
            
            # Model explainability
            print("Generating model explanations...")
            feature_names = X.columns.tolist()
            explainer = ModelExplainer(
                best_model, X_train_scaled, X_test_scaled, feature_names
            )
            
            explanation_dir = os.path.join(config.RESULTS_DIR, 'credit_data_explanations')
            importance_df = explainer.generate_explanation_report(explanation_dir)
            
            self.results['credit_data'] = {
                'results': results,
                'best_model': best_model_name,
                'feature_importance': importance_df
            }
            
            return results, best_model_name, importance_df
        except Exception as e:
            print(f"Error in run_credit_data_pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def generate_final_report(self):
        """
        Generate final comparison report.

        Returns
        -------
        str
            Path to the final report file.
        """
        print("\n=== FINAL REPORT ===")
        try:
            report_path = os.path.join(config.RESULTS_DIR, 'final_report.txt')
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write("FRAUD DETECTION PROJECT - FINAL REPORT\n")
                f.write("=" * 50 + "\n\n")
                
        
        # Load data
        print("Loading data...")
        fraud_data, ip_data, _ = self.preprocessor.load_data(
            config.FRAUD_DATA_PATH, 
            config.IP_COUNTRY_PATH, 
            config.CREDITCARD_PATH
        )
        
        # Preprocess data
        print("Preprocessing data...")
        X, y = self.preprocessor.preprocess_fraud_data()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, 
            stratify=y
        )
        
        # Handle class imbalance
        print("Handling class imbalance...")
        X_train_balanced, y_train_balanced = self.preprocessor.handle_imbalance(
            X_train, y_train, method='smote'
        )
        
        print(f"Balanced training set shape: {X_train_balanced.shape}")
        print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
            X_train_balanced, X_test
        )
        
        # Train models
        print("Training models...")
        results = self.model_trainer.train_all_models(
            X_train_scaled, y_train_balanced, X_test_scaled, y_test
        )
        
        # Get best model
        best_model_name, best_model = self.model_trainer.get_best_model(results)
        print(f"Best model: {best_model_name}")
        
        # Save best model
        model_path = self.model_trainer.save_model(
            best_model, f"fraud_data_{best_model_name}", config.MODELS_DIR
        )
        print(f"Best model saved to: {model_path}")
        
        # Model explainability
        print("Generating model explanations...")
        feature_names = X.columns.tolist()
        explainer = ModelExplainer(
            best_model, X_train_scaled, X_test_scaled, feature_names
        )
        
        explanation_dir = os.path.join(config.RESULTS_DIR, 'fraud_data_explanations')
        importance_df = explainer.generate_explanation_report(explanation_dir)
        
        self.results['fraud_data'] = {
            'results': results,
            'best_model': best_model_name,
            'feature_importance': importance_df
        }
        
        return results, best_model_name, importance_df
    
    def run_credit_data_pipeline(self):
        """Run complete pipeline for credit card data."""
        print("\n=== CREDIT CARD DATA PIPELINE ===")
        
        # Preprocess data
        print("Preprocessing data...")
        X, y = self.preprocessor.preprocess_credit_data()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, 
            stratify=y
        )
        
        # Handle class imbalance
        print("Handling class imbalance...")
        X_train_balanced, y_train_balanced = self.preprocessor.handle_imbalance(
            X_train, y_train, method='smote'
        )
        
        print(f"Balanced training set shape: {X_train_balanced.shape}")
        print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(
            X_train_balanced, X_test
        )
        
        # Train models
        print("Training models...")
        results = self.model_trainer.train_all_models(
            X_train_scaled, y_train_balanced, X_test_scaled, y_test
        )
        
        # Get best model
        best_model_name, best_model = self.model_trainer.get_best_model(results)
        print(f"Best model: {best_model_name}")
        
        # Save best model
        model_path = self.model_trainer.save_model(
            best_model, f"credit_data_{best_model_name}", config.MODELS_DIR
        )
        print(f"Best model saved to: {model_path}")
        
        # Model explainability
        print("Generating model explanations...")
        feature_names = X.columns.tolist()
        explainer = ModelExplainer(
            best_model, X_train_scaled, X_test_scaled, feature_names
        )
        
        explanation_dir = os.path.join(config.RESULTS_DIR, 'credit_data_explanations')
        importance_df = explainer.generate_explanation_report(explanation_dir)
        
        self.results['credit_data'] = {
            'results': results,
            'best_model': best_model_name,
            'feature_importance': importance_df
        }
        
        return results, best_model_name, importance_df
    
    def generate_final_report(self):
        """Generate final comparison report."""
        print("\n=== FINAL REPORT ===")
        
        report_path = os.path.join(config.RESULTS_DIR, 'final_report.txt')
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("FRAUD DETECTION PROJECT - FINAL REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset_name, result in self.results.items():
                f.write(f"{dataset_name.upper()} RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Model: {result['best_model']}\n")
                
                best_results = result['results'][result['best_model']]
                f.write(f"AUC-ROC: {best_results['evaluation']['auc_roc']:.4f}\n")
                f.write(f"AUC-PR: {best_results['evaluation']['auc_pr']:.4f}\n")
                
                f.write("\nTop 5 Important Features:\n")
                for i, row in result['feature_importance'].head().iterrows():
                    f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
                f.write("\n")
        
        print(f"Final report saved to: {report_path}")
        return report_path

def main():
    """Main execution function."""
    # Create directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    try:
        # Run fraud data pipeline
        pipeline.run_fraud_data_pipeline()
        
        # Run credit card data pipeline
        pipeline.run_credit_data_pipeline()
        
        # Generate final report
        pipeline.generate_final_report()
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()