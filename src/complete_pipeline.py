"""
Complete Fraud Detection Pipeline for all tasks
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

logger = logging.getLogger(__name__)

class CompleteFraudPipeline:
    """Complete pipeline for fraud detection tasks"""
    
    def __init__(self):
        self.fraud_pipeline = None
        self.credit_pipeline = None
        self.fraud_data = None
        self.credit_data = None
        self.best_models = {}
        
    def task1_data_analysis_preprocessing(self, fraud_path, ip_path, credit_path):
        """Task 1: Complete data analysis and preprocessing"""
        
        # Load data
        fraud_data = pd.read_csv(fraud_path)
        ip_data = pd.read_csv(ip_path)
        credit_data = pd.read_csv(credit_path)
        
        print("=== TASK 1: DATA ANALYSIS AND PREPROCESSING ===")
        
        # Process fraud data
        fraud_processed = self._process_fraud_data(fraud_data, ip_data)
        credit_processed = self._process_credit_data(credit_data)
        
        self.fraud_data = fraud_processed
        self.credit_data = credit_processed
        
        # EDA
        self._perform_eda(fraud_processed, 'Fraud Data')
        self._perform_eda(credit_processed, 'Credit Data')
        
        return fraud_processed, credit_processed
    
    def _process_fraud_data(self, df, ip_df):
        """Process fraud data with all required features"""
        df = df.copy()
        
        # Handle missing values
        print(f"Missing values before: {df.isnull().sum().sum()}")
        df = df.dropna()
        print(f"Missing values after: {df.isnull().sum().sum()}")
        
        # Remove duplicates
        print(f"Duplicates removed: {df.duplicated().sum()}")
        df = df.drop_duplicates()
        
        # Convert timestamps
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Time-based features
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # IP to integer conversion and merge
        df['ip_int'] = df['ip_address'].apply(self._ip_to_int)
        ip_df['lower_ip_int'] = ip_df['lower_bound_ip_address'].apply(self._ip_to_int)
        ip_df['upper_ip_int'] = ip_df['upper_bound_ip_address'].apply(self._ip_to_int)
        
        # Merge with country data
        df = self._merge_ip_country(df, ip_df)
        
        # Transaction frequency and velocity
        user_stats = df.groupby('user_id').agg({
            'purchase_value': ['count', 'mean', 'std', 'sum'],
            'class': 'mean'
        }).fillna(0)
        user_stats.columns = ['tx_frequency', 'avg_amount', 'std_amount', 'total_amount', 'fraud_rate']
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Drop timestamp columns
        df = df.drop(['signup_time', 'purchase_time', 'ip_address'], axis=1)
        
        return df
    
    def _process_credit_data(self, df):
        """Process credit card data"""
        df = df.copy()
        
        # Handle missing values
        print(f"Credit missing values: {df.isnull().sum().sum()}")
        
        # Remove duplicates
        print(f"Credit duplicates: {df.duplicated().sum()}")
        df = df.drop_duplicates()
        
        # Time features
        df['hour_of_day'] = (df['Time'] % (24 * 3600)) // 3600
        df['day_of_week'] = (df['Time'] // (24 * 3600)) % 7
        
        # Amount features
        df['amount_log'] = np.log1p(df['Amount'])
        
        return df
    
    def _ip_to_int(self, ip):
        """Convert IP to integer"""
        try:
            return sum(int(x) << (8 * (3 - i)) for i, x in enumerate(ip.split('.')))
        except:
            return 0
    
    def _merge_ip_country(self, df, ip_df):
        """Merge IP with country data"""
        def get_country(ip_int):
            match = ip_df[(ip_df['lower_ip_int'] <= ip_int) & (ip_df['upper_ip_int'] >= ip_int)]
            return match['country'].iloc[0] if len(match) > 0 else 'Unknown'
        
        df['country'] = df['ip_int'].apply(get_country)
        return df
    
    def _perform_eda(self, df, dataset_name):
        """Perform EDA"""
        print(f"\n=== EDA for {dataset_name} ===")
        print(f"Shape: {df.shape}")
        print(f"Target distribution:")
        target_col = 'class' if 'class' in df.columns else 'Class'
        print(df[target_col].value_counts())
        
        # Class imbalance analysis
        class_ratio = df[target_col].value_counts(normalize=True)
        print(f"Class imbalance ratio: {class_ratio}")
    
    def task2_model_building_training(self):
        """Task 2: Model building and training"""
        print("\n=== TASK 2: MODEL BUILDING AND TRAINING ===")
        
        # Process both datasets
        fraud_results = self._train_models(self.fraud_data, 'class', 'Fraud')
        credit_results = self._train_models(self.credit_data, 'Class', 'Credit')
        
        return fraud_results, credit_results
    
    def _train_models(self, df, target_col, dataset_name):
        """Train models for a dataset"""
        print(f"\n--- Training models for {dataset_name} ---")
        
        # Prepare data
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create preprocessing pipeline
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
        # Models to train
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Create pipeline with SMOTE
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Metrics
            auc_roc = roc_auc_score(y_test, y_prob)
            auc_pr = average_precision_score(y_test, y_prob)
            
            results[model_name] = {
                'pipeline': pipeline,
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'X_test': X_test
            }
            
            print(f"AUC-ROC: {auc_roc:.4f}")
            print(f"AUC-PR: {auc_pr:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_pr'])
        self.best_models[dataset_name] = results[best_model_name]
        print(f"\nBest model for {dataset_name}: {best_model_name}")
        
        return results
    
    def task3_model_explainability(self):
        """Task 3: Model explainability with SHAP"""
        print("\n=== TASK 3: MODEL EXPLAINABILITY ===")
        
        for dataset_name, model_info in self.best_models.items():
            print(f"\n--- SHAP Analysis for {dataset_name} ---")
            
            pipeline = model_info['pipeline']
            X_test = model_info['X_test']
            
            # Get preprocessed data
            X_processed = pipeline.named_steps['preprocessor'].transform(X_test)
            
            # Get feature names after preprocessing
            feature_names = self._get_feature_names(pipeline, X_test)
            X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
            
            # Create SHAP explainer
            model = pipeline.named_steps['classifier']
            
            if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_processed_df.iloc[:100])
                
                # For binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                # Summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_processed_df.iloc[:100], show=False)
                plt.title(f'SHAP Summary Plot - {dataset_name}')
                plt.tight_layout()
                plt.savefig(f'reports/{dataset_name}_shap_summary.png')
                plt.show()
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(shap_values).mean(axis=0)
                }).sort_values('importance', ascending=False)
                
                print(f"Top 10 features for {dataset_name}:")
                print(feature_importance.head(10))
    
    def _get_feature_names(self, pipeline, X):
        """Get feature names after preprocessing"""
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get numeric feature names
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Get categorical feature names
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_features:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        else:
            cat_feature_names = []
        
        return numeric_features + cat_feature_names
    
    def run_complete_pipeline(self, fraud_path, ip_path, credit_path):
        """Run the complete pipeline for all tasks"""
        # Task 1
        fraud_data, credit_data = self.task1_data_analysis_preprocessing(
            fraud_path, ip_path, credit_path
        )
        
        # Task 2
        fraud_results, credit_results = self.task2_model_building_training()
        
        # Task 3
        self.task3_model_explainability()
        
        return {
            'fraud_data': fraud_data,
            'credit_data': credit_data,
            'fraud_results': fraud_results,
            'credit_results': credit_results,
            'best_models': self.best_models
        }

def main():
    """Run the complete pipeline"""
    pipeline = CompleteFraudPipeline()
    
    results = pipeline.run_complete_pipeline(
        fraud_path='data/raw/Fraud_Data.csv',
        ip_path='data/raw/IpAddress_to_Country.csv',
        credit_path='data/raw/creditcard.csv'
    )
    
    return results

if __name__ == '__main__':
    main()