"""
Fraud Detection Pipeline
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
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import logging

logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """Complete fraud detection pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.feature_names = None
        
    def create_preprocessor(self, X):
        """Create preprocessing pipeline"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        return preprocessor
    
    def create_pipeline(self, X, model_type='random_forest'):
        """Create complete ML pipeline"""
        preprocessor = self.create_preprocessor(X)
        
        # Model selection
        if model_type == 'random_forest':
            model = RandomForestClassifier(**self.config['models']['random_forest'])
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(**self.config['models']['xgboost'])
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**self.config['models']['logistic_regression'])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        return pipeline
    
    def fit(self, X, y, model_type='random_forest'):
        """Fit the pipeline"""
        self.pipeline = self.create_pipeline(X, model_type)
        self.pipeline.fit(X, y)
        self.feature_names = X.columns.tolist()
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.pipeline.predict_proba(X)
    
    def save(self, filepath):
        """Save pipeline"""
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load(self, filepath):
        """Load pipeline"""
        self.pipeline = joblib.load(filepath)
        logger.info(f"Pipeline loaded from {filepath}")
        return self

def run_pipeline(data_path, target_col, config, model_type='random_forest'):
    """Run complete pipeline"""
    # Load data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit pipeline
    pipeline = FraudDetectionPipeline(config)
    pipeline.fit(X_train, y_train, model_type)
    
    # Evaluate
    from sklearn.metrics import classification_report, roc_auc_score
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, (X_test, y_test)