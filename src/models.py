"""Model training and evaluation for fraud detection."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

class FraudModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        
    def initialize_models(self):
        """Initialize all models."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=10
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
        }
    
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        results = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return results
    
    def cross_validate_model(self, model_name, X, y, cv_folds=5):
        """Perform cross-validation."""
        model = self.models[model_name]
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # AUC-ROC scores
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        # Precision-Recall AUC scores
        pr_scores = cross_val_score(model, X, y, cv=cv, scoring='average_precision')
        
        return {
            'auc_roc_mean': auc_scores.mean(),
            'auc_roc_std': auc_scores.std(),
            'auc_pr_mean': pr_scores.mean(),
            'auc_pr_std': pr_scores.std()
        }
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models."""
        self.initialize_models()
        results = {}
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            
            # Train model
            trained_model = self.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            evaluation = self.evaluate_model(trained_model, X_test, y_test)
            
            # Cross-validation
            cv_results = self.cross_validate_model(model_name, X_train, y_train)
            
            results[model_name] = {
                'model': trained_model,
                'evaluation': evaluation,
                'cv_results': cv_results
            }
            
            print(f"{model_name} - AUC-ROC: {evaluation['auc_roc']:.4f}, AUC-PR: {evaluation['auc_pr']:.4f}")
        
        return results
    
    def get_best_model(self, results, metric='auc_pr'):
        """Get the best performing model based on specified metric."""
        best_score = 0
        best_model_name = None
        
        for model_name, result in results.items():
            score = result['evaluation'][metric]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        return best_model_name, results[best_model_name]['model']
    
    def save_model(self, model, model_name, save_dir):
        """Save trained model."""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        return model_path
    
    def load_model(self, model_path):
        """Load saved model."""
        return joblib.load(model_path)