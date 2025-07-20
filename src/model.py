"""
Model training, evaluation, and saving for fraud detection.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Union, Tuple, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import shap

from src.utils import get_absolute_path, create_directory_if_not_exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    params: Dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a machine learning model.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Features.
    y : pandas.Series
        Target variable.
    model_type : str
        Type of model to train ('random_forest', 'xgboost', 'lightgbm', 'logistic_regression').
    params : Dict[str, Any]
        Model hyperparameters.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    Tuple[Any, Dict[str, Any]]
        Trained model and a dictionary with training information.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize model based on type
    if model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(**params)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**params)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train model
    logger.info(f"Training {model_type} model with parameters: {params}")
    model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_test, y_prob)
        metrics['average_precision'] = average_precision_score(y_test, y_prob)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Calculate feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        metrics['feature_importance'] = feature_importance.to_dict('records')
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()
    
    logger.info(f"Model evaluation metrics: {metrics}")
    
    # Training info
    training_info = {
        'model_type': model_type,
        'params': params,
        'metrics': metrics,
        'feature_names': X.columns.tolist(),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_shape': X.shape
    }
    
    return model, training_info

def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: List[str] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Parameters
    ----------
    model : Any
        Trained model.
    X : pandas.DataFrame
        Features.
    y : pandas.Series
        Target variable.
    metrics : List[str], default=None
        List of metrics to calculate.
    threshold : float, default=0.5
        Classification threshold for binary classification.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with evaluation metrics.
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision']
    
    # Get predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Apply threshold for binary classification
    if y_prob is not None:
        y_pred_threshold = (y_prob >= threshold).astype(int)
    else:
        y_pred_threshold = y_pred
    
    # Calculate metrics
    results = {}
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y, y_pred_threshold)
    
    if 'precision' in metrics:
        results['precision'] = precision_score(y, y_pred_threshold)
    
    if 'recall' in metrics:
        results['recall'] = recall_score(y, y_pred_threshold)
    
    if 'f1' in metrics:
        results['f1'] = f1_score(y, y_pred_threshold)
    
    if 'auc' in metrics and y_prob is not None:
        results['auc'] = roc_auc_score(y, y_prob)
    
    if 'average_precision' in metrics and y_prob is not None:
        results['average_precision'] = average_precision_score(y, y_prob)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred_threshold)
    results['confusion_matrix'] = cm.tolist()
    
    return results

def save_model(
    model: Any,
    model_type: str,
    dataset: str,
    metrics: Dict[str, Any],
    params: Dict[str, Any]
) -> str:
    """
    Save a trained model and its metadata.
    
    Parameters
    ----------
    model : Any
        Trained model.
    model_type : str
        Type of model.
    dataset : str
        Dataset used for training.
    metrics : Dict[str, Any]
        Model evaluation metrics.
    params : Dict[str, Any]
        Model hyperparameters.
        
    Returns
    -------
    str
        Path to the saved model.
    """
    # Create models directory if it doesn't exist
    models_dir = get_absolute_path('models')
    create_directory_if_not_exists(models_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model filename
    model_filename = f"{dataset}_{model_type}_{timestamp}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Create metadata
    metadata = {
        'model_name': f"{dataset}_{model_type}_{timestamp}",
        'model_type': model_type,
        'dataset': dataset,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': params,
        'metrics': {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    }
    
    # Save metadata
    metadata_filename = f"{dataset}_{model_type}_{timestamp}.json"
    metadata_path = os.path.join(models_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Model metadata saved to {metadata_path}")
    
    return model_path

def load_model(model_path: str) -> Any:
    """
    Load a saved model.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model.
        
    Returns
    -------
    Any
        Loaded model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def explain_model(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Generate model explanations using SHAP.
    
    Parameters
    ----------
    model : Any
        Trained model.
    X : pandas.DataFrame
        Features.
    sample_size : int, default=100
        Number of samples to use for SHAP explanations.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with SHAP values and explanations.
    """
    # Sample data if it's too large
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X
    
    try:
        # Create explainer based on model type
        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # For models that return a list of shap values (one per class), take the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate feature importance based on SHAP values
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance.to_dict('records'),
            'sample_data': X_sample
        }
    
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        return {
            'error': str(e)
        }