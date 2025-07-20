import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
import os

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target variable
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Testing target
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """
    Train a logistic regression model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    class_weight : str or dict, default='balanced'
        Class weights for imbalanced data
        
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    """
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        class_weight=class_weight,
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, class_weight='balanced'):
    """
    Train a random forest model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    class_weight : str or dict, default='balanced'
        Class weights for imbalanced data
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained random forest model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, scale_pos_weight=None):
    """
    Train an XGBoost model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    scale_pos_weight : float, default=None
        Weight of positive class for imbalanced data
        
    Returns:
    --------
    xgboost.XGBClassifier
        Trained XGBoost model
    """
    if scale_pos_weight is None:
        # Calculate scale_pos_weight based on class distribution
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : object
        Trained model with predict and predict_proba methods
    X_test : pandas.DataFrame
        Testing features
    y_test : pandas.Series
        Testing target
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Calculate precision-recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_confusion_matrix(cm, model_name):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    model_name : str
        Name of the model
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_test, y_prob, model_name):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_test : pandas.Series
        Testing target
    y_prob : numpy.ndarray
        Predicted probabilities
    model_name : str
        Name of the model
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_test, y_prob, model_name):
    """
    Plot precision-recall curve
    
    Parameters:
    -----------
    y_test : pandas.Series
        Testing target
    y_prob : numpy.ndarray
        Predicted probabilities
    model_name : str
        Name of the model
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()

def save_model(model, model_name, model_dir='models'):
    """
    Save model to disk
    
    Parameters:
    -----------
    model : object
        Trained model
    model_name : str
        Name of the model
    model_dir : str, default='models'
        Directory to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_name, model_dir='models'):
    """
    Load model from disk
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model_dir : str, default='models'
        Directory where the model is saved
        
    Returns:
    --------
    object
        Loaded model
    """
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def explain_model_with_shap(model, X_test, model_type='tree'):
    """
    Explain model predictions using SHAP
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pandas.DataFrame
        Testing features
    model_type : str, default='tree'
        Type of model ('tree', 'linear', or 'kernel')
        
    Returns:
    --------
    numpy.ndarray
        SHAP values
    """
    # Create SHAP explainer based on model type
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_test)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # For tree models, shap_values is a list where the second element contains SHAP values for class 1
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values = shap_values[1]
    
    return shap_values, explainer

def plot_shap_summary(shap_values, X_test, max_display=20):
    """
    Plot SHAP summary plot
    
    Parameters:
    -----------
    shap_values : numpy.ndarray
        SHAP values
    X_test : pandas.DataFrame
        Testing features
    max_display : int, default=20
        Maximum number of features to display
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    plt.tight_layout()
    plt.show()

def plot_shap_dependence(shap_values, X_test, feature_idx):
    """
    Plot SHAP dependence plot for a specific feature
    
    Parameters:
    -----------
    shap_values : numpy.ndarray
        SHAP values
    X_test : pandas.DataFrame
        Testing features
    feature_idx : int or str
        Index or name of the feature to plot
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_idx, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.show()