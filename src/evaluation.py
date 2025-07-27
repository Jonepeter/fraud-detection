"""
Model evaluation utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_prob = model.predict_proba(X_test)[:, 1]
        
    def print_metrics(self):
        """Print evaluation metrics"""
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        
        print(f"\nAUC-ROC: {auc(*roc_curve(self.y_test, self.y_prob)[:2]):.4f}")
        print(f"AUC-PR: {average_precision_score(self.y_test, self.y_prob):.4f}")
        
    def plot_confusion_matrix(self, figsize=(6, 4)):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
    def plot_roc_curve(self, figsize=(6, 4)):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        
    def plot_precision_recall_curve(self, figsize=(6, 4)):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        pr_auc = average_precision_score(self.y_test, self.y_prob)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        
    def generate_report(self, save_path=None):
        """Generate comprehensive evaluation report"""
        report = {
            'classification_report': classification_report(self.y_test, self.y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, self.y_pred).tolist(),
            'roc_auc': auc(*roc_curve(self.y_test, self.y_prob)[:2]),
            'pr_auc': average_precision_score(self.y_test, self.y_prob)
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report