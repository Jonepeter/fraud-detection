"""Model training and evaluation for fraud detection."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import os


class FraudModels:
    """
    A class for training, evaluating, and managing machine learning models for fraud detection.

    Methods
    -------
    initialize_models():
        Initialize all supported models.
    train_model(model_name, X_train, y_train):
        Train a specific model.
    evaluate_model(model, X_test, y_test):
        Evaluate model performance.
    cross_validate_model(model_name, X, y, cv_folds=5):
        Perform cross-validation.
    train_all_models(X_train, y_train, X_test, y_test):
        Train and evaluate all models.
    get_best_model(results, metric='auc_pr'):
        Get the best performing model based on specified metric.
    save_model(model, model_name, save_dir):
        Save trained model.
    load_model(model_path):
        Load saved model.
    """

    def __init__(self, random_state=42):
        """
        Initialize the FraudModels class.

        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}

    def initialize_models(self):
        """
        Initialize all models.

        Includes try-except to handle initialization errors.
        """
        try:
            self.models = {
                "logistic_regression": LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight="balanced",
                ),
                "random_forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                "xgboost": xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric="logloss",
                    scale_pos_weight=10,
                ),
                "lightgbm": lgb.LGBMClassifier(
                    random_state=self.random_state, class_weight="balanced", verbose=-1
                ),
            }
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.models = {}

    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model to train.
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.

        Returns
        -------
        model : trained model object

        Includes try-except to handle training errors.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        try:
            model = self.models[model_name]
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"Error training model {model_name}: {e}")
            return None

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance.

        Parameters
        ----------
        model : trained model object
            The model to evaluate.
        X_test : array-like
            Test features.
        y_test : array-like
            Test labels.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.

        Includes try-except to handle evaluation errors.
        """
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            auc_pr = average_precision_score(y_test, y_pred_proba)

            results = {
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "classification_report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
            }

            return results
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {}

    def cross_validate_model(self, model_name, X, y, cv_folds=5):
        """
        Perform cross-validation.

        Parameters
        ----------
        model_name : str
            Name of the model to cross-validate.
        X : array-like
            Features.
        y : array-like
            Labels.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        dict
            Dictionary containing mean and std of AUC-ROC and AUC-PR scores.

        Includes try-except to handle cross-validation errors.
        """
        try:
            model = self.models[model_name]
            cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=self.random_state
            )

            # AUC-ROC scores
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

            # Precision-Recall AUC scores
            pr_scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision")

            return {
                "auc_roc_mean": auc_scores.mean(),
                "auc_roc_std": auc_scores.std(),
                "auc_pr_mean": pr_scores.mean(),
                "auc_pr_std": pr_scores.std(),
            }
        except Exception as e:
            print(f"Error during cross-validation for {model_name}: {e}")
            return {}

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        X_test : array-like
            Test features.
        y_test : array-like
            Test labels.

        Returns
        -------
        dict
            Dictionary containing results for all models.

        Includes try-except to handle errors in training and evaluation.
        """
        self.initialize_models()
        results = {}

        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            try:
                # Train model
                trained_model = self.train_model(model_name, X_train, y_train)
                if trained_model is None:
                    print(f"Skipping {model_name} due to training error.")
                    continue

                # Evaluate model
                evaluation = self.evaluate_model(trained_model, X_test, y_test)

                # Cross-validation
                cv_results = self.cross_validate_model(model_name, X_train, y_train)

                results[model_name] = {
                    "model": trained_model,
                    "evaluation": evaluation,
                    "cv_results": cv_results,
                }

                if evaluation and "auc_roc" in evaluation and "auc_pr" in evaluation:
                    print(
                        f"{model_name} - AUC-ROC: {evaluation['auc_roc']:.4f}, AUC-PR: {evaluation['auc_pr']:.4f}"
                    )
                else:
                    print(f"{model_name} - Evaluation metrics unavailable.")
            except Exception as e:
                print(f"Error in training/evaluating {model_name}: {e}")

        return results

    def get_best_model(self, results, metric="auc_pr"):
        """
        Get the best performing model based on specified metric.

        Parameters
        ----------
        results : dict
            Results dictionary from train_all_models.
        metric : str, default='auc_pr'
            Metric to use for selecting the best model.

        Returns
        -------
        tuple
            (best_model_name, best_model_object)

        Includes try-except to handle errors in selection.
        """
        best_score = 0
        best_model_name = None
        try:
            for model_name, result in results.items():
                score = result["evaluation"].get(metric, 0)
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
            if best_model_name is not None:
                return best_model_name, results[best_model_name]["model"]
            else:
                print("No valid model found for selection.")
                return None, None
        except Exception as e:
            print(f"Error selecting best model: {e}")
            return None, None

    def save_model(self, model, model_name, save_dir):
        """
        Save trained model.

        Parameters
        ----------
        model : trained model object
            The model to save.
        model_name : str
            Name for the saved model file.
        save_dir : str
            Directory to save the model.

        Returns
        -------
        str
            Path to the saved model file.

        Includes try-except to handle saving errors.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            return model_path
        except Exception as e:
            print(f"Error saving model {model_name}: {e}")
            return None

    def load_model(self, model_path):
        """
        Load saved model.

        Parameters
        ----------
        model_path : str
            Path to the saved model file.

        Returns
        -------
        model : loaded model object

        Includes try-except to handle loading errors.
        """
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None
