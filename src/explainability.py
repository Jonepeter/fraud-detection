"""Model explainability using SHAP."""

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


class ModelExplainer:
    def __init__(self, model, X_train, X_test, feature_names=None):
        """
        Initialize the ModelExplainer with model, training and test data, and feature names.

        Parameters
        ----------
        model : fitted model object
            The trained model to explain.
        X_train : pd.DataFrame or np.ndarray
            Training data used for fitting the model.
        X_test : pd.DataFrame or np.ndarray
            Test data to explain.
        feature_names : list, optional
            List of feature names.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def initialize_explainer(self):
        """
        Initialize SHAP explainer based on model type.

        Uses TreeExplainer for tree-based models, LinearExplainer for linear models.
        Includes try-except to handle initialization errors.
        """
        model_name = type(self.model).__name__.lower()
        try:
            if (
                "tree" in model_name
                or "forest" in model_name
                or "xgb" in model_name
                or "lgbm" in model_name
            ):
                # Use TreeExplainer for tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For linear models
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
        except Exception as e:
            # Catch and print any error during explainer initialization
            print(f"Error initializing SHAP explainer: {e}")
            self.explainer = None

    def calculate_shap_values(self, X=None):
        """
        Calculate SHAP values for the given data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, optional
            Data to calculate SHAP values for. If None, uses self.X_test.

        Returns
        -------
        np.ndarray
            SHAP values for the data.

        Includes try-except to handle calculation errors.
        """
        if self.explainer is None:
            self.initialize_explainer()

        if X is None:
            X = self.X_test

        try:
            self.shap_values = self.explainer.shap_values(X)
            # For binary classification, get positive class SHAP values
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]
            return self.shap_values
        except Exception as e:
            # Catch and print any error during SHAP value calculation
            print(f"Error calculating SHAP values: {e}")
            self.shap_values = None

    def plot_summary(self, save_path=None, max_display=20):
        """
        Create SHAP summary plot.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot image.
        max_display : int, default=20
            Maximum number of features to display.

        Includes try-except to handle plotting errors.
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                self.X_test,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False,
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.show()
        except Exception as e:
            # Catch and print any error during summary plot
            print(f"Error creating SHAP summary plot: {e}")

    def plot_waterfall(self, instance_idx=0, save_path=None):
        """
        Create SHAP waterfall plot for a single instance.

        Parameters
        ----------
        instance_idx : int, default=0
            Index of the instance to plot.
        save_path : str, optional
            Path to save the plot image.

        Includes try-except to handle plotting errors.
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        try:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[instance_idx],
                    base_values=self.explainer.expected_value,
                    data=(
                        self.X_test.iloc[instance_idx]
                        if hasattr(self.X_test, "iloc")
                        else self.X_test[instance_idx]
                    ),
                    feature_names=self.feature_names,
                ),
                show=False,
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.show()
        except Exception as e:
            # Catch and print any error during waterfall plot
            print(f"Error creating SHAP waterfall plot: {e}")

    def plot_force(self, instance_idx=0, save_path=None):
        """
        Create SHAP force plot for a single instance.

        Parameters
        ----------
        instance_idx : int, default=0
            Index of the instance to plot.
        save_path : str, optional
            Path to save the plot image.

        Returns
        -------
        force_plot : object
            The SHAP force plot object.

        Includes try-except to handle plotting errors.
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        force_plot = None
        try:
            force_plot = shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[instance_idx],
                (
                    self.X_test.iloc[instance_idx]
                    if hasattr(self.X_test, "iloc")
                    else self.X_test[instance_idx]
                ),
                feature_names=self.feature_names,
                matplotlib=True,
                show=False,
            )
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.show()
            return force_plot
        except Exception as e:
            # Catch and print any error during force plot
            print(f"Error creating SHAP force plot: {e}")

    def get_feature_importance(self):
        """
        Get global feature importance from SHAP values.

        Returns
        -------
        pd.DataFrame
            DataFrame with features and their mean absolute SHAP values.

        Includes try-except to handle calculation errors.
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        try:
            # Calculate mean absolute SHAP values
            importance = np.abs(self.shap_values).mean(axis=0)
            # Create DataFrame
            importance_df = pd.DataFrame(
                {
                    "feature": (
                        self.feature_names
                        if self.feature_names
                        else [f"feature_{i}" for i in range(len(importance))]
                    ),
                    "importance": importance,
                }
            ).sort_values("importance", ascending=False)
            return importance_df
        except Exception as e:
            # Catch and print any error during feature importance calculation
            print(f"Error calculating feature importance: {e}")
            importance_df = pd.DataFrame()

    def plot_feature_importance(self, save_path=None, top_n=15):
        """
        Plot feature importance.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot image.
        top_n : int, default=15
            Number of top features to display.

        Includes try-except to handle plotting errors.
        """
        importance_df = self.get_feature_importance()
        try:
            plt.figure(figsize=(10, 8))
            plt.barh(
                range(min(top_n, len(importance_df))),
                importance_df["importance"].head(top_n),
            )
            plt.yticks(
                range(min(top_n, len(importance_df))),
                importance_df["feature"].head(top_n),
            )
            plt.xlabel("Mean |SHAP Value|")
            plt.title("Feature Importance (SHAP)")
            plt.gca().invert_yaxis()
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.show()
        except Exception as e:
            # Catch and print any error during feature importance plot
            print(f"Error creating feature importance plot: {e}")

    def generate_explanation_report(self, save_dir):
        """
        Generate comprehensive explanation report.

        Parameters
        ----------
        save_dir : str
            Directory to save the report files.

        Returns
        -------
        pd.DataFrame
            DataFrame of feature importances.

        Includes try-except to handle errors in report generation.
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            # Catch and print any error during directory creation
            print(f"Error creating directory {save_dir}: {e}")
            return pd.DataFrame()
        try:
            # Summary plot
            self.plot_summary(os.path.join(save_dir, "shap_summary.png"))
            # Feature importance
            self.plot_feature_importance(
                os.path.join(save_dir, "feature_importance.png")
            )
            # Force plot for first instance
            self.plot_force(0, os.path.join(save_dir, "force_plot_instance_0.png"))
            # Waterfall plot for first instance
            self.plot_waterfall(
                0, os.path.join(save_dir, "waterfall_plot_instance_0.png")
            )
            # Save feature importance as CSV
            importance_df = self.get_feature_importance()
            importance_df.to_csv(
                os.path.join(save_dir, "feature_importance.csv"), index=False
            )
            return importance_df
        except Exception as e:
            # Catch and print any error during report generation
            print(f"Error generating explanation report: {e}")
            return pd.DataFrame()
