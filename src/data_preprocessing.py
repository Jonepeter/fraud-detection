"""Data preprocessing pipeline for fraud detection."""

import pandas as pd
import numpy as np
import ipaddress
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    def __init__(self):
        """
        Initialize the DataPreprocessor object.

        Attributes:
            scaler (StandardScaler): Used to standardize numerical features to zero mean and unit variance.
            label_encoders (dict): Stores LabelEncoder objects for each categorical column to ensure consistent encoding/decoding.
        """
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, fraud_path, ip_path, credit_path):
        """
        Load all datasets from the given file paths.

        Args:
            fraud_path (str): Path to the fraud data CSV file.
            ip_path (str): Path to the IP-to-country CSV file.
            credit_path (str): Path to the credit card data CSV file.

        Returns:
            tuple: Loaded fraud_data, ip_data, and credit_data as pandas DataFrames.
        """
        try:
            self.fraud_data = pd.read_csv(fraud_path)
            self.ip_data = pd.read_csv(ip_path)
            self.credit_data = pd.read_csv(credit_path)
            return self.fraud_data, self.ip_data, self.credit_data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.

        Fills numerical columns with their median and categorical columns with their mode.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        try:
            # Fill numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                df[col].fillna(df[col].median(), inplace=True)

            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if not df[col].mode().empty:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)
            return df
        except Exception as e:
            print(f"Error handling missing values: {e}")
            raise

    def clean_data(self, df):
        """
        Clean the dataset by removing duplicates and converting datetime columns.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        try:
            # Remove duplicate rows
            df = df.drop_duplicates()

            # Convert datetime columns if present
            if "signup_time" in df.columns:
                df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
            if "purchase_time" in df.columns:
                df["purchase_time"] = pd.to_datetime(
                    df["purchase_time"], errors="coerce"
                )
            return df
        except Exception as e:
            print(f"Error cleaning data: {e}")
            raise

    def ip_to_int(self, ip):
        """
        Convert an IP address string to its integer representation.

        Args:
            ip (str): IP address in string format.

        Returns:
            int: Integer representation of the IP address, or 0 if conversion fails.
        """
        try:
            return int(ipaddress.IPv4Address(ip))
        except Exception:
            # Return 0 if IP is invalid or conversion fails
            return 0

    def merge_with_country(self, fraud_df, ip_df):
        """
        Merge fraud data with country information based on IP address ranges.

        Args:
            fraud_df (pd.DataFrame): Fraud data with 'ip_address' column.
            ip_df (pd.DataFrame): IP-to-country mapping with 'lower_bound_ip_address', 'upper_bound_ip_address', and 'country'.

        Returns:
            pd.DataFrame: Fraud data with an added 'country' column.
        """
        try:
            # Convert IP addresses to integers
            fraud_df["ip_int"] = fraud_df["ip_address"].apply(self.ip_to_int)

            # Initialize country as 'Unknown'
            merged_df = fraud_df.copy()
            merged_df["country"] = "Unknown"

            # For each IP range, assign the country if the IP falls within the range
            for _, row in ip_df.iterrows():
                mask = (merged_df["ip_int"] >= row["lower_bound_ip_address"]) & (
                    merged_df["ip_int"] <= row["upper_bound_ip_address"]
                )
                merged_df.loc[mask, "country"] = row["country"]
            return merged_df
        except Exception as e:
            print(f"Error merging with country: {e}")
            raise

    def feature_engineering_fraud(self, df):
        """
        Create new features for the fraud dataset.

        Features include time-based, transaction frequency, and velocity features.

        Args:
            df (pd.DataFrame): Fraud data DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new features added.
        """
        try:
            # Time-based features
            df["hour_of_day"] = df["purchase_time"].dt.hour
            df["day_of_week"] = df["purchase_time"].dt.dayofweek
            df["time_since_signup"] = (
                df["purchase_time"] - df["signup_time"]
            ).dt.total_seconds() / 3600

            # Transaction frequency features
            df["user_transaction_count"] = df.groupby("user_id")["user_id"].transform(
                "count"
            )
            df["device_transaction_count"] = df.groupby("device_id")[
                "device_id"
            ].transform("count")

            # Velocity features: time difference between consecutive purchases for each user
            df = df.sort_values(["user_id", "purchase_time"])
            df["time_diff"] = (
                df.groupby("user_id")["purchase_time"].diff().dt.total_seconds() / 3600
            )
            df["time_diff"].fillna(0, inplace=True)
            return df
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            raise

    def encode_categorical(self, df, categorical_cols):
        """
        Encode categorical variables using LabelEncoder.

        Args:
            df (pd.DataFrame): Input DataFrame.
            categorical_cols (list): List of column names to encode.

        Returns:
            pd.DataFrame: DataFrame with categorical columns encoded.
        """
        df_encoded = df.copy()
        try:
            for col in categorical_cols:
                if col in df_encoded.columns:
                    if col not in self.label_encoders:
                        # Fit a new LabelEncoder if not already present
                        self.label_encoders[col] = LabelEncoder()
                        df_encoded[col] = self.label_encoders[col].fit_transform(
                            df_encoded[col].astype(str)
                        )
                    else:
                        # Use existing LabelEncoder for consistency
                        df_encoded[col] = self.label_encoders[col].transform(
                            df_encoded[col].astype(str)
                        )
            return df_encoded
        except Exception as e:
            print(f"Error encoding categorical variables: {e}")
            raise

    def scale_features(self, X_train, X_test):
        """
        Scale numerical features using StandardScaler.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            X_test (pd.DataFrame or np.ndarray): Test features.

        Returns:
            tuple: Scaled X_train and X_test as numpy arrays.
        """
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        except Exception as e:
            print(f"Error scaling features: {e}")
            raise

    def handle_imbalance(self, X, y, method="smote"):
        """
        Handle class imbalance using SMOTE or random undersampling.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target vector.
            method (str): 'smote' for SMOTE oversampling, 'undersample' for random undersampling.

        Returns:
            tuple: Resampled X and y.
        """
        try:
            if method == "smote":
                sampler = SMOTE(sampling_strategy=0.3, random_state=42)
            elif method == "undersample":
                sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            else:
                raise ValueError(
                    "Invalid method for handle_imbalance. Use 'smote' or 'undersample'."
                )

            X_resampled, y_resampled = sampler.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Error handling imbalance: {e}")
            raise

    def preprocess_fraud_data(self):
        """
        Complete preprocessing pipeline for fraud data.

        Steps:
            1. Handle missing values.
            2. Clean data (remove duplicates, convert datetimes).
            3. Merge with country data using IP address.
            4. Feature engineering.
            5. Encode categorical variables.
            6. Prepare features and target.

        Returns:
            tuple: Features (X) and target (y) for fraud data.
        """
        try:
            # Handle missing values
            df = self.handle_missing_values(self.fraud_data)
            # Clean data
            df = self.clean_data(df)
            # Merge with country data
            df = self.merge_with_country(df, self.ip_data)
            # Feature engineering
            df = self.feature_engineering_fraud(df)

            # Select features for modeling
            feature_cols = [
                "purchase_value",
                "age",
                "hour_of_day",
                "day_of_week",
                "time_since_signup",
                "user_transaction_count",
                "device_transaction_count",
                "time_diff",
                "source",
                "browser",
                "sex",
                "country",
            ]

            # Encode categorical variables
            categorical_cols = ["source", "browser", "sex", "country"]
            df = self.encode_categorical(df, categorical_cols)

            # Prepare features and target
            X = df[feature_cols]
            y = df["class"]
            return X, y
        except Exception as e:
            print(f"Error in fraud data preprocessing: {e}")
            raise

    def preprocess_credit_data(self):
        """
        Complete preprocessing pipeline for credit card data.

        Steps:
            1. Handle missing values.
            2. Clean data (remove duplicates, convert datetimes).
            3. Select features and target (features are already PCA transformed).

        Returns:
            tuple: Features (X) and target (y) for credit card data.
        """
        try:
            df = self.handle_missing_values(self.credit_data)
            df = self.clean_data(df)

            # Features are already processed (PCA transformed)
            feature_cols = [col for col in df.columns if col != "Class"]
            X = df[feature_cols]
            y = df["Class"]
            return X, y
        except Exception as e:
            print(f"Error in credit data preprocessing: {e}")
            raise
