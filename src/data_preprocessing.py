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
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, fraud_path, ip_path, credit_path):
        """Load all datasets."""
        self.fraud_data = pd.read_csv(fraud_path)
        self.ip_data = pd.read_csv(ip_path)
        self.credit_data = pd.read_csv(credit_path)
        return self.fraud_data, self.ip_data, self.credit_data
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def clean_data(self, df):
        """Clean the dataset."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert datetime columns
        if 'signup_time' in df.columns:
            df['signup_time'] = pd.to_datetime(df['signup_time'])
        if 'purchase_time' in df.columns:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
            
        return df
    
    def ip_to_int(self, ip):
        """Convert IP address to integer."""
        try:
            return int(ipaddress.IPv4Address(ip))
        except:
            return 0
    
    def merge_with_country(self, fraud_df, ip_df):
        """Merge fraud data with country information."""
        # Convert IP addresses to integers
        fraud_df['ip_int'] = fraud_df['ip_address'].apply(self.ip_to_int)
        
        # Merge with country data
        merged_df = fraud_df.copy()
        merged_df['country'] = 'Unknown'
        
        for _, row in ip_df.iterrows():
            mask = (merged_df['ip_int'] >= row['lower_bound_ip_address']) & \
                   (merged_df['ip_int'] <= row['upper_bound_ip_address'])
            merged_df.loc[mask, 'country'] = row['country']
        
        return merged_df
    
    def feature_engineering_fraud(self, df):
        """Create features for fraud dataset."""
        # Time-based features
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # Transaction frequency features
        df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        df['device_transaction_count'] = df.groupby('device_id')['device_id'].transform('count')
        
        # Velocity features
        df = df.sort_values(['user_id', 'purchase_time'])
        df['time_diff'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
        df['time_diff'].fillna(0, inplace=True)
        
        return df
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables."""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X, y, method='smote'):
        """Handle class imbalance."""
        if method == 'smote':
            sampler = SMOTE(sampling_strategy=0.3, random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def preprocess_fraud_data(self):
        """Complete preprocessing pipeline for fraud data."""
        # Load and clean data
        df = self.handle_missing_values(self.fraud_data)
        df = self.clean_data(df)
        
        # Merge with country data
        df = self.merge_with_country(df, self.ip_data)
        
        # Feature engineering
        df = self.feature_engineering_fraud(df)
        
        # Select features
        feature_cols = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 
                       'time_since_signup', 'user_transaction_count', 
                       'device_transaction_count', 'time_diff', 'source', 
                       'browser', 'sex', 'country']
        
        # Encode categorical variables
        categorical_cols = ['source', 'browser', 'sex', 'country']
        df = self.encode_categorical(df, categorical_cols)
        
        # Prepare features and target
        X = df[feature_cols]
        y = df['class']
        
        return X, y
    
    def preprocess_credit_data(self):
        """Complete preprocessing pipeline for credit card data."""
        df = self.handle_missing_values(self.credit_data)
        df = self.clean_data(df)
        
        # Features are already processed (PCA transformed)
        feature_cols = [col for col in df.columns if col != 'Class']
        X = df[feature_cols]
        y = df['Class']
        
        return X, y