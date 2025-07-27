"""
Data processing pipeline for fraud detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataPipeline:
    """Data preprocessing pipeline"""
    
    def __init__(self):
        self.ip_country_map = None
        
    def load_data(self, fraud_path, ip_path=None, credit_path=None):
        """Load datasets"""
        data = {}
        
        if fraud_path:
            data['fraud'] = pd.read_csv(fraud_path)
            logger.info(f"Loaded fraud data: {data['fraud'].shape}")
            
        if ip_path:
            data['ip_country'] = pd.read_csv(ip_path)
            self.ip_country_map = data['ip_country']
            logger.info(f"Loaded IP country data: {data['ip_country'].shape}")
            
        if credit_path:
            data['credit'] = pd.read_csv(credit_path)
            logger.info(f"Loaded credit data: {data['credit'].shape}")
            
        return data
    
    def process_fraud_data(self, df):
        """Process e-commerce fraud data"""
        df = df.copy()
        
        # Convert timestamps
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Time features
        df['time_diff_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day'] = df['purchase_time'].dt.dayofweek
        
        # IP features
        if self.ip_country_map is not None:
            df = self._add_country_features(df)
        
        # User aggregations
        user_stats = df.groupby('user_id').agg({
            'purchase_value': ['count', 'mean', 'std'],
            'class': 'mean'
        }).fillna(0)
        
        user_stats.columns = ['user_tx_count', 'user_avg_amount', 'user_std_amount', 'user_fraud_rate']
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Drop original timestamp columns
        df = df.drop(['signup_time', 'purchase_time'], axis=1)
        
        return df
    
    def process_credit_data(self, df):
        """Process credit card data"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = (df['Time'] % (24 * 3600)) // 3600
        df['day'] = df['Time'] // (24 * 3600)
        
        # Amount features
        df['amount_log'] = np.log1p(df['Amount'])
        df['amount_normalized'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        
        return df
    
    def _add_country_features(self, df):
        """Add country features based on IP"""
        def ip_to_int(ip):
            try:
                return sum(int(x) << (8 * (3 - i)) for i, x in enumerate(ip.split('.')))
            except:
                return 0
        
        df['ip_int'] = df['ip_address'].apply(ip_to_int)
        
        # Simple country mapping (simplified)
        df['country'] = 'US'  # Default
        df.loc[df['ip_int'] < 1000000, 'country'] = 'Other'
        
        return df
    
    def clean_data(self, df):
        """Basic data cleaning"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        return df