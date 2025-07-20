"""
Feature engineering module for fraud detection.

This module provides functions for creating new features from the raw data,
including time-based features, IP-based features, and aggregation features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from timestamp columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with timestamp columns.
        
    Returns
    -------
    pandas.DataFrame
        Dataframe with additional time-based features.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Check if the required columns exist
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df['signup_time']):
            df['signup_time'] = pd.to_datetime(df['signup_time'])
        if not pd.api.types.is_datetime64_dtype(df['purchase_time']):
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Time difference between signup and purchase (in hours)
        df['time_diff_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # Extract hour of day, day of week, month for both timestamps
        df['signup_hour'] = df['signup_time'].dt.hour
        df['signup_day'] = df['signup_time'].dt.dayofweek
        df['signup_month'] = df['signup_time'].dt.month
        
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day'] = df['purchase_time'].dt.dayofweek
        df['purchase_month'] = df['purchase_time'].dt.month
        
        # Flag for night purchases (between 10 PM and 6 AM)
        df['night_purchase'] = ((df['purchase_hour'] >= 22) | (df['purchase_hour'] <= 6)).astype(int)
        
        # Flag for weekend purchases
        df['weekend_purchase'] = (df['purchase_day'] >= 5).astype(int)
    
    return df

def create_ip_features(df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on IP address information.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with IP address column.
    ip_country_df : pandas.DataFrame
        Dataframe mapping IP address ranges to countries.
        
    Returns
    -------
    pandas.DataFrame
        Dataframe with additional IP-based features.
    """
    from src.preprocessing import convert_ip_to_int
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Check if the required column exists
    if 'ip_address' in df.columns:
        # Convert IP address to integer
        df['ip_int'] = df['ip_address'].apply(convert_ip_to_int)
        
        # Prepare IP country dataframe
        ip_country_df['lower_bound_ip_int'] = ip_country_df['lower_bound_ip_address'].apply(convert_ip_to_int)
        ip_country_df['upper_bound_ip_int'] = ip_country_df['upper_bound_ip_address'].apply(convert_ip_to_int)
        
        # Map IP to country
        def map_ip_to_country(ip_int):
            if ip_int is None:
                return 'unknown'
            
            match = ip_country_df[
                (ip_country_df['lower_bound_ip_int'] <= ip_int) & 
                (ip_country_df['upper_bound_ip_int'] >= ip_int)
            ]
            
            if len(match) > 0:
                return match.iloc[0]['country']
            else:
                return 'unknown'
        
        df['country'] = df['ip_int'].apply(map_ip_to_country)
        
        # Count occurrences of each IP address
        ip_counts = df['ip_address'].value_counts().to_dict()
        df['ip_frequency'] = df['ip_address'].map(ip_counts)
        
        # Flag for high-frequency IPs (potential bots)
        ip_freq_threshold = df['ip_frequency'].quantile(0.95)
        df['high_freq_ip'] = (df['ip_frequency'] > ip_freq_threshold).astype(int)
    
    return df

def create_aggregation_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Create aggregation features based on the dataset type.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    dataset : str
        Dataset type ('ecommerce' or 'creditcard').
        
    Returns
    -------
    pandas.DataFrame
        Dataframe with additional aggregation features.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    if dataset == 'ecommerce':
        # Group by user_id and calculate statistics
        if 'user_id' in df.columns and 'purchase_value' in df.columns:
            user_stats = df.groupby('user_id').agg({
                'purchase_value': ['count', 'mean', 'std', 'min', 'max'],
                'is_fraud': 'mean'  # Fraud rate per user
            })
            
            # Flatten the column names
            user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
            
            # Rename columns for clarity
            user_stats.rename(columns={
                'purchase_value_count': 'user_purchase_count',
                'purchase_value_mean': 'user_avg_purchase',
                'purchase_value_std': 'user_std_purchase',
                'purchase_value_min': 'user_min_purchase',
                'purchase_value_max': 'user_max_purchase',
                'is_fraud_mean': 'user_fraud_rate'
            }, inplace=True)
            
            # Merge back to the original dataframe
            df = df.merge(user_stats, on='user_id', how='left')
            
            # Fill NaN values for users with only one purchase
            df['user_std_purchase'].fillna(0, inplace=True)
    
    elif dataset == 'creditcard':
        # For credit card data, create time-based aggregations
        if 'Time' in df.columns and 'Amount' in df.columns:
            # Create time windows (e.g., 1-hour windows)
            df['time_window'] = (df['Time'] // 3600).astype(int)
            
            # Group by time window and calculate statistics
            time_stats = df.groupby('time_window').agg({
                'Amount': ['count', 'mean', 'std', 'min', 'max'],
                'Class': 'mean'  # Fraud rate per time window
            })
            
            # Flatten the column names
            time_stats.columns = ['_'.join(col).strip() for col in time_stats.columns.values]
            
            # Rename columns for clarity
            time_stats.rename(columns={
                'Amount_count': 'window_tx_count',
                'Amount_mean': 'window_avg_amount',
                'Amount_std': 'window_std_amount',
                'Amount_min': 'window_min_amount',
                'Amount_max': 'window_max_amount',
                'Class_mean': 'window_fraud_rate'
            }, inplace=True)
            
            # Merge back to the original dataframe
            df = df.merge(time_stats, on='time_window', how='left')
    
    return df