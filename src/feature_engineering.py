import pandas as pd
import numpy as np
from src.preprocessing import convert_ip_to_int

def add_time_features(df):
    """
    Add time-based features to the dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with datetime columns
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with additional time-based features
    """
    # Check if the dataframe has the required columns
    if 'purchase_time' in df.columns:
        # Extract hour of day
        df['hour_of_day'] = df['purchase_time'].dt.hour
        
        # Extract day of week
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # Calculate time since signup if signup_time exists
        if 'signup_time' in df.columns:
            df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600  # in hours
    
    # For creditcard.csv, we can create time-based features from the 'Time' column
    elif 'Time' in df.columns:
        # Convert seconds to hours
        df['time_hours'] = df['Time'] / 3600
        
        # Create cyclic features for time of day (assuming Time is seconds from start of day)
        seconds_in_day = 24 * 60 * 60
        df['time_of_day'] = df['Time'] % seconds_in_day
        df['hour_of_day'] = (df['time_of_day'] / 3600).astype(int)
        
        # Create sin and cos transformations for cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    return df

def add_transaction_features(df, user_id_col='user_id'):
    """
    Add transaction frequency and velocity features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    user_id_col : str, default='user_id'
        Column name for user ID
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with additional transaction features
    """
    # Check if the dataframe has the required columns
    if user_id_col in df.columns and 'purchase_time' in df.columns:
        # Sort by user_id and purchase_time
        df = df.sort_values([user_id_col, 'purchase_time'])
        
        # Calculate number of transactions per user
        df['user_transaction_count'] = df.groupby(user_id_col).cumcount() + 1
        
        # Calculate time since last transaction for each user
        df['time_since_last_transaction'] = df.groupby(user_id_col)['purchase_time'].diff().dt.total_seconds() / 60  # in minutes
        
        # Calculate average transaction value per user
        df['avg_user_purchase_value'] = df.groupby(user_id_col)['purchase_value'].transform('mean')
        
        # Calculate standard deviation of transaction value per user
        df['std_user_purchase_value'] = df.groupby(user_id_col)['purchase_value'].transform('std')
        
        # Calculate ratio of current purchase to average purchase for user
        df['purchase_value_ratio'] = df['purchase_value'] / df['avg_user_purchase_value']
        df['purchase_value_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df['purchase_value_ratio'].fillna(1, inplace=True)  # For first transactions
    
    return df

def merge_ip_country_data(fraud_df, ip_country_df):
    """
    Merge fraud data with IP-to-country mapping
    
    Parameters:
    -----------
    fraud_df : pandas.DataFrame
        Fraud data with IP addresses
    ip_country_df : pandas.DataFrame
        IP-to-country mapping data
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe with country information
    """
    # Convert IP addresses to integers
    fraud_df['ip_address_int'] = fraud_df['ip_address'].apply(convert_ip_to_int)
    ip_country_df['lower_bound_int'] = ip_country_df['lower_bound_ip_address'].apply(convert_ip_to_int)
    ip_country_df['upper_bound_int'] = ip_country_df['upper_bound_ip_address'].apply(convert_ip_to_int)
    
    # Initialize country column
    fraud_df['country'] = 'Unknown'
    
    # Match IP addresses to country ranges
    for _, row in ip_country_df.iterrows():
        mask = (fraud_df['ip_address_int'] >= row['lower_bound_int']) & (fraud_df['ip_address_int'] <= row['upper_bound_int'])
        fraud_df.loc[mask, 'country'] = row['country']
    
    # Drop intermediate columns
    fraud_df = fraud_df.drop('ip_address_int', axis=1)
    
    return fraud_df

def add_amount_features(df):
    """
    Add features based on transaction amount
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with 'Amount' column
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with additional amount-based features
    """
    if 'Amount' in df.columns:
        # Log transform amount (helps with skewed distribution)
        df['log_amount'] = np.log1p(df['Amount'])
        
        # Binning amount into categories
        df['amount_bin'] = pd.qcut(df['Amount'], q=10, labels=False, duplicates='drop')
    
    return df