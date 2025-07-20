import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    strategy : str, default='mean'
        Strategy to impute missing values ('mean', 'median', 'most_frequent')
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with imputed missing values
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing values before imputation:")
    print(missing_values[missing_values > 0])
    
    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Impute numerical columns
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy=strategy)
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    # Impute categorical columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # Check for missing values after imputation
    missing_values = df.isnull().sum()
    print("Missing values after imputation:")
    print(missing_values[missing_values > 0])
    
    return df

def clean_data(df):
    """
    Clean the dataframe by removing duplicates and correcting data types
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe
    """
    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    print(f"Number of duplicates: {n_duplicates}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types
    # For Fraud_Data.csv
    if 'signup_time' in df.columns and 'purchase_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    return df

def encode_categorical_features(df, columns=None):
    """
    Encode categorical features using one-hot encoding
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, default=None
        List of categorical columns to encode. If None, all object columns are encoded.
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with encoded categorical features
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Skip if no categorical columns
    if len(columns) == 0:
        return df
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[columns])
    
    # Create a dataframe with encoded data
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns),
        index=df.index
    )
    
    # Drop original categorical columns and concatenate encoded columns
    df = pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)
    
    return df

def scale_features(df, columns=None):
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list, default=None
        List of numerical columns to scale. If None, all numerical columns are scaled.
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with scaled numerical features
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object
    """
    if columns is None:
        # Exclude target column if present
        exclude_cols = []
        if 'class' in df.columns:
            exclude_cols.append('class')
        if 'Class' in df.columns:
            exclude_cols.append('Class')
        
        columns = df.select_dtypes(include=['int64', 'float64']).columns.difference(exclude_cols)
    
    # Skip if no numerical columns
    if len(columns) == 0:
        return df, None
    
    # Apply scaling
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler

def handle_class_imbalance(X, y, method='smote', sampling_strategy=0.1):
    """
    Handle class imbalance using oversampling or undersampling
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target variable
    method : str, default='smote'
        Method to handle class imbalance ('smote', 'undersampling')
    sampling_strategy : float, default=0.1
        Ratio of minority to majority class after resampling
        
    Returns:
    --------
    X_resampled : pandas.DataFrame
        Resampled features
    y_resampled : pandas.Series
        Resampled target variable
    """
    print(f"Class distribution before resampling: {pd.Series(y).value_counts()}")
    
    if method == 'smote':
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    elif method == 'undersampling':
        resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    else:
        raise ValueError("Method must be 'smote' or 'undersampling'")
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    print(f"Class distribution after resampling: {pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled

def convert_ip_to_int(ip_str):
    """
    Convert IP address string to integer
    
    Parameters:
    -----------
    ip_str : str
        IP address string (e.g., '192.168.1.1')
        
    Returns:
    --------
    int
        Integer representation of IP address
    """
    try:
        octets = ip_str.split('.')
        if len(octets) != 4:
            return None
        
        return int(octets[0]) * 16777216 + int(octets[1]) * 65536 + int(octets[2]) * 256 + int(octets[3])
    except:
        return None