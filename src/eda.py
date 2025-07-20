import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import sidetable as stb

def load_datasets(data_dir='../data'):
    """
    Load all datasets for the fraud detection project
    
    Parameters:
    -----------
    data_dir : str, default='../data'
        Directory containing the datasets
        
    Returns:
    --------
    tuple
        (fraud_data, ip_country, creditcard) dataframes
    """
    fraud_data = pd.read_csv(f'{data_dir}/Fraud_Data.csv')
    ip_country = pd.read_csv(f'{data_dir}/IpAddress_to_Country.csv')
    creditcard = pd.read_csv(f'{data_dir}/creditcard.csv')
    
    # Convert datetime columns in fraud_data
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    return fraud_data, ip_country, creditcard

def display_basic_info(df, title):
    """
    Display basic information about a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    title : str
        Title for the output
    """
    print(f"\n{title} shape: {df.shape}")
    print("\nDataframe info:")
    display(df.info())
    print("\nFirst 5 rows:")
    display(df.head())
    print("\nMissing values:")
    display(df.stb.missing())
    print("\nSummary statistics:")
    display(df.describe())

def analyze_class_distribution(df, class_col='class'):
    """
    Analyze and visualize class distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    class_col : str, default='class'
        Name of the class column
    """
    # Check class distribution
    class_counts = df[class_col].value_counts()
    fraud_rate = class_counts[1] / len(df) * 100 if 1 in class_counts else 0
    
    print(f"\nClass distribution for {class_col}:")
    display(df.stb.freq([class_col]))
    print(f"Fraud rate: {fraud_rate:.2f}%")
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=class_col, data=df)
    plt.title(f'Class Distribution')
    plt.xlabel(f'Class (0: Non-Fraud, 1: Fraud)')
    plt.ylabel('Count')
    plt.show()

def plot_numerical_distributions(df, numerical_cols, class_col='class'):
    """
    Plot distributions of numerical features by class
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    numerical_cols : list
        List of numerical columns to plot
    class_col : str, default='class'
        Name of the class column
    """
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue=class_col, bins=30, kde=True)
        plt.title(f'Distribution of {col} by Class')
        plt.xlabel(col)
        plt.show()

def analyze_categorical_features(df, categorical_cols, class_col='class'):
    """
    Analyze categorical features and their relationship with the target
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical columns to analyze
    class_col : str, default='class'
        Name of the class column
    """
    for col in categorical_cols:
        # Value counts
        print(f"\nValue counts for {col}:")
        display(df[col].value_counts())
        
        # Plot fraud rate by category
        plt.figure(figsize=(10, 6))
        fraud_rate = df.groupby(col)[class_col].mean()
        fraud_rate.sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Fraud Rate by {col}')
        plt.xlabel(col)
        plt.ylabel('Fraud Rate')
        plt.show()
        
        # Plot count by category and class
        plt.figure(figsize=(12, 6))
        sns.countplot(x=col, hue=class_col, data=df)
        plt.title(f'Count by {col} and Class')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Class', labels=['Non-Fraud', 'Fraud'])
        plt.show()

def analyze_time_patterns(fraud_df):
    """
    Analyze time-related patterns in the fraud data
    
    Parameters:
    -----------
    fraud_df : pandas.DataFrame
        Fraud data dataframe with datetime columns
    """
    # Convert to datetime if not already
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    
    # Calculate time difference between signup and purchase
    fraud_df['time_diff_hours'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
    
    # Plot time difference distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=fraud_df, x='time_diff_hours', hue='class', bins=30, kde=True)
    plt.title('Time Difference Between Signup and Purchase by Class')
    plt.xlabel('Time Difference (hours)')
    plt.show()
    
    # Extract hour of day and day of week
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    
    # Plot fraud rate by hour of day
    plt.figure(figsize=(12, 6))
    hour_fraud_rate = fraud_df.groupby('hour_of_day')['class'].mean()
    hour_fraud_rate.plot(kind='line', marker='o')
    plt.title('Fraud Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate')
    plt.xticks(range(24))
    plt.grid(True)
    plt.show()
    
    # Plot fraud rate by day of week
    plt.figure(figsize=(10, 6))
    day_fraud_rate = fraud_df.groupby('day_of_week')['class'].mean()
    day_fraud_rate.plot(kind='bar')
    plt.title('Fraud Rate by Day of Week')
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
    plt.ylabel('Fraud Rate')
    plt.show()
    
    return fraud_df

def analyze_correlation_matrix(df, target_col=None):
    """
    Analyze and visualize correlation matrix
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str, default=None
        Target column name for correlation analysis
    """
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    
    # If target column is provided, analyze correlation with target
    if target_col and target_col in df.columns:
        # Get correlations with target
        target_corr = pd.DataFrame()
        target_corr['feature'] = df.select_dtypes(include=['int64', 'float64']).columns
        target_corr['correlation'] = [df[col].corr(df[target_col]) for col in target_corr['feature']]
        target_corr = target_corr.sort_values('correlation', ascending=False)
        
        # Plot top and bottom correlations
        plt.figure(figsize=(12, 8))
        top_features = pd.concat([target_corr.head(10), target_corr.tail(10)])
        sns.barplot(x='correlation', y='feature', data=top_features)
        plt.title(f'Top 10 Positively and Negatively Correlated Features with {target_col}')
        plt.show()
        
        return target_corr
    
    return corr_matrix

def analyze_ip_country_data(ip_country_df):
    """
    Analyze IP to country mapping data
    
    Parameters:
    -----------
    ip_country_df : pandas.DataFrame
        IP to country mapping dataframe
    """
    # Count countries
    country_counts = ip_country_df['country'].value_counts()
    print(f"Number of unique countries: {len(country_counts)}")
    
    # Plot top countries
    plt.figure(figsize=(12, 6))
    country_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Countries by IP Range Count')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.show()
    
    return country_counts