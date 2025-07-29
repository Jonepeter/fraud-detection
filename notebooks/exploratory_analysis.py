"""Exploratory Data Analysis for fraud detection datasets."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import config

class EDAAnalyzer:
    def __init__(self):
        self.fraud_data = None
        self.credit_data = None
        self.ip_data = None
        
    def load_data(self):
        """Load all datasets."""
        try:
            self.fraud_data = pd.read_csv(config.FRAUD_DATA_PATH)
            self.credit_data = pd.read_csv(config.CREDITCARD_PATH)
            self.ip_data = pd.read_csv(config.IP_COUNTRY_PATH)
            print("Data loaded successfully!")
        except FileNotFoundError as e:
            print(f"Data file not found: {e}")
            print("Please ensure data files are in the 'data' directory")
    
    def basic_info(self, df, name):
        """Display basic information about the dataset."""
        print(f"\n=== {name} DATASET INFO ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")
        
    def analyze_class_distribution(self, df, target_col, name):
        """Analyze class distribution."""
        print(f"\n=== {name} CLASS DISTRIBUTION ===")
        class_counts = df[target_col].value_counts()
        print(class_counts)
        print(f"Fraud rate: {class_counts[1] / len(df) * 100:.2f}%")
        
        # Plot
        plt.figure(figsize=(8, 6))
        class_counts.plot(kind='bar')
        plt.title(f'{name} - Class Distribution')
        plt.xlabel('Class (0=Normal, 1=Fraud)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.show()
    
    def univariate_analysis(self, df, name):
        """Perform univariate analysis."""
        print(f"\n=== {name} UNIVARIATE ANALYSIS ===")
        
        # Numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            # Distribution plots
            n_cols = min(4, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, col in enumerate(numerical_cols[:16]):  # Limit to 16 plots
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i])
                    axes[i].set_title(f'{col} Distribution')
            
            plt.tight_layout()
            plt.show()
    
    def bivariate_analysis(self, df, target_col, name):
        """Perform bivariate analysis."""
        print(f"\n=== {name} BIVARIATE ANALYSIS ===")
        
        # Numerical columns vs target
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols[:9]):  # Limit to 9 plots
                if i < len(axes):
                    df.boxplot(column=col, by=target_col, ax=axes[i])
                    axes[i].set_title(f'{col} by {target_col}')
            
            plt.tight_layout()
            plt.show()
    
    def correlation_analysis(self, df, name):
        """Analyze correlations."""
        print(f"\n=== {name} CORRELATION ANALYSIS ===")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True)
            plt.title(f'{name} - Correlation Matrix')
            plt.tight_layout()
            plt.show()
    
    def fraud_data_specific_analysis(self):
        """Specific analysis for fraud data."""
        if self.fraud_data is None:
            return
        
        print("\n=== FRAUD DATA SPECIFIC ANALYSIS ===")
        
        # Convert datetime columns
        self.fraud_data['signup_time'] = pd.to_datetime(self.fraud_data['signup_time'])
        self.fraud_data['purchase_time'] = pd.to_datetime(self.fraud_data['purchase_time'])
        
        # Time-based analysis
        self.fraud_data['hour'] = self.fraud_data['purchase_time'].dt.hour
        self.fraud_data['day_of_week'] = self.fraud_data['purchase_time'].dt.dayofweek
        
        # Fraud by hour
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        fraud_by_hour = self.fraud_data.groupby(['hour', 'class']).size().unstack(fill_value=0)
        fraud_by_hour.plot(kind='bar', ax=plt.gca())
        plt.title('Transactions by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        fraud_by_day = self.fraud_data.groupby(['day_of_week', 'class']).size().unstack(fill_value=0)
        fraud_by_day.plot(kind='bar', ax=plt.gca())
        plt.title('Transactions by Day of Week')
        plt.xlabel('Day of Week (0=Monday)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Source and browser analysis
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        source_fraud = self.fraud_data.groupby(['source', 'class']).size().unstack(fill_value=0)
        source_fraud.plot(kind='bar', ax=plt.gca())
        plt.title('Fraud by Source')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 2)
        browser_fraud = self.fraud_data.groupby(['browser', 'class']).size().unstack(fill_value=0)
        browser_fraud.plot(kind='bar', ax=plt.gca())
        plt.title('Fraud by Browser')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 3)
        sex_fraud = self.fraud_data.groupby(['sex', 'class']).size().unstack(fill_value=0)
        sex_fraud.plot(kind='bar', ax=plt.gca())
        plt.title('Fraud by Gender')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_eda(self):
        """Run complete EDA on all datasets."""
        if self.fraud_data is None:
            self.load_data()
        
        if self.fraud_data is not None:
            # Fraud data analysis
            self.basic_info(self.fraud_data, "FRAUD DATA")
            self.analyze_class_distribution(self.fraud_data, 'class', "FRAUD DATA")
            self.univariate_analysis(self.fraud_data, "FRAUD DATA")
            self.bivariate_analysis(self.fraud_data, 'class', "FRAUD DATA")
            self.correlation_analysis(self.fraud_data, "FRAUD DATA")
            self.fraud_data_specific_analysis()
        
        if self.credit_data is not None:
            # Credit card data analysis
            self.basic_info(self.credit_data, "CREDIT CARD DATA")
            self.analyze_class_distribution(self.credit_data, 'Class', "CREDIT CARD DATA")
            self.univariate_analysis(self.credit_data, "CREDIT CARD DATA")
            self.bivariate_analysis(self.credit_data, 'Class', "CREDIT CARD DATA")
            self.correlation_analysis(self.credit_data, "CREDIT CARD DATA")
        
        if self.ip_data is not None:
            self.basic_info(self.ip_data, "IP TO COUNTRY DATA")

def main():
    """Main function to run EDA."""
    analyzer = EDAAnalyzer()
    analyzer.run_complete_eda()

if __name__ == "__main__":
    main()