"""Configuration settings for the fraud detection project."""

import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FRAUD_DATA_PATH = os.path.join(DATA_DIR, 'Fraud_Data.csv')
IP_COUNTRY_PATH = os.path.join(DATA_DIR, 'IpAddress_to_Country.csv')
CREDITCARD_PATH = os.path.join(DATA_DIR, 'creditcard.csv')

# Model paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Sampling parameters
SAMPLING_STRATEGY = 0.3  # For SMOTE