"""
Main script to run the fraud detection pipeline.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_absolute_path, create_directory_if_not_exists
from src.load_data import load_data
from src.preprocessing import (
    handle_missing_values, clean_data, encode_categorical_features,
    scale_features, handle_class_imbalance
)
from src.feature_engineering import (
    create_time_features, create_ip_features, create_aggregation_features
)
from src.model import (
    train_model, evaluate_model, save_model, load_model
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(config_path: str, dataset: str) -> None:
    """
    Run the fraud detection pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    dataset : str
        Dataset to use ('ecommerce' or 'creditcard').
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Create directories if they don't exist
    create_directory_if_not_exists(get_absolute_path('data/processed'))
    create_directory_if_not_exists(get_absolute_path('models'))
    
    # Load data
    if dataset == 'ecommerce':
        logger.info("Loading e-commerce fraud data")
        data_path = get_absolute_path(config['data']['raw']['ecommerce'])
        ip_country_path = get_absolute_path(config['data']['raw']['ip_country'])
        df = load_data(data_path)
        ip_country_df = load_data(ip_country_path)
    elif dataset == 'creditcard':
        logger.info("Loading credit card fraud data")
        data_path = get_absolute_path(config['data']['raw']['creditcard'])
        df = load_data(data_path)
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'ecommerce' or 'creditcard'.")
    
    # Preprocess data
    logger.info("Preprocessing data")
    df = clean_data(df)
    df = handle_missing_values(df)
    
    # Feature engineering
    logger.info("Performing feature engineering")
    if config['feature_engineering']['time_features'] and dataset == 'ecommerce':
        df = create_time_features(df)
    
    if config['feature_engineering']['ip_features'] and dataset == 'ecommerce':
        df = create_ip_features(df, ip_country_df)
    
    if config['feature_engineering']['aggregation_features']:
        df = create_aggregation_features(df, dataset)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Scale features
    if config['preprocessing']['scaling']:
        df, scaler = scale_features(df)
    
    # Save processed data
    processed_path = get_absolute_path(config['data']['processed'][dataset])
    logger.info(f"Saving processed data to {processed_path}")
    df.to_csv(processed_path, index=False)
    
    # Split data into features and target
    if dataset == 'ecommerce':
        target_col = 'is_fraud'
    else:  # creditcard
        target_col = 'Class'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Handle class imbalance
    if config['preprocessing']['handle_imbalance']:
        logger.info("Handling class imbalance")
        imbalance_config = config['preprocessing']['handle_imbalance']
        X, y = handle_class_imbalance(
            X, y,
            method=imbalance_config['method'],
            sampling_strategy=imbalance_config['sampling_strategy']
        )
    
    # Train and evaluate models
    for model_name, model_params in config['models'].items():
        logger.info(f"Training {model_name} model")
        model = train_model(X, y, model_name, model_params)
        
        logger.info(f"Evaluating {model_name} model")
        metrics = evaluate_model(
            model, X, y,
            metrics=config['evaluation']['metrics'],
            threshold=config['evaluation']['threshold']
        )
        
        logger.info(f"Model {model_name} metrics: {metrics}")
        
        # Save model
        save_model(model, model_name, dataset, metrics, model_params)

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to the configuration file'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['ecommerce', 'creditcard'], required=True,
        help='Dataset to use (ecommerce or creditcard)'
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args.config, args.dataset)
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()