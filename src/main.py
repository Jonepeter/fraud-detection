"""
Main script to run the complete fraud detection pipeline.
"""

import argparse
import logging
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.complete_pipeline import CompleteFraudPipeline
from src.utils import create_directory_if_not_exists

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='Complete Fraud Detection Pipeline')
    parser.add_argument(
        '--task', type=str, choices=['all', '1', '2', '3'], default='all',
        help='Task to run (1: Data Analysis, 2: Model Building, 3: Explainability, all: All tasks)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create necessary directories
        create_directory_if_not_exists('data/processed')
        create_directory_if_not_exists('models')
        create_directory_if_not_exists('reports')
        
        # Initialize pipeline
        pipeline = CompleteFraudPipeline()
        
        if args.task == 'all':
            # Run complete pipeline
            results = pipeline.run_complete_pipeline(
                fraud_path='data/raw/Fraud_Data.csv',
                ip_path='data/raw/IpAddress_to_Country.csv',
                credit_path='data/raw/creditcard.csv'
            )
            logger.info("Complete pipeline executed successfully")
            
        elif args.task == '1':
            # Task 1 only
            fraud_data, credit_data = pipeline.task1_data_analysis_preprocessing(
                'data/raw/Fraud_Data.csv',
                'data/raw/IpAddress_to_Country.csv',
                'data/raw/creditcard.csv'
            )
            logger.info("Task 1 completed successfully")
            
        elif args.task == '2':
            # Task 1 + 2
            pipeline.task1_data_analysis_preprocessing(
                'data/raw/Fraud_Data.csv',
                'data/raw/IpAddress_to_Country.csv',
                'data/raw/creditcard.csv'
            )
            fraud_results, credit_results = pipeline.task2_model_building_training()
            logger.info("Task 2 completed successfully")
            
        elif args.task == '3':
            # All tasks (needed for explainability)
            results = pipeline.run_complete_pipeline(
                fraud_path='data/raw/Fraud_Data.csv',
                ip_path='data/raw/IpAddress_to_Country.csv',
                credit_path='data/raw/creditcard.csv'
            )
            logger.info("Task 3 completed successfully")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()