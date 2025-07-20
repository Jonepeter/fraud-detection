"""
Utility functions for the fraud detection project.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Parameters
    ----------
    config_path : str, default='config.yaml'
        Path to the configuration file.
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
        
    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    
    Returns
    -------
    str
        Absolute path to the project root directory.
    """
    # Assuming this file is in the src directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    return project_root

def get_absolute_path(relative_path: str) -> str:
    """
    Convert a relative path to an absolute path based on the project root.
    
    Parameters
    ----------
    relative_path : str
        Relative path from the project root.
        
    Returns
    -------
    str
        Absolute path.
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")