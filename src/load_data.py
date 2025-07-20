"""
load_data.py

This module provides utilities for loading data from various file formats into pandas DataFrames.
Supported formats include CSV, Parquet, Excel, and JSON. The main function, `load_data`, handles
file extension detection and error handling for unsupported or problematic files.
"""

import os

import pandas as pd

def load_data(filepath = None):
    """
    Load data from a file. Supports CSV, Parquet, Excel, and JSON formats.

    Parameters
    ----------
    filepath : str
        Path to the data file.

    Returns
    -------
    pandas.DataFrame
        Loaded data as a DataFrame.

    Raises
    ------
    ValueError
        If the file format is not supported or loading fails.
    """
    if filepath is None:
        raise ValueError("No filepath provided. Please specify the path to the data file.")

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    try:
        if ext in ['.csv']:
            return pd.read_csv(filepath, low_memory=False, encoding='utf-8')
        elif ext in ['.parquet']:
            return pd.read_parquet(filepath)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        elif ext in ['.json']:
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        raise ValueError(f"Failed to load file '{filepath}': {e}")
