"""
Data Loader Module
Handles loading and initial data preparation from raw data sources.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_tweet_data(file_path: str, low_memory: bool = False) -> pd.DataFrame:
    """
    Load tweet data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing tweet data
        low_memory: Whether to use low memory mode for reading CSV
        
    Returns:
        DataFrame containing tweet data
    """
    try:
        df = pd.read_csv(file_path, low_memory=low_memory)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def select_tweet_features(df: pd.DataFrame, 
                          columns: list = ["createdAt", "fullText", "likeCount"]) -> pd.DataFrame:
    """
    Select and filter required features from tweet dataframe.
    
    Args:
        df: Input dataframe
        columns: List of column names to select
        
    Returns:
        DataFrame with selected columns and dropped NA values
    """
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(f"Missing columns: {missing_cols}")
    
    selected_df = df.loc[:, columns].dropna()
    return selected_df.copy()


def load_daily_sentiment(file_path: str) -> pd.DataFrame:
    """
    Load processed daily sentiment data.
    
    Args:
        file_path: Path to the daily sentiment CSV file
        
    Returns:
        DataFrame containing daily sentiment data
    """
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Sentiment file not found at {file_path}")
    except Exception as e:
        raise Exception(f"Error loading sentiment data: {str(e)}")

