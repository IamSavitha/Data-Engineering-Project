"""
Model Training Module
Handles training and evaluation of ML models for stock price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def prepare_features(sentiment_df: pd.DataFrame,
                    stock_df: pd.DataFrame,
                    target_column: str = "close",
                    lookback_days: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for model training by merging sentiment and stock data.
    
    Args:
        sentiment_df: DataFrame with daily sentiment metrics
        stock_df: DataFrame with stock price data
        target_column: Column name for target variable
        lookback_days: Number of days to look back for prediction
        
    Returns:
        Tuple of (features_df, target_series)
    """
    # Merge sentiment and stock data on date
    merged_df = pd.merge(
        stock_df,
        sentiment_df,
        on='date',
        how='inner'
    )
    
    # Create target variable: price direction (Up/Down)
    merged_df['price_change'] = merged_df[target_column].pct_change()
    merged_df['price_direction'] = (merged_df['price_change'] > 0).astype(int)
    
    # Shift target to predict next day
    merged_df['target'] = merged_df['price_direction'].shift(-lookback_days)
    
    # Drop rows with NaN
    merged_df = merged_df.dropna()
    
    # Select features
    feature_columns = [
        'sentiment_score',
        'weighted_sentiment',
        'tweet_count',
        'total_likes',
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in merged_df.columns]
    X = merged_df[available_features]
    y = merged_df['target']
    
    return X, y


def train_linear_svc(X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    C: float = 1.0) -> Tuple[LinearSVC, dict]:
    """
    Train a Linear Support Vector Classifier.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed
        C: Regularization parameter
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model
    model = LinearSVC(C=C, random_state=random_state, max_iter=10000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return model, metrics


def calculate_correlation(sentiment_df: pd.DataFrame,
                         stock_df: pd.DataFrame,
                         sentiment_col: str = "sentiment_score",
                         price_col: str = "close") -> float:
    """
    Calculate correlation between sentiment and stock price movements.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        stock_df: DataFrame with stock data
        sentiment_col: Column name for sentiment metric
        price_col: Column name for price
        
    Returns:
        Correlation coefficient
    """
    merged_df = pd.merge(
        stock_df,
        sentiment_df,
        on='date',
        how='inner'
    )
    
    # Calculate price change
    merged_df['price_change'] = merged_df[price_col].pct_change()
    
    # Calculate correlation
    correlation = merged_df[sentiment_col].corr(merged_df['price_change'])
    
    return correlation


def predict_price_direction(model: LinearSVC,
                           features: pd.DataFrame) -> np.ndarray:
    """
    Predict price direction using trained model.
    
    Args:
        model: Trained LinearSVC model
        features: Feature matrix
        
    Returns:
        Array of predictions (0 = Down, 1 = Up)
    """
    predictions = model.predict(features)
    return predictions

