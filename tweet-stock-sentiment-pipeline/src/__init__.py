"""
Tweet Stock Sentiment Pipeline
A production-ready data engineering pipeline for analyzing Twitter sentiment
and correlating it with stock price movements.
"""

from .data_loader import (
    load_tweet_data,
    select_tweet_features,
    load_daily_sentiment
)

from .preprocessor import (
    preprocess_tweet_data,
    aggregate_daily_sentiment,
    calculate_sentiment_scores,
    calculate_weighted_sentiment
)

from .model_trainer import (
    prepare_features,
    train_linear_svc,
    calculate_correlation,
    predict_price_direction
)

__version__ = "1.0.0"
__all__ = [
    'load_tweet_data',
    'select_tweet_features',
    'load_daily_sentiment',
    'preprocess_tweet_data',
    'aggregate_daily_sentiment',
    'calculate_sentiment_scores',
    'calculate_weighted_sentiment',
    'prepare_features',
    'train_linear_svc',
    'calculate_correlation',
    'predict_price_direction'
]

