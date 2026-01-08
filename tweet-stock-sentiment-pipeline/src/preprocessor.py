"""
Data Preprocessing Module
Handles cleaning, transformation, and feature engineering for tweet data.
"""

import pandas as pd
import re
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


def convert_to_datetime(df: pd.DataFrame, date_column: str = "createdAt") -> pd.DataFrame:
    """
    Convert date column to datetime and extract date.
    
    Args:
        df: Input dataframe
        date_column: Name of the date column
        
    Returns:
        DataFrame with datetime converted and date extracted
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df['date'] = df[date_column].dt.date
    return df


def remove_retweets(df: pd.DataFrame, text_column: str = "fullText") -> pd.DataFrame:
    """
    Remove retweets (rows where text starts with 'RT').
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        
    Returns:
        DataFrame with retweets removed
    """
    df = df.copy()
    df = df[df[text_column].str.startswith('RT') == False]
    return df


def clean_text(df: pd.DataFrame, text_column: str = "fullText") -> pd.DataFrame:
    """
    Clean tweet text by removing URLs and short tweets.
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        
    Returns:
        DataFrame with cleaned text
    """
    df = df.copy()
    # Remove URLs
    df[text_column] = df[text_column].str.replace(r'http\S+', '', regex=True)
    df[text_column] = df[text_column].str.replace(r'www\S+', '', regex=True)
    
    # Remove tweets that are too short (less than 10 characters)
    df = df[df[text_column].str.len() > 10]
    
    return df


def calculate_sentiment_scores(df: pd.DataFrame, 
                               text_column: str = "fullText",
                               use_vader: bool = True) -> pd.DataFrame:
    """
    Calculate sentiment scores using VADER sentiment analyzer.
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        use_vader: Whether to use VADER (default True)
        
    Returns:
        DataFrame with sentiment_score column added
    """
    if use_vader:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Download VADER lexicon if not already downloaded
        try:
            nltk.data.find('tokenizers/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        
        sid = SentimentIntensityAnalyzer()
        df = df.copy()
        df['sentiment_score'] = df[text_column].apply(
            lambda x: sid.polarity_scores(x)['compound']
        )
    else:
        # Placeholder for other sentiment methods
        df['sentiment_score'] = 0.0
    
    return df


def calculate_weighted_sentiment(df: pd.DataFrame,
                                sentiment_column: str = "sentiment_score",
                                like_column: str = "likeCount") -> pd.DataFrame:
    """
    Calculate weighted sentiment by multiplying sentiment score with like count.
    
    Args:
        df: Input dataframe
        sentiment_column: Name of the sentiment score column
        like_column: Name of the like count column
        
    Returns:
        DataFrame with weighted_sentiment column added
    """
    df = df.copy()
    df['weighted_sentiment'] = df[sentiment_column] * df[like_column]
    return df


def aggregate_daily_sentiment(df: pd.DataFrame,
                              date_column: str = "date",
                              sentiment_column: str = "sentiment_score",
                              weighted_column: str = "weighted_sentiment",
                              like_column: str = "likeCount",
                              text_column: str = "fullText") -> pd.DataFrame:
    """
    Aggregate sentiment data by date.
    
    Args:
        df: Input dataframe
        date_column: Name of the date column
        sentiment_column: Name of the sentiment score column
        weighted_column: Name of the weighted sentiment column
        like_column: Name of the like count column
        text_column: Name of the text column (for counting tweets)
        
    Returns:
        DataFrame with daily aggregated sentiment metrics
    """
    daily_sentiment = df.groupby(date_column).agg({
        sentiment_column: 'mean',
        weighted_column: 'sum',
        like_column: 'sum',
        text_column: 'count'  # Number of tweets that day
    }).rename(columns={
        like_column: 'total_likes',
        text_column: 'tweet_count'
    }).reset_index()
    
    return daily_sentiment


def preprocess_tweet_data(df: pd.DataFrame,
                         text_column: str = "fullText",
                         date_column: str = "createdAt",
                         like_column: str = "likeCount",
                         save_intermediate: Optional[str] = None) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for tweet data.
    
    Args:
        df: Input dataframe
        text_column: Name of the text column
        date_column: Name of the date column
        like_column: Name of the like count column
        save_intermediate: Optional path to save intermediate cleaned data
        
    Returns:
        Fully preprocessed dataframe with sentiment scores
    """
    # Step 1: Convert to datetime
    df = convert_to_datetime(df, date_column)
    
    # Step 2: Remove retweets
    df = remove_retweets(df, text_column)
    
    # Step 3: Clean text
    df = clean_text(df, text_column)
    
    # Step 4: Calculate sentiment scores
    df = calculate_sentiment_scores(df, text_column)
    
    # Step 5: Calculate weighted sentiment
    df = calculate_weighted_sentiment(df)
    
    # Save intermediate if requested
    if save_intermediate:
        df.to_csv(save_intermediate, index=False)
    
    return df

