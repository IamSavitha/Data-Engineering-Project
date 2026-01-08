from airflow import DAG
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from airflow.models import Variable
from airflow.utils.dates import days_ago
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests
import time  # Optional: To handle rate limiting
import os
import sys
from pathlib import Path

# Add src directory to path for importing modular pipeline
# This assumes the DAGs folder is at the same level as src folder
dag_folder = Path(__file__).parent
project_root = dag_folder.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def return_snowflake_conn():
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    conn = hook.get_conn()
    return conn.cursor()

@task
def load_stock_data_to_snowflake(target_table: str, start_year=2023):
    
    cur = return_snowflake_conn()

    try:
        ALPHA_VANTAGE_API_KEY = Variable.get('ALPHA_VANTAGE_API_KEY')
        STOCK_SYMBOL = "TSLA"
        # Alpha Vantage API URL
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={STOCK_SYMBOL}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"

        response = requests.get(url)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
                raise ValueError("Invalid API response: 'Time Series (Daily)' key not found.")
        
        extracted_data = {"symbol": STOCK_SYMBOL, "data": data["Time Series (Daily)"]}
        symbol = extracted_data["symbol"]
        time_series = extracted_data["data"]

    # Convert to DataFrame


        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df[df.index.year >= start_year]
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        df['Date'] = df.index.date
        stock_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        stock_df = stock_df.astype({
            "Open": float, "High": float, "Low": float,
            "Close": float, "Volume": int
        })

        cur.execute("BEGIN;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_table} (
                date DATE PRIMARY KEY,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT
            );
        """)
        cur.execute(f"DELETE FROM {target_table};")

        for _, r in stock_df.iterrows():
            cur.execute(f"""
                            INSERT INTO {target_table}
                            (date, open, high, low, close, volume)
                            VALUES ('{r['Date']}', {r['Open']}, {r['High']}, {r['Low']}, {r['Close']}, {r['Volume']});
                        """)

        cur.execute("COMMIT;")
        print(f"✅ Stock data (via Alpha Vantage) loaded into {target_table}.")

    except Exception as e:
        cur.execute("ROLLBACK;")
        raise e
    finally:
        cur.close()


@task
def process_tweet_sentiment(tweet_data_path: str = None, output_path: str = None):
    """
    Process raw tweet data to generate daily sentiment metrics.
    Uses the modular preprocessing pipeline.
    """
    # Import modular pipeline functions
    # Since we added src to sys.path above, we can import directly
    from data_loader import load_tweet_data, select_tweet_features
    from preprocessor import preprocess_tweet_data, aggregate_daily_sentiment
    
    # Get paths from Airflow Variables or use defaults
    if tweet_data_path is None:
        try:
            tweet_data_path = Variable.get('TWEET_DATA_PATH')
        except KeyError:
            tweet_data_path = '/opt/airflow/data/raw/all_musk_posts.csv'
    
    if output_path is None:
        try:
            output_path = Variable.get('DAILY_SENTIMENT_OUTPUT_PATH')
        except KeyError:
            output_path = '/opt/airflow/data/daily_sentiment.csv'
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and process tweet data
        print(f"Loading tweet data from: {tweet_data_path}")
        tweet_df = load_tweet_data(tweet_data_path, low_memory=False)
        tweet_df = select_tweet_features(tweet_df, columns=["createdAt", "fullText", "likeCount"])
        
        # Preprocess tweets
        print("Preprocessing tweet data...")
        processed_df = preprocess_tweet_data(tweet_df)
        
        # Aggregate daily sentiment
        print("Aggregating daily sentiment...")
        daily_sentiment = aggregate_daily_sentiment(processed_df)
        
        # Save to output path
        daily_sentiment.to_csv(output_path, index=False)
        print(f"✅ Daily sentiment data saved to: {output_path}")
        print(f"   Processed {len(daily_sentiment)} days of sentiment data")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error processing tweet sentiment: {str(e)}")
        raise


@task
def load_sentiment_data_to_snowflake(target_table: str, sentiment_file_path: str = None, start_year=2023):
    """
    Load processed sentiment data from CSV to Snowflake.
    """
    cur = return_snowflake_conn()
    try:
        # Get path from Airflow Variable or use provided path
        if sentiment_file_path is None:
            try:
                sentiment_file_path = Variable.get('DAILY_SENTIMENT_OUTPUT_PATH')
            except KeyError:
                sentiment_file_path = '/opt/airflow/data/daily_sentiment.csv'
        
        if not os.path.exists(sentiment_file_path):
            raise FileNotFoundError(
                f"Sentiment data file not found at {sentiment_file_path}. "
                f"Ensure the preprocessing task runs first."
            )
        
        print(f"Loading sentiment data from: {sentiment_file_path}")
        df = pd.read_csv(sentiment_file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= start_year]
        df['date'] = df['date'].dt.date

        cur.execute("BEGIN;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_table} (
                date DATE PRIMARY KEY,
                sentiment_score FLOAT,
                weighted_sentiment FLOAT,
                tweet_count INT,
                total_likes INT
            );
        """)
        cur.execute(f"DELETE FROM {target_table};")

        for _, r in df.iterrows():
            cur.execute(f"""
                INSERT INTO {target_table}
                (date, sentiment_score, weighted_sentiment, tweet_count, total_likes)
                VALUES ('{r['date']}', {r['sentiment_score']}, {r['weighted_sentiment']},
                        {r['tweet_count']}, {r['total_likes']});
            """)
        cur.execute("COMMIT;")
        print(f"✅ Sentiment data loaded into {target_table}.")

    except Exception as e:
        cur.execute("ROLLBACK;")
        raise e
    finally:
        cur.close()

@task
def merge_stock_and_sentiment_to_raw(raw_table: str):
    cur = return_snowflake_conn()
    try:
        cur.execute("BEGIN;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {raw_table} (
                date DATE PRIMARY KEY,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                sentiment_score FLOAT,
                weighted_sentiment FLOAT,
                tweet_count INT,
                total_likes INT,
                price_change_pct FLOAT
            );
        """)

        cur.execute(f"""
            MERGE INTO {raw_table} AS target
            USING (
                SELECT
                    stock.date,
                    stock.open,
                    stock.high,
                    stock.low,
                    stock.close,
                    stock.volume,
                    sentiment.sentiment_score,
                    sentiment.weighted_sentiment,
                    sentiment.tweet_count,
                    sentiment.total_likes,
                    ((stock.close - stock.open) / stock.open) * 100 AS price_change_pct
                FROM
                    dev.raw.tesla_stock_data AS stock
                LEFT JOIN
                    dev.raw.tesla_tweet_data AS sentiment
                ON stock.date = sentiment.date
            ) AS source
            ON target.date = source.date
            WHEN MATCHED THEN UPDATE SET
                open = source.open,
                high = source.high,
                low = source.low,
                close = source.close,
                volume = source.volume,
                sentiment_score = source.sentiment_score,
                weighted_sentiment = source.weighted_sentiment,
                tweet_count = source.tweet_count,
                total_likes = source.total_likes,
                price_change_pct = source.price_change_pct
            WHEN NOT MATCHED THEN INSERT (
                date, open, high, low, close, volume,
                sentiment_score, weighted_sentiment,
                tweet_count, total_likes, price_change_pct
            )
            VALUES (
                source.date, source.open, source.high, source.low, source.close, source.volume,
                source.sentiment_score, source.weighted_sentiment,
                source.tweet_count, source.total_likes, source.price_change_pct
            );
        """)
        cur.execute("COMMIT;")
        print(f"✅ Incrementally merged into {raw_table}.")

    except Exception as e:
        cur.execute("ROLLBACK;")
        raise e
    finally:
        cur.close()

with DAG(
    dag_id='tesla_stock_sentiment_api_etl',
    start_date=datetime(2025, 2, 21),
    catchup=False,
    schedule='30 2 * * *',  # Daily at 2:30 AM
    tags=['ETL', 'tesla', 'sentiment', 'snowflake']
) as dag:

    target_table = "dev.raw.tesla_stock_data"
    target_table_sentiment = "dev.raw.tesla_tweet_data"
    raw_table = "dev.raw.tsla_sentiment_merged"
    
    # Get configurable paths from Airflow Variables (with defaults)
    try:
        tweet_data_path = Variable.get('TWEET_DATA_PATH')
    except KeyError:
        tweet_data_path = '/opt/airflow/data/raw/all_musk_posts.csv'
    
    try:
        daily_sentiment_path = Variable.get('DAILY_SENTIMENT_OUTPUT_PATH')
    except KeyError:
        daily_sentiment_path = '/opt/airflow/data/daily_sentiment.csv'

    # Task 1: Process tweet data to generate daily sentiment
    process_sentiment_task = process_tweet_sentiment(
        tweet_data_path=tweet_data_path,
        output_path=daily_sentiment_path
    )
    
    # Task 2: Load stock data to Snowflake
    stock_task = load_stock_data_to_snowflake(target_table)
    
    # Task 3: Load sentiment data to Snowflake (depends on processing task)
    sentiment_task = load_sentiment_data_to_snowflake(
        target_table_sentiment,
        sentiment_file_path=daily_sentiment_path
    )
    
    # Task 4: Merge stock and sentiment data
    merge_task = merge_stock_and_sentiment_to_raw(raw_table)

    # Define task dependencies
    process_sentiment_task >> sentiment_task
    [stock_task, sentiment_task] >> merge_task
