# Step-by-Step Execution Guide: Repository to Snowflake

This guide provides detailed instructions for setting up and executing the Twitter Sentiment & Stock Correlation Pipeline from repository clone to data loading in Snowflake.

## Prerequisites

- Python 3.8 or higher
- Git installed
- Access to Snowflake account
- Apache Airflow installed (for production pipeline)
- dbt installed (for transformations)

---

## Step 1: Clone and Setup Repository

### 1.1 Clone the Repository
```bash
git clone https://github.com/IamSavitha/Data-Engineering-Project.git
cd Data-Engineering-Project/tweet-stock-sentiment-pipeline
```

### 1.2 Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 1.3 Install Dependencies
```bash
pip install -r requirements.txt

# Install NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Step 2: Prepare Data Files

### 2.1 Place Raw Data Files
Place your raw tweet data CSV file in the `data/raw/` directory:
```bash
# Ensure the directory exists
mkdir -p data/raw

# Place your tweet data file (e.g., all_musk_posts.csv)
# The file should contain columns: createdAt, fullText, likeCount
```

### 2.2 Verify Data Structure
Your raw data file should have at minimum these columns:
- `createdAt`: Timestamp of the tweet
- `fullText`: Full text content of the tweet
- `likeCount`: Number of likes/engagement

---

## Step 3: Local Testing (Optional - Before Airflow Setup)

### 3.1 Run Preprocessing Pipeline Locally
```bash
# Using Python directly
python -c "
from src.data_loader import load_tweet_data, select_tweet_features
from src.preprocessor import preprocess_tweet_data, aggregate_daily_sentiment

# Load data
tweet_df = load_tweet_data('data/raw/all_musk_posts.csv')
tweet_df = select_tweet_features(tweet_df)

# Preprocess
processed_df = preprocess_tweet_data(tweet_df)
daily_sentiment = aggregate_daily_sentiment(processed_df)

# Save output
daily_sentiment.to_csv('data/processed/daily_sentiment.csv', index=False)
print('✅ Preprocessing complete!')
"
```

### 3.2 Verify Output
```bash
# Check the processed data
head data/processed/daily_sentiment.csv
```

Expected columns in output:
- `date`: Date of aggregation
- `sentiment_score`: Mean sentiment score for the day
- `weighted_sentiment`: Sum of weighted sentiment
- `tweet_count`: Number of tweets that day
- `total_likes`: Total likes for the day

---

## Step 4: Configure Airflow

### 4.1 Install Airflow (if not already installed)
```bash
# Install Airflow with required providers
pip install apache-airflow==2.7.0
pip install apache-airflow-providers-snowflake
pip install apache-airflow-providers-common-sql
```

### 4.2 Initialize Airflow
```bash
# Set Airflow home (if not set)
export AIRFLOW_HOME=$(pwd)

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 4.3 Configure Airflow Variables
Access Airflow UI at `http://localhost:8080` and set the following Variables:

**Admin → Variables → Add:**

1. **TWEET_DATA_PATH**
   - Key: `TWEET_DATA_PATH`
   - Value: `/opt/airflow/data/raw/all_musk_posts.csv` (or your path)

2. **DAILY_SENTIMENT_OUTPUT_PATH**
   - Key: `DAILY_SENTIMENT_OUTPUT_PATH`
   - Value: `/opt/airflow/data/daily_sentiment.csv` (or your path)

3. **ALPHA_VANTAGE_API_KEY**
   - Key: `ALPHA_VANTAGE_API_KEY`
   - Value: `your_alpha_vantage_api_key_here`

### 4.4 Configure Snowflake Connection
**Admin → Connections → Add:**

- **Connection Id**: `snowflake_conn`
- **Connection Type**: `Snowflake`
- **Host**: `your_account.snowflakecomputing.com`
- **Schema**: `dev`
- **Login**: `your_username`
- **Password**: `your_password`
- **Extra** (JSON):
```json
{
  "account": "your_account",
  "warehouse": "your_warehouse",
  "database": "your_database",
  "role": "your_role"
}
```

---

## Step 5: Setup dbt Project

### 5.1 Navigate to dbt Directory
```bash
cd sentiment_analysis_dbt
```

### 5.2 Configure dbt Profile
Edit `profiles.yml` with your Snowflake credentials:
```yaml
sentiment_analysis:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: your_account
      user: your_username
      password: your_password
      role: your_role
      database: your_database
      warehouse: your_warehouse
      schema: dev
      threads: 4
```

### 5.3 Test dbt Connection
```bash
dbt debug --profiles-dir . --project-dir .
```

---

## Step 6: Prepare Airflow Environment

### 6.1 Create Required Directories
```bash
# Create data directories in Airflow home
mkdir -p /opt/airflow/data/raw
mkdir -p /opt/airflow/data/processed

# Copy raw data file to Airflow data directory
cp data/raw/all_musk_posts.csv /opt/airflow/data/raw/
```

### 6.2 Ensure src Module is Accessible
The DAG automatically adds the `src` directory to Python path. Verify the structure:
```
tweet-stock-sentiment-pipeline/
├── dags/
│   └── tesla_stock_etl_api.py
└── src/
    ├── data_loader.py
    ├── preprocessor.py
    └── model_trainer.py
```

---

## Step 7: Execute the Pipeline

### 7.1 Start Airflow Services
```bash
# Start Airflow webserver (in one terminal)
airflow webserver --port 8080

# Start Airflow scheduler (in another terminal)
airflow scheduler
```

### 7.2 Trigger the Pipeline

**Option A: Via Airflow UI**
1. Navigate to `http://localhost:8080`
2. Login with admin credentials
3. Find DAG: `trigger_etl_then_elt`
4. Click "Play" button to trigger manually
5. Monitor execution in the Graph View

**Option B: Via Command Line**
```bash
# Trigger the main orchestration DAG
airflow dags trigger trigger_etl_then_elt
```

### 7.3 Pipeline Execution Flow

The pipeline executes in this order:

1. **ETL DAG** (`tesla_stock_sentiment_api_etl`):
   - ✅ **process_tweet_sentiment**: Processes raw tweets → generates `daily_sentiment.csv`
   - ✅ **load_stock_data_to_snowflake**: Fetches stock data from Alpha Vantage API → loads to `dev.raw.tesla_stock_data`
   - ✅ **load_sentiment_data_to_snowflake**: Loads processed sentiment data → `dev.raw.tesla_tweet_data`
   - ✅ **merge_stock_and_sentiment_to_raw**: Merges both datasets → `dev.raw.tsla_sentiment_merged`

2. **ELT DAG** (`sentiment_analysis_stock_ELT_dbt`):
   - ✅ **dbt_run**: Executes dbt transformations
   - ✅ **dbt_test**: Runs data quality tests
   - ✅ **dbt_snapshot**: Creates historical snapshots

---

## Step 8: Verify Data in Snowflake

### 8.1 Connect to Snowflake
```bash
# Using Snowflake CLI or SQL client
snowsql -a your_account -u your_username
```

### 8.2 Query Raw Tables
```sql
-- Check stock data
SELECT * FROM dev.raw.tesla_stock_data 
ORDER BY date DESC 
LIMIT 10;

-- Check sentiment data
SELECT * FROM dev.raw.tesla_tweet_data 
ORDER BY date DESC 
LIMIT 10;

-- Check merged data
SELECT * FROM dev.raw.tsla_sentiment_merged 
ORDER BY date DESC 
LIMIT 10;
```

### 8.3 Query Transformed Tables (dbt Output)
```sql
-- Check feature engineering outputs
SELECT * FROM dev.input.feature_engineering__sentiment_features 
ORDER BY date DESC 
LIMIT 10;

SELECT * FROM dev.input.feature_engineering__price_features 
ORDER BY date DESC 
LIMIT 10;

SELECT * FROM dev.input.feature_engineering__combined_features 
ORDER BY date DESC 
LIMIT 10;

-- Check prediction model output
SELECT * FROM dev.input.model__predict_price 
ORDER BY date DESC 
LIMIT 10;

-- Check evaluation output
SELECT * FROM dev.output.evaluation__prediction_output 
ORDER BY date DESC 
LIMIT 10;
```

---

## Step 9: Schedule Automated Execution

### 9.1 Configure DAG Schedule
The main trigger DAG (`trigger_etl_then_elt`) is scheduled to run daily. To modify:

Edit `dags/Etl_elt_trigger.py`:
```python
schedule_interval='@daily',  # Change to your preferred schedule
```

Common schedule options:
- `@daily` - Every day at midnight
- `@hourly` - Every hour
- `0 2 * * *` - Daily at 2 AM (cron format)
- `None` - Manual trigger only

### 9.2 Enable DAG in Airflow UI
1. Go to Airflow UI → DAGs
2. Toggle ON the `trigger_etl_then_elt` DAG
3. The scheduler will automatically run it according to schedule

---

## Step 10: Monitor and Troubleshoot

### 10.1 Check Task Logs
In Airflow UI:
1. Click on the DAG run
2. Click on individual task
3. Click "Log" to view execution logs

### 10.2 Common Issues and Solutions

**Issue: ModuleNotFoundError for src modules**
- **Solution**: Ensure `src/` directory is at the same level as `dags/` directory

**Issue: FileNotFoundError for daily_sentiment.csv**
- **Solution**: Check that `process_tweet_sentiment` task completed successfully before `load_sentiment_data_to_snowflake`

**Issue: Snowflake connection timeout**
- **Solution**: Verify Snowflake connection credentials and network access

**Issue: dbt run fails**
- **Solution**: Check dbt profiles.yml configuration and Snowflake permissions

### 10.3 Verify Pipeline Health
```bash
# Check Airflow DAG status
airflow dags list

# Check recent DAG runs
airflow dags list-runs -d trigger_etl_then_elt --state success --no-backfill

# Check task instances
airflow tasks list trigger_etl_then_elt
```

---

## Step 11: Connect Superset for Visualization (Optional)

### 11.1 Install Superset
```bash
pip install apache-superset
superset db upgrade
superset fab create-admin
superset init
```

### 11.2 Add Snowflake Database Connection
1. Open Superset UI
2. Settings → Database Connections → Add Database
3. Select "Snowflake"
4. Enter connection string:
```
snowflake://username:password@account/database/schema?warehouse=warehouse&role=role
```

### 11.3 Create Dashboards
1. Create new charts using Snowflake tables
2. Visualize:
   - Daily sentiment trends
   - Stock price vs sentiment correlation
   - Prediction accuracy metrics
   - Time series analysis

---

## Summary Checklist

- [ ] Repository cloned and dependencies installed
- [ ] Raw data files placed in `data/raw/`
- [ ] Virtual environment activated
- [ ] Airflow installed and initialized
- [ ] Airflow Variables configured (TWEET_DATA_PATH, DAILY_SENTIMENT_OUTPUT_PATH, ALPHA_VANTAGE_API_KEY)
- [ ] Snowflake connection configured in Airflow
- [ ] dbt profiles.yml configured
- [ ] Data directories created in Airflow home
- [ ] DAGs visible in Airflow UI
- [ ] Pipeline executed successfully
- [ ] Data verified in Snowflake
- [ ] DAG scheduled for automated runs
- [ ] Monitoring and logging configured

---

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Airflow webserver
airflow webserver --port 8080

# Start Airflow scheduler
airflow scheduler

# Trigger DAG manually
airflow dags trigger trigger_etl_then_elt

# Check DAG status
airflow dags list-runs -d trigger_etl_then_elt

# Run dbt transformations manually
cd sentiment_analysis_dbt
dbt run --profiles-dir . --project-dir .

# Test dbt connection
dbt debug --profiles-dir . --project-dir .
```

---

## Support

For issues or questions:
1. Check Airflow task logs for detailed error messages
2. Verify all configuration files (Variables, Connections, dbt profiles)
3. Ensure all dependencies are installed correctly
4. Review the README.md for architecture overview

---

**Last Updated**: 2025-01-07 
**Pipeline Version**: 1.0.0

