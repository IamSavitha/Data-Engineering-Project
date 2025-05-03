from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
}

with DAG(
    dag_id='trigger_etl_then_elt',
    default_args=default_args,
    description='Triggers ETL DAG then ELT DAG sequentially',
    schedule_interval='@daily',
    start_date=datetime(2025, 5, 1),
    catchup=False,
    tags=['trigger', 'ETL', 'ELT'],
) as dag:

    trigger_etl = TriggerDagRunOperator(
        task_id='trigger_etl_dag',
        trigger_dag_id='tesla_stock_sentiment_api_etl',
        wait_for_completion=True,
        poke_interval=60,   # Check every 60 seconds for completion
        reset_dag_run=True  # Start fresh even if previous DAG run exists
    )

    trigger_elt = TriggerDagRunOperator(
        task_id='trigger_elt_dag',
        trigger_dag_id='sentiment_analysis_stock_ELT_dbt',
        wait_for_completion=False  # No need to wait unless required
    )

    trigger_etl >> trigger_elt
