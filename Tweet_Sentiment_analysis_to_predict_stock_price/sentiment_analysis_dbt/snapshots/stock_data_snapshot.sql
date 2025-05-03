{% snapshot stock_data_snapshot %}

{{
    config(
      target_database='dev',
      target_schema='snapshots',
      unique_key='date',
      strategy='check',
      check_cols='all'
    )
}}

SELECT * 
FROM {{ source('raw', 'tsla_sentiment_merged') }}

{% endsnapshot %}
