WITH base AS (
    SELECT *
    FROM {{ source('raw', 'tsla_sentiment_merged') }}
)

SELECT
    date,
    open AS open_price,
    close AS close_price,
    high AS high_price,
    low AS low_price,
    volume,
    ((close - open) / NULLIF(open, 0)) * 100 AS price_change_pct,
    (high - low) / NULLIF(open, 0) * 100 AS high_low_range_pct,
    AVG(close) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS avg_7d_close
FROM base
