WITH base AS (
    SELECT *
    FROM {{ source('raw', 'tsla_sentiment_merged') }}
)

SELECT
    date,
    sentiment_score,
    weighted_sentiment,
    tweet_count,
    total_likes,
    LAG(sentiment_score) OVER (ORDER BY date) AS lag_1_sentiment,
    AVG(sentiment_score) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS avg_3d_sentiment
FROM base
