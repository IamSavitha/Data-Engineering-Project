SELECT
    date,
    sentiment_score,
    weighted_sentiment,
    tweet_count,
    total_likes,
    open_price,
    close_price AS actual_close_price,
    predicted_close_price,
    ABS(close_price - predicted_close_price) AS prediction_error,
    ABS(close_price - predicted_close_price) / close_price AS relative_error,
    volume
FROM {{ ref('model__predict_price') }}
ORDER BY date
