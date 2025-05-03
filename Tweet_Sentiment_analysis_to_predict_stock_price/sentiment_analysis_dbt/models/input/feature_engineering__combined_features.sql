SELECT
    s.date,
    sentiment_score,
    weighted_sentiment,
    tweet_count,
    total_likes,
    lag_1_sentiment,
    avg_3d_sentiment,
    open_price,
    close_price,
    high_price,
    low_price,
    volume,
    price_change_pct,
    high_low_range_pct,
    avg_7d_close,
    weighted_sentiment * tweet_count AS weighted_tweet_impact
FROM {{ ref('feature_engineering__sentiment_features') }} s
JOIN {{ ref('feature_engineering__price_features') }} p
  ON s.date = p.date
