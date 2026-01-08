SELECT *,
    -- Replace with actual ML PREDICT if applicable
    0.5 * sentiment_score +
    0.2 * weighted_sentiment +
    0.05 * tweet_count +
    0.25 * avg_7d_close AS predicted_close_price
FROM {{ ref('feature_engineering__combined_features') }}
