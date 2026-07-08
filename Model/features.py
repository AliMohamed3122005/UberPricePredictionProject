"""Feature column contract shared between preprocessing and model training.

Engineered columns must match the outputs of Preprocessing/featuring.py:
  - DatetimeFeatures      → hour, day, month, year, dayofweek
  - DistanceFeature       → dist_travel_km
  - BearingFeature        → bearing
  - DistanceFromCityCenterFeature → pickup_dist_from_center, dropoff_dist_from_center
  - WeekendFeature        → is_weekend
  - RushHourFeature       → is_rush_hour
"""

TARGET_COLUMN = "fare_amount"

RAW_COLUMNS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

ENGINEERED_COLUMNS = [
    "hour",
    "day",
    "month",
    "year",
    "dayofweek",
    "dist_travel_km",
    "bearing",
    "pickup_dist_from_center",
    "dropoff_dist_from_center",
    "is_weekend",
    "is_rush_hour",
]

FEATURE_COLUMNS = RAW_COLUMNS + ENGINEERED_COLUMNS
