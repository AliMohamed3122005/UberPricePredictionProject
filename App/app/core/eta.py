"""Estimated trip duration — a simple presentation-layer calculation.

Explicitly NOT part of the trained regression model (the model was never
given a duration feature). This exists purely so the results panel can
show "~14 min" the way a real ride-hailing app would.
"""
from config import AVG_SPEED_KMH, AVG_SPEED_KMH_RUSH_HOUR


def estimate_minutes(distance_km: float, is_rush_hour: bool) -> int:
    speed = AVG_SPEED_KMH_RUSH_HOUR if is_rush_hour else AVG_SPEED_KMH
    hours = distance_km / speed
    return max(1, round(hours * 60))
