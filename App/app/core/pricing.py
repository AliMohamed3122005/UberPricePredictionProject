"""Turns a raw model prediction into a final quoted price.

Kept separate from core/predictor.py on purpose: the model predicts a
*base* fare from historical data, while ride-type multiplier and surge
are business rules layered on top for the UI — never fed back into the
model and never confused with what the model actually learned.
"""
from dataclasses import dataclass

from config import RIDE_TYPES, RideType, RUSH_HOUR_SURGE_MULTIPLIER
from core.predictor import PredictionResult


@dataclass(frozen=True)
class PriceQuote:
    base_fare: float
    ride_type: RideType
    is_rush_hour: bool
    final_fare: float
    low_estimate: float
    high_estimate: float
    model_name: str


def quote_price(
    prediction: PredictionResult, ride_type: RideType, is_rush_hour: bool,
) -> PriceQuote:
    multiplier = ride_type.multiplier
    if is_rush_hour:
        multiplier *= RUSH_HOUR_SURGE_MULTIPLIER

    final_fare = max(0.0, prediction.fare * multiplier)

    # Confidence range from the model's own test-set MAE, scaled by the
    # same multiplier so it reflects the quoted price, not the raw base fare.
    spread = (prediction.test_mae or 0.0) * multiplier
    return PriceQuote(
        base_fare=prediction.fare,
        ride_type=ride_type,
        is_rush_hour=is_rush_hour,
        final_fare=final_fare,
        low_estimate=max(0.0, final_fare - spread),
        high_estimate=final_fare + spread,
        model_name=prediction.model_name,
    )
