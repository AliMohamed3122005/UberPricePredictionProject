"""Builds the model-ready feature vector for a single trip request.

Mirrors Preprocessing/featuring.py EXACTLY (same formulas, same column
names) so a live prediction is computed identically to how the model
was trained. If the teammate changes a formula there, mirror the change
here — the contract is Model/features.py::FEATURE_COLUMNS.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from config import NYC_CENTER, RUSH_HOUR_WINDOWS, WEEKEND_DAYS

EARTH_RADIUS_KM = 6371


@dataclass(frozen=True)
class TripRequest:
    """Raw input captured from the user (map clicks + sidebar inputs)."""

    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    passenger_count: int
    pickup_datetime: datetime


class BaseFeatureBuilder(ABC):
    """Abstraction so `predictor.py` never depends on the concrete formulas."""

    @abstractmethod
    def build(self, trip: TripRequest) -> pd.DataFrame:
        """Return a single-row DataFrame matching Model.features.FEATURE_COLUMNS."""
        ...


class TripFeatureBuilder(BaseFeatureBuilder):
    """Concrete feature builder — one row in, one row out."""

    def build(self, trip: TripRequest) -> pd.DataFrame:
        row = {
            "pickup_longitude": trip.pickup_lon,
            "pickup_latitude": trip.pickup_lat,
            "dropoff_longitude": trip.dropoff_lon,
            "dropoff_latitude": trip.dropoff_lat,
            "passenger_count": trip.passenger_count,
        }

        dt = trip.pickup_datetime
        row["hour"] = dt.hour
        row["day"] = dt.day
        row["month"] = dt.month
        row["year"] = dt.year
        row["dayofweek"] = dt.isoweekday()  # 1=Mon ... 7=Sun, matches +1 convention upstream

        row["dist_travel_km"] = self._haversine(
            trip.pickup_lon, trip.pickup_lat, trip.dropoff_lon, trip.dropoff_lat
        )
        row["bearing"] = self._bearing(
            trip.pickup_lon, trip.pickup_lat, trip.dropoff_lon, trip.dropoff_lat
        )
        row["pickup_dist_from_center"] = self._haversine(
            trip.pickup_lon, trip.pickup_lat, NYC_CENTER.lon, NYC_CENTER.lat
        )
        row["dropoff_dist_from_center"] = self._haversine(
            trip.dropoff_lon, trip.dropoff_lat, NYC_CENTER.lon, NYC_CENTER.lat
        )
        row["is_weekend"] = int(row["dayofweek"] in WEEKEND_DAYS)
        row["is_rush_hour"] = int(
            any(start <= row["hour"] <= end for start, end in RUSH_HOUR_WINDOWS)
        )

        return pd.DataFrame([row])

    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2) -> float:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return float(EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a)))

    @staticmethod
    def _bearing(lon1, lat1, lon2, lat2) -> float:
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(x, y))
        return float((bearing + 360) % 360)
