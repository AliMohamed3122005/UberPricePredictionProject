import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base Classes (DIP)
# ──────────────────────────────────────────────

class BaseFeatureEngineer(ABC):
    """Abstraction for feature engineering.

    High-level modules (e.g. PreprocessingPipeline) depend on this
    interface — not on any concrete implementation (DIP).
    """

    @abstractmethod
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and return the modified DataFrame."""
        ...


# ──────────────────────────────────────────────
# Abstract Base Class for Feature Steps
# ──────────────────────────────────────────────

class FeatureStep(ABC):
    """Abstract base class for all feature engineering steps.

    To add a new feature, create a subclass and implement apply().
    No existing code needs to change (Open/Closed Principle).
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add one or more feature columns and return the modified DataFrame."""
        ...


# ──────────────────────────────────────────────
# Concrete Feature Steps
# ──────────────────────────────────────────────

class DatetimeFeatures(FeatureStep):
    """Extracts hour, day, month, year, and dayofweek from pickup_datetime."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

        df['hour'] = df['pickup_datetime'].dt.hour
        df['day'] = df['pickup_datetime'].dt.day
        df['month'] = df['pickup_datetime'].dt.month
        df['year'] = df['pickup_datetime'].dt.year
        df['dayofweek'] = df['pickup_datetime'].dt.dayofweek + 1

        logger.info("DatetimeFeatures: added hour, day, month, year, dayofweek")
        return df


class DistanceFeature(FeatureStep):
    """Adds dist_travel_km using the Haversine formula."""

    EARTH_RADIUS_KM = 6371

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['dist_travel_km'] = self._haversine(
            df['pickup_longitude'],
            df['pickup_latitude'],
            df['dropoff_longitude'],
            df['dropoff_latitude'],
        )
        logger.info("DistanceFeature: added dist_travel_km")
        return df

    def _haversine(self, lon1, lat1, lon2, lat2):
        """Vectorised Haversine distance in kilometres."""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return self.EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


class WeekendFeature(FeatureStep):
    """Adds an integer is_weekend column (1 if Saturday/Sunday, else 0)."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['is_weekend'] = df['dayofweek'].isin([6, 7]).astype(int)
        logger.info("WeekendFeature: added is_weekend (int)")
        return df


class RushHourFeature(FeatureStep):
    """Adds an integer is_rush_hour column (1 if 7-9 AM or 4-7 PM, else 0)."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['is_rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9))
            | ((df['hour'] >= 16) & (df['hour'] <= 19))
        ).astype(int)
        logger.info("RushHourFeature: added is_rush_hour (int)")
        return df


class BearingFeature(FeatureStep):
    """Adds a bearing (direction) column in degrees from pickup to dropoff.

    Bearing is calculated using the forward azimuth formula.
    Result is in degrees [0, 360).
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lon1 = np.radians(df['pickup_longitude'])
        lat1 = np.radians(df['pickup_latitude'])
        lon2 = np.radians(df['dropoff_longitude'])
        lat2 = np.radians(df['dropoff_latitude'])

        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.degrees(np.arctan2(x, y))
        df['bearing'] = (bearing + 360) % 360

        logger.info("BearingFeature: added bearing (degrees)")
        return df


class DistanceFromCityCenterFeature(FeatureStep):
    """Adds pickup and dropoff distance from NYC city center.

    Uses Times Square (40.7580, -73.9855) as the reference point.
    Distances are computed via the Haversine formula.
    """

    NYC_CENTER_LAT = 40.7580
    NYC_CENTER_LON = -73.9855
    EARTH_RADIUS_KM = 6371

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['pickup_dist_from_center'] = self._haversine(
            df['pickup_longitude'], df['pickup_latitude'],
            self.NYC_CENTER_LON, self.NYC_CENTER_LAT,
        )
        df['dropoff_dist_from_center'] = self._haversine(
            df['dropoff_longitude'], df['dropoff_latitude'],
            self.NYC_CENTER_LON, self.NYC_CENTER_LAT,
        )
        logger.info("DistanceFromCityCenterFeature: added pickup_dist_from_center, "
                     "dropoff_dist_from_center")
        return df

    def _haversine(self, lon1, lat1, lon2, lat2):
        """Vectorised Haversine distance in kilometres."""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return self.EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))