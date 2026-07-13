"""SQLite schema for stored prediction requests."""
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class PredictionRecord:
    """One row = one prediction request, as required by the project brief."""

    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    pickup_datetime: str          # ISO string — the trip time the user picked
    passenger_count: int
    ride_type: str
    dist_travel_km: float
    is_weekend: int
    is_rush_hour: int
    model_name: str
    predicted_fare: float
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: int | None = None         # set by SQLite AUTOINCREMENT after insert


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    pickup_lat        REAL NOT NULL,
    pickup_lon        REAL NOT NULL,
    dropoff_lat       REAL NOT NULL,
    dropoff_lon       REAL NOT NULL,
    pickup_datetime   TEXT NOT NULL,
    passenger_count   INTEGER NOT NULL,
    ride_type         TEXT NOT NULL,
    dist_travel_km    REAL NOT NULL,
    is_weekend        INTEGER NOT NULL,
    is_rush_hour      INTEGER NOT NULL,
    model_name        TEXT NOT NULL,
    predicted_fare    REAL NOT NULL,
    created_at        TEXT NOT NULL
);
"""
