"""Persistence layer for prediction requests (SQLite3, per project brief)."""
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from db.models import CREATE_TABLE_SQL, PredictionRecord

logger = logging.getLogger(__name__)


class BasePredictionRepository(ABC):
    """Abstraction so UI code never touches sqlite3 directly (DIP)."""

    @abstractmethod
    def save(self, record: PredictionRecord) -> int:
        """Persist a record, return its new row id."""
        ...

    @abstractmethod
    def fetch_recent(self, limit: int = 20) -> list[PredictionRecord]:
        """Return the most recent records, newest first."""
        ...


class SQLitePredictionRepository(BasePredictionRepository):
    """Concrete SQLite-backed repository."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)
        logger.info("SQLitePredictionRepository: schema ready at %s", self.db_path)

    def save(self, record: PredictionRecord) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
                    pickup_datetime, passenger_count, ride_type,
                    dist_travel_km, is_weekend, is_rush_hour,
                    model_name, predicted_fare, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.pickup_lat, record.pickup_lon,
                    record.dropoff_lat, record.dropoff_lon,
                    record.pickup_datetime, record.passenger_count, record.ride_type,
                    record.dist_travel_km, record.is_weekend, record.is_rush_hour,
                    record.model_name, record.predicted_fare, record.created_at,
                ),
            )
            row_id = cursor.lastrowid
        logger.info("SQLitePredictionRepository: saved prediction id=%d fare=%.2f",
                    row_id, record.predicted_fare)
        return row_id

    def fetch_recent(self, limit: int = 20) -> list[PredictionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            PredictionRecord(
                id=row["id"],
                pickup_lat=row["pickup_lat"], pickup_lon=row["pickup_lon"],
                dropoff_lat=row["dropoff_lat"], dropoff_lon=row["dropoff_lon"],
                pickup_datetime=row["pickup_datetime"],
                passenger_count=row["passenger_count"], ride_type=row["ride_type"],
                dist_travel_km=row["dist_travel_km"],
                is_weekend=row["is_weekend"], is_rush_hour=row["is_rush_hour"],
                model_name=row["model_name"], predicted_fare=row["predicted_fare"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
