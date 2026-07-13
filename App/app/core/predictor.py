"""Loads a trained model artifact and produces fare predictions.

Depends only on the artifact CONTRACT produced by Model/training.py:
    {
        "model_name": str,
        "model": BaseRegressor,        # has .fit / .predict
        "scaler": StandardScaler,
        "feature_columns": list[str],
        "target_column": str,
        "metrics": {"train": {...}, "test": {...}},
    }
If that contract changes, update `PredictionResult`/`JoblibPredictor` here —
nothing else in the app should need to know the artifact's internal shape.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    fare: float
    model_name: str
    test_mae: float | None
    test_rmse: float | None


class BasePredictor(ABC):
    """Abstraction so UI code never touches joblib/sklearn directly."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> PredictionResult:
        ...

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        ...


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when the expected .joblib artifact hasn't been produced yet."""


class JoblibPredictor(BasePredictor):
    """Loads one artifact file and serves predictions from it."""

    def __init__(self, artifact_path: Path):
        if not artifact_path.exists():
            raise ModelArtifactNotFoundError(
                f"No trained model found at {artifact_path}. "
                "Run Model/training.py first (teammate's ML pipeline)."
            )
        self._artifact = joblib.load(artifact_path)
        logger.info(
            "JoblibPredictor: loaded '%s' from %s",
            self._artifact.get("model_name", "unknown"), artifact_path,
        )

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        ordered = features[self.feature_columns]
        scaled = self._artifact["scaler"].transform(ordered)
        raw_pred = self._artifact["model"].predict(scaled)
        fare = float(raw_pred[0])

        test_metrics = self._artifact.get("metrics", {}).get("test", {})
        return PredictionResult(
            fare=fare,
            model_name=self._artifact.get("model_name", "unknown"),
            test_mae=test_metrics.get("mae"),
            test_rmse=test_metrics.get("rmse"),
        )

    @property
    def feature_columns(self) -> list[str]:
        return self._artifact["feature_columns"]
