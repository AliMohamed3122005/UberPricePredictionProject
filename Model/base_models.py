import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseRegressor(ABC):
    """Abstraction for regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRegressor":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class BaseEvaluator(ABC):
    """Abstraction for model evaluation."""

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        ...


class CoefficientRegressor(BaseRegressor):
    """Linear models that expose learned coefficients."""

    @property
    @abstractmethod
    def coefficients(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def intercept(self) -> float:
        ...

    @property
    def alpha(self) -> float | None:
        """Regularization strength, if applicable."""
        return getattr(self._model, "alpha_", None)


class FeatureImportanceRegressor(BaseRegressor):
    """Tree-based models that expose feature importances."""

    @property
    @abstractmethod
    def feature_importances(self) -> np.ndarray:
        ...


class RegressionEvaluator(BaseEvaluator):
    """Computes standard regression metrics."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info("RegressionEvaluator: MAE=%.4f  RMSE=%.4f  R²=%.4f",
                    mae, rmse, r2)
        return metrics
