import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.ensemble import RandomForestRegressor as SklearnRFR
from sklearn.model_selection import GridSearchCV

from base_models import FeatureImportanceRegressor

logger = logging.getLogger(__name__)

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [16, 20],
    "min_samples_leaf": [5, 10],
}

GRADIENT_BOOSTING_PARAM_GRID = {
    "n_estimators": [100],
    "learning_rate": [0.1],
    "max_depth": [5, 7],
}


@dataclass
class RandomForestRegressionModel(FeatureImportanceRegressor):
    """Random forest regressor with optional grid-search tuning."""

    tune_hyperparameters: bool = True
    param_grid: dict = field(default_factory=lambda: RANDOM_FOREST_PARAM_GRID.copy())
    cv: int = 3
    random_state: int = 42
    _model: SklearnRFR | GridSearchCV = field(init=False, repr=False)

    def __post_init__(self) -> None:
        base = SklearnRFR(random_state=self.random_state, n_jobs=-1)
        if self.tune_hyperparameters:
            self._model = GridSearchCV(
                base,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
        else:
            self._model = base

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRegressionModel":
        logger.info("RandomForestRegressionModel: fitting on %d samples, %d features",
                    X.shape[0], X.shape[1])
        self._model.fit(X, y)

        if isinstance(self._model, GridSearchCV):
            logger.info("RandomForestRegressionModel: best params=%s", self._model.best_params_)
            logger.info("RandomForestRegressionModel: best CV RMSE=%.4f",
                        -self._model.best_score_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def estimator(self) -> SklearnRFR:
        if isinstance(self._model, GridSearchCV):
            return self._model.best_estimator_
        return self._model

    @property
    def feature_importances(self) -> np.ndarray:
        return self.estimator.feature_importances_


@dataclass
class GradientBoostingRegressionModel(FeatureImportanceRegressor):
    """Gradient boosting regressor with optional grid-search tuning."""

    tune_hyperparameters: bool = False
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    param_grid: dict = field(default_factory=lambda: GRADIENT_BOOSTING_PARAM_GRID.copy())
    cv: int = 3
    random_state: int = 42
    _model: SklearnGBR | GridSearchCV = field(init=False, repr=False)

    def __post_init__(self) -> None:
        base = SklearnGBR(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        if self.tune_hyperparameters:
            self._model = GridSearchCV(
                SklearnGBR(random_state=self.random_state),
                param_grid=self.param_grid,
                cv=self.cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
        else:
            self._model = base

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressionModel":
        logger.info("GradientBoostingRegressionModel: fitting on %d samples, %d features",
                    X.shape[0], X.shape[1])
        self._model.fit(X, y)

        if isinstance(self._model, GridSearchCV):
            logger.info("GradientBoostingRegressionModel: best params=%s",
                        self._model.best_params_)
            logger.info("GradientBoostingRegressionModel: best CV RMSE=%.4f",
                        -self._model.best_score_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def estimator(self) -> SklearnGBR:
        if isinstance(self._model, GridSearchCV):
            return self._model.best_estimator_
        return self._model

    @property
    def feature_importances(self) -> np.ndarray:
        return self.estimator.feature_importances_
