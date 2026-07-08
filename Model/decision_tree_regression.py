import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor as SklearnDTR

from base_models import FeatureImportanceRegressor

logger = logging.getLogger(__name__)

DEFAULT_PARAM_GRID = {
    "max_depth": [8, 12, 16, 20],
    "min_samples_leaf": [5, 10, 20],
}


@dataclass
class DecisionTreeRegressionModel(FeatureImportanceRegressor):
    """Decision tree regressor with optional grid-search tuning."""

    tune_hyperparameters: bool = True
    param_grid: dict = field(default_factory=lambda: DEFAULT_PARAM_GRID.copy())
    cv: int = 3
    random_state: int = 42
    _model: SklearnDTR | GridSearchCV = field(init=False, repr=False)

    def __post_init__(self) -> None:
        base = SklearnDTR(random_state=self.random_state)
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressionModel":
        logger.info("DecisionTreeRegressionModel: fitting on %d samples, %d features",
                    X.shape[0], X.shape[1])
        self._model.fit(X, y)

        if isinstance(self._model, GridSearchCV):
            logger.info("DecisionTreeRegressionModel: best params=%s", self._model.best_params_)
            logger.info("DecisionTreeRegressionModel: best CV RMSE=%.4f",
                        -self._model.best_score_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def estimator(self) -> SklearnDTR:
        if isinstance(self._model, GridSearchCV):
            return self._model.best_estimator_
        return self._model

    @property
    def feature_importances(self) -> np.ndarray:
        return self.estimator.feature_importances_

    @property
    def tree_depth(self) -> int:
        return int(self.estimator.get_depth())

    @property
    def n_leaves(self) -> int:
        return int(self.estimator.get_n_leaves())
