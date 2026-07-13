#edited

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import LassoCV as SklearnLassoCV
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import RidgeCV as SklearnRidgeCV

from base_models import CoefficientRegressor

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = np.logspace(-2, 3, 50)


@dataclass
class LinearRegressionModel(CoefficientRegressor):
    """Thin wrapper around scikit-learn's LinearRegression."""

    fit_intercept: bool = True

    def __post_init__(self) -> None:
        self._model = SklearnLR(fit_intercept=self.fit_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        logger.info(
            "LinearRegressionModel: fitting on %d samples, %d features",
            X.shape[0], X.shape[1]
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coefficients(self) -> np.ndarray:
        return self._model.coef_

    @property
    def intercept(self) -> float:
        return float(self._model.intercept_)


@dataclass
class RidgeRegressionModel(CoefficientRegressor):
    """Ridge regression with optional fixed alpha or CV-tuned alpha."""

    fixed_alpha: float | None = None
    fit_intercept: bool = True
    alphas: np.ndarray = field(default_factory=lambda: DEFAULT_ALPHAS.copy())
    _model: SklearnRidge | SklearnRidgeCV = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fixed_alpha is None:
            self._model = SklearnRidgeCV(
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
            )
        else:
            self._model = SklearnRidge(
                alpha=self.fixed_alpha,
                fit_intercept=self.fit_intercept,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionModel":
        logger.info(
            "RidgeRegressionModel: fitting on %d samples, %d features",
            X.shape[0], X.shape[1]
        )
        self._model.fit(X, y)

        if self.fixed_alpha is None:
            logger.info(
                "RidgeRegressionModel: best alpha=%.6f",
                self._model.alpha_,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coefficients(self) -> np.ndarray:
        return self._model.coef_

    @property
    def intercept(self) -> float:
        return float(self._model.intercept_)


@dataclass
class LassoRegressionModel(CoefficientRegressor):
    """Lasso regression with optional fixed alpha or CV-tuned alpha."""

    fixed_alpha: float | None = None
    fit_intercept: bool = True
    alphas: np.ndarray = field(default_factory=lambda: DEFAULT_ALPHAS.copy())
    max_iter: int = 5000
    _model: SklearnLasso | SklearnLassoCV = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fixed_alpha is None:
            self._model = SklearnLassoCV(
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                n_jobs=-1,
            )
        else:
            self._model = SklearnLasso(
                alpha=self.fixed_alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressionModel":
        logger.info(
            "LassoRegressionModel: fitting on %d samples, %d features",
            X.shape[0], X.shape[1]
        )
        self._model.fit(X, y)

        if self.fixed_alpha is None:
            logger.info(
                "LassoRegressionModel: best alpha=%.6f",
                self._model.alpha_,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def coefficients(self) -> np.ndarray:
        return self._model.coef_

    @property
    def intercept(self) -> float:
        return float(self._model.intercept_)