import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from base_models import (
    BaseEvaluator,
    BaseRegressor,
    CoefficientRegressor,
    FeatureImportanceRegressor,
    RegressionEvaluator,
)
from decision_tree_regression import DecisionTreeRegressionModel
from ensemble_regression import (
    GradientBoostingRegressionModel,
    RandomForestRegressionModel,
)
from features import FEATURE_COLUMNS, TARGET_COLUMN
from linear_regression import (
    LassoRegressionModel,
    LinearRegressionModel,
    RidgeRegressionModel,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data I/O (ISP)
# ──────────────────────────────────────────────

class DataReader(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...


class CsvDataReader(DataReader):
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> pd.DataFrame:
        logger.info("Loading data from %s", self.path)
        return pd.read_csv(self.path)


class ModelWriter(ABC):
    @abstractmethod
    def save(self, artifact: dict) -> None:
        ...


class JoblibModelWriter(ModelWriter):
    def __init__(self, path: Path):
        self.path = path

    def save(self, artifact: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, self.path)
        logger.info("Model artifact saved to %s", self.path)


# ──────────────────────────────────────────────
# Feature preparation
# ──────────────────────────────────────────────

class FeaturePreparer:
    """Splits a DataFrame into feature matrix X and target vector y."""

    def __init__(self, feature_columns: list[str], target_column: str):
        self.feature_columns = feature_columns
        self.target_column = target_column

    def prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        missing = [c for c in self.feature_columns + [self.target_column] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        logger.info("FeaturePreparer: %d rows, %d features", len(X), len(self.feature_columns))
        return X, y


class DataSplitter:
    """Train/test split with optional scaling."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logger.info(
            "DataSplitter: train=%d  test=%d",
            len(X_train_scaled), len(X_test_scaled),
        )
        return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), scaler


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

class PredictionPlotter:
    """Plots actual vs predicted fares and residual distribution."""

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path

    def plot(self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str) -> None:
        plt.style.use("dark_background")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_true, y_pred, alpha=0.3, color="cyan", s=8)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lims, lims, color="magenta", linestyle="--", linewidth=1.5)
        axes[0].set_title(f"Actual vs Predicted ({split_name})", color="cyan")
        axes[0].set_xlabel("Actual Fare")
        axes[0].set_ylabel("Predicted Fare")
        axes[0].grid(alpha=0.25)

        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=50, color="lime", edgecolor="black")
        axes[1].set_title(f"Residuals ({split_name})", color="lime")
        axes[1].set_xlabel("Residual (Actual - Predicted)")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_path, dpi=150, bbox_inches="tight")
            logger.info("PredictionPlotter: saved plots to %s", self.output_path)

        plt.show()
        logger.info("PredictionPlotter: finished for %s set", split_name)


# ──────────────────────────────────────────────
# Training Pipeline
# ──────────────────────────────────────────────

class TrainingPipeline:
    """Orchestrates load → prepare → split → train → evaluate → save."""

    def __init__(
        self,
        reader: DataReader,
        preparer: FeaturePreparer,
        splitter: DataSplitter,
        model: BaseRegressor,
        evaluator: BaseEvaluator,
        writer: ModelWriter,
        plotter: PredictionPlotter | None = None,
        model_name: str = "model",
    ):
        self.reader = reader
        self.preparer = preparer
        self.splitter = splitter
        self.model = model
        self.evaluator = evaluator
        self.writer = writer
        self.plotter = plotter
        self.model_name = model_name

    def run(self) -> dict[str, dict[str, float]]:
        df = self.reader.load()
        X, y = self.preparer.prepare(df)
        X_train, X_test, y_train, y_test, scaler = self.splitter.split(X, y)

        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        metrics = {
            "train": self.evaluator.evaluate(y_train, train_pred),
            "test": self.evaluator.evaluate(y_test, test_pred),
        }

        if self.plotter:
            self.plotter.plot(y_test, test_pred, "test")

        artifact = {
            "model_name": self.model_name,
            "model": self.model,
            "scaler": scaler,
            "feature_columns": self.preparer.feature_columns,
            "target_column": self.preparer.target_column,
            "metrics": metrics,
        }
        self.writer.save(artifact)

        self._log_model_details(self.preparer.feature_columns)
        return metrics

    def _log_model_details(self, feature_names: list[str]) -> None:
        if isinstance(self.model, CoefficientRegressor):
            coefs = self.model.coefficients
            ranked = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
            logger.info("%s: top feature coefficients (standardised scale):", self.model_name)
            for name, coef in ranked[:5]:
                logger.info("  %s: %.4f", name, coef)
            logger.info("  intercept: %.4f", self.model.intercept)

            alpha = self.model.alpha
            if alpha is not None:
                logger.info("  alpha: %.6f", alpha)

            if isinstance(self.model, LassoRegressionModel):
                zero_coefs = int(np.sum(coefs == 0))
                logger.info("  zeroed coefficients: %d / %d", zero_coefs, len(coefs))
            return

        if isinstance(self.model, FeatureImportanceRegressor):
            importances = self.model.feature_importances
            ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            logger.info("%s: top feature importances:", self.model_name)
            for name, importance in ranked[:5]:
                logger.info("  %s: %.4f", name, importance)

            if isinstance(self.model, DecisionTreeRegressionModel):
                logger.info("  tree depth: %d", self.model.tree_depth)
                logger.info("  leaf nodes: %d", self.model.n_leaves)
                if self.model.estimator.max_depth is not None:
                    logger.info("  max_depth: %d", self.model.estimator.max_depth)
                logger.info("  min_samples_leaf: %d", self.model.estimator.min_samples_leaf)


def print_results(model_name: str, results: dict[str, dict[str, float]]) -> None:
    print(f"\n── {model_name} Results ──")
    for split, m in results.items():
        print(f"  {split:>5}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    BASE_DIR = Path(__file__).resolve().parents[1]
    data_path = BASE_DIR / "Data" / "processed" / "uber_processed.csv"
    artifacts_dir = BASE_DIR / "Model" / "artifacts"

    reader = CsvDataReader(data_path)
    preparer = FeaturePreparer(FEATURE_COLUMNS, TARGET_COLUMN)
    splitter = DataSplitter(test_size=0.2, random_state=42)
    evaluator = RegressionEvaluator()

    model_configs = [
        ("Linear Regression", LinearRegressionModel(), "linear_regression.joblib", "linear_regression_plots.png"),
        ("Ridge Regression", RidgeRegressionModel(), "ridge_regression.joblib", "ridge_regression_plots.png"),
        ("Lasso Regression", LassoRegressionModel(), "lasso_regression.joblib", "lasso_regression_plots.png"),
        ("Decision Tree", DecisionTreeRegressionModel(), "decision_tree_regression.joblib", "decision_tree_regression_plots.png"),
        ("Random Forest", RandomForestRegressionModel(), "random_forest_regression.joblib", "random_forest_regression_plots.png"),
        ("Gradient Boosting", GradientBoostingRegressionModel(), "gradient_boosting_regression.joblib", "gradient_boosting_regression_plots.png"),
    ]

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, model, artifact_file, plot_file in model_configs:
        pipeline = TrainingPipeline(
            reader=reader,
            preparer=preparer,
            splitter=splitter,
            model=model,
            evaluator=evaluator,
            writer=JoblibModelWriter(artifacts_dir / artifact_file),
            plotter=PredictionPlotter(output_path=artifacts_dir / plot_file),
            model_name=model_name,
        )
        all_results[model_name] = pipeline.run()
        print_results(model_name, all_results[model_name])

    print("\n── Model Comparison (test set) ──")
    print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 50)
    for model_name, results in all_results.items():
        test = results["test"]
        print(f"{model_name:<22} {test['mae']:8.4f} {test['rmse']:8.4f} {test['r2']:8.4f}")
