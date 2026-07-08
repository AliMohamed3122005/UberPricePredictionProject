import logging
import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base Classes (DIP)
# ──────────────────────────────────────────────

class BaseVisualizer(ABC):
    """Abstraction for visualization.

    High-level modules (e.g. PreprocessingPipeline) depend on this
    interface — not on any concrete implementation (DIP).
    """

    @abstractmethod
    def run(self, df: pd.DataFrame) -> None:
        """Execute all visualization steps."""
        ...


# ──────────────────────────────────────────────
# Abstract Base Class for Plot Steps
# ──────────────────────────────────────────────

class PlotStep(ABC):
    """Abstract base class for all visualization steps.

    Each subclass draws onto a provided Axes object so the Visualizer
    can compose all plots into a single figure.
    """

    @abstractmethod
    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        """Draw the plot onto the given Axes."""
        ...

    def can_plot(self, df: pd.DataFrame) -> bool:
        """Return True if this step has the required columns to plot."""
        required = getattr(self, "REQUIRED_COLUMN", None)
        if required and required not in df.columns:
            logger.warning("%s: '%s' not in columns, skipping",
                           self.__class__.__name__, required)
            return False
        return True


# ──────────────────────────────────────────────
# Concrete Plot Steps
# ──────────────────────────────────────────────

class FareDistributionPlot(PlotStep):
    """Histogram of fare_amount."""

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        df["fare_amount"].hist(ax=ax, color="cyan", bins=50, edgecolor="black")
        ax.set_title("Fare Amount Distribution", fontsize=13, color="cyan")
        ax.set_xlabel("Fare Amount", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(color="gray", alpha=0.25)


class FareBoxplotByPassengerCount(PlotStep):
    """Boxplot of fare_amount by passenger_count."""

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        grouped = df.groupby("passenger_count")["fare_amount"]
        data = [grouped.get_group(k).values for k in sorted(grouped.groups.keys())]
        labels = sorted(grouped.groups.keys())

        ax.boxplot(data, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="cyan", alpha=0.6),
                   medianprops=dict(color="black"))
        ax.set_title("Fare by Passenger Count", fontsize=13, color="cyan")
        ax.set_xlabel("Passenger Count", fontsize=11)
        ax.set_ylabel("Fare Amount", fontsize=11)
        ax.grid(axis="y", color="gray", alpha=0.25)


class FareBoxplotByWeekend(PlotStep):
    """Boxplot of fare_amount by is_weekend."""

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        if "is_weekend" not in df.columns:
            logger.warning("FareBoxplotByWeekend: 'is_weekend' not in columns, skipping")
            return

        groups = [df.loc[df["is_weekend"] == 0, "fare_amount"].values,
                  df.loc[df["is_weekend"] == 1, "fare_amount"].values]
        ax.boxplot(groups, labels=["Weekday", "Weekend"], patch_artist=True,
                   boxprops=dict(facecolor="magenta", alpha=0.6),
                   medianprops=dict(color="black"))
        ax.set_title("Fare by Weekend vs Weekday", fontsize=13, color="magenta")
        ax.set_ylabel("Fare Amount", fontsize=11)
        ax.grid(axis="y", color="gray", alpha=0.25)


class FareBoxplotByRushHour(PlotStep):
    """Boxplot of fare_amount by is_rush_hour."""

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        if "is_rush_hour" not in df.columns:
            logger.warning("FareBoxplotByRushHour: 'is_rush_hour' not in columns, skipping")
            return

        groups = [df.loc[df["is_rush_hour"] == 0, "fare_amount"].values,
                  df.loc[df["is_rush_hour"] == 1, "fare_amount"].values]
        ax.boxplot(groups, labels=["Off-peak", "Rush hour"], patch_artist=True,
                   boxprops=dict(facecolor="lime", alpha=0.6),
                   medianprops=dict(color="black"))
        ax.set_title("Fare by Rush Hour", fontsize=13, color="lime")
        ax.set_ylabel("Fare Amount", fontsize=11)
        ax.grid(axis="y", color="gray", alpha=0.25)


class DistanceDistributionPlot(PlotStep):
    """Histogram of dist_travel_km."""

    REQUIRED_COLUMN = "dist_travel_km"

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        df["dist_travel_km"].hist(ax=ax, color="lime", bins=50, edgecolor="black", range=(0, 25))
        ax.set_title("Distance Distribution", fontsize=13, color="lime")
        ax.set_xlabel("Distance KM", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_xlim(0, 25)
        ax.grid(color="gray", alpha=0.25)


class FareVsDistancePlot(PlotStep):
    """Scatter plot of fare_amount vs dist_travel_km."""

    REQUIRED_COLUMN = "dist_travel_km"

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        ax.scatter(
            df["dist_travel_km"], df["fare_amount"],
            color="magenta", alpha=0.3,
        )
        ax.set_title("Fare Amount vs Distance", fontsize=13, color="magenta")
        ax.set_xlabel("Distance KM", fontsize=11)
        ax.set_ylabel("Fare Amount", fontsize=11)
        ax.grid(color="gray", alpha=0.25)


class TripsByHourPlot(PlotStep):
    """Bar chart of trip counts per hour."""

    REQUIRED_COLUMN = "hour"

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        trips_by_hour = df["hour"].value_counts().sort_index()
        ax.bar(trips_by_hour.index, trips_by_hour.values,
               color="cyan", edgecolor="black")
        ax.set_title("Trips by Hour", fontsize=13, color="cyan")
        ax.set_xlabel("Hour", fontsize=11)
        ax.set_ylabel("Number of Trips", fontsize=11)
        ax.set_xticks(range(0, 24))
        ax.grid(axis="y", color="gray", alpha=0.25)


class AvgFareByHourPlot(PlotStep):
    """Line plot of average fare per hour."""

    REQUIRED_COLUMN = "hour"

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        avg_fare_by_hour = df.groupby("hour")["fare_amount"].mean()
        ax.plot(avg_fare_by_hour.index, avg_fare_by_hour.values,
                color="lime", marker="o", linewidth=2)
        ax.set_title("Average Fare by Hour", fontsize=13, color="lime")
        ax.set_xlabel("Hour", fontsize=11)
        ax.set_ylabel("Average Fare", fontsize=11)
        ax.set_xticks(range(0, 24))
        ax.grid(color="gray", alpha=0.25)


class PassengerCountPlot(PlotStep):
    """Bar chart of trips by passenger count."""

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        passenger_counts = df["passenger_count"].value_counts().sort_index()
        ax.bar(passenger_counts.index, passenger_counts.values,
               color="cyan", edgecolor="black")
        ax.set_title("Trips by Passenger Count", fontsize=13, color="cyan")
        ax.set_xlabel("Passenger Count", fontsize=11)
        ax.set_ylabel("Number of Trips", fontsize=11)
        ax.grid(axis="y", color="gray", alpha=0.25)


class NumericFeatureHistograms(PlotStep):
    """Histograms for selected numeric features other than fare and distance."""

    FEATURES = [
        "passenger_count",
        "bearing",
        "pickup_dist_from_center",
        "dropoff_dist_from_center",
    ]

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        # Choose the first available feature from the list for this subplot.
        for col in self.FEATURES:
            if col in df.columns:
                series = df[col].dropna()
                series.hist(ax=ax, bins=50, color="orange", edgecolor="black")
                ax.set_title(f"{col} Distribution", fontsize=13, color="orange")
                ax.set_xlabel(col, fontsize=11)
                ax.set_ylabel("Frequency", fontsize=11)
                ax.grid(color="gray", alpha=0.25)
                return

        logger.warning("NumericFeatureHistograms: none of %s found, skipping", self.FEATURES)


class CorrelationHeatmapPlot(PlotStep):
    """Correlation heatmap for numeric features including fare_amount."""

    def plot(self, df: pd.DataFrame, ax: plt.Axes) -> None:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty or "fare_amount" not in numeric_df.columns:
            logger.warning("CorrelationHeatmapPlot: no numeric data with 'fare_amount', skipping")
            return

        corr = numeric_df.corr(numeric_only=True)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Correlation Heatmap", fontsize=13, color="cyan")

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns, fontsize=8)

        fig = ax.get_figure()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)



# ──────────────────────────────────────────────
# Visualizer — displays all plots in one figure
# ──────────────────────────────────────────────

class Visualizer(BaseVisualizer):
    """Renders all PlotStep objects in a single figure grid.

    Plots are arranged in a grid with `n_cols` columns. Only steps
    whose required columns exist in the DataFrame are rendered.
    One call to plt.show() at the end displays everything at once.
    """

    def __init__(self, steps: list[PlotStep], n_cols: int = 3):
        self.steps = steps
        self.n_cols = n_cols

    def run(self, df: pd.DataFrame) -> None:
        # Filter to only plottable steps
        plottable = [s for s in self.steps if s.can_plot(df)]
        n = len(plottable)

        if n == 0:
            logger.warning("Visualizer: no plottable steps, skipping")
            return

        logger.info("Visualizer: rendering %d plots in a single figure", n)

        n_rows = math.ceil(n / self.n_cols)
        plt.style.use("dark_background")
        fig, axes = plt.subplots(n_rows, self.n_cols, figsize=(8 * self.n_cols, 5 * n_rows))

        # Flatten axes to a 1-D array for easy indexing (handles 1-row case)
        if n_rows == 1 and self.n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, step in enumerate(plottable):
            step.plot(df, axes[i])

        # Hide any unused subplot slots
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        plt.show()
        logger.info("Visualizer: finished")