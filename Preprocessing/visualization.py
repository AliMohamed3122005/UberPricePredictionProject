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
    """Renders PlotStep objects across multiple readable figures."""

    def __init__(
        self,
        steps: list[PlotStep],
        n_cols: int = 2,
        plots_per_figure: int = 4,
    ):
        self.steps = steps
        self.n_cols = n_cols
        self.plots_per_figure = plots_per_figure

    def run(self, df: pd.DataFrame) -> None:
        
        plottable = [
            step for step in self.steps
            if step.can_plot(df)
        ]

        total_plots = len(plottable)

        if total_plots == 0:
            logger.warning(
                "Visualizer: no plottable steps, skipping"
            )
            return

        plt.style.use("dark_background")

        total_pages = math.ceil(
            total_plots / self.plots_per_figure
        )

        logger.info(
            "Visualizer: rendering %d plots across %d figures",
            total_plots,
            total_pages,
        )

        
        for page_index in range(total_pages):
            start = page_index * self.plots_per_figure
            end = start + self.plots_per_figure

            page_steps = plottable[start:end]
            page_plot_count = len(page_steps)

            n_rows = math.ceil(
                page_plot_count / self.n_cols
            )

            fig, axes = plt.subplots(
                nrows=n_rows,
                ncols=self.n_cols,
                figsize=(16, 10),
                squeeze=False,
                constrained_layout=True,
            )

            axes = axes.flatten()

            
            for index, step in enumerate(page_steps):
                current_axis = axes[index]

                step.plot(df, current_axis)

                current_axis.set_title(
                    current_axis.get_title(),
                    fontsize=15,
                    pad=12
                )
                current_axis.xaxis.label.set_size(11)
                current_axis.yaxis.label.set_size(11)

                current_axis.tick_params(
                    axis="both",
                    labelsize=9,
                )

                current_axis.grid(
                    alpha=0.2,
                )

            
            for index in range(page_plot_count, len(axes)):
                axes[index].remove()

            fig.suptitle(
                f"Uber Data Visualization "
                f"({page_index + 1}/{total_pages})",
                fontsize=20,
                fontweight="bold",
            )

        
        plt.show()

        logger.info("Visualizer: finished")