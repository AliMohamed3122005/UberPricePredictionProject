import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# DIP: At module level, import only ABSTRACTIONS.
# Concrete classes are imported in __main__ (the composition root).
from featuring import BaseFeatureEngineer, FeatureStep
from visualization import BaseVisualizer, PlotStep

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Abstract Base Class for Cleaning Steps
# ──────────────────────────────────────────────

class CleaningStep(ABC):
    """Abstract base class for all cleaning steps."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the cleaning step and return the modified DataFrame."""
        ...


# ──────────────────────────────────────────────
# Concrete Cleaning Steps
# ──────────────────────────────────────────────

class ColumnDropper(CleaningStep):
    """Drops specified columns from the DataFrame."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in self.columns if c in df.columns]
        df = df.drop(columns=existing)
        logger.info("ColumnDropper: dropped columns %s", existing)
        return df


class PositiveFilter(CleaningStep):
    """Keeps only rows where a column's value is strictly positive."""

    def __init__(self, column: str):
        self.column = column

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df[self.column] > 0]
        logger.info("PositiveFilter on '%s': removed %d rows", self.column, before - len(df))
        return df


class RangeFilter(CleaningStep):
    """Filters rows where a column's value falls within [min_val, max_val]."""

    def __init__(self, column: str, min_val: float, max_val: float):
        self.column = column
        self.min_val = min_val
        self.max_val = max_val

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[(df[self.column] >= self.min_val) & (df[self.column] <= self.max_val)]
        logger.info(
            "RangeFilter on '%s' [%s, %s]: removed %d rows",
            self.column, self.min_val, self.max_val, before - len(df),
        )
        return df


class NonZeroFilter(CleaningStep):
    """Removes rows where any of the specified columns have a zero value."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        for col in self.columns:
            df = df[df[col] != 0]
        logger.info("NonZeroFilter on %s: removed %d rows", self.columns, before - len(df))
        return df


class NullHandler(CleaningStep):
    """Fills null values in specified columns with their column mean.

    WARNING: Do NOT use this for geographical coordinates — mean imputation
    creates fake locations. Use DropNullsStep instead for coordinate columns.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        total_nulls = df[self.columns].isnull().sum().sum()
        if total_nulls == 0:
            logger.info("NullHandler: no null values found in %s", self.columns)
        else:
            logger.info("NullHandler: filling %d null values in %s", total_nulls, self.columns)
            for col in self.columns:
                df[col] = df[col].fillna(value=df[col].mean())
        return df


class DropNullsStep(CleaningStep):
    """Drops rows that contain null values in the specified columns.

    Use this instead of NullHandler for essential columns (e.g. coordinates)
    where mean imputation would produce meaningless data.
    """

    def __init__(self, columns: list[str]):
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=self.columns)
        logger.info("DropNullsStep on %s: dropped %d rows", self.columns, before - len(df))
        return df


class DuplicateDropper(CleaningStep):
    """Drops duplicate rows from the DataFrame."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        logger.info("DuplicateDropper: dropped %d duplicate rows", before - len(df))
        return df


# ──────────────────────────────────────────────
# BaseCleaner ABC + DataCleaner (DIP)
# ──────────────────────────────────────────────

class BaseCleaner(ABC):
    """Abstraction for data cleaning.

    High-level modules (e.g. PreprocessingPipeline) depend on this
    interface — not on any concrete implementation (DIP).
    """

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame and return the result."""
        ...


class DataCleaner(BaseCleaner):
    """Applies an ordered sequence of CleaningStep objects to a DataFrame."""

    def __init__(self, steps: list[CleaningStep]):
        self.steps = steps

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("DataCleaner: starting %d cleaning steps", len(self.steps))
        for step in self.steps:
            df = step.apply(df)
        logger.info("DataCleaner: finished. Rows remaining: %d", len(df))
        return df


# ──────────────────────────────────────────────
# Data I/O — segregated interfaces (ISP)
# ──────────────────────────────────────────────

class DataReader(ABC):
    """Interface for reading data. Consumers that only read
    depend on this — not on any write logic."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return a DataFrame."""
        ...


class DataWriter(ABC):
    """Interface for writing data. Consumers that only write
    depend on this — not on any read logic."""

    @abstractmethod
    def save(self, df: pd.DataFrame) -> None:
        """Persist a DataFrame."""
        ...


class CsvDataHandler(DataReader, DataWriter):
    """Concrete handler that reads/writes CSV files.

    Implements both DataReader and DataWriter, so it can be injected
    wherever either (or both) interface is needed.
    """

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path

    def load(self) -> pd.DataFrame:
        logger.info("Loading data from %s", self.input_path)
        return pd.read_csv(self.input_path)

    def save(self, df: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info("Processed data saved to %s", self.output_path)


# ──────────────────────────────────────────────
# Explorer — segregated interfaces (ISP)
# ──────────────────────────────────────────────

class InitialExplorer(ABC):
    """Interface for initial (pre-cleaning) data exploration.
    Components that only need initial EDA depend on this."""

    @abstractmethod
    def explore_initial(self, df: pd.DataFrame) -> None: ...


class PostCleaningExplorer(ABC):
    """Interface for post-cleaning data exploration.
    Components that only need post-clean EDA depend on this."""

    @abstractmethod
    def explore_post_cleaning(self, df: pd.DataFrame) -> None: ...


class DataExplorer(InitialExplorer, PostCleaningExplorer):
    """Concrete explorer implementing both EDA interfaces.

    Parameters
    ----------
    describe_columns : list[str] | None
        Columns to run .describe() on individually. If None, describe() is
        called on the entire DataFrame.
    """

    def __init__(self, describe_columns: list[str] | None = None):
        self.describe_columns = describe_columns

    def explore_initial(self, df: pd.DataFrame) -> None:
        rows, cols = df.shape
        logger.info("Initial data shape: %d rows × %d columns", rows, cols)
        logger.info("First 5 rows:\n%s", df.iloc[:, :5])
        logger.info("Column types:\n%s", df.dtypes)
        logger.info("Initial null values per column:\n%s", df.isnull().sum())
        logger.info("Duplicate rows: %d", df.duplicated().sum())

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
        logger.info("Numeric columns (%d): %s", len(numeric_cols), numeric_cols)
        logger.info("Non-numeric columns (%d): %s", len(non_numeric_cols), non_numeric_cols)

        if self.describe_columns:
            for col in self.describe_columns:
                if col in df.columns:
                    logger.info("Describe '%s':\n%s", col, df[col].describe())
        else:
            logger.info("Describe all:\n%s", df.describe())

        coord_cols = ['pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude', 'dropoff_latitude']
        existing_coords = [c for c in coord_cols if c in df.columns]
        if existing_coords:
            zero_rows = df[(df[existing_coords] == 0).any(axis=1)]
            logger.info("Rows with zero coordinates: %d", len(zero_rows))

    def explore_post_cleaning(self, df: pd.DataFrame) -> None:
        logger.info("Post-cleaning shape: %d rows × %d columns", *df.shape)
        logger.info("Null values after cleaning:\n%s", df.isnull().sum())
        logger.info("Duplicate rows: %d", df.duplicated().sum())
        logger.info("Describe:\n%s", df.describe())

        # Simple feature-importance hypothesis: correlation magnitude with fare_amount.
        if "fare_amount" in df.columns:
            numeric_df = df.select_dtypes(include="number")
            corr = numeric_df.corr(numeric_only=True)["fare_amount"].sort_values(key=lambda s: s.abs(), ascending=False)
            logger.info("Correlation with fare_amount (descending |corr|):\n%s", corr)

        coord_cols = ['pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude', 'dropoff_latitude']
        existing_coords = [c for c in coord_cols if c in df.columns]
        if existing_coords:
            logger.info("Zero coordinates:\n%s", (df[existing_coords] == 0).sum())


# ──────────────────────────────────────────────
# FeatureEngineer — extends BaseFeatureEngineer (DIP)
# ──────────────────────────────────────────────

class FeatureEngineer(BaseFeatureEngineer):
    """Applies an ordered sequence of FeatureStep objects to a DataFrame.

    To add a new feature, create a FeatureStep subclass and append it
    to the list — no existing code needs to change (Open/Closed Principle).
    """

    def __init__(self, steps: list[FeatureStep]):
        self.steps = steps

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("FeatureEngineer: starting %d feature steps", len(self.steps))
        for step in self.steps:
            df = step.apply(df)
        logger.info("FeatureEngineer: finished")
        return df


# ──────────────────────────────────────────────
# PreprocessingPipeline — orchestration
# ──────────────────────────────────────────────

class PreprocessingPipeline:
    """Orchestrates the data processing steps.

    Every dependency is typed against an ABSTRACTION, never a
    concrete class (Dependency Inversion Principle).

    Parameters
    ----------
    reader : DataReader
        Loads the raw data.
    writer : DataWriter
        Persists the processed data.
    cleaner : BaseCleaner
        Pre-feature cleaning steps.
    engineer : BaseFeatureEngineer
        Feature engineering steps.
    post_feature_cleaner : BaseCleaner
        Post-feature cleaning steps (e.g. distance range filter).
    initial_explorer : InitialExplorer
        Pre-cleaning EDA.
    post_cleaning_explorer : PostCleaningExplorer
        Post-cleaning EDA.
    visualizer : BaseVisualizer
        Visualization steps.
    """

    def __init__(
        self,
        reader: DataReader,
        writer: DataWriter,
        cleaner: BaseCleaner,
        engineer: BaseFeatureEngineer,
        post_feature_cleaner: BaseCleaner,
        initial_explorer: InitialExplorer,
        post_cleaning_explorer: PostCleaningExplorer,
        visualizer: BaseVisualizer,
    ):
        self.reader = reader
        self.writer = writer
        self.cleaner = cleaner
        self.engineer = engineer
        self.post_feature_cleaner = post_feature_cleaner
        self.initial_explorer = initial_explorer
        self.post_cleaning_explorer = post_cleaning_explorer
        self.visualizer = visualizer

    def run(self):
        # 1. Load Data
        df = self.reader.load()

        # 2. Explore Initial Data
        self.initial_explorer.explore_initial(df)

        # 3. Pre-feature Cleaning
        df = self.cleaner.clean(df)

        # 4. Explore Post-Cleaning Data
        self.post_cleaning_explorer.explore_post_cleaning(df)

        # 5. Feature Engineering
        df = self.engineer.engineer(df)

        # 6. Post-feature Cleaning (e.g. distance bounds)
        df = self.post_feature_cleaner.clean(df)

        # 7. Visualization
        self.visualizer.run(df)

        # 8. Save Data
        self.writer.save(df)


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Configure logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # DIP: Concrete classes are imported ONLY in the composition root (__main__).
    # The pipeline itself never sees them — it depends only on abstractions.
    from featuring import (
        DatetimeFeatures, DistanceFeature, WeekendFeature, RushHourFeature,
        BearingFeature, DistanceFromCityCenterFeature,
    )
    from visualization import (
        Visualizer,
        FareDistributionPlot,
        FareBoxplotByPassengerCount,
        FareBoxplotByWeekend,
        FareBoxplotByRushHour,
        DistanceDistributionPlot,
        FareVsDistancePlot,
        TripsByHourPlot,
        AvgFareByHourPlot,
        PassengerCountPlot,
        NumericFeatureHistograms,
        CorrelationHeatmapPlot,
    )

    BASE_DIR = Path(__file__).resolve().parents[1]
    input_file = BASE_DIR / "Preprocessing" / "uber.csv"
    output_file = BASE_DIR / "Data" / "processed" / "uber_processed.csv"

    # --- Wire up components (composition root) ---

    data_handler = CsvDataHandler(input_file, output_file)

    pre_feature_steps = [
        ColumnDropper(columns=["Unnamed: 0", "key"]),
        DuplicateDropper(),                         # Fix 2: remove duplicates
        PositiveFilter(column="fare_amount"),
        RangeFilter(column="passenger_count", min_val=1, max_val=4),
        RangeFilter(column="pickup_longitude", min_val=-180, max_val=180),
        RangeFilter(column="dropoff_longitude", min_val=-180, max_val=180),
        RangeFilter(column="pickup_latitude", min_val=-90, max_val=90),
        RangeFilter(column="dropoff_latitude", min_val=-90, max_val=90),
        NonZeroFilter(columns=[
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude",
        ]),
        DropNullsStep(columns=[                     # Fix 1: drop nulls, don't impute
            "dropoff_longitude", "dropoff_latitude",
        ]),
    ]
    cleaner = DataCleaner(steps=pre_feature_steps)

    feature_steps = [
        DatetimeFeatures(),
        DistanceFeature(),
        BearingFeature(),                           # Fix 4: direction/bearing
        DistanceFromCityCenterFeature(),             # Fix 4: distance from NYC center
        WeekendFeature(),
        RushHourFeature(),
    ]
    engineer = FeatureEngineer(steps=feature_steps)

    post_feature_steps = [
        RangeFilter(column="dist_travel_km", min_val=0.01, max_val=150),
    ]
    post_feature_cleaner = DataCleaner(steps=post_feature_steps)

    explorer = DataExplorer(describe_columns=[
        "fare_amount", "passenger_count",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ])

    plot_steps = [
        FareDistributionPlot(),
        FareBoxplotByPassengerCount(),
        FareBoxplotByWeekend(),
        FareBoxplotByRushHour(),
        PassengerCountPlot(),
        DistanceDistributionPlot(),
        FareVsDistancePlot(),
        TripsByHourPlot(),
        AvgFareByHourPlot(),
        CorrelationHeatmapPlot(),
    ]
    visualizer = Visualizer(
    steps=plot_steps,
    n_cols=2,
    plots_per_figure=4,
)

    # Pipeline depends ONLY on abstractions — concrete wiring happens here.
    pipeline = PreprocessingPipeline(
        reader=data_handler,                # DataReader (ABC)
        writer=data_handler,                # DataWriter (ABC)
        cleaner=cleaner,                    # BaseCleaner (ABC)
        engineer=engineer,                  # BaseFeatureEngineer (ABC)
        post_feature_cleaner=post_feature_cleaner,  # BaseCleaner (ABC)
        initial_explorer=explorer,          # InitialExplorer (ABC)
        post_cleaning_explorer=explorer,    # PostCleaningExplorer (ABC)
        visualizer=visualizer,              # BaseVisualizer (ABC)
    )
    pipeline.run()