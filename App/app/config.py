"""Central configuration for the Uber Fare Prediction app.

Single source of truth for paths, constants, and app-wide settings.
Nothing in ui/, core/, or db/ should hardcode a path or magic number
that belongs here — import from config instead (Open/Closed friendly:
change a value once, it propagates everywhere).
"""
from dataclasses import dataclass
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[1]          # uber-fare-app/
MODEL_DIR = BASE_DIR / "Model" / "artifacts"
DB_PATH = BASE_DIR / "app" / "db" / "predictions.db"

DEFAULT_MODEL_ARTIFACT = MODEL_DIR / "random_forest_regression.joblib"

AVAILABLE_MODEL_ARTIFACTS: dict[str, str] = {
    "Random Forest": "random_forest_regression.joblib",
    "Gradient Boosting": "gradient_boosting_regression.joblib",
    "Decision Tree": "decision_tree_regression.joblib",
    "Linear Regression": "linear_regression.joblib",
    "Ridge Regression": "ridge_regression.joblib",
    "Lasso Regression": "lasso_regression.joblib",
}


# ──────────────────────────────────────────────
# Geography (must match Preprocessing/featuring.py)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class CityCenter:
    name: str
    lat: float
    lon: float


NYC_CENTER = CityCenter(name="New York City", lat=40.7580, lon=-73.9855)

MAP_DEFAULT_ZOOM = 12
MAP_DEFAULT_LAT = NYC_CENTER.lat
MAP_DEFAULT_LON = NYC_CENTER.lon

COORDINATE_BOUNDS = {
    "lat_min": 40.4,
    "lat_max": 41.1,
    "lon_min": -74.5,
    "lon_max": -73.3,
}


# ──────────────────────────────────────────────
# Trip / fare business rules
# ──────────────────────────────────────────────

RUSH_HOUR_WINDOWS = ((7, 9), (16, 19))
WEEKEND_DAYS = (6, 7)

MAX_PASSENGERS = 4
MIN_PASSENGERS = 1


@dataclass(frozen=True)
class RideType:
    key: str
    label: str
    description: str
    multiplier: float
    capacity: int


RIDE_TYPES: tuple[RideType, ...] = (
    RideType(
        key="economy",
        label="UberX",
        description="Affordable rides for 1-4",
        multiplier=1.0,
        capacity=4,
    ),
    RideType(
        key="comfort",
        label="Comfort",
        description="Newer cars with extra legroom",
        multiplier=1.25,
        capacity=4,
    ),
    RideType(
        key="xl",
        label="UberXL",
        description="Spacious rides for up to 6",
        multiplier=1.55,
        capacity=6,
    ),
)

RUSH_HOUR_SURGE_MULTIPLIER = 1.15

AVG_SPEED_KMH = 24.0
AVG_SPEED_KMH_RUSH_HOUR = 14.0


# ──────────────────────────────────────────────
# App metadata / theme tokens
# ──────────────────────────────────────────────

APP_TITLE = "FareCast"
APP_TAGLINE = "Plan your trip. Know your fare."
APP_CITY = "New York City"


@dataclass(frozen=True)
class Theme:
    bg: str = "#000000"
    surface: str = "#121212"
    surface_alt: str = "#1A1A1A"
    surface_hover: str = "#242424"
    text_primary: str = "#FFFFFF"
    text_secondary: str = "#AFAFAF"
    text_muted: str = "#6B6B6B"
    accent: str = "#FFFFFF"
    accent_dark: str = "#000000"
    success: str = "#06C167"
    warning: str = "#F5A623"
    danger: str = "#E54949"
    border: str = "#2B2B2B"
    map_route: str = "#FFFFFF"
    pickup: str = "#06C167"
    dropoff: str = "#FFFFFF"


THEME = Theme()

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
