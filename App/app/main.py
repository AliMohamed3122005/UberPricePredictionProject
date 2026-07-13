"""Entry point: `streamlit run app/main.py`"""
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import APP_TITLE, DB_PATH, DEFAULT_MODEL_ARTIFACT, LOG_FORMAT  # noqa: E402
from core.eta import estimate_minutes  # noqa: E402
from core.feature_builder import TripFeatureBuilder, TripRequest  # noqa: E402
from core.predictor import JoblibPredictor, ModelArtifactNotFoundError  # noqa: E402
from core.pricing import quote_price  # noqa: E402
from db.models import PredictionRecord  # noqa: E402
from db.repository import SQLitePredictionRepository  # noqa: E402
from ui.history import render_history  # noqa: E402
from ui.map_view import render_map  # noqa: E402
from ui.model_comparison import render_model_comparison  # noqa: E402
from ui.results import render_results  # noqa: E402
from ui.sidebar import render_trip_panel  # noqa: E402
from ui.styles import inject_global_styles, render_header  # noqa: E402

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_repository() -> SQLitePredictionRepository:
    return SQLitePredictionRepository(DB_PATH)


@st.cache_resource
def get_predictor() -> JoblibPredictor | None:
    try:
        return JoblibPredictor(DEFAULT_MODEL_ARTIFACT)
    except ModelArtifactNotFoundError as exc:
        logger.warning("%s", exc)
        return None


def render_predict_tab(predictor, repository) -> None:
    col_map, col_panel = st.columns([1.7, 1], gap="large")

    with col_map:
        pickup, dropoff = render_map()

    with col_panel:
        panel_values = render_trip_panel()

        if predictor is None:
            st.warning(
                "No trained model found yet. Drop a `.joblib` artifact into "
                f"`Model/artifacts/` to enable fare predictions."
            )

        ready = pickup is not None and dropoff is not None
        get_estimate = st.button(
            "See prices",
            type="primary",
            use_container_width=True,
            disabled=not ready,
        )

        quote = None
        eta_minutes = None

        if get_estimate and ready and predictor is not None:
            trip = TripRequest(
                pickup_lat=pickup[0], pickup_lon=pickup[1],
                dropoff_lat=dropoff[0], dropoff_lon=dropoff[1],
                passenger_count=panel_values["passenger_count"],
                pickup_datetime=panel_values["pickup_datetime"],
            )
            features = TripFeatureBuilder().build(trip)
            prediction = predictor.predict(features)

            is_rush_hour = bool(features["is_rush_hour"].iloc[0])
            ride_type = panel_values["ride_type"]
            quote = quote_price(prediction, ride_type, is_rush_hour)
            eta_minutes = estimate_minutes(
                float(features["dist_travel_km"].iloc[0]), is_rush_hour,
            )

            repository.save(PredictionRecord(
                pickup_lat=trip.pickup_lat, pickup_lon=trip.pickup_lon,
                dropoff_lat=trip.dropoff_lat, dropoff_lon=trip.dropoff_lon,
                pickup_datetime=trip.pickup_datetime.isoformat(),
                passenger_count=trip.passenger_count, ride_type=ride_type.key,
                dist_travel_km=float(features["dist_travel_km"].iloc[0]),
                is_weekend=int(features["is_weekend"].iloc[0]),
                is_rush_hour=int(is_rush_hour),
                model_name=quote.model_name, predicted_fare=quote.final_fare,
            ))
        elif get_estimate and predictor is None:
            st.error("Cannot estimate fare — no trained model artifact available.")

        render_results(quote, eta_minutes)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    inject_global_styles()
    render_header()

    repository = get_repository()
    predictor = get_predictor()

    tab_predict, tab_history, tab_models = st.tabs(["Plan trip", "Trip history", "Models"])

    with tab_predict:
        render_predict_tab(predictor, repository)
    with tab_history:
        render_history(repository)
    with tab_models:
        render_model_comparison()


if __name__ == "__main__":
    main()
