"""Trip controls panel: passengers, ride type, pickup date/time."""
from datetime import date, datetime, time

import streamlit as st

from config import MAX_PASSENGERS, MIN_PASSENGERS, RIDE_TYPES, RideType


def render_trip_panel() -> dict:
    """Render trip controls in the main panel and return collected values."""
    st.markdown('<p class="panel-title">Passengers</p>', unsafe_allow_html=True)
    passenger_count = st.number_input(
        "Passengers",
        min_value=MIN_PASSENGERS,
        max_value=MAX_PASSENGERS,
        value=1,
        step=1,
        label_visibility="collapsed",
    )

    st.markdown('<p class="panel-title">Choose a ride</p>', unsafe_allow_html=True)
    ride_labels = [
        f"{rt.label}  ·  {rt.description}  ·  {rt.capacity} seats"
        for rt in RIDE_TYPES
    ]
    chosen_idx = st.radio(
        "Ride type",
        options=range(len(RIDE_TYPES)),
        format_func=lambda i: ride_labels[i],
        label_visibility="collapsed",
    )
    ride_type: RideType = RIDE_TYPES[chosen_idx]

    if passenger_count > ride_type.capacity:
        st.warning(
            f"{ride_type.label} fits up to {ride_type.capacity} passengers. "
            "Choose a larger ride or reduce the passenger count."
        )

    st.markdown('<p class="panel-title">Pickup time</p>', unsafe_allow_html=True)
    col_date, col_time = st.columns(2)
    with col_date:
        trip_date = st.date_input("Date", value=date.today(), label_visibility="collapsed")
    with col_time:
        trip_time = st.time_input(
            "Time", value=time(hour=datetime.now().hour), label_visibility="collapsed",
        )

    return {
        "passenger_count": passenger_count,
        "ride_type": ride_type,
        "pickup_datetime": datetime.combine(trip_date, trip_time),
    }
