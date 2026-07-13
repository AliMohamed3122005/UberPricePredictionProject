"""History tab: recent prediction requests from SQLite."""
import pandas as pd
import streamlit as st

from db.repository import BasePredictionRepository


def render_history(repository: BasePredictionRepository, limit: int = 20) -> None:
    records = repository.fetch_recent(limit=limit)

    if not records:
        st.markdown(
            """
            <div class="empty-state">
                No trips saved yet.<br>
                Request a fare estimate from the Plan trip tab.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total_fare = sum(r.predicted_fare for r in records)
    avg_fare = total_fare / len(records)
    total_dist = sum(r.dist_travel_km for r in records)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{len(records)}</div>'
            f'<div class="stat-label">Trips</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">${avg_fare:.2f}</div>'
            f'<div class="stat-label">Avg fare</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-value">{total_dist:.1f} km</div>'
            f'<div class="stat-label">Total distance</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="panel-title" style="margin-top: 1.5rem;">Recent trips</p>', unsafe_allow_html=True)

    df = pd.DataFrame([
        {
            "Time": r.created_at,
            "Pickup": f"{r.pickup_lat:.4f}, {r.pickup_lon:.4f}",
            "Dropoff": f"{r.dropoff_lat:.4f}, {r.dropoff_lon:.4f}",
            "Distance (km)": round(r.dist_travel_km, 2),
            "Passengers": r.passenger_count,
            "Ride": r.ride_type,
            "Model": r.model_name,
            "Fare": f"${r.predicted_fare:.2f}",
        }
        for r in records
    ])

    st.dataframe(df, use_container_width=True, hide_index=True)

    map_df = pd.DataFrame([
        {"lat": r.pickup_lat, "lon": r.pickup_lon} for r in records
    ] + [
        {"lat": r.dropoff_lat, "lon": r.dropoff_lon} for r in records
    ])
    st.map(map_df, size=20, color="#06C167")
