"""Interactive map: click once for pickup, click again for dropoff."""
import folium
import streamlit as st
from streamlit_folium import st_folium

from config import COORDINATE_BOUNDS, MAP_DEFAULT_LAT, MAP_DEFAULT_LON, MAP_DEFAULT_ZOOM, THEME

PICKUP_KEY = "trip_pickup"
DROPOFF_KEY = "trip_dropoff"


def _in_bounds(lat: float, lon: float) -> bool:
    b = COORDINATE_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


def _handle_click(lat: float, lon: float) -> None:
    if not _in_bounds(lat, lon):
        st.warning(
            "That location is outside the NYC metro area this model was trained on. "
            "Please pick a point within the map bounds."
        )
        return

    pickup = st.session_state.get(PICKUP_KEY)
    dropoff = st.session_state.get(DROPOFF_KEY)

    if pickup is None:
        st.session_state[PICKUP_KEY] = (lat, lon)
    elif dropoff is None:
        st.session_state[DROPOFF_KEY] = (lat, lon)
    else:
        st.session_state[PICKUP_KEY] = (lat, lon)
        st.session_state[DROPOFF_KEY] = None


def _circle_marker(
    lat: float, lon: float, color: str, fill: str, tooltip: str,
) -> folium.CircleMarker:
    return folium.CircleMarker(
        location=(lat, lon),
        radius=9,
        color=color,
        weight=3,
        fill=True,
        fill_color=fill,
        fill_opacity=1.0,
        tooltip=tooltip,
    )


def _build_map() -> folium.Map:
    pickup = st.session_state.get(PICKUP_KEY)
    dropoff = st.session_state.get(DROPOFF_KEY)

    center = pickup if pickup else (MAP_DEFAULT_LAT, MAP_DEFAULT_LON)
    fmap = folium.Map(
        location=center,
        zoom_start=MAP_DEFAULT_ZOOM,
        tiles="CartoDB dark_matter",
        control_scale=False,
    )

    if pickup:
        _circle_marker(pickup[0], pickup[1], THEME.pickup, THEME.pickup, "Pickup").add_to(fmap)

    if dropoff:
        _circle_marker(
            dropoff[0], dropoff[1], THEME.text_primary, THEME.surface, "Drop-off",
        ).add_to(fmap)

    if pickup and dropoff:
        folium.PolyLine(
            locations=[pickup, dropoff],
            color=THEME.map_route,
            weight=4,
            opacity=0.9,
        ).add_to(fmap)

    return fmap


def _render_location_panel(
    pickup: tuple[float, float] | None,
    dropoff: tuple[float, float] | None,
) -> None:
    if not pickup and not dropoff:
        return

    pickup_html = ""
    if pickup:
        pickup_html = f"""
            <div class="location-row">
                <div class="location-dot pickup"></div>
                <div class="location-text">
                    <div class="location-label">Pickup</div>
                    <div class="location-coords">{pickup[0]:.5f}, {pickup[1]:.5f}</div>
                </div>
            </div>
        """

    divider_html = '<div class="location-divider"></div>' if pickup and dropoff else ""
    dropoff_html = ""
    if dropoff:
        dropoff_html = f"""
            <div class="location-row">
                <div class="location-dot dropoff"></div>
                <div class="location-text">
                    <div class="location-label">Drop-off</div>
                    <div class="location-coords">{dropoff[0]:.5f}, {dropoff[1]:.5f}</div>
                </div>
            </div>
        """

    st.markdown(
        f'<div style="margin-bottom: 0.75rem;">{pickup_html}{divider_html}{dropoff_html}</div>',
        unsafe_allow_html=True,
    )


def render_map() -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Render the map, handle clicks, and return (pickup, dropoff)."""
    st.session_state.setdefault(PICKUP_KEY, None)
    st.session_state.setdefault(DROPOFF_KEY, None)

    pickup = st.session_state[PICKUP_KEY]
    dropoff = st.session_state[DROPOFF_KEY]

    if pickup is None:
        hint = "Tap the map to set your pickup location"
    elif dropoff is None:
        hint = "Now tap to set your drop-off location"
    else:
        hint = "Route ready — tap anywhere to start a new trip"

    st.markdown(f'<p class="map-hint">{hint}</p>', unsafe_allow_html=True)
    _render_location_panel(pickup, dropoff)

    fmap = _build_map()
    map_state = st_folium(
        fmap,
        height=520,
        width=None,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="trip_map",
    )

    clicked = map_state.get("last_clicked") if map_state else None
    if clicked:
        lat, lon = clicked["lat"], clicked["lng"]
        last_processed = st.session_state.get("_last_map_click")
        if last_processed != (lat, lon):
            st.session_state["_last_map_click"] = (lat, lon)
            _handle_click(lat, lon)
            st.rerun()

    if st.button("Clear route", use_container_width=True):
        st.session_state[PICKUP_KEY] = None
        st.session_state[DROPOFF_KEY] = None
        st.session_state["_last_map_click"] = None
        st.rerun()

    return pickup, dropoff
