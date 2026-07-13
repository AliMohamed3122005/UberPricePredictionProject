"""Results panel: predicted fare, confidence range, ETA, surge indicator."""
import streamlit as st

from core.pricing import PriceQuote


def render_results(quote: PriceQuote | None, eta_minutes: int | None) -> None:
    if quote is None:
        st.markdown(
            """
            <div class="empty-state">
                Select pickup and drop-off on the map,<br>
                then request a fare estimate.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    eta_line = f" · {eta_minutes} min" if eta_minutes is not None else ""
    surge_html = ""
    if quote.is_rush_hour:
        surge_html = """
            <div class="surge-banner">
                <span class="surge-dot"></span>
                Higher demand — fares are elevated
            </div>
        """

    st.markdown(
        f"""
        <div class="fare-panel">
            <div class="fare-amount">${quote.final_fare:.2f}</div>
            <div class="fare-meta">{quote.ride_type.label}{eta_line}</div>
            <div class="fare-range">
                Estimated range ${quote.low_estimate:.2f} – ${quote.high_estimate:.2f}
            </div>
            {surge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Price breakdown"):
        st.write(f"Base model estimate: ${quote.base_fare:.2f}")
        st.write(f"Ride type multiplier: x{quote.ride_type.multiplier:.2f}")
        if quote.is_rush_hour:
            st.write("Rush-hour surge applied")
        st.write(f"Model: {quote.model_name}")
