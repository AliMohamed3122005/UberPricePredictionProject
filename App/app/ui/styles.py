"""Global Uber-inspired styling for the Streamlit app."""
import streamlit as st

from config import THEME


def inject_global_styles() -> None:
    t = THEME
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        .stApp {{
            background-color: {t.bg};
            color: {t.text_primary};
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        #MainMenu, footer, .stDeployButton {{
            visibility: hidden;
        }}

        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {t.text_primary};
        }}

        .stCaption, small {{
            color: {t.text_secondary} !important;
        }}

        div[data-testid="stTabs"] button {{
            background: transparent;
            color: {t.text_secondary};
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0;
            font-weight: 500;
            padding: 0.65rem 1.25rem;
        }}

        div[data-testid="stTabs"] button[aria-selected="true"] {{
            color: {t.text_primary};
            border-bottom-color: {t.text_primary};
            background: transparent;
        }}

        div[data-testid="stTabs"] button:hover {{
            color: {t.text_primary};
            background: {t.surface_alt};
        }}

        .stButton > button {{
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.01em;
            transition: all 0.15s ease;
            border: 1px solid {t.border};
        }}

        .stButton > button[kind="primary"] {{
            background-color: {t.accent};
            color: {t.accent_dark};
            border: none;
        }}

        .stButton > button[kind="primary"]:hover {{
            background-color: #E8E8E8;
            color: {t.accent_dark};
        }}

        .stButton > button[kind="secondary"] {{
            background-color: {t.surface_alt};
            color: {t.text_primary};
            border: 1px solid {t.border};
        }}

        .stButton > button:disabled {{
            background-color: {t.surface_alt} !important;
            color: {t.text_muted} !important;
            border: 1px solid {t.border} !important;
        }}

        div[data-baseweb="input"], div[data-baseweb="select"] {{
            background-color: {t.surface_alt};
            border-color: {t.border};
        }}

        .stNumberInput input, .stDateInput input, .stTimeInput input {{
            background-color: {t.surface_alt} !important;
            color: {t.text_primary} !important;
            border-color: {t.border} !important;
        }}

        div[data-testid="stMetric"] {{
            background-color: {t.surface};
            border: 1px solid {t.border};
            border-radius: 12px;
            padding: 1rem;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {t.border};
            border-radius: 12px;
            overflow: hidden;
        }}

        .app-header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            padding: 0.25rem 0 1.5rem 0;
            border-bottom: 1px solid {t.border};
            margin-bottom: 1.25rem;
        }}

        .app-logo {{
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: {t.text_primary};
        }}

        .app-city {{
            font-size: 0.875rem;
            font-weight: 500;
            color: {t.text_secondary};
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .panel-title {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: {t.text_muted};
            margin: 0 0 0.75rem 0;
        }}

        .location-row {{
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.65rem 0;
        }}

        .location-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-top: 0.35rem;
            flex-shrink: 0;
        }}

        .location-dot.pickup {{
            background: {t.pickup};
        }}

        .location-dot.dropoff {{
            background: {t.dropoff};
            border-radius: 2px;
        }}

        .location-text {{
            flex: 1;
        }}

        .location-label {{
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {t.text_muted};
            margin-bottom: 0.15rem;
        }}

        .location-coords {{
            font-size: 0.85rem;
            color: {t.text_secondary};
            font-variant-numeric: tabular-nums;
        }}

        .location-divider {{
            border-left: 2px dashed {t.border};
            margin-left: 4px;
            height: 12px;
        }}

        .ride-card {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.9rem 1rem;
            border: 1px solid {t.border};
            border-radius: 10px;
            background: {t.surface};
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: border-color 0.15s ease, background 0.15s ease;
        }}

        .ride-card.selected {{
            border-color: {t.text_primary};
            background: {t.surface_hover};
        }}

        .ride-card:hover {{
            border-color: {t.text_secondary};
        }}

        .ride-name {{
            font-size: 1rem;
            font-weight: 600;
            color: {t.text_primary};
        }}

        .ride-desc {{
            font-size: 0.8rem;
            color: {t.text_secondary};
            margin-top: 0.15rem;
        }}

        .ride-capacity {{
            font-size: 0.75rem;
            color: {t.text_muted};
            white-space: nowrap;
        }}

        .fare-panel {{
            background: {t.surface};
            border: 1px solid {t.border};
            border-radius: 14px;
            padding: 1.25rem;
            margin-top: 1rem;
        }}

        .fare-amount {{
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            color: {t.text_primary};
            line-height: 1.1;
        }}

        .fare-meta {{
            font-size: 0.95rem;
            color: {t.text_secondary};
            margin-top: 0.35rem;
        }}

        .fare-range {{
            font-size: 0.85rem;
            color: {t.text_muted};
            margin-top: 0.5rem;
        }}

        .surge-banner {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.85rem;
            padding: 0.65rem 0.85rem;
            background: rgba(245, 166, 35, 0.12);
            border: 1px solid rgba(245, 166, 35, 0.35);
            border-radius: 8px;
            font-size: 0.85rem;
            color: {t.warning};
            font-weight: 500;
        }}

        .surge-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: {t.warning};
            flex-shrink: 0;
        }}

        .empty-state {{
            text-align: center;
            padding: 2rem 1rem;
            color: {t.text_muted};
            font-size: 0.9rem;
            border: 1px dashed {t.border};
            border-radius: 12px;
            margin-top: 1rem;
        }}

        .map-hint {{
            font-size: 0.85rem;
            color: {t.text_secondary};
            padding: 0.5rem 0 0.75rem 0;
        }}

        .stat-card {{
            background: {t.surface};
            border: 1px solid {t.border};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {t.text_primary};
        }}

        .stat-label {{
            font-size: 0.75rem;
            color: {t.text_muted};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.25rem;
        }}

        .best-model-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            background: rgba(6, 193, 103, 0.12);
            border: 1px solid rgba(6, 193, 103, 0.35);
            border-radius: 8px;
            color: {t.success};
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 1rem;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: {t.border};
            background: {t.surface};
            border-radius: 12px;
        }}

        div[data-testid="stRadio"] > div {{
            gap: 0.5rem;
        }}

        div[data-testid="stRadio"] label {{
            background: {t.surface};
            border: 1px solid {t.border};
            border-radius: 10px;
            padding: 0.85rem 1rem !important;
            font-weight: 500;
            transition: border-color 0.15s ease, background 0.15s ease;
        }}

        div[data-testid="stRadio"] label:hover {{
            border-color: {t.text_secondary};
            background: {t.surface_hover};
        }}

        div[data-testid="stRadio"] label[data-checked="true"],
        div[data-testid="stRadio"] label:has(input:checked) {{
            border-color: {t.text_primary};
            background: {t.surface_hover};
        }}

        div[data-testid="stRadio"] label > div:first-child {{
            display: none;
        }}

        .stAlert {{
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    from config import APP_CITY, APP_TITLE

    st.markdown(
        f"""
        <div class="app-header">
            <span class="app-logo">{APP_TITLE}</span>
            <span class="app-city">{APP_CITY}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
