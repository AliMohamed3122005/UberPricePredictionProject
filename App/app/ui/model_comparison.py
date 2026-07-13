"""Model comparison tab: MAE / RMSE / R2 for every trained artifact."""
import logging

import joblib
import pandas as pd
import streamlit as st

from config import AVAILABLE_MODEL_ARTIFACTS, MODEL_DIR

logger = logging.getLogger(__name__)


def render_model_comparison() -> None:
    st.markdown(
        '<p class="panel-title">Model performance on test set</p>',
        unsafe_allow_html=True,
    )

    rows = []
    for display_name, filename in AVAILABLE_MODEL_ARTIFACTS.items():
        path = MODEL_DIR / filename
        if not path.exists():
            rows.append({"Model": display_name, "Status": "Not trained"})
            continue
        try:
            artifact = joblib.load(path)
        except Exception as exc:
            logger.warning("Model comparison: failed to load %s (%s)", path, exc)
            rows.append({"Model": display_name, "Status": "Load failed"})
            continue

        test = artifact.get("metrics", {}).get("test", {})
        rows.append({
            "Model": display_name,
            "Status": "Ready",
            "MAE": round(test.get("mae", float("nan")), 3),
            "RMSE": round(test.get("rmse", float("nan")), 3),
            "R2": round(test.get("r2", float("nan")), 3),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    ready = df[df.get("Status") == "Ready"] if "Status" in df else pd.DataFrame()
    if not ready.empty:
        best = ready.loc[ready["RMSE"].idxmin(), "Model"]
        best_rmse = ready.loc[ready["RMSE"].idxmin(), "RMSE"]
        st.markdown(
            f'<div class="best-model-badge">Best model: {best} (RMSE {best_rmse})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No trained artifacts found in Model/artifacts/ yet.")
