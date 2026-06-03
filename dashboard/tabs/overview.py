from pathlib import Path
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager

class OverviewTab(BaseTab):
    """Orchestrates the Overview tab rendering including best models and saved plots."""
    
    def __init__(self, root_dir: Path):
        self.figures_dir = root_dir / "outputs" / "group5" / "figures"

    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        metrics = data["metrics"]
        best = metrics[metrics["acorn"] == "ALL"].sort_values(["frequency", "rmse"]).groupby("frequency").head(1)
        
        half_rmse = best.loc[best["frequency"] == "half_hourly", "rmse"].iloc[0]
        daily_rmse = best.loc[best["frequency"] == "daily", "rmse"].iloc[0]
        half_model = best.loc[best["frequency"] == "half_hourly", "model"].iloc[0]
        daily_model = best.loc[best["frequency"] == "daily", "model"].iloc[0]

        # Render premium glassmorphic cards
        kpi_html = f"""
        <div class="kpi-container">
            {StyleManager.render_kpi_card("Half-hourly Predict Rows", f"{len(data['half_pred'])}", "Short-term horizon")}
            {StyleManager.render_kpi_card("Daily Predict Rows", f"{len(data['daily_pred'])}", "Medium-term horizon")}
            {StyleManager.render_kpi_card("Best 48h RMSE", f"{half_rmse:.4f}", f"Model: {half_model}")}
            {StyleManager.render_kpi_card("Best Daily RMSE", f"{daily_rmse:.4f}", f"Model: {daily_model}")}
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.subheader("Selected Best Performing Models")
        st.dataframe(best[["frequency", "model", "rmse", "n"]], width='stretch')

        st.subheader("Key Visual Trends")
        figure_cols = st.columns(2)
        trend_path = self.figures_dir / "01_daily_consumption_trend.png"
        rmse_path = self.figures_dir / "06_validation_rmse.png"

        if trend_path.exists():
            figure_cols[0].image(str(trend_path), caption="Historical Daily Consumption Trend")
        else:
            figure_cols[0].warning(f"Trend plot not found at: {trend_path}")

        if rmse_path.exists():
            figure_cols[1].image(str(rmse_path), caption="Validation RMSE Performance Matrix")
        else:
            figure_cols[1].warning(f"RMSE plot not found at: {rmse_path}")
