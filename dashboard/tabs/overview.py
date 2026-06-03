from pathlib import Path
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager

class OverviewTab(BaseTab):
    """Orchestrates the Overview tab rendering including best models and interactive plots."""
    
    def __init__(self, root_dir: Path | None = None):
        pass

    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        metrics = data["metrics"]
        best = metrics[metrics["acorn"] == "ALL"].sort_values(["frequency", "rmse"]).groupby("frequency").head(1)
        
        half_rmse = best.loc[best["frequency"] == "half_hourly", "rmse"].iloc[0]
        daily_rmse = best.loc[best["frequency"] == "daily", "rmse"].iloc[0]
        half_model = best.loc[best["frequency"] == "half_hourly", "model"].iloc[0]
        daily_model = best.loc[best["frequency"] == "daily", "model"].iloc[0]

        # Get human-readable names for KPIs
        nice_half_model = StyleManager.MODEL_MAP.get(half_model, half_model)
        nice_daily_model = StyleManager.MODEL_MAP.get(daily_model, daily_model)
        nice_half_freq = StyleManager.FREQUENCY_MAP.get("half_hourly", "Half-Hourly")
        nice_daily_freq = StyleManager.FREQUENCY_MAP.get("daily", "Daily")

        # Render premium glassmorphic cards
        kpi_html = f"""
        <div class="kpi-container">
            {StyleManager.render_kpi_card(f"{nice_half_freq} Predict Rows", f"{len(data['half_pred'])}", "Short-term horizon")}
            {StyleManager.render_kpi_card(f"{nice_daily_freq} Predict Rows", f"{len(data['daily_pred'])}", "Medium-term horizon")}
            {StyleManager.render_kpi_card("Best 48h RMSE", f"{half_rmse:.4f}", f"Model: {nice_half_model}")}
            {StyleManager.render_kpi_card("Best Daily RMSE", f"{daily_rmse:.4f}", f"Model: {nice_daily_model}")}
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.subheader("Selected Best Performing Models")
        display_best = best[["frequency", "model", "rmse", "n"]].copy()
        display_best["frequency"] = display_best["frequency"].map(StyleManager.FREQUENCY_MAP)
        display_best["model"] = display_best["model"].map(StyleManager.MODEL_MAP)
        display_best["rmse"] = display_best["rmse"].map(lambda x: f"{x:.4f}")
        display_best = display_best.rename(columns={
            "frequency": "Forecast Frequency",
            "model": "Best Model",
            "rmse": "Validation RMSE",
            "n": "Observations (N)"
        })
        st.markdown(f'<div class="scrollable-table-wrapper">{display_best.to_html(index=False)}</div>', unsafe_allow_html=True)

        st.subheader("Key Visual Trends")
        figure_cols = st.columns(2)

        # 14-day rolling mean daily consumption trend (calculated dynamically)
        daily_enriched = data["daily_enriched"]
        df_list = []
        for acorn, group in daily_enriched[daily_enriched["Acorn"].isin(selected_acorns)].groupby("Acorn"):
            group_sorted = group.sort_values("Date").copy()
            group_sorted["rolling_mean"] = group_sorted["Conso_kWh"].rolling(14, min_periods=1).mean()
            df_list.append(group_sorted)

        if df_list:
            df_rolling = pd.concat(df_list)
            fig_trend = px.line(
                df_rolling,
                x="Date",
                y="rolling_mean",
                color="Acorn",
                color_discrete_map=StyleManager.PALETTE,
                title="Daily electricity consumption, 14-day rolling mean",
                labels={
                    "rolling_mean": "14-day Rolling Mean Consumption (kWh)",
                    "Date": "Date",
                    "Acorn": "ACORN Segment"
                }
            )
            StyleManager.style_plotly_chart(fig_trend, st.session_state.theme_mode)
            figure_cols[0].plotly_chart(fig_trend, width='stretch')
        else:
            figure_cols[0].warning("No data selected to display trend.")

        # Interactive Validation RMSE Comparison
        metric_subset = metrics[metrics["acorn"] == "ALL"].copy()
        metric_subset["frequency"] = metric_subset["frequency"].map(StyleManager.FREQUENCY_MAP)
        metric_subset["model"] = metric_subset["model"].map(StyleManager.MODEL_MAP)
        fig_rmse = px.bar(
            metric_subset,
            x="model",
            y="rmse",
            color="frequency",
            barmode="group",
            title="Validation RMSE by model",
            labels={
                "model": "Model",
                "rmse": "Root Mean Squared Error (RMSE)",
                "frequency": "Forecast Frequency"
            }
        )
        StyleManager.style_plotly_chart(fig_rmse, st.session_state.theme_mode)
        figure_cols[1].plotly_chart(fig_rmse, width='stretch')
