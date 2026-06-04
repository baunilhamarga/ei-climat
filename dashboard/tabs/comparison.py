import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager

class ComparisonTab(BaseTab):
    """Orchestrates comparison across different ACORN segments."""
    
    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        daily_pred = data["daily_pred"]
        filtered_pred = daily_pred[daily_pred["Acorn"].isin(selected_acorns)]
        
        daily_summary = (
            filtered_pred.groupby("Acorn")["Conso_kWh_predict"]
            .agg(["mean", "min", "max"])
            .round(3)
            .reset_index()
        )
        
        # Sort by mean consumption descending
        daily_summary = daily_summary.sort_values("mean", ascending=False)
        
        st.subheader("Statistical Summary of Forecasts by Segment")
        display_summary = daily_summary.copy()
        for col in ["mean", "min", "max"]:
            display_summary[col] = display_summary[col].map(lambda x: f"{x:.3f}")
        display_summary = display_summary.rename(columns={
            "Acorn": "ACORN Segment",
            "mean": "Mean Predicted Daily Consumption (kWh)",
            "min": "Minimum Predicted Daily Consumption (kWh)",
            "max": "Maximum Predicted Daily Consumption (kWh)"
        })
        st.markdown(f'<div class="scrollable-table-wrapper">{display_summary.to_html(index=False)}</div>', unsafe_allow_html=True)
        
        st.subheader("Comparison of Mean Daily Forecasted Consumption")
        fig_bar = px.bar(
            daily_summary,
            x="Acorn",
            y="mean",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="Mean Forecast Daily Consumption by Segment",
            labels={
                "Acorn": "ACORN Segment",
                "mean": "Mean Forecasted Daily Consumption (kWh)"
            }
        )
        fig_bar.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_bar, st.session_state.theme_mode)
        st.plotly_chart(fig_bar, width='stretch', theme=None)
