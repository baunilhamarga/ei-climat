import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager

class ForecastsTab(BaseTab):
    """Orchestrates rendering the model prediction forecasts."""
    
    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        half_pred = data["half_pred"][data["half_pred"]["Acorn"].isin(selected_acorns)]
        daily_pred = data["daily_pred"][data["daily_pred"]["Acorn"].isin(selected_acorns)]
        
        st.subheader("Short-Term Forecast (48 Hours)")
        fig_half = px.line(
            half_pred,
            x="DateTime",
            y="Conso_moy_predict",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="48-Hour Half-Hourly Forecast",
            labels={
                "DateTime": "Date & Time",
                "Conso_moy_predict": "Predicted Average Consumption (kW)",
                "Acorn": "ACORN Segment"
            }
        )
        st.plotly_chart(fig_half, width='stretch')

        st.subheader("Medium-Term Forecast (1 Month)")
        fig_daily = px.line(
            daily_pred,
            x="Date",
            y="Conso_kWh_predict",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="One-Month Daily Forecast",
            labels={
                "Date": "Date",
                "Conso_kWh_predict": "Predicted Daily Consumption (kWh)",
                "Acorn": "ACORN Segment"
            }
        )
        st.plotly_chart(fig_daily, width='stretch')
