import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager

class EDATab(BaseTab):
    """Orchestrates the Exploratory Data Analysis (EDA) tab rendering."""
    
    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        daily_enriched = data["daily_enriched"]
        filtered_daily = daily_enriched[daily_enriched["Acorn"].isin(selected_acorns)]
        
        # Historical Daily Consumption Chart
        fig = px.line(
            filtered_daily,
            x="Date",
            y="Conso_kWh",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="Historical Daily Consumption",
        )
        st.plotly_chart(fig, width='stretch')

        # Half-Hourly and Weekly Profiles Side-by-Side
        col1, col2 = st.columns(2)
        half_profile = data["half_profile"][data["half_profile"]["Acorn"].isin(selected_acorns)].copy()
        half_profile["time_of_day"] = half_profile["half_hour_slot"] / 2
        
        fig_half = px.line(
            half_profile,
            x="time_of_day",
            y="mean_conso_moy",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="Typical Half-Hourly Profile",
        )
        col1.plotly_chart(fig_half, width='stretch')

        weekly = data["weekly"][data["weekly"]["Acorn"].isin(selected_acorns)]
        fig_weekly = px.line(
            weekly,
            x="weekday",
            y="mean_conso_kwh",
            color="Acorn",
            markers=True,
            color_discrete_map=StyleManager.PALETTE,
            title="Weekly Pattern",
        )
        col2.plotly_chart(fig_weekly, width='stretch')

        # Temperature Scatter and Autocorrelation Plots Side-by-Side
        col3, col4 = st.columns(2)
        fig_temp = px.scatter(
            filtered_daily,
            x="temperatureMean",
            y="Conso_kWh",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="Temperature vs Daily Consumption",
        )
        col3.plotly_chart(fig_temp, width='stretch')

        autocorr = data["autocorr"][data["autocorr"]["Acorn"].isin(selected_acorns)]
        fig_autocorr = px.line(
            autocorr,
            x="lag_days",
            y="autocorrelation",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="Daily Autocorrelation",
        )
        col4.plotly_chart(fig_autocorr, width='stretch')
