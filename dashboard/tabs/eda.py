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
            labels={
                "Conso_kWh": "Daily Consumption (kWh)",
                "Date": "Date",
                "Acorn": "ACORN Segment"
            }
        )
        StyleManager.style_plotly_chart(fig, st.session_state.theme_mode)
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
            labels={
                "time_of_day": "Hour of Day",
                "mean_conso_moy": "Average Consumption (kW)",
                "Acorn": "ACORN Segment"
            }
        )
        StyleManager.style_plotly_chart(fig_half, st.session_state.theme_mode)
        col1.plotly_chart(fig_half, width='stretch')

        # Map weekday numbers (0=Monday, ..., 6=Sunday) to name strings
        day_names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }
        weekly = data["weekly"][data["weekly"]["Acorn"].isin(selected_acorns)].copy()
        weekly["weekday_name"] = weekly["weekday"].map(day_names)

        fig_weekly = px.line(
            weekly,
            x="weekday_name",
            y="mean_conso_kwh",
            color="Acorn",
            markers=True,
            color_discrete_map=StyleManager.PALETTE,
            title="Weekly Pattern",
            labels={
                "weekday_name": "Day of Week",
                "mean_conso_kwh": "Average Daily Consumption (kWh)",
                "Acorn": "ACORN Segment"
            }
        )
        fig_weekly.update_xaxes(
            categoryorder="array",
            categoryarray=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        StyleManager.style_plotly_chart(fig_weekly, st.session_state.theme_mode)
        col2.plotly_chart(fig_weekly, width='stretch')

        # Temperature Scatter and Correlation Matrix Side-by-Side
        col3, col4 = st.columns(2)
        fig_temp = px.scatter(
            filtered_daily,
            x="temperatureMean",
            y="Conso_kWh",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="Temperature vs Daily Consumption",
            labels={
                "temperatureMean": "Mean Temperature (°C)",
                "Conso_kWh": "Daily Consumption (kWh)",
                "Acorn": "ACORN Segment"
            }
        )
        StyleManager.style_plotly_chart(fig_temp, st.session_state.theme_mode)
        col3.plotly_chart(fig_temp, width='stretch')

        # Correlation Matrix
        corr_cols = [
            "Conso_kWh",
            "temperatureMean",
            "temperatureMin",
            "temperatureMax",
            "is_weekend",
            "is_holiday"
        ]
        available_cols = [col for col in corr_cols if col in filtered_daily.columns]
        corr_data = filtered_daily[available_cols].copy()
        for col in ["is_weekend", "is_holiday"]:
            if col in corr_data.columns:
                corr_data[col] = corr_data[col].astype(int)

        corr_df = corr_data.corr()

        column_mapping = {
            "Conso_kWh": "Consumption",
            "temperatureMean": "Mean Temp",
            "temperatureMin": "Min Temp",
            "temperatureMax": "Max Temp",
            "is_weekend": "Weekend",
            "is_holiday": "Holiday"
        }
        corr_df.index = [column_mapping.get(idx, idx) for idx in corr_df.index]
        corr_df.columns = [column_mapping.get(col, col) for col in corr_df.columns]

        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1.0,
            zmax=1.0,
            title="Correlation Matrix of Daily Variables",
            labels=dict(color="Correlation")
        )
        StyleManager.style_plotly_chart(fig_corr, st.session_state.theme_mode)
        col4.plotly_chart(fig_corr, width='stretch')

        # Daily Autocorrelation (Full Width)
        autocorr = data["autocorr"][data["autocorr"]["Acorn"].isin(selected_acorns)]
        fig_autocorr = px.line(
            autocorr,
            x="lag_days",
            y="autocorrelation",
            color="Acorn",
            color_discrete_map=StyleManager.PALETTE,
            title="Daily Autocorrelation",
            labels={
                "lag_days": "Lag (Days)",
                "autocorrelation": "Autocorrelation Coefficient",
                "Acorn": "ACORN Segment"
            }
        )
        StyleManager.style_plotly_chart(fig_autocorr, st.session_state.theme_mode)
        st.plotly_chart(fig_autocorr, width='stretch')
