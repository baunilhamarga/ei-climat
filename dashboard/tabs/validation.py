# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager

class ValidationTab(BaseTab):
    """Orchestrates the model validation performance rendering."""
    
    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        metric_filter = data["metrics"][data["metrics"]["acorn"].isin(["ALL", *selected_acorns])]
        
        st.subheader("Model Metrics Matrix")
        st.dataframe(metric_filter.sort_values(["frequency", "acorn", "rmse"]), width='stretch')
        
        st.subheader("Overall Performance (acorn = ALL)")
        fig_bar = px.bar(
            metric_filter[metric_filter["acorn"] == "ALL"],
            x="model",
            y="rmse",
            color="frequency",
            barmode="group",
            title="Overall Validation RMSE",
        )
        st.plotly_chart(fig_bar, width='stretch')

        st.subheader("Interactive Predictions Comparison")
        frequency = st.radio("Validation series frequency", ["daily", "half_hourly"], horizontal=True)
        valid = data["daily_valid" if frequency == "daily" else "half_valid"]
        valid = valid[valid["Acorn"].isin(selected_acorns)]
        
        model_options = [
            model
            for model in [
                "autogluon_timeseries",
                "autogluon",
                "lightgbm",
                "catboost",
                "xgboost",
                "ridge",
                "seasonal_mean",
                "previous_week",
                "previous_day",
            ]
            if model in valid.columns
        ]
        
        if not model_options:
            st.warning("No candidate models found in validation prediction columns.")
            return

        model_col = st.selectbox("Select model to compare", model_options)
        
        plot_df = valid[["timestamp", "Acorn", "actual", model_col]].melt(
            id_vars=["timestamp", "Acorn"], var_name="series", value_name="value"
        )
        
        fig_line = px.line(
            plot_df,
            x="timestamp",
            y="value",
            color="Acorn",
            line_dash="series",
            color_discrete_map=StyleManager.PALETTE,
            title=f"Validation Actual vs {model_col}",
        )
        st.plotly_chart(fig_line, width='stretch')
