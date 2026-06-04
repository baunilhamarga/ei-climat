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
        display_metrics = metric_filter.sort_values(["frequency", "acorn", "rmse"]).copy()
        display_metrics = display_metrics[["frequency", "acorn", "model", "rmse", "n"]]
        display_metrics["frequency"] = display_metrics["frequency"].map(StyleManager.FREQUENCY_MAP)
        display_metrics["model"] = display_metrics["model"].map(StyleManager.MODEL_MAP)
        display_metrics["rmse"] = display_metrics["rmse"].map(lambda x: f"{x:.4f}")
        display_metrics = display_metrics.rename(columns={
            "frequency": "Forecast Frequency",
            "acorn": "ACORN Segment",
            "model": "Model",
            "rmse": "Validation RMSE",
            "n": "Observations (N)"
        })
        st.markdown(f'<div class="scrollable-table-wrapper">{display_metrics.to_html(index=False)}</div>', unsafe_allow_html=True)
        
        st.subheader("Overall Performance (All ACORN Segments)")
        overall_cols = st.columns(2)
        overall_metrics = metric_filter[metric_filter["acorn"] == "ALL"].copy()
        
        # Half-Hourly Chart
        overall_half = overall_metrics[overall_metrics["frequency"] == "half_hourly"].copy()
        overall_half["model"] = overall_half["model"].map(StyleManager.MODEL_MAP)
        overall_half = overall_half.sort_values("rmse", ascending=False)
        fig_half = px.bar(
            overall_half,
            x="model",
            y="rmse",
            title="Overall Half-Hourly Validation RMSE",
            labels={
                "model": "Model",
                "rmse": "Root Mean Squared Error (RMSE)"
            },
            color_discrete_sequence=["#4da4a9"]
        )
        fig_half.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_half, st.session_state.theme_mode)
        overall_cols[0].plotly_chart(fig_half, width='stretch')
        
        # Daily Chart
        overall_daily = overall_metrics[overall_metrics["frequency"] == "daily"].copy()
        overall_daily["model"] = overall_daily["model"].map(StyleManager.MODEL_MAP)
        overall_daily = overall_daily.sort_values("rmse", ascending=False)
        fig_daily = px.bar(
            overall_daily,
            x="model",
            y="rmse",
            title="Overall Daily Validation RMSE",
            labels={
                "model": "Model",
                "rmse": "Root Mean Squared Error (RMSE)"
            },
            color_discrete_sequence=["#d49c5e"]
        )
        fig_daily.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_daily, st.session_state.theme_mode)
        overall_cols[1].plotly_chart(fig_daily, width='stretch')

        st.subheader("Interactive Predictions Comparison")
        frequency = st.radio(
            "Validation series frequency", 
            ["daily", "half_hourly"], 
            horizontal=True,
            format_func=lambda x: StyleManager.FREQUENCY_MAP.get(x, x)
        )
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

        model_col = st.selectbox(
            "Select model to compare", 
            model_options,
            format_func=lambda x: StyleManager.MODEL_MAP.get(x, x)
        )
        
        plot_df = valid[["timestamp", "Acorn", "actual", model_col]].melt(
            id_vars=["timestamp", "Acorn"], var_name="series", value_name="value"
        )
        
        nice_model_name = StyleManager.MODEL_MAP.get(model_col, model_col)
        series_map = {
            "actual": "Actual",
            model_col: nice_model_name
        }
        plot_df["series"] = plot_df["series"].map(series_map)
        
        fig_line = px.line(
            plot_df,
            x="timestamp",
            y="value",
            color="Acorn",
            line_dash="series",
            color_discrete_map=StyleManager.PALETTE,
            title=f"Validation Actual vs {nice_model_name}",
            labels={
                "timestamp": "Timestamp",
                "value": "Electricity Consumption",
                "Acorn": "ACORN Segment",
                "series": "Series Type"
            }
        )
        StyleManager.style_plotly_chart(fig_line, st.session_state.theme_mode)
        st.plotly_chart(fig_line, width='stretch')
