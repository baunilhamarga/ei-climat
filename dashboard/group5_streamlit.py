from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group5_energy.config import ACORN_GROUPS

OUTPUT_DIR = ROOT / "outputs" / "group5"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

PALETTE = {"ACORN-E": "#2f6f73", "ACORN-F": "#7b5f39", "ACORN-Q": "#8f3f46"}


@st.cache_data
def load_artifacts() -> dict[str, pd.DataFrame]:
    half_pred = pd.read_csv(PREDICTION_DIR / "group_5_half_hourly_predict.csv", parse_dates=["DateTime"])
    daily_pred = pd.read_csv(PREDICTION_DIR / "group_5_daily_predict.csv", parse_dates=["Date"])
    metrics = pd.read_csv(METRICS_DIR / "validation_metrics.csv")
    half_profile = pd.read_csv(METRICS_DIR / "eda_half_hour_profile.csv")
    weekly = pd.read_csv(METRICS_DIR / "eda_weekly_profile.csv")
    daily_enriched = pd.read_csv(METRICS_DIR / "eda_daily_enriched.csv", parse_dates=["Date"])
    autocorr = pd.read_csv(METRICS_DIR / "eda_daily_autocorrelation.csv")
    half_valid = pd.read_csv(METRICS_DIR / "validation_predictions_half_hourly.csv", parse_dates=["timestamp"])
    daily_valid = pd.read_csv(METRICS_DIR / "validation_predictions_daily.csv", parse_dates=["timestamp"])
    return {
        "half_pred": half_pred,
        "daily_pred": daily_pred,
        "metrics": metrics,
        "half_profile": half_profile,
        "weekly": weekly,
        "daily_enriched": daily_enriched,
        "autocorr": autocorr,
        "half_valid": half_valid,
        "daily_valid": daily_valid,
    }


def main() -> None:
    st.set_page_config(page_title="Group 5 Energy Forecasts", layout="wide")
    st.title("Group 5 Energy Forecasts")

    try:
        data = load_artifacts()
    except FileNotFoundError:
        st.error("Run `EI-climat/bin/python scripts/group5_run_pipeline.py` before opening the dashboard.")
        st.stop()

    acorn_options = list(ACORN_GROUPS)
    selected_acorns = st.sidebar.multiselect(
        "ACORN segments",
        acorn_options,
        default=acorn_options,
        format_func=lambda value: f"{value} - {ACORN_GROUPS[value]}",
    )
    if not selected_acorns:
        selected_acorns = acorn_options

    overview, eda, validation, forecasts, comparison = st.tabs(
        ["Overview", "EDA", "Validation", "Forecasts", "ACORN Comparison"]
    )

    with overview:
        metrics = data["metrics"]
        best = metrics[metrics["acorn"] == "ALL"].sort_values(["frequency", "rmse"]).groupby("frequency").head(1)
        cols = st.columns(4)
        cols[0].metric("Half-hourly rows", len(data["half_pred"]))
        cols[1].metric("Daily rows", len(data["daily_pred"]))
        half_rmse = best.loc[best["frequency"] == "half_hourly", "rmse"].iloc[0]
        daily_rmse = best.loc[best["frequency"] == "daily", "rmse"].iloc[0]
        cols[2].metric("Best 48h RMSE", f"{half_rmse:.4f}")
        cols[3].metric("Best daily RMSE", f"{daily_rmse:.4f}")
        st.dataframe(best[["frequency", "model", "rmse", "n"]], width='stretch')

        figure_cols = st.columns(2)
        figure_cols[0].image(str(FIGURES_DIR / "01_daily_consumption_trend.png"))
        figure_cols[1].image(str(FIGURES_DIR / "06_validation_rmse.png"))

    with eda:
        daily_enriched = data["daily_enriched"]
        filtered_daily = daily_enriched[daily_enriched["Acorn"].isin(selected_acorns)]
        fig = px.line(
            filtered_daily,
            x="Date",
            y="Conso_kWh",
            color="Acorn",
            color_discrete_map=PALETTE,
            title="Historical Daily Consumption",
        )
        st.plotly_chart(fig, width='stretch')

        left, right = st.columns(2)
        half_profile = data["half_profile"][data["half_profile"]["Acorn"].isin(selected_acorns)].copy()
        half_profile["time_of_day"] = half_profile["half_hour_slot"] / 2
        left.plotly_chart(
            px.line(
                half_profile,
                x="time_of_day",
                y="mean_conso_moy",
                color="Acorn",
                color_discrete_map=PALETTE,
                title="Typical Half-Hourly Profile",
            ),
            width='stretch',
        )
        weekly = data["weekly"][data["weekly"]["Acorn"].isin(selected_acorns)]
        right.plotly_chart(
            px.line(
                weekly,
                x="weekday",
                y="mean_conso_kwh",
                color="Acorn",
                markers=True,
                color_discrete_map=PALETTE,
                title="Weekly Pattern",
            ),
            width='stretch',
        )

        left, right = st.columns(2)
        left.plotly_chart(
            px.scatter(
                filtered_daily,
                x="temperatureMean",
                y="Conso_kWh",
                color="Acorn",
                color_discrete_map=PALETTE,
                title="Temperature vs Daily Consumption",
            ),
            width='stretch',
        )
        autocorr = data["autocorr"][data["autocorr"]["Acorn"].isin(selected_acorns)]
        right.plotly_chart(
            px.line(
                autocorr,
                x="lag_days",
                y="autocorrelation",
                color="Acorn",
                color_discrete_map=PALETTE,
                title="Daily Autocorrelation",
            ),
            width='stretch',
        )

    with validation:
        metric_filter = data["metrics"][data["metrics"]["acorn"].isin(["ALL", *selected_acorns])]
        st.dataframe(metric_filter.sort_values(["frequency", "acorn", "rmse"]), width='stretch')
        st.plotly_chart(
            px.bar(
                metric_filter[metric_filter["acorn"] == "ALL"],
                x="model",
                y="rmse",
                color="frequency",
                barmode="group",
                title="Overall Validation RMSE",
            ),
            width='stretch',
        )

        frequency = st.radio("Validation series", ["daily", "half_hourly"], horizontal=True)
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
        model_col = st.selectbox("Model", model_options)
        plot_df = valid[["timestamp", "Acorn", "actual", model_col]].melt(
            id_vars=["timestamp", "Acorn"], var_name="series", value_name="value"
        )
        st.plotly_chart(
            px.line(
                plot_df,
                x="timestamp",
                y="value",
                color="Acorn",
                line_dash="series",
                color_discrete_map=PALETTE,
                title=f"Validation Actual vs {model_col}",
            ),
            width='stretch',
        )

    with forecasts:
        half_pred = data["half_pred"][data["half_pred"]["Acorn"].isin(selected_acorns)]
        daily_pred = data["daily_pred"][data["daily_pred"]["Acorn"].isin(selected_acorns)]
        st.plotly_chart(
            px.line(
                half_pred,
                x="DateTime",
                y="Conso_moy_predict",
                color="Acorn",
                color_discrete_map=PALETTE,
                title="48-Hour Half-Hourly Forecast",
            ),
            width='stretch',
        )
        st.plotly_chart(
            px.line(
                daily_pred,
                x="Date",
                y="Conso_kWh_predict",
                color="Acorn",
                color_discrete_map=PALETTE,
                title="One-Month Daily Forecast",
            ),
            width='stretch',
        )

    with comparison:
        daily_summary = (
            data["daily_pred"][data["daily_pred"]["Acorn"].isin(selected_acorns)]
            .groupby("Acorn")["Conso_kWh_predict"]
            .agg(["mean", "min", "max"])
            .round(3)
            .reset_index()
        )
        st.dataframe(daily_summary, width='stretch')
        st.plotly_chart(
            px.bar(
                daily_summary,
                x="Acorn",
                y="mean",
                color="Acorn",
                color_discrete_map=PALETTE,
                title="Mean Forecast Daily Consumption by Segment",
            ),
            width='stretch',
        )


if __name__ == "__main__":
    main()
