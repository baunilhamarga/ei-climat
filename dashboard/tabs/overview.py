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
        # Calculate row counts dynamically based on selected ACORN segments
        half_pred_filtered = data["half_pred"][data["half_pred"]["Acorn"].isin(selected_acorns)]
        daily_pred_filtered = data["daily_pred"][data["daily_pred"]["Acorn"].isin(selected_acorns)]
        half_rows = len(half_pred_filtered)
        daily_rows = len(daily_pred_filtered)

        # Calculate best validation RMSE dynamically for selected ACORN segments
        metrics_acorns = metrics[metrics["acorn"].isin(selected_acorns)].copy()
        metrics_acorns["sse"] = metrics_acorns["n"] * (metrics_acorns["rmse"] ** 2)
        
        combined_metrics = metrics_acorns.groupby(["frequency", "model"]).agg(
            total_sse=("sse", "sum"),
            total_n=("n", "sum")
        ).reset_index()
        
        combined_metrics["rmse"] = (combined_metrics["total_sse"] / combined_metrics["total_n"]) ** 0.5
        combined_metrics["n"] = combined_metrics["total_n"]
        
        best_combined = combined_metrics.sort_values(["frequency", "rmse"]).groupby("frequency").head(1)
        
        half_rmse = best_combined.loc[best_combined["frequency"] == "half_hourly", "rmse"].iloc[0]
        daily_rmse = best_combined.loc[best_combined["frequency"] == "daily", "rmse"].iloc[0]
        half_model = best_combined.loc[best_combined["frequency"] == "half_hourly", "model"].iloc[0]
        daily_model = best_combined.loc[best_combined["frequency"] == "daily", "model"].iloc[0]

        # Get human-readable names for KPIs
        nice_half_model = StyleManager.MODEL_MAP.get(half_model, half_model)
        nice_daily_model = StyleManager.MODEL_MAP.get(daily_model, daily_model)
        nice_half_freq = StyleManager.FREQUENCY_MAP.get("half_hourly", "Half-Hourly")
        nice_daily_freq = StyleManager.FREQUENCY_MAP.get("daily", "Daily")

        # Render premium glassmorphic cards
        kpi_html = f"""
        <div class="kpi-container">
            {StyleManager.render_kpi_card(f"{nice_half_freq} Predict Rows", f"{half_rows}", "Short-term horizon")}
            {StyleManager.render_kpi_card(f"{nice_daily_freq} Predict Rows", f"{daily_rows}", "Medium-term horizon")}
            {StyleManager.render_kpi_card("Best 48h RMSE", f"{half_rmse:.4f}", f"Model: {nice_half_model}")}
            {StyleManager.render_kpi_card("Best Daily RMSE", f"{daily_rmse:.4f}", f"Model: {nice_daily_model}")}
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.subheader("Selected Best Performing Models")
        display_best = best_combined[["frequency", "model", "rmse", "n"]].copy()
        display_best["frequency"] = display_best["frequency"].map(StyleManager.FREQUENCY_MAP)
        display_best["model"] = display_best["model"].map(StyleManager.MODEL_MAP)
        display_best["rmse"] = display_best["rmse"].map(lambda x: f"{x:.4f}")
        display_best["n"] = display_best["n"].map(lambda x: f"{int(x)}")
        display_best = display_best.rename(columns={
            "frequency": "Forecast Frequency",
            "model": "Best Model",
            "rmse": "Validation RMSE",
            "n": "Observations (N)"
        })
        st.markdown(f'<div class="scrollable-table-wrapper">{display_best.to_html(index=False)}</div>', unsafe_allow_html=True)

        st.subheader("Key Visual Trends")

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
            st.plotly_chart(fig_trend, width='stretch', theme=None)
        else:
            st.warning("No data selected to display trend.")

        # Interactive Validation RMSE Comparison (separated by frequency for better readability)
        rmse_cols = st.columns(2)
        metric_subset = metrics[metrics["acorn"] == "ALL"].copy()
        
        is_light = (st.session_state.theme_mode == "light")
        baseline_text_color = "#b93c4b" if is_light else "#ff7675"
        baselines = {"previous_day", "previous_week", "seasonal_mean"}

        # Half-Hourly RMSE Chart
        metric_half = metric_subset[metric_subset["frequency"] == "half_hourly"].copy()
        metric_half["model_type"] = metric_half["model"].map(lambda m: "Baseline" if m in baselines else "Trained Model")
        metric_half["model_display"] = metric_half["model"].map(
            lambda m: f'<span style="color: {baseline_text_color}; font-weight: bold;">{StyleManager.MODEL_MAP.get(m, m)}</span>'
            if m in baselines else StyleManager.MODEL_MAP.get(m, m)
        )
        metric_half = metric_half.sort_values("rmse", ascending=False)
        fig_rmse_half = px.bar(
            metric_half,
            x="model_display",
            y="rmse",
            color="model_type",
            color_discrete_map={
                "Baseline": "#c85a64",
                "Trained Model": "#4da4a9"
            },
            title="Half-Hourly Validation RMSE by model",
            labels={
                "model_display": "Model",
                "rmse": "Root Mean Squared Error (RMSE)",
                "model_type": "Model Class"
            }
        )
        fig_rmse_half.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_rmse_half, st.session_state.theme_mode)
        rmse_cols[0].plotly_chart(fig_rmse_half, width='stretch', theme=None)

        # Daily RMSE Chart
        metric_daily = metric_subset[metric_subset["frequency"] == "daily"].copy()
        metric_daily["model_type"] = metric_daily["model"].map(lambda m: "Baseline" if m in baselines else "Trained Model")
        metric_daily["model_display"] = metric_daily["model"].map(
            lambda m: f'<span style="color: {baseline_text_color}; font-weight: bold;">{StyleManager.MODEL_MAP.get(m, m)}</span>'
            if m in baselines else StyleManager.MODEL_MAP.get(m, m)
        )
        metric_daily = metric_daily.sort_values("rmse", ascending=False)
        fig_rmse_daily = px.bar(
            metric_daily,
            x="model_display",
            y="rmse",
            color="model_type",
            color_discrete_map={
                "Baseline": "#c85a64",
                "Trained Model": "#d49c5e"
            },
            title="Daily Validation RMSE by model",
            labels={
                "model_display": "Model",
                "rmse": "Root Mean Squared Error (RMSE)",
                "model_type": "Model Class"
            }
        )
        fig_rmse_daily.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_rmse_daily, st.session_state.theme_mode)
        rmse_cols[1].plotly_chart(fig_rmse_daily, width='stretch', theme=None)
