from html import escape
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
        
        best_models_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th>Forecast Horizon</th>'
            '<th>Best Model</th>'
            '<th>Validation RMSE</th>'
            '<th>Observations (N)</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            '<tr>'
            '<td>'
            f'<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">{nice_half_freq}</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Short-term forecast</div>'
            '</td>'
            '<td>'
            f'<code style="font-family: monospace; font-size: 0.82rem; background: var(--kpi-border); color: #4da4a9; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">{nice_half_model}</code>'
            '</td>'
            '<td>'
            f'<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">{half_rmse:.6f}</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Lower is better</div>'
            '</td>'
            '<td>'
            f'<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">{int(best_combined.loc[best_combined["frequency"] == "half_hourly", "n"].iloc[0])}</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Validation steps</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            f'<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">{nice_daily_freq}</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Medium-term forecast</div>'
            '</td>'
            '<td>'
            f'<code style="font-family: monospace; font-size: 0.82rem; background: var(--kpi-border); color: #d49c5e; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">{nice_daily_model}</code>'
            '</td>'
            '<td>'
            f'<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">{daily_rmse:.6f}</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Lower is better</div>'
            '</td>'
            '<td>'
            f'<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">{int(best_combined.loc[best_combined["frequency"] == "daily", "n"].iloc[0])}</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Validation steps</div>'
            '</td>'
            '</tr>'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(best_models_html, unsafe_allow_html=True)

        st.subheader("Best Model Forecast vs Actual Validation")
        st.markdown(
            "These charts compare actual consumption with the forecasts from the best performing model. "
            "The curves represent the **total consumption summed across all selected ACORN categories** over the validation period."
        )
        
        # Load validation predictions and actuals
        half_valid_df = data["half_valid"]
        daily_valid_df = data["daily_valid"]

        half_valid_models = [c for c in half_valid_df.columns if c not in ["frequency", "Acorn", "timestamp", "actual"]]
        daily_valid_models = [c for c in daily_valid_df.columns if c not in ["frequency", "Acorn", "timestamp", "actual"]]

        # Find the best models dynamically based on selected ACORN segments
        half_best_model = None
        for model in combined_metrics[combined_metrics["frequency"] == "half_hourly"].sort_values("rmse")["model"]:
            if model in half_valid_models:
                half_best_model = model
                break
        if half_best_model is None:
            half_best_model = half_valid_models[0] if half_valid_models else "xgboost"

        daily_best_model = None
        for model in combined_metrics[combined_metrics["frequency"] == "daily"].sort_values("rmse")["model"]:
            if model in daily_valid_models:
                daily_best_model = model
                break
        if daily_best_model is None:
            daily_best_model = daily_valid_models[0] if daily_valid_models else "xgboost"

        # Sum of consumption across selected ACORN segments
        half_v_filtered = half_valid_df[half_valid_df["Acorn"].isin(selected_acorns)]
        daily_v_filtered = daily_valid_df[daily_valid_df["Acorn"].isin(selected_acorns)]

        # Group by timestamp and sum actual and prediction columns
        half_summed = half_v_filtered.groupby("timestamp").agg(
            actual=("actual", "sum"),
            pred=(half_best_model, "sum")
        ).reset_index()

        daily_summed = daily_v_filtered.groupby("timestamp").agg(
            actual=("actual", "sum"),
            pred=(daily_best_model, "sum")
        ).reset_index()

        nice_half_model = StyleManager.MODEL_MAP.get(half_best_model, half_best_model)
        nice_daily_model = StyleManager.MODEL_MAP.get(daily_best_model, daily_best_model)

        actual_color = "#4da4a9"
        half_pred_color = "#c85a64"
        daily_pred_color = "#d49c5e"

        # Half-Hourly Chart
        half_plot_df = half_summed.melt(
            id_vars=["timestamp"],
            value_vars=["actual", "pred"],
            var_name="Series",
            value_name="Consumption"
        )
        half_plot_df["Series"] = half_plot_df["Series"].map({
            "actual": "Actual",
            "pred": f"Predicted ({nice_half_model})"
        })

        fig_half_forecast = px.line(
            half_plot_df,
            x="timestamp",
            y="Consumption",
            color="Series",
            color_discrete_map={
                "Actual": actual_color,
                f"Predicted ({nice_half_model})": half_pred_color
            },
            title="Half-Hourly Validation: Total Consumption across selected ACORN categories",
            labels={
                "timestamp": "Validation Timeline (Half-Hourly)",
                "Consumption": "Summed Consumption of all ACORNs (kW)",
                "Series": "Legend"
            }
        )
        for trace in fig_half_forecast.data:
            if trace.name.startswith("Predicted"):
                trace.line.dash = "dash"
        StyleManager.style_plotly_chart(fig_half_forecast, st.session_state.theme_mode)

        # Daily Chart
        daily_plot_df = daily_summed.melt(
            id_vars=["timestamp"],
            value_vars=["actual", "pred"],
            var_name="Series",
            value_name="Consumption"
        )
        daily_plot_df["Series"] = daily_plot_df["Series"].map({
            "actual": "Actual",
            "pred": f"Predicted ({nice_daily_model})"
        })

        fig_daily_forecast = px.line(
            daily_plot_df,
            x="timestamp",
            y="Consumption",
            color="Series",
            color_discrete_map={
                "Actual": actual_color,
                f"Predicted ({nice_daily_model})": daily_pred_color
            },
            title="Daily Validation: Total Consumption across selected ACORN categories",
            labels={
                "timestamp": "Validation Timeline (Daily)",
                "Consumption": "Summed Consumption of all ACORNs (kWh)",
                "Series": "Legend"
            }
        )
        for trace in fig_daily_forecast.data:
            if trace.name.startswith("Predicted"):
                trace.line.dash = "dash"
        StyleManager.style_plotly_chart(fig_daily_forecast, st.session_state.theme_mode)

        # Render forecast graphs side-by-side
        forecast_cols = st.columns(2)
        forecast_cols[0].plotly_chart(fig_half_forecast, width='stretch', theme=None)
        forecast_cols[1].plotly_chart(fig_daily_forecast, width='stretch', theme=None)

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

        self._render_model_reference(metrics)

    def _render_model_reference(self, metrics: pd.DataFrame) -> None:
        st.subheader("Model Reference")
        st.markdown(
            "This table is a quick glossary for the benchmark models shown across the dashboard. "
            "It includes the simple baselines as well as the trained models used in validation and final forecasting."
        )

        model_order = [
            "previous_day",
            "previous_week",
            "seasonal_mean",
            "ridge",
            "xgboost",
            "xgboost_by_acorn",
            "catboost",
            "lightgbm",
            "stack_regressor",
            "autogluon",
            "autogluon_best",
            "autogluon_timeseries",
            "autogluon_timeseries_best",
        ]
        descriptions = {
            "previous_day": "Repeats the same ACORN value from one day earlier. This is the main persistence baseline.",
            "previous_week": "Repeats the same ACORN value from seven days earlier to test weekly repetition.",
            "seasonal_mean": "Uses the historical mean for the same ACORN and calendar slot or day.",
            "ridge": "Regularized linear regression on the engineered weather, calendar, holiday, lag, and rolling features.",
            "xgboost": "Pooled gradient-boosted tree model trained across the three ACORN segments on the engineered feature table.",
            "xgboost_by_acorn": "Three isolated XGBoost models, one per ACORN segment, trained only on that segment's history.",
            "catboost": "Gradient-boosted tree model with strong categorical-feature handling, trained on the same tabular features.",
            "lightgbm": "Fast gradient-boosted tree model; this is the selected half-hourly model in the current benchmark.",
            "stack_regressor": "Sklearn stacking ensemble that blends Ridge, XGBoost, and LightGBM with a Ridge meta-model.",
            "autogluon": "Mid-effort AutoGluon Tabular AutoML run on the engineered feature table; selected for the daily forecast.",
            "autogluon_best": "High-effort AutoGluon Tabular calibration with more extensive ensembling and stacking.",
            "autogluon_timeseries": "AutoGluon TimeSeries run using target history plus known future calendar, holiday, and weather covariates.",
            "autogluon_timeseries_best": "High-effort AutoGluon TimeSeries calibration. It improved over the mid-effort time-series run but was not selected.",
        }

        available_models = [str(model) for model in metrics["model"].dropna().unique().tolist()]
        ordered_models = [model for model in model_order if model in available_models]
        ordered_models.extend(sorted(model for model in available_models if model not in ordered_models))

        rows = []
        for model in ordered_models:
            display_name = StyleManager.MODEL_MAP.get(model, model)
            description = descriptions.get(model, "Model included in the saved validation metrics for comparison.")
            rows.append(
                "<tr>"
                f'<td style="vertical-align: top;"><div style="font-weight: 600; color: var(--theme-text);">{escape(display_name)}</div>'
                f'<div style="font-size: 0.78rem; color: var(--kpi-sub); margin-top: 2px;"><code>{escape(model)}</code></div></td>'
                f'<td style="vertical-align: top; color: var(--theme-text);">{escape(description)}</td>'
                "</tr>"
            )

        table_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead><tr><th>Model</th><th>Short description</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(table_html, unsafe_allow_html=True)
