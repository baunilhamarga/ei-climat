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

        st.markdown(
            '<div class="intro-panel">'
            '<p>Validation is chronological: models train on earlier history and are scored on the later '
            'holdout period before the final forecast window. RMSE is the main comparison metric, so lower '
            'values indicate better validation performance.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        
        st.subheader("Model Metrics Matrix")
        st.markdown(
            "The table reports RMSE for each model at the overall level (`ALL`) and for each selected ACORN "
            "segment. Baseline rows are simple repetition or seasonal averages; trained-model rows show the "
            "added value of feature engineering and model fitting."
        )
        display_metrics = metric_filter.sort_values(["frequency", "acorn", "rmse"]).copy()
        display_metrics = display_metrics[["frequency", "acorn", "model", "rmse", "n"]]
        
        matrix_rows_html = ""
        for _, row in display_metrics.iterrows():
            freq = row["frequency"]
            acorn = row["acorn"]
            model_key = row["model"]
            rmse_val = f"{row['rmse']:.6f}" if pd.notna(row['rmse']) else "n/a"
            obs_n = f"{int(row['n'])}" if pd.notna(row['n']) else "0"
            
            # Format frequency badge
            if freq == "half_hourly":
                freq_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.78rem; background: rgba(77, 164, 169, 0.12); color: #4da4a9; border: 1px solid rgba(77, 164, 169, 0.25);">Half-Hourly</span>'
            else:
                freq_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.78rem; background: rgba(212, 156, 94, 0.12); color: #d49c5e; border: 1px solid rgba(212, 156, 94, 0.25);">Daily</span>'
                
            # Format ACORN badge
            if acorn == "ALL":
                acorn_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; background: var(--kpi-border); color: var(--theme-text);">All Segments</span>'
            elif acorn == "ACORN-E":
                acorn_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; background: rgba(77, 164, 169, 0.12); color: #4da4a9; border: 1px solid rgba(77, 164, 169, 0.25);">ACORN-E</span>'
            elif acorn == "ACORN-F":
                acorn_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; background: rgba(212, 156, 94, 0.12); color: #d49c5e; border: 1px solid rgba(212, 156, 94, 0.25);">ACORN-F</span>'
            else: # ACORN-Q
                acorn_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; background: rgba(200, 90, 100, 0.12); color: #c85a64; border: 1px solid rgba(200, 90, 100, 0.25);">ACORN-Q</span>'
                
            # Format model label
            nice_model_name = StyleManager.MODEL_MAP.get(model_key, model_key)
            if model_key in {"previous_day", "previous_week", "seasonal_mean"}:
                model_display = f'<span style="font-weight: 500; color: #c85a64;">{nice_model_name}</span>'
            else:
                code_color = "#4da4a9" if freq == "half_hourly" else "#d49c5e"
                model_display = f'<code style="font-family: monospace; font-size: 0.8rem; background: var(--kpi-border); color: {code_color}; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">{nice_model_name}</code>'
                
            matrix_rows_html += (
                f'<tr>'
                f'<td style="vertical-align: middle; padding: 10px 14px;">{freq_badge}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px;">{acorn_badge}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px;">{model_display}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px;">'
                f'<div style="font-weight: 600; color: var(--theme-text);">{rmse_val}</div>'
                f'</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--kpi-sub);">{obs_n}</td>'
                f'</tr>'
            )

        matrix_html = (
            '<div class="scrollable-table-wrapper" style="max-height: 480px;">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th>Forecast Frequency</th>'
            '<th>ACORN Segment</th>'
            '<th>Model</th>'
            '<th>Validation RMSE</th>'
            '<th>Observations (N)</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{matrix_rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(matrix_html, unsafe_allow_html=True)
        
        st.subheader("Overall Performance (All ACORN Segments)")
        st.markdown(
            "These charts focus on the aggregate score across all three ACORN segments. They make it easy "
            "to see whether a model beats the simple baselines and whether the best method changes between "
            "the half-hourly and daily forecasting tasks."
        )
        overall_cols = st.columns(2)
        overall_metrics = metric_filter[metric_filter["acorn"] == "ALL"].copy()
        
        is_light = (st.session_state.theme_mode == "light")
        baseline_text_color = "#b93c4b" if is_light else "#ff7675"
        baselines = {"previous_day", "previous_week", "seasonal_mean"}

        # Half-Hourly Chart
        overall_half = overall_metrics[overall_metrics["frequency"] == "half_hourly"].copy()
        overall_half["model_type"] = overall_half["model"].map(lambda m: "Baseline" if m in baselines else "Trained Model")
        overall_half["model_display"] = overall_half["model"].map(
            lambda m: f'<span style="color: {baseline_text_color}; font-weight: bold;">{StyleManager.MODEL_MAP.get(m, m)}</span>'
            if m in baselines else StyleManager.MODEL_MAP.get(m, m)
        )
        overall_half = overall_half.sort_values("rmse", ascending=False)
        fig_half = px.bar(
            overall_half,
            x="model_display",
            y="rmse",
            color="model_type",
            color_discrete_map={
                "Baseline": "#c85a64",
                "Trained Model": "#4da4a9"
            },
            title="Overall Half-Hourly Validation RMSE",
            labels={
                "model_display": "Model",
                "rmse": "Root Mean Squared Error (RMSE)",
                "model_type": "Model Class"
            }
        )
        fig_half.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_half, st.session_state.theme_mode)
        overall_cols[0].plotly_chart(fig_half, width='stretch', theme=None)
        
        # Daily Chart
        overall_daily = overall_metrics[overall_metrics["frequency"] == "daily"].copy()
        overall_daily["model_type"] = overall_daily["model"].map(lambda m: "Baseline" if m in baselines else "Trained Model")
        overall_daily["model_display"] = overall_daily["model"].map(
            lambda m: f'<span style="color: {baseline_text_color}; font-weight: bold;">{StyleManager.MODEL_MAP.get(m, m)}</span>'
            if m in baselines else StyleManager.MODEL_MAP.get(m, m)
        )
        overall_daily = overall_daily.sort_values("rmse", ascending=False)
        fig_daily = px.bar(
            overall_daily,
            x="model_display",
            y="rmse",
            color="model_type",
            color_discrete_map={
                "Baseline": "#c85a64",
                "Trained Model": "#d49c5e"
            },
            title="Overall Daily Validation RMSE",
            labels={
                "model_display": "Model",
                "rmse": "Root Mean Squared Error (RMSE)",
                "model_type": "Model Class"
            }
        )
        fig_daily.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_daily, st.session_state.theme_mode)
        overall_cols[1].plotly_chart(fig_daily, width='stretch', theme=None)

        st.subheader("Actual vs Predicted Validation Series")
        st.markdown(
            "The line chart compares the held-out actual consumption with one model's validation prediction. "
            "Good models should follow both the level and the shape of the actual curve; systematic gaps or "
            "missed peaks explain why a model's RMSE is higher."
        )
        frequency = st.radio(
            "Validation series frequency", 
            ["daily", "half_hourly"], 
            horizontal=True,
            format_func=lambda x: StyleManager.FREQUENCY_MAP.get(x, x)
        )
        valid = data["daily_valid" if frequency == "daily" else "half_valid"]
        valid = valid[valid["Acorn"].isin(selected_acorns)]
        
        available_models = [
            model
            for model in valid.columns
            if model not in {"timestamp", "Acorn", "actual"}
        ]
        rmse_order = (
            data["metrics"][
                (data["metrics"]["frequency"] == frequency)
                & (data["metrics"]["acorn"] == "ALL")
                & (data["metrics"]["model"].isin(available_models))
            ]
            .sort_values(["rmse", "model"])["model"]
            .tolist()
        )
        model_options = rmse_order + [
            model for model in available_models if model not in rmse_order
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
            model_col: "Predicted"
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
        st.plotly_chart(fig_line, width='stretch', theme=None)
