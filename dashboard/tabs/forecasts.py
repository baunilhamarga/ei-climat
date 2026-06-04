import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager
from group5_energy.config import ACORN_GROUPS

class ForecastsTab(BaseTab):
    """Orchestrates rendering the model prediction forecasts."""
    
    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        half_pred = data["half_pred"][data["half_pred"]["Acorn"].isin(selected_acorns)]
        daily_pred = data["daily_pred"][data["daily_pred"]["Acorn"].isin(selected_acorns)]

        st.markdown(
            '<div class="intro-panel">'
            '<p>This tab shows the final forecast deliverables and the saved-model outputs behind them. We '
            'compare the model predictions with the provided truth values.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        
        st.subheader("Short-Term Forecast (48 Hours)")
        st.markdown(
            "The short-term output predicts average consumption every 30 minutes from 2014-01-13 00:00 "
            "through 2014-01-14 23:30. This view is useful for checking intraday shape and evening peaks."
        )
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
        StyleManager.style_plotly_chart(fig_half, st.session_state.theme_mode)
        st.plotly_chart(fig_half, width='stretch', theme=None)

        st.subheader("Medium-Term Forecast (1 Month)")
        st.markdown(
            "The medium-term output predicts total daily consumption from 2014-01-13 through 2014-02-13. "
            "Because this horizon is longer, recursive lag updates mean earlier predictions can influence "
            "later forecast features."
        )
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
        StyleManager.style_plotly_chart(fig_daily, st.session_state.theme_mode)
        st.plotly_chart(fig_daily, width='stretch', theme=None)

        self._render_model_comparison(data, selected_acorns)

    def _render_model_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
    ) -> None:
        st.subheader("Saved Model Comparison")
        st.markdown(
            "The model comparison keeps every saved model's predictions available, not only the selected "
            "winner. This helps separate three questions: how models performed on the validation holdout, "
            "what each model predicts for the final horizon, and how final predictions compare with available "
            "ground truth."
        )
        view_mode = st.radio(
            "View",
            ["test", "final", "validation"],
            horizontal=True,
            format_func=lambda value: {
                "test": "Test Actual vs Prediction",
                "final": "Final Forecast",
                "validation": "Validation Holdout",
            }[value],
        )
        frequency = st.radio(
            "Frequency",
            ["half_hourly", "daily"],
            horizontal=True,
            format_func=lambda value: StyleManager.FREQUENCY_MAP.get(value, value),
        )

        if view_mode == "test":
            self._render_test_comparison(data, selected_acorns, frequency)
        elif view_mode == "validation":
            self._render_validation_comparison(data, selected_acorns, frequency)
        else:
            self._render_final_forecast_comparison(data, selected_acorns, frequency)

    def _render_test_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
        frequency: str,
    ) -> None:
        config = self._forecast_config(frequency)
        st.markdown(
            "This view joins saved final-horizon predictions with the provided actual values for the same "
            "dates and ACORN segments. It is the closest check to real final-test performance when those "
            "actual values are available."
        )
        actuals = data[config["actual_data_key"]].copy()
        actuals = actuals[actuals["Acorn"].isin(selected_acorns)]
        if actuals.empty:
            st.warning(
                f"No final test ground-truth file was found for {StyleManager.FREQUENCY_MAP.get(frequency, frequency).lower()}. "
                f"Expected `data/02_processed/csv/{config['provided_file']}` or "
                f"`outputs/group5/metrics/{config['actuals_file']}` with columns `Acorn`, "
                f"`{config['time_col']}`, and `{config['actual_col']}`."
            )
            return

        predictions = data[config["data_key"]].copy()
        predictions = predictions[predictions["Acorn"].isin(selected_acorns)]
        if predictions.empty:
            st.warning("No saved-model forecasts are available for the selected ACORN segments.")
            return

        join_cols = ["Acorn", config["time_col"]]
        merged = predictions.merge(actuals, on=join_cols, how="inner")
        if merged.empty:
            st.warning("Final test actuals were found, but their dates or ACORN segments do not overlap the saved forecasts.")
            return

        model_options = self._ordered_models(
            merged["model"].dropna().unique().tolist(),
            data["metrics"],
            frequency,
        )
        selected_models = self._model_selector(model_options, model_options[:1], f"test_{frequency}")
        if not selected_models:
            st.warning("Select at least one model to display.")
            return

        merged = merged[merged["model"].isin(selected_models)].copy()
        plot_df = self._test_plot_frame(merged, config)

        fig = px.line(
            plot_df,
            x=config["time_col"],
            y="value",
            color="series_label",
            line_dash="series_type",
            facet_col="Acorn",
            facet_col_wrap=3,
            facet_col_spacing=0.07,
            title=config["test_title"],
            labels={
                config["time_col"]: "Timestamp",
                "value": config["y_label"],
                "series_label": "Series",
                "series_type": "Type",
                "Acorn": "ACORN Segment",
            },
        )
        fig.update_yaxes(matches=None)
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True))
        fig.for_each_annotation(
            lambda a: a.update(
                text=f"{a.text.split('=')[-1]} ({ACORN_GROUPS.get(a.text.split('=')[-1], '')})"
                if "=" in a.text else a.text
            )
        )
        StyleManager.style_plotly_chart(fig, st.session_state.theme_mode)
        st.plotly_chart(fig, width='stretch', theme=None)

        summary = self._test_metric_table(merged, config)
        st.markdown(self._format_test_metric_table(summary, frequency), unsafe_allow_html=True)
        self._render_test_rmse_chart(summary, frequency)

    def _render_final_forecast_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
        frequency: str,
    ) -> None:
        config = self._forecast_config(frequency)
        st.markdown(
            "This view shows the final assignment-period forecasts produced by each saved model. It does not "
            "score the models by itself; the RMSE chart below comes from the chronological validation split."
        )
        model_predictions = data[config["data_key"]]
        filtered = model_predictions[model_predictions["Acorn"].isin(selected_acorns)].copy()
        if filtered.empty:
            st.warning("No saved-model forecasts are available for the selected ACORN segments.")
            return

        model_options = self._ordered_models(filtered["model"].dropna().unique().tolist(), data["metrics"], frequency)
        selected_models = self._model_selector(model_options, model_options, f"final_{frequency}")
        if not selected_models:
            st.warning("Select at least one model to display.")
            return

        filtered = filtered[filtered["model"].isin(selected_models)].copy()
        filtered["model_label"] = filtered["model"].map(self._model_label)

        fig = px.line(
            filtered,
            x=config["time_col"],
            y=config["prediction_col"],
            color="model_label",
            facet_col="Acorn",
            facet_col_wrap=3,
            facet_col_spacing=0.07,
            title=config["title"],
            labels={
                config["time_col"]: "Timestamp",
                config["prediction_col"]: config["prediction_label"],
                "model_label": "Model",
                "Acorn": "ACORN Segment",
            },
        )
        fig.update_yaxes(matches=None)
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True))
        fig.for_each_annotation(
            lambda a: a.update(
                text=f"{a.text.split('=')[-1]} ({ACORN_GROUPS.get(a.text.split('=')[-1], '')})"
                if "=" in a.text else a.text
            )
        )
        StyleManager.style_plotly_chart(fig, st.session_state.theme_mode)
        st.plotly_chart(fig, width='stretch', theme=None)

        summary_html = self._forecast_metric_table(filtered, data["metrics"], frequency, config["prediction_col"])
        st.markdown(summary_html, unsafe_allow_html=True)
        self._render_rmse_chart(data["metrics"], selected_models, frequency)

    def _render_validation_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
        frequency: str,
    ) -> None:
        st.markdown(
            "This view returns to the chronological validation window, where actual values were held out "
            "from training. It is the fair model-comparison view used to choose the selected forecasting models."
        )
        valid = data["daily_valid" if frequency == "daily" else "half_valid"].copy()
        valid = valid[valid["Acorn"].isin(selected_acorns)]
        model_options = self._ordered_models(
            [model for model in data["metrics"]["model"].dropna().unique().tolist() if model in valid.columns],
            data["metrics"],
            frequency,
        )
        if not model_options:
            st.warning("No validation prediction columns are available for this frequency.")
            return

        selected_models = self._model_selector(model_options, model_options[:1], f"validation_{frequency}")
        if not selected_models:
            st.warning("Select at least one model to display.")
            return

        plot_df = valid[["timestamp", "Acorn", "actual", *selected_models]].melt(
            id_vars=["timestamp", "Acorn"],
            var_name="series",
            value_name="value",
        )
        plot_df["series_label"] = plot_df["series"].map(lambda value: "Actual" if value == "actual" else self._model_label(value))
        plot_df["series_type"] = plot_df["series"].map(lambda value: "Actual" if value == "actual" else "Prediction")

        fig = px.line(
            plot_df,
            x="timestamp",
            y="value",
            color="series_label",
            line_dash="series_type",
            facet_col="Acorn",
            facet_col_wrap=3,
            facet_col_spacing=0.07,
            title="Validation Actual vs Prediction",
            labels={
                "timestamp": "Timestamp",
                "value": "Electricity Consumption",
                "series_label": "Series",
                "series_type": "Type",
                "Acorn": "ACORN Segment",
            },
        )
        fig.update_yaxes(matches=None)
        fig.for_each_yaxis(lambda y: y.update(showticklabels=True))
        fig.for_each_annotation(
            lambda a: a.update(
                text=f"{a.text.split('=')[-1]} ({ACORN_GROUPS.get(a.text.split('=')[-1], '')})"
                if "=" in a.text else a.text
            )
        )
        StyleManager.style_plotly_chart(fig, st.session_state.theme_mode)
        st.plotly_chart(fig, width='stretch', theme=None)

        summary_html = self._validation_metric_table(data["metrics"], selected_models, frequency)
        st.markdown(summary_html, unsafe_allow_html=True)
        self._render_rmse_chart(data["metrics"], selected_models, frequency)

    def _model_selector(self, model_options: list[str], default_models: list[str], key_suffix: str) -> list[str]:
        return st.multiselect(
            "Models",
            model_options,
            default=default_models,
            format_func=self._model_label,
            key=f"forecast_models_{key_suffix}",
        )

    def _render_rmse_chart(self, metrics: pd.DataFrame, selected_models: list[str], frequency: str) -> None:
        metric_subset = metrics[
            (metrics["frequency"] == frequency)
            & (metrics["acorn"] == "ALL")
            & (metrics["model"].isin(selected_models))
        ].copy()
        if metric_subset.empty:
            return
        
        is_light = (st.session_state.theme_mode == "light")
        baseline_text_color = "#b93c4b" if is_light else "#ff7675"
        baselines = {"previous_day", "previous_week", "seasonal_mean"}
        trained_color = "#4da4a9" if frequency == "half_hourly" else "#d49c5e"

        metric_subset["model_type"] = metric_subset["model"].map(lambda m: "Baseline" if m in baselines else "Trained Model")
        metric_subset["model_display"] = metric_subset["model"].map(
            lambda m: f'<span style="color: {baseline_text_color}; font-weight: bold;">{self._model_label(m)}</span>'
            if m in baselines else self._model_label(m)
        )

        fig_rmse = px.bar(
            metric_subset.sort_values("rmse", ascending=False),
            x="model_display",
            y="rmse",
            color="model_type",
            color_discrete_map={
                "Baseline": "#c85a64",
                "Trained Model": trained_color
            },
            title="Validation RMSE for Displayed Models",
            labels={
                "model_display": "Model", 
                "rmse": "Validation RMSE",
                "model_type": "Model Class"
            },
        )
        fig_rmse.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_rmse, st.session_state.theme_mode)
        st.plotly_chart(fig_rmse, width='stretch', theme=None)

    def _render_test_rmse_chart(self, summary: pd.DataFrame, frequency: str) -> None:
        if summary.empty:
            return
        
        is_light = (st.session_state.theme_mode == "light")
        baseline_text_color = "#b93c4b" if is_light else "#ff7675"
        baselines = {"previous_day", "previous_week", "seasonal_mean"}
        trained_color = "#4da4a9" if frequency == "half_hourly" else "#d49c5e"

        chart_df = summary.copy()
        chart_df["model_type"] = chart_df["model"].map(lambda m: "Baseline" if m in baselines else "Trained Model")
        chart_df["model_display"] = chart_df["model"].map(
            lambda m: f'<span style="color: {baseline_text_color}; font-weight: bold;">{self._model_label(m)}</span>'
            if m in baselines else self._model_label(m)
        )

        fig_rmse = px.bar(
            chart_df.sort_values("rmse", ascending=False),
            x="model_display",
            y="rmse",
            color="model_type",
            color_discrete_map={
                "Baseline": "#c85a64",
                "Trained Model": trained_color
            },
            title="Final Test RMSE for Displayed Models",
            labels={
                "model_display": "Model", 
                "rmse": "Test RMSE",
                "model_type": "Model Class"
            },
        )
        fig_rmse.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_rmse, st.session_state.theme_mode)
        st.plotly_chart(fig_rmse, width='stretch', theme=None)

    def _forecast_config(self, frequency: str) -> dict[str, str]:
        return {
            "half_hourly": {
                "data_key": "half_model_pred",
                "actual_data_key": "half_test_actual",
                "time_col": "DateTime",
                "prediction_col": "Conso_moy_predict",
                "actual_col": "Conso_moy",
                "actuals_file": "test_actuals_half_hourly.csv",
                "provided_file": "group_5_half_hourly_predict.csv",
                "prediction_label": "Predicted Average Consumption (kW)",
                "y_label": "Average Consumption (kW)",
                "title": "Final 48-Hour Forecast by Saved Model",
                "test_title": "Final Test Actual vs Prediction",
            },
            "daily": {
                "data_key": "daily_model_pred",
                "actual_data_key": "daily_test_actual",
                "time_col": "Date",
                "prediction_col": "Conso_kWh_predict",
                "actual_col": "Conso_kWh",
                "actuals_file": "test_actuals_daily.csv",
                "provided_file": "group_5_daily_predict.csv",
                "prediction_label": "Predicted Daily Consumption (kWh)",
                "y_label": "Daily Consumption (kWh)",
                "title": "Final Daily Forecast by Saved Model",
                "test_title": "Final Test Actual vs Prediction",
            },
        }[frequency]

    def _test_plot_frame(self, merged: pd.DataFrame, config: dict[str, str]) -> pd.DataFrame:
        actual_plot = merged[[config["time_col"], "Acorn", config["actual_col"]]].drop_duplicates().copy()
        actual_plot = actual_plot.rename(columns={config["actual_col"]: "value"})
        actual_plot["series_label"] = "Actual"
        actual_plot["series_type"] = "Actual"

        prediction_plot = merged[[config["time_col"], "Acorn", "model", config["prediction_col"]]].copy()
        prediction_plot = prediction_plot.rename(columns={config["prediction_col"]: "value"})
        prediction_plot["series_label"] = prediction_plot["model"].map(self._model_label)
        prediction_plot["series_type"] = "Prediction"

        return pd.concat([actual_plot, prediction_plot], ignore_index=True, sort=False)

    def _test_metric_table(self, merged: pd.DataFrame, config: dict[str, str]) -> pd.DataFrame:
        rows = []
        for model, model_df in merged.groupby("model"):
            errors = model_df[config["prediction_col"]] - model_df[config["actual_col"]]
            rows.append(
                {
                    "model": model,
                    "rmse": (errors.pow(2).mean()) ** 0.5,
                    "mae": errors.abs().mean(),
                    "bias": errors.mean(),
                    "n": len(model_df),
                }
            )
        return pd.DataFrame(rows).sort_values(["rmse", "model"], na_position="last")

    def _format_test_metric_table(self, summary: pd.DataFrame, frequency: str) -> str:
        rows_html = ""
        for _, row in summary.iterrows():
            model_key = row["model"]
            rmse_val = f"{row['rmse']:.6f}" if pd.notna(row['rmse']) else "n/a"
            mae_val = f"{row['mae']:.6f}" if pd.notna(row['mae']) else "n/a"
            bias_val = f"{row['bias']:.6f}" if pd.notna(row['bias']) else "n/a"
            rows_count = f"{int(row['n'])}" if pd.notna(row['n']) else "0"
            
            nice_model_name = self._model_label(model_key)
            if model_key in {"previous_day", "previous_week", "seasonal_mean"}:
                model_display = f'<span style="font-weight: 500; color: #c85a64;">{nice_model_name}</span>'
            else:
                code_color = "#4da4a9" if frequency == "half_hourly" else "#d49c5e"
                model_display = f'<code style="font-family: monospace; font-size: 0.8rem; background: var(--kpi-border); color: {code_color}; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">{nice_model_name}</code>'
                
            rows_html += (
                f'<tr>'
                f'<td style="vertical-align: middle; padding: 10px 14px;">{model_display}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; font-weight: 600; color: var(--theme-text);">{rmse_val}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--theme-text);">{mae_val}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--theme-text);">{bias_val}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--kpi-sub);">{rows_count}</td>'
                f'</tr>'
            )
            
        html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th>Model</th>'
            '<th>Test RMSE</th>'
            '<th>Test MAE</th>'
            '<th>Test Bias</th>'
            '<th>Test Rows</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )
        return html

    def _forecast_metric_table(
        self,
        forecast_df: pd.DataFrame,
        metrics: pd.DataFrame,
        frequency: str,
        prediction_col: str,
    ) -> str:
        forecast_summary = forecast_df.groupby("model", as_index=False).agg(
            forecast_mean=(prediction_col, "mean"),
            forecast_min=(prediction_col, "min"),
            forecast_max=(prediction_col, "max"),
        )
        metric_subset = metrics[(metrics["frequency"] == frequency) & (metrics["acorn"] == "ALL")][
            ["model", "rmse", "n"]
        ]
        summary = forecast_summary.merge(metric_subset, on="model", how="left")
        summary = summary.sort_values(["rmse", "model"], na_position="last")
        
        rows_html = ""
        for _, row in summary.iterrows():
            model_key = row["model"]
            rmse_val = f"{row['rmse']:.6f}" if pd.notna(row['rmse']) else "n/a"
            rows_count = f"{int(row['n'])}" if pd.notna(row['n']) else "0"
            f_mean = f"{row['forecast_mean']:.3f}" if pd.notna(row['forecast_mean']) else "n/a"
            f_min = f"{row['forecast_min']:.3f}" if pd.notna(row['forecast_min']) else "n/a"
            f_max = f"{row['forecast_max']:.3f}" if pd.notna(row['forecast_max']) else "n/a"
            
            nice_model_name = self._model_label(model_key)
            if model_key in {"previous_day", "previous_week", "seasonal_mean"}:
                model_display = f'<span style="font-weight: 500; color: #c85a64;">{nice_model_name}</span>'
            else:
                code_color = "#4da4a9" if frequency == "half_hourly" else "#d49c5e"
                model_display = f'<code style="font-family: monospace; font-size: 0.8rem; background: var(--kpi-border); color: {code_color}; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">{nice_model_name}</code>'
                
            rows_html += (
                f'<tr>'
                f'<td style="vertical-align: middle; padding: 10px 14px;">{model_display}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; font-weight: 600; color: var(--theme-text);">{rmse_val}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--theme-text);">{f_mean}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--kpi-sub);">{f_min} – {f_max}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--kpi-sub);">{rows_count}</td>'
                f'</tr>'
            )
            
        html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th>Model</th>'
            '<th>Validation RMSE</th>'
            '<th>Forecast Mean</th>'
            '<th>Forecast Range (Min – Max)</th>'
            '<th>Validation Rows</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )
        return html

    def _validation_metric_table(self, metrics: pd.DataFrame, models: list[str], frequency: str) -> str:
        summary = metrics[
            (metrics["frequency"] == frequency)
            & (metrics["acorn"] == "ALL")
            & (metrics["model"].isin(models))
        ][["model", "rmse", "n"]].copy()
        summary = summary.sort_values(["rmse", "model"], na_position="last")
        
        rows_html = ""
        for _, row in summary.iterrows():
            model_key = row["model"]
            rmse_val = f"{row['rmse']:.6f}" if pd.notna(row['rmse']) else "n/a"
            rows_count = f"{int(row['n'])}" if pd.notna(row['n']) else "0"
            
            nice_model_name = self._model_label(model_key)
            if model_key in {"previous_day", "previous_week", "seasonal_mean"}:
                model_display = f'<span style="font-weight: 500; color: #c85a64;">{nice_model_name}</span>'
            else:
                code_color = "#4da4a9" if frequency == "half_hourly" else "#d49c5e"
                model_display = f'<code style="font-family: monospace; font-size: 0.8rem; background: var(--kpi-border); color: {code_color}; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">{nice_model_name}</code>'
                
            rows_html += (
                f'<tr>'
                f'<td style="vertical-align: middle; padding: 10px 14px;">{model_display}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; font-weight: 600; color: var(--theme-text);">{rmse_val}</td>'
                f'<td style="vertical-align: middle; padding: 10px 14px; color: var(--kpi-sub);">{rows_count}</td>'
                f'</tr>'
            )
            
        html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th>Model</th>'
            '<th>Validation RMSE</th>'
            '<th>Validation Rows</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )
        return html

    def _ordered_models(self, models: list[str], metrics: pd.DataFrame, frequency: str) -> list[str]:
        metric_subset = metrics[(metrics["frequency"] == frequency) & (metrics["acorn"] == "ALL")].copy()
        order = metric_subset.sort_values("rmse")["model"].tolist()
        return [model for model in order if model in models] + [model for model in models if model not in order]

    def _model_label(self, model: str) -> str:
        return StyleManager.MODEL_MAP.get(model, model)
