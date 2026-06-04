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
        StyleManager.style_plotly_chart(fig_half, st.session_state.theme_mode)
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
        StyleManager.style_plotly_chart(fig_daily, st.session_state.theme_mode)
        st.plotly_chart(fig_daily, width='stretch')

        self._render_model_comparison(data, selected_acorns)

    def _render_model_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
    ) -> None:
        st.subheader("Saved Model Comparison")
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
        st.plotly_chart(fig, width='stretch')

        summary = self._test_metric_table(merged, config)
        display_summary = self._format_test_metric_table(summary)
        st.markdown(f'<div class="scrollable-table-wrapper">{display_summary.to_html(index=False)}</div>', unsafe_allow_html=True)
        self._render_test_rmse_chart(summary)

    def _render_final_forecast_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
        frequency: str,
    ) -> None:
        config = self._forecast_config(frequency)
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
        st.plotly_chart(fig, width='stretch')

        summary = self._forecast_metric_table(filtered, data["metrics"], frequency, config["prediction_col"])
        st.markdown(f'<div class="scrollable-table-wrapper">{summary.to_html(index=False)}</div>', unsafe_allow_html=True)
        self._render_rmse_chart(data["metrics"], selected_models, frequency)

    def _render_validation_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
        frequency: str,
    ) -> None:
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
        st.plotly_chart(fig, width='stretch')

        summary = self._validation_metric_table(data["metrics"], selected_models, frequency)
        st.markdown(f'<div class="scrollable-table-wrapper">{summary.to_html(index=False)}</div>', unsafe_allow_html=True)
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
        metric_subset["model_label"] = metric_subset["model"].map(self._model_label)
        fig_rmse = px.bar(
            metric_subset.sort_values("rmse", ascending=False),
            x="model_label",
            y="rmse",
            title="Validation RMSE for Displayed Models",
            labels={"model_label": "Model", "rmse": "Validation RMSE"},
        )
        fig_rmse.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_rmse, st.session_state.theme_mode)
        st.plotly_chart(fig_rmse, width='stretch')

    def _render_test_rmse_chart(self, summary: pd.DataFrame) -> None:
        if summary.empty:
            return
        chart_df = summary.copy()
        chart_df["model_label"] = chart_df["model"].map(self._model_label)
        fig_rmse = px.bar(
            chart_df.sort_values("rmse", ascending=False),
            x="model_label",
            y="rmse",
            title="Final Test RMSE for Displayed Models",
            labels={"model_label": "Model", "rmse": "Test RMSE"},
        )
        fig_rmse.update_xaxes(categoryorder="total descending")
        StyleManager.style_plotly_chart(fig_rmse, st.session_state.theme_mode)
        st.plotly_chart(fig_rmse, width='stretch')

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

    def _format_test_metric_table(self, summary: pd.DataFrame) -> pd.DataFrame:
        display = summary.copy()
        display["model"] = display["model"].map(self._model_label)
        display = display.rename(
            columns={
                "model": "Model",
                "rmse": "Test RMSE",
                "mae": "Test MAE",
                "bias": "Test Bias",
                "n": "Test Rows",
            }
        )
        for col in ["Test RMSE", "Test MAE", "Test Bias"]:
            display[col] = display[col].map(lambda value: "n/a" if pd.isna(value) else f"{value:.4f}")
        display["Test Rows"] = display["Test Rows"].map(lambda value: "n/a" if pd.isna(value) else f"{int(value)}")
        return display

    def _forecast_metric_table(
        self,
        forecast_df: pd.DataFrame,
        metrics: pd.DataFrame,
        frequency: str,
        prediction_col: str,
    ) -> pd.DataFrame:
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
        summary["model"] = summary["model"].map(self._model_label)
        summary = summary.rename(
            columns={
                "model": "Model",
                "rmse": "Validation RMSE",
                "n": "Validation Rows",
                "forecast_mean": "Forecast Mean",
                "forecast_min": "Forecast Min",
                "forecast_max": "Forecast Max",
            }
        )
        for col in ["Validation RMSE", "Forecast Mean", "Forecast Min", "Forecast Max"]:
            summary[col] = summary[col].map(lambda value: "n/a" if pd.isna(value) else f"{value:.4f}")
        summary["Validation Rows"] = summary["Validation Rows"].map(
            lambda value: "n/a" if pd.isna(value) else f"{int(value)}"
        )
        return summary

    def _validation_metric_table(self, metrics: pd.DataFrame, models: list[str], frequency: str) -> pd.DataFrame:
        summary = metrics[
            (metrics["frequency"] == frequency)
            & (metrics["acorn"] == "ALL")
            & (metrics["model"].isin(models))
        ][["model", "rmse", "n"]].copy()
        summary = summary.sort_values(["rmse", "model"], na_position="last")
        summary["model"] = summary["model"].map(self._model_label)
        summary = summary.rename(columns={"model": "Model", "rmse": "Validation RMSE", "n": "Validation Rows"})
        summary["Validation RMSE"] = summary["Validation RMSE"].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:.4f}"
        )
        summary["Validation Rows"] = summary["Validation Rows"].map(
            lambda value: "n/a" if pd.isna(value) else f"{int(value)}"
        )
        return summary

    def _ordered_models(self, models: list[str], metrics: pd.DataFrame, frequency: str) -> list[str]:
        metric_subset = metrics[(metrics["frequency"] == frequency) & (metrics["acorn"] == "ALL")].copy()
        order = metric_subset.sort_values("rmse")["model"].tolist()
        return [model for model in order if model in models] + [model for model in models if model not in order]

    def _model_label(self, model: str) -> str:
        return StyleManager.MODEL_MAP.get(model, model)
