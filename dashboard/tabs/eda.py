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

        st.markdown(
            '<div class="intro-panel">'
            '<p>This tab checks the structure of the consumption data before modeling: long-term level, '
            'within-day and weekly rhythms, weather sensitivity, autocorrelation, and differences between '
            'the assigned ACORN segments. These patterns explain why the forecasting models use calendar, '
            'weather, lag, and rolling features.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Historical Consumption Levels")
        st.markdown(
            "The historical daily series shows the baseline level of each segment and the seasonal rise in "
            "winter demand. In this group, ACORN-E is generally the highest-consuming segment, ACORN-F is in "
            "the middle, and ACORN-Q is lower."
        )
        
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
        st.plotly_chart(fig, width='stretch', theme=None)

        st.subheader("Daily and Weekly Shape")
        st.markdown(
            "The profile charts summarize repeated behaviour rather than individual dates. The half-hourly "
            "profile captures the typical daily load curve, while the weekly profile shows whether weekday "
            "and weekend consumption differ by segment."
        )

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
        col1.plotly_chart(fig_half, width='stretch', theme=None)

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
        col2.plotly_chart(fig_weekly, width='stretch', theme=None)

        st.subheader("Weather Relationship")
        st.markdown(
            "Electricity demand is strongly linked to temperature in this dataset: colder days tend to "
            "correspond to higher daily consumption. The scatter plot shows that relationship directly, and "
            "the correlation matrix checks whether weather, weekend, and holiday variables carry useful signal."
        )

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
        col3.plotly_chart(fig_temp, width='stretch', theme=None)

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
        col4.plotly_chart(fig_corr, width='stretch', theme=None)

        st.subheader("Autocorrelation and Lag Evidence")
        st.markdown(
            "Autocorrelation measures how much today's consumption resembles previous days. Strong values "
            "around short lags and weekly lags justify features such as yesterday, last week, and rolling "
            "averages in the tabular models."
        )

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
        st.plotly_chart(fig_autocorr, width='stretch', theme=None)

        self._render_acorn_comparison(data, selected_acorns, filtered_daily)
        self._render_model_feature_rationale()

    def _render_acorn_comparison(
        self,
        data: dict[str, pd.DataFrame],
        selected_acorns: list[str],
        filtered_daily: pd.DataFrame,
    ) -> None:
        st.subheader("ACORN Segment Comparison")
        st.markdown(
            "This comparison keeps the segment story in the EDA context: historical daily levels show how "
            "the segments differ in the training data, while the forecast summary shows whether the final "
            "one-month predictions preserve the same ordering and scale."
        )

        daily_pred = data["daily_pred"][data["daily_pred"]["Acorn"].isin(selected_acorns)].copy()
        if filtered_daily.empty or daily_pred.empty:
            st.info("No ACORN comparison data is available for the current selection.")
            return

        historical_summary = (
            filtered_daily.groupby("Acorn")["Conso_kWh"]
            .agg(historical_mean="mean", historical_min="min", historical_max="max")
            .reset_index()
        )
        forecast_summary = (
            daily_pred.groupby("Acorn")["Conso_kWh_predict"]
            .agg(forecast_mean="mean", forecast_min="min", forecast_max="max")
            .reset_index()
        )
        summary = historical_summary.merge(forecast_summary, on="Acorn", how="outer").sort_values(
            "forecast_mean", ascending=False
        )

        summary_rows_html = ""
        for _, row in summary.iterrows():
            acorn = row["Acorn"]
            hist_mean = f"{row['historical_mean']:.3f}" if pd.notna(row['historical_mean']) else "n/a"
            hist_min = f"{row['historical_min']:.3f}" if pd.notna(row['historical_min']) else "n/a"
            hist_max = f"{row['historical_max']:.3f}" if pd.notna(row['historical_max']) else "n/a"
            
            fore_mean = f"{row['forecast_mean']:.3f}" if pd.notna(row['forecast_mean']) else "n/a"
            fore_min = f"{row['forecast_min']:.3f}" if pd.notna(row['forecast_min']) else "n/a"
            fore_max = f"{row['forecast_max']:.3f}" if pd.notna(row['forecast_max']) else "n/a"
            
            if acorn == "ACORN-E":
                badge_bg = "rgba(77, 164, 169, 0.12)"
                badge_text_color = "#4da4a9"
                border_color = "rgba(77, 164, 169, 0.25)"
            elif acorn == "ACORN-F":
                badge_bg = "rgba(212, 156, 94, 0.12)"
                badge_text_color = "#d49c5e"
                border_color = "rgba(212, 156, 94, 0.25)"
            else:
                badge_bg = "rgba(200, 90, 100, 0.12)"
                badge_text_color = "#c85a64"
                border_color = "rgba(200, 90, 100, 0.25)"
                
            summary_rows_html += (
                f'<tr>'
                f'<td>'
                f'<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; background: {badge_bg}; color: {badge_text_color}; border: 1px solid {border_color};">{acorn}</span>'
                f'</td>'
                f'<td>'
                f'<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">{hist_mean} kWh</div>'
                f'<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Range: {hist_min} – {hist_max}</div>'
                f'</td>'
                f'<td>'
                f'<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">{fore_mean} kWh</div>'
                f'<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Range: {fore_min} – {fore_max}</div>'
                f'</td>'
                f'</tr>'
            )

        summary_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th>ACORN Segment</th>'
            '<th>Historical Daily Usage</th>'
            '<th>Forecasted Daily Usage</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{summary_rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )

        col1, col2 = st.columns([3, 2], vertical_alignment="center")
        with col1:
            st.markdown(summary_html, unsafe_allow_html=True)
        with col2:
            fig_bar = px.bar(
                summary,
                x="Acorn",
                y="forecast_mean",
                color="Acorn",
                color_discrete_map=StyleManager.PALETTE,
                title="Mean Forecast Daily Consumption by Segment",
                labels={
                    "Acorn": "ACORN Segment",
                    "forecast_mean": "Mean Forecast Daily Consumption (kWh)",
                },
            )
            fig_bar.update_xaxes(categoryorder="total descending")
            StyleManager.style_plotly_chart(fig_bar, st.session_state.theme_mode)
            st.plotly_chart(fig_bar, width="stretch", theme=None)


    def _render_model_feature_rationale(self) -> None:
        st.subheader("Model Feature Set and Rationale")
        st.markdown(
            "The forecasting models are not trained directly on raw timestamps alone. The pipeline turns the "
            "historical consumption, ACORN segment, calendar, holiday, weather, and recent-consumption "
            "history into model-ready feature tables. Numeric columns are median-imputed when needed; "
            "categorical columns are filled with `missing` and one-hot encoded for the sklearn-style models. "
            "AutoGluon Tabular receives the same tabular feature columns."
        )

        tabular_rows = [
            {
                "Frequency": "Half-hourly",
                "Feature group": "Segment and scale",
                "Columns": "Acorn, Acorn_grouped, nb_clients",
                "Rationale": "ACORN identifies the customer segment and nb_clients controls for changes in the number of households represented by the segment average.",
            },
            {
                "Frequency": "Half-hourly",
                "Feature group": "Calendar rhythm",
                "Columns": "hour, minute, half_hour_slot, weekday, is_weekend, month, dayofyear, weekofyear",
                "Rationale": "Electricity demand has strong time-of-day, weekday-weekend, weekly, and seasonal patterns.",
            },
            {
                "Frequency": "Half-hourly",
                "Feature group": "Holiday",
                "Columns": "is_holiday",
                "Rationale": "Bank holidays can shift occupancy and therefore the daily load profile.",
            },
            {
                "Frequency": "Half-hourly",
                "Feature group": "Weather",
                "Columns": "temperature, temperature_half_hour, apparentTemperature, dewPoint, humidity, windSpeed, pressure, visibility, windBearing, icon, precipType",
                "Rationale": "Weather affects heating needs and behaviour. Temperature-like fields capture thermal demand, while humidity, wind, pressure, visibility, and weather labels add context on conditions.",
            },
            {
                "Frequency": "Half-hourly",
                "Feature group": "Recent target history",
                "Columns": "lag_1, lag_2, lag_48, lag_336, rolling_48_mean, rolling_336_mean",
                "Rationale": "The latest half-hours, the same slot yesterday, the same slot last week, and recent rolling averages capture persistence and weekly repetition.",
            },
            {
                "Frequency": "Daily",
                "Feature group": "Segment and scale",
                "Columns": "Acorn, nb_clients",
                "Rationale": "ACORN separates the three socioeconomic groups, while nb_clients keeps the target comparable when the segment sample size changes.",
            },
            {
                "Frequency": "Daily",
                "Feature group": "Calendar rhythm",
                "Columns": "weekday, is_weekend, month, dayofyear, weekofyear",
                "Rationale": "Daily consumption varies by weekday, weekend, week of year, and broader winter seasonality.",
            },
            {
                "Frequency": "Daily",
                "Feature group": "Holiday",
                "Columns": "is_holiday",
                "Rationale": "Public holidays can behave differently from ordinary weekdays and weekends.",
            },
            {
                "Frequency": "Daily",
                "Feature group": "Weather",
                "Columns": "temperatureMax, temperatureMin, temperatureHigh, temperatureLow, temperatureMean, temperatureRange, apparentTemperatureMax, apparentTemperatureMin, humidity, windSpeed, cloudCover, pressure, visibility, uvIndex, moonPhase, icon, precipType",
                "Rationale": "Daily weather aggregates explain changes in total daily demand, especially through temperature and apparent-temperature variables related to heating needs.",
            },
            {
                "Frequency": "Daily",
                "Feature group": "Recent target history",
                "Columns": "lag_1, lag_7, lag_14, rolling_7_mean, rolling_28_mean",
                "Rationale": "Yesterday, last week, two weeks ago, and recent rolling means summarize persistence, weekly recurrence, and short-term trend.",
            },
        ]
        tabular_rows_html = ""
        for row in tabular_rows:
            freq = row["Frequency"]
            feat_group = row["Feature group"]
            cols = row["Columns"]
            rationale = row["Rationale"]
            
            if freq.lower().startswith("half"):
                freq_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.78rem; background: rgba(77, 164, 169, 0.12); color: #4da4a9; border: 1px solid rgba(77, 164, 169, 0.25);">Half-Hourly</span>'
            else:
                freq_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.78rem; background: rgba(212, 156, 94, 0.12); color: #d49c5e; border: 1px solid rgba(212, 156, 94, 0.25);">Daily</span>'
                
            col_badges = ", ".join([f'<code style="font-family: monospace; font-size: 0.8rem; background: var(--kpi-border); color: var(--theme-text); padding: 2px 6px; border-radius: 4px; font-weight: 500;">{c.strip()}</code>' for c in cols.split(",")])
            
            tabular_rows_html += (
                f'<tr>'
                f'<td style="vertical-align: top; padding: 12px 14px;">{freq_badge}</td>'
                f'<td style="vertical-align: top; padding: 12px 14px;">'
                f'<div style="font-weight: 600; font-size: 0.92rem; color: var(--theme-text);">{feat_group}</div>'
                f'</td>'
                f'<td style="vertical-align: top; padding: 12px 14px;">'
                f'<div style="line-height: 1.6; margin-bottom: 4px;">{col_badges}</div>'
                f'</td>'
                f'<td style="vertical-align: top; padding: 12px 14px;">'
                f'<div style="font-size: 0.85rem; color: var(--kpi-sub); line-height: 1.45;">{rationale}</div>'
                f'</td>'
                f'</tr>'
            )

        tabular_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th style="width: 15%;">Frequency</th>'
            '<th style="width: 20%;">Feature Group</th>'
            '<th style="width: 35%;">Columns</th>'
            '<th style="width: 30%;">Rationale</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{tabular_rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(tabular_html, unsafe_allow_html=True)

        st.markdown(
            "AutoGluon TimeSeries is handled separately. It receives each ACORN as an item series, the "
            "historical target sequence, and known future numeric covariates. It does not receive the manual "
            "lag or rolling columns because the time-series models build their own sequence representation from "
            "the target history."
        )
        ts_rows = [
            {
                "Frequency": "Half-hourly",
                "Input type": "Target history",
                "Columns": "Acorn as item id, DateTime as timestamp, Conso_moy as target",
                "Rationale": "The model learns persistence, daily shape, and weekly repetition from the ordered consumption sequence.",
            },
            {
                "Frequency": "Half-hourly",
                "Input type": "Known future covariates",
                "Columns": "nb_clients, hour, minute, half_hour_slot, weekday, is_weekend, month, dayofyear, weekofyear, is_holiday, temperature, temperature_half_hour, apparentTemperature, dewPoint, humidity, windSpeed, pressure, visibility, windBearing",
                "Rationale": "These values are available for the forecast horizon and help the time-series model adjust its sequence forecast for calendar, holiday, and weather conditions.",
            },
            {
                "Frequency": "Daily",
                "Input type": "Target history",
                "Columns": "Acorn as item id, Date as timestamp, Conso_kWh as target",
                "Rationale": "The model learns segment-specific daily persistence and recurrence from the historical daily series.",
            },
            {
                "Frequency": "Daily",
                "Input type": "Known future covariates",
                "Columns": "nb_clients, weekday, is_weekend, month, dayofyear, weekofyear, is_holiday, temperatureMax, temperatureMin, temperatureHigh, temperatureLow, temperatureMean, temperatureRange, apparentTemperatureMax, apparentTemperatureMin, humidity, windSpeed, cloudCover, pressure, visibility, uvIndex, moonPhase",
                "Rationale": "These known future variables provide calendar and weather context for the one-month daily forecast.",
            },
        ]
        ts_rows_html = ""
        for row in ts_rows:
            freq = row["Frequency"]
            input_type = row["Input type"]
            cols = row["Columns"]
            rationale = row["Rationale"]
            
            if freq.lower().startswith("half"):
                freq_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.78rem; background: rgba(77, 164, 169, 0.12); color: #4da4a9; border: 1px solid rgba(77, 164, 169, 0.25);">Half-Hourly</span>'
            else:
                freq_badge = '<span style="display: inline-block; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.78rem; background: rgba(212, 156, 94, 0.12); color: #d49c5e; border: 1px solid rgba(212, 156, 94, 0.25);">Daily</span>'
                
            col_badges = ", ".join([f'<code style="font-family: monospace; font-size: 0.8rem; background: var(--kpi-border); color: var(--theme-text); padding: 2px 6px; border-radius: 4px; font-weight: 500;">{c.strip()}</code>' for c in cols.split(",")])
            
            ts_rows_html += (
                f'<tr>'
                f'<td style="vertical-align: top; padding: 12px 14px;">{freq_badge}</td>'
                f'<td style="vertical-align: top; padding: 12px 14px;">'
                f'<div style="font-weight: 600; font-size: 0.92rem; color: var(--theme-text);">{input_type}</div>'
                f'</td>'
                f'<td style="vertical-align: top; padding: 12px 14px;">'
                f'<div style="line-height: 1.6; margin-bottom: 4px;">{col_badges}</div>'
                f'</td>'
                f'<td style="vertical-align: top; padding: 12px 14px;">'
                f'<div style="font-size: 0.85rem; color: var(--kpi-sub); line-height: 1.45;">{rationale}</div>'
                f'</td>'
                f'</tr>'
            )

        ts_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th style="width: 15%;">Frequency</th>'
            '<th style="width: 20%;">Input Type</th>'
            '<th style="width: 35%;">Columns</th>'
            '<th style="width: 30%;">Rationale</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            f'{ts_rows_html}'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(ts_html, unsafe_allow_html=True)

        st.markdown(
            "The simple baselines are included for comparison rather than feature learning: previous-day and "
            "previous-week baselines reuse the corresponding lag values, while the seasonal mean baseline uses "
            "the historical average for the same ACORN and calendar slot or weekday."
        )
