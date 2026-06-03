# Group 5 Energy Forecasting Audit Report

Generated from the current project artifacts on 2026-06-02.

This file is meant to make the final report easier to write. It gathers the evidence we have already produced: data quality findings, preprocessing choices, feature design, validation results, model selection, final output checks, and the files that support each claim.

## Current status

The Group 5 pipeline is implemented and has generated predictions, metrics, figures, notebooks, saved models, and a concise report. The exported prediction files passed `scripts/group5_validate_outputs.py` during this audit.

Group 5 ACORN segments:

| ACORN | Label |
| --- | --- |
| ACORN-E | Affluent |
| ACORN-F | Comfortable |
| ACORN-Q | Adversity |

Forecast scope:

| Horizon | Period | Resolution | Expected rows | Prediction column |
| --- | --- | --- | ---: | --- |
| Short-term | 2014-01-13 00:00:00 to 2014-01-14 23:30:00 | 30 minutes | 288 | `Conso_moy_predict` |
| Medium-term | 2014-01-13 to 2014-02-13 | daily | 96 | `Conso_kWh_predict` |

## Reproducibility commands

Run from the repository root:

```bash
EI-climat/bin/pip install -r requirements-group5.txt
EI-climat/bin/python scripts/group5_run_pipeline.py
EI-climat/bin/python scripts/group5_validate_outputs.py
EI-climat/bin/streamlit run dashboard/group5_streamlit.py
```

The pipeline writes new generated files and does not overwrite the client-provided raw data or the original prediction templates.

## Data ownership and output policy

The client-provided files are treated as read-only inputs:

- `data/00_raw/`
- `data/02_processed/csv/`
- `data/02_processed/parquet/`

Generated data is written separately:

- `data/01_interim/group5/csv/` and `data/01_interim/group5/parquet/` for cleaned weather, holidays, and joined frames.
- `data/02_processed/group5_modeling/csv/` and `data/02_processed/group5_modeling/parquet/` for model-ready feature frames.
- `outputs/group5/predictions/` for filled prediction files.
- `outputs/group5/metrics/` for validation and EDA tables.
- `outputs/group5/figures/` for saved plots.
- `models/short_term/` and `models/medium_term/` for fitted scikit-learn pipelines.

## Data quality audit

Notebook `notebooks/00_data_quality_irregularities.ipynb` documents the discovery process. The main findings were:

1. Consumption targets parse cleanly and have no missing target values.
2. Consumption ACORN/timestamp keys are unique and the time series is regular enough for lag features.
3. Weather covariates have small gaps. Examples include hourly weather gaps, daily weather grid gaps, and half-hourly temperature nulls.
4. Daily weather has duplicate normalized dates around daylight-saving transitions when raw weather timestamps are converted to calendar dates.
5. Raw weather joins can multiply rows if daily weather duplicates are not handled before joining.
6. Some target values are high by simple outlier rules, but they match plausible winter demand and evening peaks.

Preprocessing decisions:

1. Keep all valid consumption target rows.
2. Do not interpolate, cap, or delete `Conso_moy` or `Conso_kWh` target values.
3. Sort daily weather by raw timestamp and keep the earliest record per normalized date.
4. Complete hourly, half-hourly, and daily weather grids before joining to consumption.
5. Interpolate numeric weather covariates over time for short gaps.
6. Forward/backward fill categorical weather labels such as `icon`, `summary`, and `precipType`.
7. Keep scikit-learn imputers inside the model pipeline as a fallback for unexpected missing covariates.
8. Preserve chronological validation so preprocessing does not leak future target values.

## Generated frame audit

| path | rows | columns | range | acorns | duplicate_key_rows | top_missing_counts |
| --- | --- | --- | --- | --- | --- | --- |
| data/01_interim/group5/csv/group_5_half_hourly_joined.csv | 80796 | 28 | 2012-06-30 22:00:00 to 2014-01-12 23:30:00 | ACORN-E, ACORN-F, ACORN-Q | 0 | Acorn: 0, Acorn_grouped: 0, DateTime: 0, nb_clients: 0, Conso_moy: 0 |
| data/01_interim/group5/csv/group_5_daily_joined.csv | 1683 | 28 | 2012-07-01 00:00:00 to 2014-01-12 00:00:00 | ACORN-E, ACORN-F, ACORN-Q | 0 | Acorn: 0, Date: 0, nb_pts: 0, nb_clients: 0, Conso_kWh: 0 |
| data/02_processed/group5_modeling/csv/group_5_half_hourly_features.csv | 80796 | 34 | 2012-06-30 22:00:00 to 2014-01-12 23:30:00 | ACORN-E, ACORN-F, ACORN-Q | 0 | lag_336: 1008, lag_48: 144, rolling_336_mean: 144, rolling_48_mean: 24, lag_2: 6 |
| data/02_processed/group5_modeling/csv/group_5_daily_features.csv | 1683 | 33 | 2012-07-01 00:00:00 to 2014-01-12 00:00:00 | ACORN-E, ACORN-F, ACORN-Q | 0 | lag_14: 42, rolling_28_mean: 21, lag_7: 21, rolling_7_mean: 6, lag_1: 3 |

The joined interim frames have no duplicate ACORN/time keys. The model-ready feature frames contain expected missing values in lag and rolling columns at the start of each ACORN history. These are not source-data defects; they are created by lag construction.

## Exploratory evidence

Daily consumption level by segment:

| Acorn | mean | min | max |
| --- | --- | --- | --- |
| ACORN-E | 10.248 | 7.087 | 16.050 |
| ACORN-F | 9.207 | 7.147 | 13.495 |
| ACORN-Q | 7.515 | 6.118 | 10.350 |

Half-hourly consumption level by segment:

| Acorn | mean | min | max |
| --- | --- | --- | --- |
| ACORN-E | 0.2135 | 0.0879 | 0.5002 |
| ACORN-F | 0.1918 | 0.0832 | 0.4357 |
| ACORN-Q | 0.1566 | 0.0712 | 0.3253 |

Temperature relationship:

| Acorn | temperature_consumption_corr |
| --- | --- |
| ACORN-E | -0.916 |
| ACORN-F | -0.885 |
| ACORN-Q | -0.868 |

Interpretation for report writing:

1. ACORN-E has the highest average consumption, ACORN-F sits in the middle, and ACORN-Q is lowest.
2. Daily demand rises in colder periods and falls in warmer periods.
3. The average half-hourly profile has low overnight demand and a higher evening period.
4. The strong negative temperature correlations support using weather variables.
5. Lag and rolling features are justified because consumption repeats across recent days and weeks.

## Feature design

Half-hourly model target: `Conso_moy`.

Half-hourly features include:

- ACORN segment and grouped ACORN label.
- Client count.
- Calendar fields: hour, minute, half-hour slot, weekday, weekend flag, month, day of year, week of year.
- Holiday flag.
- Hourly weather and half-hourly temperature.
- Lag features: `lag_1`, `lag_2`, `lag_48`, `lag_336`.
- Rolling features: `rolling_48_mean`, `rolling_336_mean`.

Daily model target: `Conso_kWh`.

Daily features include:

- ACORN segment.
- Client count.
- Calendar fields: weekday, weekend flag, month, day of year, week of year.
- Holiday flag.
- Daily weather features such as temperature, apparent temperature, humidity, wind, pressure, visibility, cloud cover, UV index, and moon phase.
- Lag features: `lag_1`, `lag_7`, `lag_14`.
- Rolling features: `rolling_7_mean`, `rolling_28_mean`.

The forecast step uses recursive lag updates. This means earlier model predictions can become lag inputs for later forecast timestamps.

## Modeling and validation

Validation is chronological, not shuffled.

| Horizon | Validation starts | Reason |
| --- | --- | --- |
| Half-hourly | 2013-12-16 00:00:00 | tests the 30-minute model on later historical data before the forecast period |
| Daily | 2013-12-13 | tests the daily model on later historical data before the forecast period |

Compared models:

- `previous_day`: same ACORN value from the previous day.
- `previous_week`: same ACORN value from seven days earlier.
- `seasonal_mean`: historical mean by ACORN and calendar slot.
- `ridge`: regularized linear regression.
- `gradient_boosting`: tree-based regression using histogram gradient boosting.

The model pipelines impute numeric and categorical features. Categorical features are one-hot encoded. The ridge model also scales transformed features. Predictions are clipped at zero to avoid invalid negative electricity values.

Selected models from current validation:

| Horizon | Selected model | Overall RMSE | Validation rows |
| --- | --- | ---: | ---: |
| Half-hourly | gradient_boosting | 0.007596 | 4032 |
| Daily | ridge | 0.377257 | 93 |

Overall validation metrics:

| frequency | model | acorn | rmse | n |
| --- | --- | --- | --- | --- |
| daily | ridge | ALL | 0.377257 | 93 |
| daily | gradient_boosting | ALL | 0.384853 | 93 |
| daily | previous_day | ALL | 0.483489 | 93 |
| daily | previous_week | ALL | 0.494148 | 93 |
| daily | seasonal_mean | ALL | 1.537880 | 93 |
| half_hourly | gradient_boosting | ALL | 0.007596 | 4032 |
| half_hourly | ridge | ALL | 0.009935 | 4032 |
| half_hourly | previous_day | ALL | 0.021174 | 4032 |
| half_hourly | previous_week | ALL | 0.024217 | 4032 |
| half_hourly | seasonal_mean | ALL | 0.040723 | 4032 |

Trainable model metrics by segment:

| frequency | model | acorn | rmse | n |
| --- | --- | --- | --- | --- |
| daily | gradient_boosting | ACORN-E | 0.417996 | 31 |
| daily | ridge | ACORN-E | 0.420502 | 31 |
| daily | ridge | ACORN-F | 0.424276 | 31 |
| daily | gradient_boosting | ACORN-F | 0.457392 | 31 |
| daily | gradient_boosting | ACORN-Q | 0.245779 | 31 |
| daily | ridge | ACORN-Q | 0.264832 | 31 |
| daily | ridge | ALL | 0.377257 | 93 |
| daily | gradient_boosting | ALL | 0.384853 | 93 |
| half_hourly | gradient_boosting | ACORN-E | 0.007536 | 1344 |
| half_hourly | ridge | ACORN-E | 0.010833 | 1344 |
| half_hourly | gradient_boosting | ACORN-F | 0.008292 | 1344 |
| half_hourly | ridge | ACORN-F | 0.010313 | 1344 |
| half_hourly | gradient_boosting | ACORN-Q | 0.006897 | 1344 |
| half_hourly | ridge | ACORN-Q | 0.008508 | 1344 |
| half_hourly | gradient_boosting | ALL | 0.007596 | 4032 |
| half_hourly | ridge | ALL | 0.009935 | 4032 |

Interpretation for report writing:

1. The half-hourly gradient boosting model clearly improves on previous-day, previous-week, seasonal mean, and ridge baselines.
2. The daily ridge model has the best overall daily RMSE, slightly ahead of gradient boosting.
3. Gradient boosting still performs best on ACORN-E and ACORN-Q in the daily split, but ridge wins overall because it is more stable across all three groups.
4. The baseline comparison is important because it shows the model adds value beyond simple repetition.

## Final prediction checks

Half-hourly output checks:

| check | actual | expected | status |
| --- | --- | --- | --- |
| row count | 288 | 288 | pass |
| ACORN segments | ACORN-E, ACORN-F, ACORN-Q | ACORN-E, ACORN-F, ACORN-Q | pass |
| forecast period | 2014-01-13 00:00:00 to 2014-01-14 23:30:00 | 2014-01-13 00:00:00 to 2014-01-14 23:30:00 | pass |
| missing predictions | 0 | 0 | pass |
| duplicate keys | 0 | 0 | pass |

Daily output checks:

| check | actual | expected | status |
| --- | --- | --- | --- |
| row count | 96 | 96 | pass |
| ACORN segments | ACORN-E, ACORN-F, ACORN-Q | ACORN-E, ACORN-F, ACORN-Q | pass |
| forecast period | 2014-01-13 to 2014-02-13 | 2014-01-13 to 2014-02-13 | pass |
| missing predictions | 0 | 0 | pass |
| duplicate keys | 0 | 0 | pass |

Final output files:

- `outputs/group5/predictions/group_5_half_hourly_predict.csv`
- `outputs/group5/predictions/group_5_half_hourly_predict.parquet`
- `outputs/group5/predictions/group_5_daily_predict.csv`
- `outputs/group5/predictions/group_5_daily_predict.parquet`

Saved models:

- `models/short_term/group5_half_hourly_selected.joblib`
- `models/short_term/group5_half_hourly_gradient_boosting.joblib`
- `models/medium_term/group5_daily_selected.joblib`
- `models/medium_term/group5_daily_ridge.joblib`
- `models/medium_term/group5_daily_gradient_boosting.joblib`

## Notebook map

| notebook | role |
| --- | --- |
| 00_data_quality_irregularities.ipynb | data irregularity discovery and local demo fix |
| 01_data_exploration.ipynb | EDA on segment levels, profiles, weather relationship |
| 02_data_preparation_features.ipynb | feature table construction and lag missingness review |
| 03_short_term_modeling.ipynb | 48-hour half-hourly model review and forecast inspection |
| 04_medium_term_modeling.ipynb | one-month daily model review and forecast inspection |

## Figure map

| figure | use_in_report |
| --- | --- |
| 01_daily_consumption_trend.png | historical daily trend and winter seasonality |
| 02_half_hourly_profile.png | typical day shape and evening peak |
| 03_weekly_pattern.png | weekday/weekend behavior |
| 04_weather_relationship.png | temperature link with consumption |
| 05_daily_autocorrelation.png | lag choice justification |
| 06_validation_rmse.png | model comparison |
| 07_final_half_hourly_forecast.png | short-term forecast output |
| 08_final_daily_forecast.png | medium-term forecast output |

## Dashboard and report artifacts

Dashboard:

- `dashboard/group5_streamlit.py`
- Reads saved metrics, predictions, and figures instead of retraining on page load.
- Tabs cover overview, EDA, validation, final forecasts, and ACORN comparison.

Concise report:

- `reports/group5_report.md`
- Good base for the final write-up, but this audit file contains more supporting detail.

## Pending AutoGluon extensions

After this audit was generated, the pipeline was extended with two AutoGluon trainable models:

- `autogluon`: AutoGluon TabularPredictor on the engineered feature table.
- `autogluon_timeseries`: AutoGluon TimeSeriesPredictor using target history plus known future calendar, holiday, and weather covariates.

The pipeline has not been rerun yet, so the metrics and selected-model results in this audit do not include either AutoGluon model.

Do not cite AutoGluon validation results until `EI-climat/bin/python scripts/group5_run_pipeline.py` has been rerun and `outputs/group5/metrics/validation_metrics.csv` contains both `autogluon` and `autogluon_timeseries` rows.

## Limits and risks to mention

1. Forecasts are point forecasts. They do not include uncertainty intervals.
2. Real weather data is used for the forecast period, following the project assumption.
3. Recursive forecasting means later predictions depend partly on earlier predicted values.
4. The project uses practical scikit-learn models, not deep learning. This is appropriate for the assignment size and makes the method easier to explain.
5. Target outliers were kept because they look plausible. This choice avoids underpredicting high winter and evening demand, but it also means the model must learn those peaks rather than ignore them.
6. Weather gaps were filled conservatively. This is better than dropping consumption rows, but interpolated weather values are still approximations.

## Suggested final report structure

1. Introduce Group 5 scope, ACORN segments, and forecast periods.
2. Explain data audit findings and why raw client files were not overwritten.
3. Summarize EDA: segment ranking, daily/weekly seasonality, half-hourly profile, temperature effect.
4. Describe preprocessing: weather grid completion, deterministic duplicate handling, holiday join, lag and rolling features.
5. Describe chronological validation and baseline comparisons.
6. Present RMSE table and selected models.
7. Interpret final short-term and medium-term forecasts.
8. Close with limits and possible improvements.

## Copy-ready short methodology paragraph

We used a reproducible Python pipeline for Group 5, covering ACORN-E, ACORN-F, and ACORN-Q. The raw client files were treated as read-only. Cleaned weather and joined intermediate frames were written to `data/01_interim/group5`, while model-ready feature tables were written to `data/02_processed/group5_modeling`. The preprocessing keeps all valid consumption targets, fixes weather irregularities before joining, creates calendar, holiday, weather, lag, and rolling features, and validates models on later historical windows without shuffling.

## Copy-ready short results paragraph

The selected short-term model is gradient boosting, with an overall half-hourly validation RMSE of 0.00760 over 4032 validation rows. The selected medium-term model is ridge regression, with an overall daily validation RMSE of 0.37726 over 93 validation rows. Both selected models improve on simple repetition and seasonal-mean baselines. The final exported files contain 288 half-hourly predictions and 96 daily predictions, with no missing values and no duplicate ACORN/time keys.
