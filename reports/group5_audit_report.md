# Group 5 Energy Forecasting Audit Report

Generated from the project artifacts on 2026-06-02 and updated after the 2026-06-04 AutoGluon refresh.

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

Standard random cross-validation was not used because the target is a time series. Random folds would mix past and future timestamps, which can leak future behavior into training and make the RMSE look better than it would be in a real forecast setting. A rolling or expanding-window temporal cross-validation would be a valid extension, but it would multiply training time across all trainable models. For this project, one fixed chronological holdout gives a direct and explainable test on the most recent historical period before the forecast window.

Compared models:

- `previous_day`: same ACORN value from the previous day.
- `previous_week`: same ACORN value from seven days earlier.
- `seasonal_mean`: historical mean by ACORN and calendar slot.
- `ridge`: regularized linear regression.
- `xgboost`: pooled gradient-boosted tree regression using calendar, weather, holiday, lag, and rolling features.
- `xgboost_by_acorn`: three isolated XGBoost models, one per ACORN, each trained and validated only on that ACORN history.
- `catboost`: CatBoost gradient boosting on the engineered feature table.
- `lightgbm`: LightGBM gradient boosting on the engineered feature table.
- `stack_regressor`: sklearn StackingRegressor with ridge, XGBoost, and LightGBM base learners and a ridge meta-model.
- `autogluon`: AutoGluon TabularPredictor on the engineered feature table.
- `autogluon_timeseries`: AutoGluon TimeSeriesPredictor using target history plus known future calendar, holiday, and weather covariates.

The sklearn-style model pipelines impute numeric and categorical features. Categorical features are one-hot encoded. The ridge model also scales transformed features. Predictions are clipped at zero to avoid invalid negative electricity values.

Selected models from current validation:

| Horizon | Selected model | Overall RMSE | Validation rows |
| --- | --- | ---: | ---: |
| Half-hourly | lightgbm | 0.007372 | 4032 |
| Daily | autogluon | 0.347658 | 93 |

Overall validation metrics:

| frequency | model | acorn | rmse | n |
| --- | --- | --- | --- | --- |
| daily | autogluon | ALL | 0.347658 | 93 |
| daily | xgboost_by_acorn | ALL | 0.352796 | 93 |
| daily | xgboost | ALL | 0.353151 | 93 |
| daily | catboost | ALL | 0.353230 | 93 |
| daily | lightgbm | ALL | 0.374496 | 93 |
| daily | ridge | ALL | 0.377257 | 93 |
| daily | stack_regressor | ALL | 0.395510 | 93 |
| daily | previous_day | ALL | 0.483489 | 93 |
| daily | autogluon_timeseries | ALL | 0.491688 | 93 |
| daily | previous_week | ALL | 0.494148 | 93 |
| daily | seasonal_mean | ALL | 1.537880 | 93 |
| half_hourly | lightgbm | ALL | 0.007372 | 4032 |
| half_hourly | autogluon | ALL | 0.007426 | 4032 |
| half_hourly | xgboost_by_acorn | ALL | 0.007953 | 4032 |
| half_hourly | catboost | ALL | 0.008033 | 4032 |
| half_hourly | xgboost | ALL | 0.008256 | 4032 |
| half_hourly | stack_regressor | ALL | 0.008294 | 4032 |
| half_hourly | ridge | ALL | 0.009935 | 4032 |
| half_hourly | autogluon_timeseries | ALL | 0.016361 | 4032 |
| half_hourly | previous_day | ALL | 0.021174 | 4032 |
| half_hourly | previous_week | ALL | 0.024217 | 4032 |
| half_hourly | seasonal_mean | ALL | 0.040723 | 4032 |

Trainable model metrics by segment:

| frequency | model | acorn | rmse | n |
| --- | --- | --- | --- | --- |
| daily | autogluon | ACORN-E | 0.370784 | 31 |
| daily | autogluon | ACORN-F | 0.406974 | 31 |
| daily | autogluon | ACORN-Q | 0.243903 | 31 |
| daily | stack_regressor | ACORN-E | 0.463011 | 31 |
| daily | stack_regressor | ACORN-F | 0.425364 | 31 |
| daily | stack_regressor | ACORN-Q | 0.271977 | 31 |
| daily | xgboost_by_acorn | ACORN-E | 0.369403 | 31 |
| daily | xgboost_by_acorn | ACORN-F | 0.399533 | 31 |
| daily | xgboost_by_acorn | ACORN-Q | 0.278049 | 31 |
| daily | xgboost | ACORN-E | 0.387952 | 31 |
| daily | xgboost | ACORN-F | 0.415606 | 31 |
| daily | xgboost | ACORN-Q | 0.225638 | 31 |
| half_hourly | autogluon | ACORN-E | 0.007334 | 1344 |
| half_hourly | autogluon | ACORN-F | 0.008315 | 1344 |
| half_hourly | autogluon | ACORN-Q | 0.006518 | 1344 |
| half_hourly | lightgbm | ACORN-E | 0.007126 | 1344 |
| half_hourly | lightgbm | ACORN-F | 0.008091 | 1344 |
| half_hourly | lightgbm | ACORN-Q | 0.006841 | 1344 |
| half_hourly | stack_regressor | ACORN-E | 0.008520 | 1344 |
| half_hourly | stack_regressor | ACORN-F | 0.008906 | 1344 |
| half_hourly | stack_regressor | ACORN-Q | 0.007380 | 1344 |
| half_hourly | xgboost_by_acorn | ACORN-E | 0.007948 | 1344 |
| half_hourly | xgboost_by_acorn | ACORN-F | 0.008804 | 1344 |
| half_hourly | xgboost_by_acorn | ACORN-Q | 0.007005 | 1344 |

AutoGluon and stack-regressor details:

1. The selected daily AutoGluon Tabular predictor is `WeightedEnsemble_L3_FULL`, the full-data refit of the validation-selected level-3 weighted ensemble.
2. The daily selected ensemble combines `CatBoost_BAG_L1` at 50.0%, `LightGBMXT_BAG_L1` at 27.3%, `CatBoost_BAG_L2` at 18.2%, and `RandomForestMSE_BAG_L2` at 4.5%.
3. The saved AutoGluon instances use these final prediction models: half-hourly `autogluon` uses `WeightedEnsemble_L2_FULL`; daily `autogluon` uses `WeightedEnsemble_L3_FULL`; half-hourly and daily `autogluon_best` use `WeightedEnsemble_L4`; mid-effort TimeSeries uses `WeightedEnsemble`; max-effort TimeSeries predicts with the best non-full `WeightedEnsemble` even though AutoGluon also created `_FULL` refits.
4. Runtime evidence: mid-effort Tabular was about 17 minutes per half-hourly fit and about 7 minutes per daily fit, estimated from saved predictor timestamps. Max-effort Tabular was about 43-45 minutes per fit. TimeSeries logs show about 8m18s/8m16s for half-hourly validation/final mid-effort, 2m21s/2m28s for daily validation/final mid-effort, 1h02m17s + 28s refit and 1h05m07s + 39s refit for half-hourly max-effort, and 27m28s + 3s refit and 29m24s + 3s refit for daily max-effort.
5. The stack regressor blends ridge, XGBoost, and LightGBM with a ridge meta-model. It performed competitively on the half-hourly task but did not beat LightGBM or AutoGluon.
6. AutoGluon TimeSeries was evaluated with rolling final-horizon validation, but it remained weaker than the tabular models in this benchmark.

Interpretation for report writing:

1. LightGBM has the best overall half-hourly RMSE, narrowly ahead of AutoGluon Tabular.
2. AutoGluon Tabular has the best overall daily RMSE, narrowly ahead of the isolated per-ACORN XGBoost models.
3. The stack regressor is useful as an ensemble benchmark, but the simple ridge meta-model did not outperform the strongest individual tree models.
4. The baseline comparison is important because it shows the trained models add value beyond simple repetition.

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
- `models/short_term/group5_half_hourly_lightgbm.joblib`
- `models/short_term/group5_half_hourly_stack_regressor.joblib`
- `models/short_term/group5_half_hourly_autogluon/`
- `models/medium_term/group5_daily_selected/`
- `models/medium_term/group5_daily_autogluon/`
- `models/medium_term/group5_daily_stack_regressor.joblib`
- `models/medium_term/group5_daily_xgboost_by_acorn.joblib`

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
| 01_daily_consumption_trend.pdf | historical daily trend and winter seasonality |
| 02_half_hourly_profile.pdf | typical day shape and evening peak |
| 03_weekly_pattern.pdf | weekday/weekend behavior |
| 04_weather_relationship.pdf | temperature link with consumption |
| 05_daily_autocorrelation.pdf | lag choice justification |
| 06_validation_rmse.pdf | model comparison |
| 07_final_half_hourly_forecast.pdf | short-term forecast output |
| 08_final_daily_forecast.pdf | medium-term forecast output |

## Dashboard and report artifacts

Dashboard:

- `dashboard/group5_streamlit.py`
- Reads saved metrics, predictions, and figures instead of retraining on page load.
- Tabs cover overview, EDA, validation, final forecasts, and ACORN comparison.

Concise report:

- `reports/group5_report.md`
- Good base for the final write-up, but this audit file contains more supporting detail.

## AutoGluon refresh notes

The AutoGluon-only refresh was run after the broader model benchmark so that tabular AutoML and time-series AutoML could be retrained without rerunning all non-AutoGluon models. The refresh updates `outputs/group5/metrics/validation_metrics.csv`, the all-model prediction files, and the selected daily model directory.

AutoGluon Tabular now wins the daily chronological validation benchmark with RMSE `0.347658`. The selected daily model is the full-data refit `WeightedEnsemble_L3_FULL`. It should be described as a stacked weighted ensemble, not as a single CatBoost or XGBoost model. The saved final daily Tabular fit took about 7m05s under the mid-effort settings; the max-effort daily Tabular run took about 45m22s and selected `WeightedEnsemble_L4`, but it did not beat the mid-effort daily Tabular validation score.

AutoGluon TimeSeries is still included in the comparison, but it is not selected. Its half-hourly validation uses rolling 96-step chunks to match the 48-hour assignment horizon. The max-effort TimeSeries runs did improve over the mid-effort TimeSeries scores, but final prediction uses the non-full `WeightedEnsemble` path because the `_FULL` ensemble referenced an unavailable Chronos fine-tuned checkpoint during prediction.

## Limits and risks to mention

1. Forecasts are point forecasts. They do not include uncertainty intervals.
2. Real weather data is used for the forecast period, following the project assumption.
3. Recursive forecasting means later predictions depend partly on earlier predicted values.
4. Most selected performance comes from tabular tree and ensemble methods. AutoGluon also tried neural-network and time-series variants, but the strongest final choices remained tabular.
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

The selected short-term model is LightGBM, with an overall half-hourly validation RMSE of 0.00737 over 4032 validation rows. The selected medium-term model is AutoGluon Tabular, with an overall daily validation RMSE of 0.34766 over 93 validation rows. The daily AutoGluon predictor is `WeightedEnsemble_L3_FULL`, a stacked weighted ensemble led by CatBoost and LightGBM components. Both selected models improve on simple repetition and seasonal-mean baselines. The final exported files contain 288 half-hourly predictions and 96 daily predictions, with no missing values and no duplicate ACORN/time keys.
