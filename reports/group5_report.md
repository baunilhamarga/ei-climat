# Group 5 Energy Forecasting Report

## Project Scope

Group 5 covers `ACORN-E` (Affluent), `ACORN-F` (Comfortable), and `ACORN-Q` (Adversity). The historical consumption data runs up to `2014-01-12`, and the final outputs forecast:

- `2014-01-13 00:00` to `2014-01-14 23:30` at 30-minute resolution.
- `2014-01-13` to `2014-02-13` at daily resolution.

## Exploratory Findings

Daily consumption levels by ACORN:

| Acorn | mean | min | max |
| --- | --- | --- | --- |
| ACORN-E | 10.248 | 7.087 | 16.05 |
| ACORN-F | 9.207 | 7.147 | 13.495 |
| ACORN-Q | 7.515 | 6.118 | 10.35 |

Half-hourly consumption levels by ACORN:

| Acorn | mean | min | max |
| --- | --- | --- | --- |
| ACORN-E | 0.2135 | 0.0879 | 0.5002 |
| ACORN-F | 0.1918 | 0.0832 | 0.4357 |
| ACORN-Q | 0.1566 | 0.0712 | 0.3253 |

Correlation between daily mean temperature and daily consumption:

| Acorn | temperature_consumption_corr |
| --- | --- |
| ACORN-E | -0.916 |
| ACORN-F | -0.885 |
| ACORN-Q | -0.868 |

The visual outputs in `outputs/group5/figures` show three useful dynamics: ACORN-E has the highest average consumption, all three ACORN segments have stable daily and weekly seasonality, and short lags plus weekly repetition are strong reference points for forecasting.

## Modeling Approach

The validation strategy is chronological: the models train only on dates before the validation window and are tested on later historical data. The half-hourly validation window starts at `2013-12-16 00:00:00`. The daily validation window starts at `2013-12-13`.

The compared models are:

- `previous_day`: same ACORN value from the previous day.
- `previous_week`: same ACORN value from seven days earlier.
- `seasonal_mean`: historical mean by ACORN and calendar slot.
- `ridge`: regularized linear regression.
- `xgboost`: gradient-boosted tree regression using calendar, weather, holiday, lag, and rolling features.
- `xgboost_by_acorn`: three isolated XGBoost models, one per ACORN, each trained only on that ACORN history.
- `catboost`: CatBoost gradient boosting regression on the same engineered feature table.
- `lightgbm`: LightGBM gradient boosting regression on the same engineered feature table.
- `stack_regressor`: sklearn StackingRegressor using ridge, XGBoost, and LightGBM base learners with a ridge meta-model.
- `autogluon`: AutoGluon TabularPredictor AutoML regression on the same engineered feature table.
- `autogluon_timeseries`: AutoGluon TimeSeriesPredictor using the target history plus known future calendar, holiday, and weather covariates.

The stack regressor is a second ensemble benchmark built inside sklearn. It combines ridge, XGBoost, and LightGBM base learners, then fits a ridge meta-model on their predictions. This gives a useful check that a simple blended model can compete with the individual models, but it did not improve the final validation score in either horizon.

The selected daily AutoGluon Tabular model is `WeightedEnsemble_L3_FULL`. It is the refit version of AutoGluon's level-3 weighted ensemble, trained after the validation comparison on all available labeled daily history. In the saved daily predictor, the ensemble is mainly CatBoost and LightGBM-based: `CatBoost_BAG_L1` has 50.0% weight, `LightGBMXT_BAG_L1` 27.3%, `CatBoost_BAG_L2` 18.2%, and `RandomForestMSE_BAG_L2` 4.5%.

Saved AutoGluon prediction models and training-time evidence:

| horizon | dashboard model | final prediction model | validation fit | final fit | validation RMSE |
| --- | --- | --- | --- | --- | --- |
| half-hourly | `autogluon` | `WeightedEnsemble_L2_FULL` | 16m57s estimated | 16m47s estimated | 0.007426 |
| daily | `autogluon` | `WeightedEnsemble_L3_FULL` | 7m08s estimated | 7m05s estimated | 0.347658 |
| half-hourly | `autogluon_best` | `WeightedEnsemble_L4` | 42m38s estimated | 43m48s estimated | 0.007410 |
| daily | `autogluon_best` | `WeightedEnsemble_L4` | 45m22s estimated | 45m22s logged | 0.364007 |
| half-hourly | `autogluon_timeseries` | `WeightedEnsemble` | 8m18s logged | 8m16s logged | 0.016361 |
| daily | `autogluon_timeseries` | `WeightedEnsemble` | 2m21s logged | 2m28s logged | 0.491688 |
| half-hourly | `autogluon_timeseries_best` | `WeightedEnsemble` | 1h02m17s + 28s refit logged | 1h05m07s + 39s refit logged | 0.015508 |
| daily | `autogluon_timeseries_best` | `WeightedEnsemble` | 27m28s + 3s refit logged | 29m24s + 3s refit logged | 0.412653 |

The Tabular runtimes marked as estimates were reconstructed from saved predictor timestamps because the plain AutoGluon training logs were not preserved for every tabular run. The TimeSeries runtimes come from the saved AutoGluon predictor logs. The max-effort TimeSeries predictors created `_FULL` refits, but final predictions use the best non-full `WeightedEnsemble` because the default `_FULL` ensemble path referenced an unavailable Chronos fine-tuned checkpoint during prediction.

Overall validation RMSE:

| frequency | model | acorn | rmse | n |
| --- | --- | --- | --- | --- |
| daily | autogluon | ALL | 0.34766 | 93 |
| daily | xgboost_by_acorn | ALL | 0.35280 | 93 |
| daily | xgboost | ALL | 0.35315 | 93 |
| daily | catboost | ALL | 0.35323 | 93 |
| daily | lightgbm | ALL | 0.37450 | 93 |
| daily | ridge | ALL | 0.37726 | 93 |
| daily | stack_regressor | ALL | 0.39551 | 93 |
| daily | previous_day | ALL | 0.48349 | 93 |
| daily | autogluon_timeseries | ALL | 0.49169 | 93 |
| daily | previous_week | ALL | 0.49415 | 93 |
| daily | seasonal_mean | ALL | 1.53788 | 93 |
| half_hourly | lightgbm | ALL | 0.00737 | 4032 |
| half_hourly | autogluon | ALL | 0.00743 | 4032 |
| half_hourly | xgboost_by_acorn | ALL | 0.00795 | 4032 |
| half_hourly | catboost | ALL | 0.00803 | 4032 |
| half_hourly | xgboost | ALL | 0.00826 | 4032 |
| half_hourly | stack_regressor | ALL | 0.00829 | 4032 |
| half_hourly | ridge | ALL | 0.00993 | 4032 |
| half_hourly | autogluon_timeseries | ALL | 0.01636 | 4032 |
| half_hourly | previous_day | ALL | 0.02117 | 4032 |
| half_hourly | previous_week | ALL | 0.02422 | 4032 |
| half_hourly | seasonal_mean | ALL | 0.04072 | 4032 |

Selected final trainable models:

- Half-hourly: `lightgbm` with RMSE `0.00737`.
- Daily: `autogluon` with RMSE `0.34766`.

## Final Outputs

The filled forecast files are available in:

- `outputs/group5/predictions/group_5_half_hourly_predict.csv`
- `outputs/group5/predictions/group_5_daily_predict.csv`
- `outputs/group5/predictions/group_5_half_hourly_all_models_predict.csv`
- `outputs/group5/predictions/group_5_daily_all_models_predict.csv`

Cleaned weather and joined intermediate frames are written to `data/01_interim/group5`. Model-ready feature files are written to `data/02_processed/group5_modeling`. The original client-provided files under `data/00_raw` and the existing `data/02_processed/csv` and `data/02_processed/parquet` template files are treated as read-only inputs.

Final fitted trainable models are saved by name under `models/short_term` and `models/medium_term`. The selected forecast model is also saved with the `selected` suffix for compatibility with downstream tools.

## Limits

The forecasts use real weather data for the forecast period, as allowed by the assignment. Future lag features are generated recursively, so later daily predictions depend partly on earlier model predictions. The models are practical and reproducible, but they are not calibrated probabilistic forecasts and do not estimate uncertainty intervals.
