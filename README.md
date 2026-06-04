# EI Climat - Energy Forecasting

Code for the "Enseignement d'intégration : Analyse de la consommation électrique par la Data Science".

Our group forecasts UK household electricity consumption for:

- `ACORN-E` - Affluent
- `ACORN-F` - Comfortable
- `ACORN-Q` - Adversity

The project includes exploratory analysis, validation metrics, filled forecast templates, a short report, and a Streamlit dashboard.
The modeling pipeline compares baselines, ridge regression, XGBoost, CatBoost, LightGBM, AutoGluon Tabular AutoML, and AutoGluon TimeSeries.

## Quick Setup

Use the existing project virtual environment for the full local setup:

```bash
EI-climat/bin/pip install -r requirements.txt
```

This installs the complete project environment, including notebooks, sklearn models, CatBoost, LightGBM, and both AutoGluon models.

For lighter installs, use the split requirement files:

```bash
# Streamlit dashboard only, recommended for Render
EI-climat/bin/pip install -r requirements-dashboard.txt

# Local modeling and notebooks without the full frozen environment
EI-climat/bin/pip install -r requirements-modeling.txt
```

Generate the forecasts, metrics, figures, and report:

```bash
EI-climat/bin/python scripts/group5_run_pipeline.py
```

This keeps the client-provided data files unchanged. It writes cleaned/intermediate data to `data/01_interim/group5`, model-ready feature files to `data/02_processed/group5_modeling`, and saved model pipelines under `models/short_term` and `models/medium_term`. The pipeline saves every final trainable model by name, plus a `selected` copy for the model used to produce the final forecasts.

AutoGluon is pinned to the latest stable `1.5.0` release checked for this project. The pipeline includes both `autogluon.tabular` and `autogluon.timeseries`.

The pipeline auto-detects the usable CPU count from CPU quota and affinity, then passes that count to XGBoost, CatBoost, LightGBM, and AutoGluon where supported. If PyTorch can see CUDA GPUs, AutoGluon Tabular also receives the detected GPU count. Override these only when you want to pin resources manually:

```bash
export GROUP5_NUM_CPUS=6
export GROUP5_MODEL_NUM_CPUS=6
export GROUP5_AUTOGLUON_NUM_CPUS=6
export GROUP5_AUTOGLUON_NUM_GPUS=0
```

Training defaults can be adjusted before running the pipeline with:

```bash
export GROUP5_AUTOGLUON_PRESETS=medium_quality
export GROUP5_AUTOGLUON_HALF_HOURLY_TIME_LIMIT=300
export GROUP5_AUTOGLUON_DAILY_TIME_LIMIT=120
export GROUP5_AUTOGLUON_TS_PRESETS=medium_quality
export GROUP5_AUTOGLUON_TS_HALF_HOURLY_TIME_LIMIT=300
export GROUP5_AUTOGLUON_TS_DAILY_TIME_LIMIT=120
```

Native model progress/log output is enabled by default alongside the Group 5 heartbeat messages. To quiet the model libraries and keep only the custom pipeline progress, run with:

```bash
export GROUP5_NATIVE_MODEL_PROGRESS=0
```

You can also tune individual libraries:

```bash
export GROUP5_AUTOGLUON_VERBOSITY=2
export GROUP5_AUTOGLUON_TS_VERBOSITY=2
export GROUP5_CATBOOST_VERBOSE=50
export GROUP5_LIGHTGBM_VERBOSITY=1
export GROUP5_XGBOOST_VERBOSITY=1
```

For a mid-effort AutoGluon-only refresh that leaves the already-trained non-AutoGluon models alone and updates the dashboard artifacts, run:

```bash
GROUP5_AUTOGLUON_PRESETS=high_quality \
GROUP5_AUTOGLUON_HALF_HOURLY_TIME_LIMIT=900 \
GROUP5_AUTOGLUON_DAILY_TIME_LIMIT=300 \
GROUP5_AUTOGLUON_NUM_BAG_FOLDS=5 \
GROUP5_AUTOGLUON_NUM_STACK_LEVELS=1 \
GROUP5_AUTOGLUON_DYNAMIC_STACKING=0 \
GROUP5_AUTOGLUON_VERBOSITY=2 \
GROUP5_AUTOGLUON_TS_VERBOSITY=2 \
GROUP5_AUTOGLUON_TS_PRESETS=medium_quality \
GROUP5_AUTOGLUON_TS_HALF_HOURLY_TIME_LIMIT=900 \
GROUP5_AUTOGLUON_TS_DAILY_TIME_LIMIT=300 \
GROUP5_AUTOGLUON_TS_NUM_VAL_WINDOWS=3 \
EI-climat/bin/python scripts/group5_update_ag_models.py
```

`autogluon_timeseries` validation uses rolling final-horizon chunks: 96 half-hourly steps for the 48-hour task, and the available daily validation horizon for the daily task. This avoids scoring the half-hourly TimeSeries model on a single 28-day direct forecast when the assignment only asks it to forecast 48 hours. The mid-effort command disables AutoGluon Tabular dynamic stacking (`GROUP5_AUTOGLUON_DYNAMIC_STACKING=0`) because DyStack starts Ray and can emit harmless metrics-exporter errors in this local environment; explicit bagging and one stack level are still enabled.

For a high-cost AutoGluon experiment that does not overwrite the mid-effort AutoGluon rows, run the separate best-run updater. It writes `autogluon_best` and `autogluon_timeseries_best` model directories, metrics, validation predictions, and final-horizon predictions, then lets them compete on the dashboard against the existing models:

```bash
GROUP5_AUTOGLUON_NUM_GPUS=1 \
GROUP5_AUTOGLUON_PRESETS=best_quality \
GROUP5_AUTOGLUON_HALF_HOURLY_TIME_LIMIT=0 \
GROUP5_AUTOGLUON_DAILY_TIME_LIMIT=0 \
GROUP5_AUTOGLUON_NUM_BAG_FOLDS=10 \
GROUP5_AUTOGLUON_NUM_BAG_SETS=3 \
GROUP5_AUTOGLUON_NUM_STACK_LEVELS=2 \
GROUP5_AUTOGLUON_DYNAMIC_STACKING=1 \
GROUP5_AUTOGLUON_FIT_WEIGHTED_ENSEMBLE=1 \
GROUP5_AUTOGLUON_VERBOSITY=3 \
GROUP5_AUTOGLUON_TS_PRESETS=best_quality \
GROUP5_AUTOGLUON_TS_HALF_HOURLY_TIME_LIMIT=0 \
GROUP5_AUTOGLUON_TS_DAILY_TIME_LIMIT=0 \
GROUP5_AUTOGLUON_TS_NUM_VAL_WINDOWS=5 \
GROUP5_AUTOGLUON_TS_REFIT_FULL=1 \
GROUP5_AUTOGLUON_TS_ENABLE_ENSEMBLE=1 \
GROUP5_AUTOGLUON_TS_VERBOSITY=3 \
EI-climat/bin/python scripts/group5_update_ag_best_models.py
```

The `0` time limits mean the wrapper does not pass an internal AutoGluon time cap. Let this run finish cleanly so the final model directories and dashboard artifacts are written.

To run only a subset of trainable models, set:

```bash
export GROUP5_TRAINABLE_MODELS=ridge,xgboost,xgboost_by_acorn,catboost,lightgbm,stack_regressor
```

This is useful on Python environments where AutoGluon is not available. The `xgboost_by_acorn` option trains three isolated XGBoost models, one per ACORN segment, while the regular `xgboost` option trains one pooled model with ACORN as a categorical feature. The `stack_regressor` option uses ridge, XGBoost, and LightGBM base learners with a ridge meta-model.

The TimeSeries model is compared on the same chronological validation subset and the same RMSE metric as the other models, but half-hourly validation is rolled in 48-hour chunks to match the operational forecast horizon. It uses the target history plus known future calendar, holiday, and weather covariates instead of the manual lag/rolling feature table.

Validate the generated prediction files:

```bash
EI-climat/bin/python scripts/group5_validate_outputs.py
```

Run the dashboard:

```bash
EI-climat/bin/streamlit run dashboard/group5_streamlit.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

For Render, set the build command to:

```bash
pip install -r requirements-dashboard.txt
```

The deployed dashboard reads saved CSV metrics, predictions, and figures. It should not install AutoGluon because it does not train models at runtime.

## Main Outputs

- `notebooks/00_data_quality_irregularities.ipynb`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preparation_features.ipynb`
- `notebooks/03_short_term_modeling.ipynb`
- `notebooks/04_medium_term_modeling.ipynb`
- `data/01_interim/group5/`
- `data/02_processed/group5_modeling/`
- `models/short_term/group5_half_hourly_selected.joblib`
- `models/medium_term/group5_daily_selected.joblib`
- `outputs/group5/predictions/group_5_half_hourly_predict.csv`
- `outputs/group5/predictions/group_5_daily_predict.csv`
- `outputs/group5/predictions/group_5_half_hourly_all_models_predict.csv`
- `outputs/group5/predictions/group_5_daily_all_models_predict.csv`
- `outputs/group5/metrics/validation_metrics.csv`
- `outputs/group5/figures/`
- `reports/group5_report.md`

The filled prediction templates are written to `outputs/group5/predictions`; the original templates in `data/02_processed` are left unchanged.
