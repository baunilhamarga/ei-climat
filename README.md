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
export GROUP5_AUTOGLUON_TS_PRESETS=fast_training
export GROUP5_AUTOGLUON_TS_HALF_HOURLY_TIME_LIMIT=300
export GROUP5_AUTOGLUON_TS_DAILY_TIME_LIMIT=120
```

To run only a subset of trainable models, set:

```bash
export GROUP5_TRAINABLE_MODELS=ridge,xgboost,catboost,lightgbm
```

This is useful on Python environments where AutoGluon is not available.

The TimeSeries model is compared on the same chronological validation subset and the same RMSE metric as the other models. It uses the target history plus known future calendar, holiday, and weather covariates instead of the manual lag/rolling feature table.

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
