# EI Climat - Group 5 Energy Forecasting

Code for the "Enseignement d'intégration : Analyse de la consommation électrique par la Data Science".

Group 5 forecasts UK household electricity consumption for:

- `ACORN-E` - Affluent
- `ACORN-F` - Comfortable
- `ACORN-Q` - Adversity

The project includes exploratory analysis, validation metrics, filled forecast templates, a short report, and a Streamlit dashboard.

## Quick Setup

Use the existing project virtual environment:

```bash
EI-climat/bin/pip install -r requirements-group5.txt
```

Generate the forecasts, metrics, figures, and report:

```bash
EI-climat/bin/python scripts/group5_run_pipeline.py
```

This keeps the client-provided data files unchanged. It writes cleaned/intermediate data to `data/01_interim/group5`, model-ready feature files to `data/02_processed/group5_modeling`, and saved model pipelines under `models/short_term` and `models/medium_term`.

Validate the generated prediction files:

```bash
EI-climat/bin/python scripts/group5_validate_outputs.py
```

Run the dashboard:

```bash
EI-climat/bin/streamlit run dashboard/group5_streamlit.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

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
- `outputs/group5/metrics/validation_metrics.csv`
- `outputs/group5/figures/`
- `reports/group5_report.md`

The filled prediction templates are written to `outputs/group5/predictions`; the original templates in `data/02_processed` are left unchanged.
