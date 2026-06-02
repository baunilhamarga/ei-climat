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

- `outputs/group5/predictions/group_5_half_hourly_predict.csv`
- `outputs/group5/predictions/group_5_daily_predict.csv`
- `outputs/group5/metrics/validation_metrics.csv`
- `outputs/group5/figures/`
- `reports/group5_report.md`

The filled prediction templates are also written back to `data/02_processed/csv` and `data/02_processed/parquet`.
