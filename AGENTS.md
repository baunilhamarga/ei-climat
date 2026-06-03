# Codex Project Context

This repository contains the Group 5 energy forecasting project for the EI Climat assignment. Future Codex sessions should read this file first, then `README.md`, then `reports/group5_audit_report.md`.

## Project Scope

Group 5 ACORN segments:

- `ACORN-E`: Affluent
- `ACORN-F`: Comfortable
- `ACORN-Q`: Adversity

Forecast deliverables:

- Short-term half-hourly forecast from `2014-01-13 00:00:00` to `2014-01-14 23:30:00`.
- Medium-term daily forecast from `2014-01-13` to `2014-02-13`.

The filled prediction outputs must preserve these columns:

- Half-hourly: `Acorn`, `DateTime`, `Conso_moy_predict`
- Daily: `Acorn`, `Date`, `Conso_kWh_predict`

## User Preferences

- Keep notebook/report writing clear and a bit explanatory.
- In notebooks, explain the intention before code or plots, then write a short conclusion after plots/tables.
- Avoid over-polished AI-sounding prose. Use direct wording.
- Avoid flashy emphasis patterns in notebooks such as bold callouts, em dashes, and "key insight" style labels.
- Do not overwrite original client data.

## Data Ownership Rules

Treat these as read-only client or template inputs:

- `data/00_raw/`
- `data/02_processed/csv/`
- `data/02_processed/parquet/`

Generated project data belongs in:

- `data/01_interim/group5/` for cleaned weather, cleaned holidays, and joined intermediate frames.
- `data/02_processed/group5_modeling/` for model-ready feature frames.
- `outputs/group5/` for predictions, metrics, and figures.
- `models/short_term/` and `models/medium_term/` for fitted model files.

If preprocessing needs to be rerun, use the project pipeline. Do not write cleaned replacements into the original raw/template folders.

## Main Commands

Use the existing virtual environment:

```bash
EI-climat/bin/pip install -r requirements.txt
EI-climat/bin/python scripts/group5_run_pipeline.py
EI-climat/bin/python scripts/group5_validate_outputs.py
EI-climat/bin/streamlit run dashboard/group5_streamlit.py
```

The validation command should print:

```text
Group 5 output validation passed.
```

## Current Model State

Current run summary from `outputs/group5/metrics/run_summary.json`:

- Half-hourly output rows: `288`
- Daily output rows: `96`
- Selected half-hourly model: `gradient_boosting`
- Selected daily model: `ridge`
- Best half-hourly validation RMSE: `0.007596` over `4032` validation rows
- Best daily validation RMSE: `0.377257` over `93` validation rows
- Parquet output is available

Saved selected model paths:

- `models/short_term/group5_half_hourly_selected.joblib`
- `models/medium_term/group5_daily_selected.joblib`

## Important Files

Source and scripts:

- `src/group5_energy/config.py`
- `src/group5_energy/pipeline.py`
- `scripts/group5_run_pipeline.py`
- `scripts/group5_validate_outputs.py`
- `dashboard/group5_streamlit.py`

Notebooks:

- `notebooks/00_data_quality_irregularities.ipynb`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preparation_features.ipynb`
- `notebooks/03_short_term_modeling.ipynb`
- `notebooks/04_medium_term_modeling.ipynb`

Reports:

- `reports/group5_report.md`
- `reports/group5_audit_report.md`

Final predictions:

- `outputs/group5/predictions/group_5_half_hourly_predict.csv`
- `outputs/group5/predictions/group_5_half_hourly_predict.parquet`
- `outputs/group5/predictions/group_5_daily_predict.csv`
- `outputs/group5/predictions/group_5_daily_predict.parquet`

## Data Quality Decisions

The detailed audit is in `notebooks/00_data_quality_irregularities.ipynb` and `reports/group5_audit_report.md`.

Current production choices:

- Keep all valid consumption target rows.
- Do not interpolate, cap, or delete `Conso_moy` or `Conso_kWh`.
- Resolve daily weather duplicate normalized dates by sorting on raw weather timestamp and keeping the earliest record per date.
- Complete hourly, half-hourly, and daily weather grids before joining to consumption.
- Interpolate numeric weather covariates over time for short gaps.
- Forward/backward fill categorical weather labels such as `icon`, `summary`, and `precipType`.
- Keep sklearn imputers in the model pipeline as a fallback.
- Use chronological validation only. Do not shuffle time series rows.

Expected lag missingness remains in model-ready feature files at the start of each ACORN history. This is normal and not a source-data defect.

## Notebook Plot Rendering

Avoid plain `plt.show()` in notebooks because this environment can emit:

```text
FigureCanvasAgg is non-interactive, and thus cannot be shown
```

Use the existing `show_plot(fig)` helper pattern in plotting notebooks. It displays the figure through IPython and closes it afterward.

## Worktree Notes

The user may have local changes. Never revert files unless explicitly asked.

Treat those as user/project changes and work with them.

## Best Next Step For Future Sessions

For any report-writing or presentation task, start from:

1. `reports/group5_audit_report.md`
2. `reports/group5_report.md`
3. The saved figures in `outputs/group5/figures/`
4. The notebooks, especially `00` for data quality and `03`/`04` for modeling interpretation

For any modeling or output task, rerun:

```bash
EI-climat/bin/python scripts/group5_run_pipeline.py
EI-climat/bin/python scripts/group5_validate_outputs.py
```

Then check `outputs/group5/metrics/run_summary.json`.
