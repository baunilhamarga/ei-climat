from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group5_energy.config import ACORNS, METRICS_DIR, PREDICTION_DIR
from group5_energy.pipeline import (
    AUTOGLUON_TABULAR_MODEL,
    AUTOGLUON_TIMESERIES_MODEL,
    BASELINE_MODEL_NAMES,
    CATBOOST_MODEL,
    DAILY_TARGET,
    DAILY_VALIDATION_START,
    HALF_HOURLY_VALIDATION_START,
    HALF_TARGET,
    LIGHTGBM_MODEL,
    STACK_REGRESSOR_MODEL,
    XGBOOST_BY_ACORN_MODEL,
    autogluon_timeseries_validation_horizon,
    autogluon_timeseries_validation_predictions,
    autogluon_validation_path,
    best_model,
    clip_predictions,
    feature_matrix,
    forecast_horizon_length,
    forecast_with_model,
    make_model,
    metric_record,
    model_dir,
    model_stem,
    read_model_ready_frame,
    reset_path,
    write_all_model_prediction_tables,
    write_prediction_table,
    write_table,
)

TARGETED_TRAINABLE_MODELS = (
    "ridge",
    "xgboost",
    XGBOOST_BY_ACORN_MODEL,
    CATBOOST_MODEL,
    LIGHTGBM_MODEL,
    STACK_REGRESSOR_MODEL,
    AUTOGLUON_TABULAR_MODEL,
    AUTOGLUON_TIMESERIES_MODEL,
)

AUTOGLUON_BEST_TABULAR_MODEL = "autogluon_best"
AUTOGLUON_BEST_TIMESERIES_MODEL = "autogluon_timeseries_best"
AUTOGLUON_EXPERIMENT_MODELS = (AUTOGLUON_BEST_TABULAR_MODEL, AUTOGLUON_BEST_TIMESERIES_MODEL)
MODEL_ALIAS_BASE = {
    AUTOGLUON_BEST_TABULAR_MODEL: AUTOGLUON_TABULAR_MODEL,
    AUTOGLUON_BEST_TIMESERIES_MODEL: AUTOGLUON_TIMESERIES_MODEL,
}
SELECTABLE_TRAINABLE_MODELS = (*TARGETED_TRAINABLE_MODELS, *AUTOGLUON_EXPERIMENT_MODELS)
AUTOGLUON_MODELS = {AUTOGLUON_TABULAR_MODEL, AUTOGLUON_TIMESERIES_MODEL}
MODEL_ORDER = (*BASELINE_MODEL_NAMES, *SELECTABLE_TRAINABLE_MODELS)


def main(model_name: str | None = None, output_model_name: str | None = None) -> None:
    if model_name is None:
        parser = argparse.ArgumentParser(description="Update one saved Group 5 model in dashboard artifacts.")
        parser.add_argument("model", choices=SELECTABLE_TRAINABLE_MODELS)
        parser.add_argument(
            "--output-model-name",
            help="Optional dashboard/artifact name to write instead of the training model name.",
        )
        args = parser.parse_args()
        model_name = args.model
        output_model_name = args.output_model_name

    base_model_name, dashboard_model_name = resolve_update_names(model_name, output_model_name)
    print(
        f"Updating {dashboard_model_name} artifacts only "
        f"(training backend: {base_model_name}).",
        flush=True,
    )
    half_result = update_frequency(
        model_name=base_model_name,
        output_model_name=dashboard_model_name,
        frequency="half_hourly",
        feature_name="group_5_half_hourly_features",
        future_name="group_5_half_hourly_forecast_features",
        target_col=HALF_TARGET,
        time_col="DateTime",
        validation_start=pd.Timestamp(HALF_HOURLY_VALIDATION_START),
        validation_predictions_file="validation_predictions_half_hourly.csv",
        all_model_file="group_5_half_hourly_all_models_predict.csv",
        selected_file="group_5_half_hourly_predict.csv",
        selected_parquet="group_5_half_hourly_predict.parquet",
        prediction_col="Conso_moy_predict",
    )
    daily_result = update_frequency(
        model_name=base_model_name,
        output_model_name=dashboard_model_name,
        frequency="daily",
        feature_name="group_5_daily_features",
        future_name="group_5_daily_forecast_features",
        target_col=DAILY_TARGET,
        time_col="Date",
        validation_start=pd.Timestamp(DAILY_VALIDATION_START),
        validation_predictions_file="validation_predictions_daily.csv",
        all_model_file="group_5_daily_all_models_predict.csv",
        selected_file="group_5_daily_predict.csv",
        selected_parquet="group_5_daily_predict.parquet",
        prediction_col="Conso_kWh_predict",
    )
    write_all_model_prediction_tables(half_result["all_models"], daily_result["all_models"])
    metrics = write_updated_metrics([half_result["metric_rows"], daily_result["metric_rows"]])
    update_run_summary(metrics, half_result, daily_result)
    print(
        "Done. Selected models: "
        f"half-hourly={half_result['selected_model']}, daily={daily_result['selected_model']}.",
        flush=True,
    )


def resolve_update_names(model_name: str, output_model_name: str | None = None) -> tuple[str, str]:
    base_model_name = MODEL_ALIAS_BASE.get(model_name, model_name)
    if base_model_name not in TARGETED_TRAINABLE_MODELS:
        raise ValueError(f"Unsupported training model: {model_name}")
    dashboard_model_name = output_model_name or model_name
    validate_dashboard_model_name(dashboard_model_name)
    return base_model_name, dashboard_model_name


def validate_dashboard_model_name(model_name: str) -> None:
    if not model_name:
        raise ValueError("Output model name cannot be empty.")
    if model_name in BASELINE_MODEL_NAMES:
        raise ValueError(f"Output model name {model_name!r} conflicts with a baseline model.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(char not in allowed for char in model_name):
        raise ValueError(
            "Output model name may only contain letters, numbers, underscores, and hyphens. "
            f"Got {model_name!r}."
        )


def update_frequency(
    *,
    model_name: str,
    output_model_name: str,
    frequency: str,
    feature_name: str,
    future_name: str,
    target_col: str,
    time_col: str,
    validation_start: pd.Timestamp,
    validation_predictions_file: str,
    all_model_file: str,
    selected_file: str,
    selected_parquet: str,
    prediction_col: str,
) -> dict[str, object]:
    history = read_model_ready_frame(feature_name, frequency)
    future = read_model_ready_frame(future_name, frequency)
    train = history[history[time_col] < validation_start].copy()
    valid = history[history[time_col] >= validation_start].copy()
    if train.empty or valid.empty:
        raise ValueError(f"Invalid validation split for {frequency}.")

    print(f"[{frequency}] fitting validation {output_model_name}...", flush=True)
    validation_predictions = fit_validation_predictions(
        model_name,
        output_model_name,
        frequency,
        train,
        valid,
        target_col,
        time_col,
    )
    updated_valid = update_validation_predictions(
        model_name=output_model_name,
        frequency=frequency,
        valid=valid,
        time_col=time_col,
        predictions=validation_predictions,
        path=METRICS_DIR / validation_predictions_file,
    )
    metric_rows = new_metric_rows(updated_valid, output_model_name)

    print(f"[{frequency}] fitting final {output_model_name}...", flush=True)
    final_model = fit_final_update_model(model_name, output_model_name, frequency, history, future, target_col, time_col)

    print(f"[{frequency}] forecasting final horizon...", flush=True)
    forecast = forecast_with_model(
        final_model,
        model_name,
        history,
        future,
        frequency,
        target_col,
        prediction_col,
        time_col,
    )
    forecast.insert(0, "model", output_model_name)
    all_models = update_all_model_predictions(
        model_name=output_model_name,
        forecast=forecast,
        time_col=time_col,
        path=PREDICTION_DIR / all_model_file,
    )

    selected_model = selected_model_after_metric_update(frequency, metric_rows, extra_model_names=[output_model_name])
    selected = all_models[all_models["model"] == selected_model][["Acorn", time_col, prediction_col]].copy()
    if selected.empty:
        raise ValueError(f"Selected model {selected_model} is missing from {frequency} all-model predictions.")
    write_selected_predictions(
        selected=selected.reset_index(drop=True),
        time_col=time_col,
        csv_path=PREDICTION_DIR / selected_file,
        parquet_path=PREDICTION_DIR / selected_parquet,
    )
    sync_selected_model_artifact(frequency, selected_model)

    return {
        "frequency": frequency,
        "metric_rows": metric_rows,
        "all_models": all_models,
        "selected_model": selected_model,
        "updated_model": output_model_name,
    }


def fit_validation_predictions(
    model_name: str,
    output_model_name: str,
    frequency: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_col: str,
    time_col: str,
) -> pd.Series | list[float]:
    if model_name == AUTOGLUON_TIMESERIES_MODEL:
        prediction_length = autogluon_timeseries_validation_horizon(frequency, valid, time_col)
        model = make_model(
            model_name,
            frequency,
            autogluon_validation_path(frequency, output_model_name),
            prediction_length=prediction_length,
        )
        model.fit_history(train, target_col, time_col)
        return clip_predictions(
            autogluon_timeseries_validation_predictions(model, train, valid, frequency, target_col, time_col)
        )

    model_path = autogluon_validation_path(frequency, output_model_name) if model_name == AUTOGLUON_TABULAR_MODEL else None
    model = make_model(model_name, frequency, model_path)
    model.fit(feature_matrix(train, frequency), train[target_col])
    return clip_predictions(model.predict(feature_matrix(valid, frequency)))


def fit_final_update_model(
    model_name: str,
    output_model_name: str,
    frequency: str,
    history: pd.DataFrame,
    future: pd.DataFrame,
    target_col: str,
    time_col: str,
):
    model_path = update_model_storage_path(frequency, output_model_name, base_model_name=model_name)
    if model_name == AUTOGLUON_TIMESERIES_MODEL:
        model = make_model(
            model_name,
            frequency,
            model_path,
            prediction_length=forecast_horizon_length(future, time_col),
        )
        model.fit_history(history, target_col, time_col)
        return model

    model = make_model(model_name, frequency, model_path if model_name == AUTOGLUON_TABULAR_MODEL else None)
    model.fit(feature_matrix(history, frequency), history[target_col])
    if model_name not in AUTOGLUON_MODELS:
        joblib.dump(model, model_path)
    return model


def update_validation_predictions(
    *,
    model_name: str,
    frequency: str,
    valid: pd.DataFrame,
    time_col: str,
    predictions: pd.Series | list[float],
    path: Path,
) -> pd.DataFrame:
    existing = pd.read_csv(path, parse_dates=["timestamp"])
    new_col = valid[["Acorn", time_col]].copy()
    new_col = new_col.rename(columns={time_col: "timestamp"})
    new_col["timestamp"] = pd.to_datetime(new_col["timestamp"])
    if frequency == "daily":
        new_col["timestamp"] = new_col["timestamp"].dt.normalize()
    new_col[model_name] = predictions

    existing = existing.drop(columns=[model_name], errors="ignore")
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    if frequency == "daily":
        existing["timestamp"] = existing["timestamp"].dt.normalize()
    merged = existing.merge(new_col, on=["Acorn", "timestamp"], how="left")
    if merged[model_name].isna().any():
        missing = int(merged[model_name].isna().sum())
        raise ValueError(f"Missing {missing} validation predictions for {frequency}.")
    merged = order_validation_columns(merged)
    write_table(merged, path)
    return merged


def new_metric_rows(valid_predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = [metric_record(valid_predictions, model_name, "ALL")]
    for acorn in ACORNS:
        rows.append(metric_record(valid_predictions[valid_predictions["Acorn"] == acorn], model_name, acorn))
    return pd.DataFrame(rows)


def write_updated_metrics(metric_frames: list[pd.DataFrame]) -> pd.DataFrame:
    path = METRICS_DIR / "validation_metrics.csv"
    existing = pd.read_csv(path)
    new_metrics = pd.concat(metric_frames, ignore_index=True)
    key_pairs = set(zip(new_metrics["frequency"], new_metrics["model"]))
    keep_mask = ~existing[["frequency", "model"]].apply(tuple, axis=1).isin(key_pairs)
    metrics = pd.concat([existing[keep_mask], new_metrics], ignore_index=True)
    metrics = sort_metrics(metrics)
    write_table(metrics, path)
    return metrics


def update_all_model_predictions(*, model_name: str, forecast: pd.DataFrame, time_col: str, path: Path) -> pd.DataFrame:
    existing = pd.read_csv(path, parse_dates=[time_col])
    if time_col == "Date":
        existing[time_col] = pd.to_datetime(existing[time_col]).dt.normalize()
        forecast[time_col] = pd.to_datetime(forecast[time_col]).dt.normalize()
    else:
        existing[time_col] = pd.to_datetime(existing[time_col])
        forecast[time_col] = pd.to_datetime(forecast[time_col])
    existing = existing[existing["model"] != model_name].copy()
    out = pd.concat([existing, forecast], ignore_index=True, sort=False)
    out["_model_order"] = out["model"].map(model_order_map(extra_model_names=[model_name])).fillna(99)
    out = out.sort_values(["_model_order", "model", "Acorn", time_col]).drop(columns="_model_order")
    return out.reset_index(drop=True)


def selected_model_after_metric_update(
    frequency: str,
    new_rows: pd.DataFrame,
    extra_model_names: list[str] | None = None,
) -> str:
    existing = pd.read_csv(METRICS_DIR / "validation_metrics.csv")
    key_pairs = set(zip(new_rows["frequency"], new_rows["model"]))
    keep_mask = ~existing[["frequency", "model"]].apply(tuple, axis=1).isin(key_pairs)
    metrics = pd.concat([existing[keep_mask], new_rows], ignore_index=True)
    return selected_trainable_model_from_metrics(metrics, frequency, extra_model_names=extra_model_names)


def selected_trainable_model_from_metrics(
    metrics: pd.DataFrame,
    frequency: str,
    extra_model_names: list[str] | None = None,
) -> str:
    candidates = selectable_model_names(extra_model_names)
    subset = metrics[
        (metrics["frequency"] == frequency)
        & (metrics["acorn"] == "ALL")
        & (metrics["model"].isin(candidates))
    ].dropna(subset=["rmse"])
    if subset.empty:
        raise ValueError(f"No trainable model metrics found for {frequency}.")
    return str(subset.sort_values("rmse").iloc[0]["model"])


def sync_selected_model_artifact(frequency: str, model_name: str) -> None:
    source = existing_model_storage_path(frequency, model_name)
    if not source.exists():
        raise FileNotFoundError(f"Selected model source does not exist: {source}")

    selected_dir = model_dir(frequency) / f"{model_stem(frequency)}_selected"
    selected_joblib = model_dir(frequency) / f"{model_stem(frequency)}_selected.joblib"
    reset_path(selected_dir)
    reset_path(selected_joblib)

    destination = selected_dir if source.is_dir() else selected_joblib
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def write_selected_predictions(*, selected: pd.DataFrame, time_col: str, csv_path: Path, parquet_path: Path) -> None:
    csv_df = selected.copy()
    if time_col == "Date":
        csv_df[time_col] = pd.to_datetime(csv_df[time_col]).dt.strftime("%Y-%m-%d")
    else:
        csv_df[time_col] = pd.to_datetime(csv_df[time_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    write_prediction_table(csv_df, selected, csv_path, parquet_path)


def update_run_summary(metrics: pd.DataFrame, half_result: dict[str, object], daily_result: dict[str, object]) -> None:
    summary_path = METRICS_DIR / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    half_models = half_result["all_models"]
    daily_models = daily_result["all_models"]
    half_selected = str(half_result["selected_model"])
    daily_selected = str(daily_result["selected_model"])
    metric_models = metric_trainable_model_names(metrics)
    summary.update(
        {
            "half_hourly_rows": 288,
            "daily_rows": 96,
            "half_hourly_all_model_rows": int(len(half_models)),
            "daily_all_model_rows": int(len(daily_models)),
            "selected_half_hourly_model": half_selected,
            "selected_daily_model": daily_selected,
            "active_trainable_models": metric_models,
            "short_term_model_path": str(selected_artifact_path("half_hourly")),
            "medium_term_model_path": str(selected_artifact_path("daily")),
            "short_term_model_paths": model_paths("half_hourly", metric_models),
            "medium_term_model_paths": model_paths("daily", metric_models),
            "short_term_all_model_prediction_path": str(PREDICTION_DIR / "group_5_half_hourly_all_models_predict.csv"),
            "medium_term_all_model_prediction_path": str(PREDICTION_DIR / "group_5_daily_all_models_predict.csv"),
            "best_half_hourly_overall": best_model(metrics, "half_hourly"),
            "best_daily_overall": best_model(metrics, "daily"),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def selectable_model_names(extra_model_names: list[str] | None = None) -> tuple[str, ...]:
    names = list(SELECTABLE_TRAINABLE_MODELS)
    for model_name in extra_model_names or []:
        if model_name not in names:
            names.append(model_name)
    return tuple(names)


def metric_trainable_model_names(metrics: pd.DataFrame) -> list[str]:
    metric_models = set(metrics["model"].dropna().unique()) - set(BASELINE_MODEL_NAMES)
    ordered = [model for model in SELECTABLE_TRAINABLE_MODELS if model in metric_models]
    ordered.extend(sorted(metric_models - set(ordered)))
    return ordered


def model_order_map(extra_model_names: list[str] | None = None) -> dict[str, int]:
    return {model: idx for idx, model in enumerate((*BASELINE_MODEL_NAMES, *selectable_model_names(extra_model_names)))}


def update_model_storage_path(
    frequency: str,
    model_name: str,
    base_model_name: str | None = None,
    selected: bool = False,
) -> Path:
    suffix = "selected" if selected else model_name
    base = base_model_name or MODEL_ALIAS_BASE.get(model_name, model_name)
    if base in AUTOGLUON_MODELS:
        return model_dir(frequency) / f"{model_stem(frequency)}_{suffix}"
    return model_dir(frequency) / f"{model_stem(frequency)}_{suffix}.joblib"


def existing_model_storage_path(frequency: str, model_name: str) -> Path:
    directory_path = model_dir(frequency) / f"{model_stem(frequency)}_{model_name}"
    if directory_path.exists():
        return directory_path
    joblib_path = model_dir(frequency) / f"{model_stem(frequency)}_{model_name}.joblib"
    if joblib_path.exists():
        return joblib_path
    return update_model_storage_path(frequency, model_name)


def selected_artifact_path(frequency: str) -> Path:
    directory_path = model_dir(frequency) / f"{model_stem(frequency)}_selected"
    if directory_path.exists():
        return directory_path
    return model_dir(frequency) / f"{model_stem(frequency)}_selected.joblib"


def model_paths(frequency: str, model_names: list[str] | None = None) -> dict[str, str]:
    names = model_names or list(SELECTABLE_TRAINABLE_MODELS)
    return {name: str(existing_model_storage_path(frequency, name)) for name in names}


def sort_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    model_order = model_order_map(extra_model_names=metric_trainable_model_names(metrics))
    acorn_order = {"ALL": 0, "ACORN-E": 1, "ACORN-F": 2, "ACORN-Q": 3}
    frequency_order = {"half_hourly": 0, "daily": 1}
    out = metrics.copy()
    out["_frequency_order"] = out["frequency"].map(frequency_order).fillna(99)
    out["_model_order"] = out["model"].map(model_order).fillna(99)
    out["_acorn_order"] = out["acorn"].map(acorn_order).fillna(99)
    out = out.sort_values(["_frequency_order", "_model_order", "_acorn_order", "frequency", "model", "acorn"])
    return out.drop(columns=["_frequency_order", "_model_order", "_acorn_order"]).reset_index(drop=True)


def order_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    extra_models = [col for col in df.columns if col not in {"frequency", "timestamp", "Acorn", "actual", *MODEL_ORDER}]
    preferred = ["frequency", "timestamp", "Acorn", "actual", *MODEL_ORDER, *extra_models]
    columns = [col for col in preferred if col in df.columns]
    columns.extend(col for col in df.columns if col not in columns)
    return df[columns]


if __name__ == "__main__":
    main()
