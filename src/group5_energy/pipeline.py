from __future__ import annotations

import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-group5")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from .config import (
    ACORN_GROUPS,
    ACORNS,
    DAILY_FORECAST_END,
    DAILY_FORECAST_START,
    DAILY_VALIDATION_START,
    FIGURES_DIR,
    HALF_HOURLY_FORECAST_END,
    HALF_HOURLY_FORECAST_START,
    HALF_HOURLY_VALIDATION_START,
    INTERIM_CSV_DIR,
    INTERIM_PARQUET_DIR,
    METRICS_DIR,
    MEDIUM_TERM_MODEL_DIR,
    MODEL_READY_CSV_DIR,
    MODEL_READY_PARQUET_DIR,
    PREDICTION_DIR,
    PROCESSED_CSV_DIR,
    PROCESSED_PARQUET_DIR,
    RANDOM_STATE,
    RAW_DIR,
    REPORT_DIR,
    SHORT_TERM_MODEL_DIR,
)


HALF_TARGET = "Conso_moy"
DAILY_TARGET = "Conso_kWh"

HALF_NUMERIC_FEATURES = [
    "nb_clients",
    "hour",
    "minute",
    "half_hour_slot",
    "weekday",
    "is_weekend",
    "month",
    "dayofyear",
    "weekofyear",
    "is_holiday",
    "temperature",
    "temperature_half_hour",
    "apparentTemperature",
    "dewPoint",
    "humidity",
    "windSpeed",
    "pressure",
    "visibility",
    "windBearing",
    "lag_1",
    "lag_2",
    "lag_48",
    "lag_336",
    "rolling_48_mean",
    "rolling_336_mean",
]

HALF_CATEGORICAL_FEATURES = ["Acorn", "Acorn_grouped", "icon", "precipType"]

DAILY_NUMERIC_FEATURES = [
    "nb_clients",
    "weekday",
    "is_weekend",
    "month",
    "dayofyear",
    "weekofyear",
    "is_holiday",
    "temperatureMax",
    "temperatureMin",
    "temperatureHigh",
    "temperatureLow",
    "temperatureMean",
    "temperatureRange",
    "apparentTemperatureMax",
    "apparentTemperatureMin",
    "humidity",
    "windSpeed",
    "cloudCover",
    "pressure",
    "visibility",
    "uvIndex",
    "moonPhase",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_28_mean",
]

DAILY_CATEGORICAL_FEATURES = ["Acorn", "icon", "precipType"]
AUTOGLUON_TABULAR_MODEL = "autogluon"
AUTOGLUON_TIMESERIES_MODEL = "autogluon_timeseries"
AUTOGLUON_MODEL_NAMES = (AUTOGLUON_TABULAR_MODEL, AUTOGLUON_TIMESERIES_MODEL)
CATBOOST_MODEL = "catboost"
LIGHTGBM_MODEL = "lightgbm"
TRAINABLE_MODEL_NAMES = (
    "ridge",
    "xgboost",
    CATBOOST_MODEL,
    LIGHTGBM_MODEL,
    AUTOGLUON_TABULAR_MODEL,
    AUTOGLUON_TIMESERIES_MODEL,
)
BASELINE_MODEL_NAMES = ("previous_day", "previous_week", "seasonal_mean")
AUTOGLUON_LABEL = "__target__"
TIME_SERIES_ID = "item_id"
TIME_SERIES_TIMESTAMP = "timestamp"
TIME_SERIES_TARGET = "target"

HALF_TIME_SERIES_KNOWN_COVARIATES = [
    "nb_clients",
    "hour",
    "minute",
    "half_hour_slot",
    "weekday",
    "is_weekend",
    "month",
    "dayofyear",
    "weekofyear",
    "is_holiday",
    "temperature",
    "temperature_half_hour",
    "apparentTemperature",
    "dewPoint",
    "humidity",
    "windSpeed",
    "pressure",
    "visibility",
    "windBearing",
]

DAILY_TIME_SERIES_KNOWN_COVARIATES = [
    "nb_clients",
    "weekday",
    "is_weekend",
    "month",
    "dayofyear",
    "weekofyear",
    "is_holiday",
    "temperatureMax",
    "temperatureMin",
    "temperatureHigh",
    "temperatureLow",
    "temperatureMean",
    "temperatureRange",
    "apparentTemperatureMax",
    "apparentTemperatureMin",
    "humidity",
    "windSpeed",
    "cloudCover",
    "pressure",
    "visibility",
    "uvIndex",
    "moonPhase",
]


@dataclass
class PipelineResult:
    half_hourly_predictions: pd.DataFrame
    daily_predictions: pd.DataFrame
    metrics: pd.DataFrame
    validation_predictions_half_hourly: pd.DataFrame
    validation_predictions_daily: pd.DataFrame


def run_pipeline() -> PipelineResult:
    ensure_output_dirs()

    half_history = load_half_hourly_history()
    daily_history = load_daily_history()
    half_template = load_half_hourly_template()
    daily_template = load_daily_template()
    weather_hourly = load_hourly_weather()
    temperatures = load_temperatures()
    weather_daily = load_daily_weather()
    holidays = load_holidays()
    write_clean_interim_sources(weather_hourly, temperatures, weather_daily, holidays)

    half_clients = latest_clients(half_history, "DateTime")
    daily_clients = latest_clients(daily_history, "Date")

    half_joined = prepare_half_hourly_frame(
        half_history, weather_hourly, temperatures, holidays, half_clients
    )
    daily_joined = prepare_daily_frame(daily_history, weather_daily, holidays, daily_clients)

    half_future = prepare_half_hourly_frame(
        half_template, weather_hourly, temperatures, holidays, half_clients
    )
    daily_future = prepare_daily_frame(daily_template, weather_daily, holidays, daily_clients)

    write_joined_interim_frames(half_joined, daily_joined, half_future, daily_future)

    half_features = add_history_lags(half_joined, HALF_TARGET, "half_hourly")
    daily_features = add_history_lags(daily_joined, DAILY_TARGET, "daily")
    write_model_ready_frames(half_features, daily_features, half_future, daily_future)

    half_features = read_model_ready_frame("group_5_half_hourly_features", "half_hourly")
    daily_features = read_model_ready_frame("group_5_daily_features", "daily")
    half_future = read_model_ready_frame("group_5_half_hourly_forecast_features", "half_hourly")
    daily_future = read_model_ready_frame("group_5_daily_forecast_features", "daily")

    half_model_name, half_metrics, half_valid_predictions = fit_and_evaluate(
        half_features,
        frequency="half_hourly",
        target_col=HALF_TARGET,
        time_col="DateTime",
        validation_start=pd.Timestamp(HALF_HOURLY_VALIDATION_START),
    )
    daily_model_name, daily_metrics, daily_valid_predictions = fit_and_evaluate(
        daily_features,
        frequency="daily",
        target_col=DAILY_TARGET,
        time_col="Date",
        validation_start=pd.Timestamp(DAILY_VALIDATION_START),
    )

    half_models = fit_final_models(
        "half_hourly",
        half_features,
        half_future,
        HALF_TARGET,
        "DateTime",
    )
    daily_models = fit_final_models(
        "daily",
        daily_features,
        daily_future,
        DAILY_TARGET,
        "Date",
    )
    save_models(half_models, daily_models, half_model_name, daily_model_name)
    half_model = half_models[half_model_name]
    daily_model = daily_models[daily_model_name]

    half_predictions = forecast_with_model(
        half_model,
        half_model_name,
        half_features,
        half_future,
        frequency="half_hourly",
        target_col=HALF_TARGET,
        prediction_col="Conso_moy_predict",
        time_col="DateTime",
    )
    daily_predictions = forecast_with_model(
        daily_model,
        daily_model_name,
        daily_features,
        daily_future,
        frequency="daily",
        target_col=DAILY_TARGET,
        prediction_col="Conso_kWh_predict",
        time_col="Date",
    )

    metrics = pd.concat([half_metrics, daily_metrics], ignore_index=True)
    write_core_outputs(
        half_predictions,
        daily_predictions,
        metrics,
        half_valid_predictions,
        daily_valid_predictions,
        half_model_name,
        daily_model_name,
    )
    write_eda_tables(half_features, daily_features)
    generate_figures(half_features, daily_features, half_predictions, daily_predictions, metrics)
    write_report(half_features, daily_features, metrics)
    validate_outputs(half_predictions, daily_predictions, metrics)

    return PipelineResult(
        half_hourly_predictions=half_predictions,
        daily_predictions=daily_predictions,
        metrics=metrics,
        validation_predictions_half_hourly=half_valid_predictions,
        validation_predictions_daily=daily_valid_predictions,
    )


def ensure_output_dirs() -> None:
    for directory in (
        PREDICTION_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        REPORT_DIR,
        SHORT_TERM_MODEL_DIR,
        MEDIUM_TERM_MODEL_DIR,
        INTERIM_CSV_DIR,
        INTERIM_PARQUET_DIR,
        MODEL_READY_CSV_DIR,
        MODEL_READY_PARQUET_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def model_dir(frequency: str) -> Path:
    if frequency == "half_hourly":
        return SHORT_TERM_MODEL_DIR
    if frequency == "daily":
        return MEDIUM_TERM_MODEL_DIR
    raise ValueError(f"Unsupported frequency: {frequency}")


def model_stem(frequency: str) -> str:
    if frequency == "half_hourly":
        return "group5_half_hourly"
    if frequency == "daily":
        return "group5_daily"
    raise ValueError(f"Unsupported frequency: {frequency}")


def model_storage_path(frequency: str, model_name: str, selected: bool = False) -> Path:
    suffix = "selected" if selected else model_name
    if model_name in AUTOGLUON_MODEL_NAMES:
        return model_dir(frequency) / f"{model_stem(frequency)}_{suffix}"
    return model_dir(frequency) / f"{model_stem(frequency)}_{suffix}.joblib"


def autogluon_validation_path(frequency: str, model_name: str) -> Path:
    return model_dir(frequency) / f"{model_stem(frequency)}_{model_name}_validation"


def active_trainable_model_names() -> tuple[str, ...]:
    raw_names = os.environ.get("GROUP5_TRAINABLE_MODELS", "").strip()
    if not raw_names:
        return TRAINABLE_MODEL_NAMES
    names = tuple(name.strip() for name in raw_names.split(",") if name.strip())
    unknown = sorted(set(names) - set(TRAINABLE_MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unsupported trainable model names in GROUP5_TRAINABLE_MODELS: {unknown}.")
    if not names:
        raise ValueError("GROUP5_TRAINABLE_MODELS did not contain any model names.")
    return names


def final_model_paths(frequency: str) -> dict[str, str]:
    return {
        model_name: str(model_storage_path(frequency, model_name, selected=False))
        for model_name in active_trainable_model_names()
    }


def save_models(
    half_models: dict[str, Any],
    daily_models: dict[str, Any],
    half_model_name: str,
    daily_model_name: str,
) -> None:
    save_final_models(half_models, "half_hourly", half_model_name)
    save_final_models(daily_models, "daily", daily_model_name)


def save_final_models(models: dict[str, Any], frequency: str, selected_model_name: str) -> None:
    if selected_model_name not in models:
        raise ValueError(f"Selected model {selected_model_name} was not fitted for {frequency}.")
    reset_stale_selected_model(frequency, selected_model_name)
    for model_name, model in models.items():
        save_named_final_model(model, frequency, model_name)
    save_selected_model(models[selected_model_name], frequency, selected_model_name)


def save_named_final_model(model: Any, frequency: str, model_name: str) -> None:
    named_path = model_storage_path(frequency, model_name, selected=False)
    if model_name in AUTOGLUON_MODEL_NAMES:
        if model.path.resolve() != named_path.resolve():
            model.copy_to(named_path)
        return
    joblib.dump(model, named_path)


def save_selected_model(model: Any, frequency: str, model_name: str) -> None:
    selected_path = model_storage_path(frequency, model_name, selected=True)
    if model_name in AUTOGLUON_MODEL_NAMES:
        model.copy_to(selected_path)
        return
    joblib.dump(model, selected_path)


def reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def reset_stale_selected_model(frequency: str, model_name: str) -> None:
    stem = model_stem(frequency)
    if model_name in AUTOGLUON_MODEL_NAMES:
        reset_path(model_dir(frequency) / f"{stem}_selected.joblib")
    else:
        reset_path(model_dir(frequency) / f"{stem}_selected")


def parquet_available() -> bool:
    return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None


def read_processed_table(name: str) -> pd.DataFrame:
    parquet_path = PROCESSED_PARQUET_DIR / f"{name}.parquet"
    csv_path = PROCESSED_CSV_DIR / f"{name}.csv"
    if parquet_available() and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def read_generated_table(name: str, csv_dir: Path, parquet_dir: Path) -> pd.DataFrame:
    parquet_path = parquet_dir / f"{name}.parquet"
    csv_path = csv_dir / f"{name}.csv"
    if parquet_available() and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def read_model_ready_frame(name: str, frequency: str) -> pd.DataFrame:
    df = read_generated_table(name, MODEL_READY_CSV_DIR, MODEL_READY_PARQUET_DIR)
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    if frequency == "half_hourly" and HALF_TARGET in df.columns:
        df[HALF_TARGET] = pd.to_numeric(df[HALF_TARGET], errors="coerce")
    if frequency == "daily" and DAILY_TARGET in df.columns:
        df[DAILY_TARGET] = pd.to_numeric(df[DAILY_TARGET], errors="coerce")
    return df


def write_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if parquet_path is not None and parquet_available():
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)


def write_generated_table(df: pd.DataFrame, name: str, csv_dir: Path, parquet_dir: Path) -> None:
    write_table(df, csv_dir / f"{name}.csv", parquet_dir / f"{name}.parquet")


def write_clean_interim_sources(
    weather_hourly: pd.DataFrame,
    temperatures: pd.DataFrame,
    weather_daily: pd.DataFrame,
    holidays: pd.DataFrame,
) -> None:
    write_generated_table(weather_hourly, "weather_hourly_clean", INTERIM_CSV_DIR, INTERIM_PARQUET_DIR)
    write_generated_table(temperatures, "temperatures_half_hour_clean", INTERIM_CSV_DIR, INTERIM_PARQUET_DIR)
    write_generated_table(weather_daily, "weather_daily_clean", INTERIM_CSV_DIR, INTERIM_PARQUET_DIR)
    write_generated_table(holidays, "uk_bank_holidays_clean", INTERIM_CSV_DIR, INTERIM_PARQUET_DIR)


def write_joined_interim_frames(
    half_joined: pd.DataFrame,
    daily_joined: pd.DataFrame,
    half_future: pd.DataFrame,
    daily_future: pd.DataFrame,
) -> None:
    write_generated_table(half_joined, "group_5_half_hourly_joined", INTERIM_CSV_DIR, INTERIM_PARQUET_DIR)
    write_generated_table(daily_joined, "group_5_daily_joined", INTERIM_CSV_DIR, INTERIM_PARQUET_DIR)
    write_generated_table(
        half_future,
        "group_5_half_hourly_forecast_joined",
        INTERIM_CSV_DIR,
        INTERIM_PARQUET_DIR,
    )
    write_generated_table(
        daily_future,
        "group_5_daily_forecast_joined",
        INTERIM_CSV_DIR,
        INTERIM_PARQUET_DIR,
    )


def write_model_ready_frames(
    half_features: pd.DataFrame,
    daily_features: pd.DataFrame,
    half_future: pd.DataFrame,
    daily_future: pd.DataFrame,
) -> None:
    write_generated_table(
        half_features,
        "group_5_half_hourly_features",
        MODEL_READY_CSV_DIR,
        MODEL_READY_PARQUET_DIR,
    )
    write_generated_table(daily_features, "group_5_daily_features", MODEL_READY_CSV_DIR, MODEL_READY_PARQUET_DIR)
    write_generated_table(
        half_future,
        "group_5_half_hourly_forecast_features",
        MODEL_READY_CSV_DIR,
        MODEL_READY_PARQUET_DIR,
    )
    write_generated_table(
        daily_future,
        "group_5_daily_forecast_features",
        MODEL_READY_CSV_DIR,
        MODEL_READY_PARQUET_DIR,
    )


def load_half_hourly_history() -> pd.DataFrame:
    df = read_processed_table("group_5_half_hourly")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.normalize()
    df[HALF_TARGET] = pd.to_numeric(df[HALF_TARGET], errors="coerce")
    df["nb_clients"] = pd.to_numeric(df["nb_clients"], errors="coerce")
    return df[df["Acorn"].isin(ACORNS)].sort_values(["Acorn", "DateTime"]).reset_index(drop=True)


def load_daily_history() -> pd.DataFrame:
    df = read_processed_table("group_5_daily")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df[DAILY_TARGET] = pd.to_numeric(df[DAILY_TARGET], errors="coerce")
    df["nb_clients"] = pd.to_numeric(df["nb_clients"], errors="coerce")
    return df[df["Acorn"].isin(ACORNS)].sort_values(["Acorn", "Date"]).reset_index(drop=True)


def load_half_hourly_template() -> pd.DataFrame:
    df = read_processed_table("group_5_half_hourly_predict")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.normalize()
    return df[df["Acorn"].isin(ACORNS)].sort_values(["Acorn", "DateTime"]).reset_index(drop=True)


def load_daily_template() -> pd.DataFrame:
    df = read_processed_table("group_5_daily_predict")
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df[df["Acorn"].isin(ACORNS)].sort_values(["Acorn", "Date"]).reset_index(drop=True)


def load_hourly_weather() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "weather_hourly_darksky.csv")
    df["WeatherHour"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    numeric_cols = [
        "visibility",
        "windBearing",
        "temperature",
        "dewPoint",
        "pressure",
        "apparentTemperature",
        "windSpeed",
        "humidity",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates("WeatherHour").sort_values("WeatherHour")
    return complete_time_series(
        df,
        time_col="WeatherHour",
        freq="h",
        numeric_cols=numeric_cols,
        categorical_cols=["precipType", "icon", "summary"],
    )


def load_temperatures() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "temperatures.csv", sep=";", decimal=",")
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="mixed")
    df = df.rename(columns={"Temperature": "temperature_half_hour"})
    df["temperature_half_hour"] = pd.to_numeric(df["temperature_half_hour"], errors="coerce")
    df = df.drop_duplicates("DateTime").sort_values("DateTime")
    return complete_time_series(
        df,
        time_col="DateTime",
        freq="30min",
        numeric_cols=["temperature_half_hour"],
        categorical_cols=[],
    )


def load_daily_weather() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "weather_daily_darksky.csv")
    df["WeatherTime"] = pd.to_datetime(df["time"])
    df["Date"] = df["WeatherTime"].dt.normalize()
    numeric_cols = [
        "temperatureMax",
        "dewPoint",
        "cloudCover",
        "windSpeed",
        "pressure",
        "visibility",
        "humidity",
        "apparentTemperatureHigh",
        "apparentTemperatureLow",
        "apparentTemperatureMax",
        "temperatureLow",
        "temperatureMin",
        "temperatureHigh",
        "uvIndex",
        "apparentTemperatureMin",
        "moonPhase",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("WeatherTime").drop_duplicates("Date", keep="first")
    df["temperatureMean"] = df[["temperatureMin", "temperatureMax"]].mean(axis=1)
    df["temperatureRange"] = df["temperatureMax"] - df["temperatureMin"]
    keep_cols = [
        "Date",
        "temperatureMax",
        "temperatureMin",
        "temperatureHigh",
        "temperatureLow",
        "temperatureMean",
        "temperatureRange",
        "apparentTemperatureMax",
        "apparentTemperatureMin",
        "humidity",
        "windSpeed",
        "cloudCover",
        "pressure",
        "visibility",
        "uvIndex",
        "moonPhase",
        "icon",
        "precipType",
    ]
    numeric_clean_cols = [
        col
        for col in keep_cols
        if col not in {"Date", "icon", "precipType"}
    ]
    return complete_time_series(
        df[keep_cols].sort_values("Date"),
        time_col="Date",
        freq="D",
        numeric_cols=numeric_clean_cols,
        categorical_cols=["icon", "precipType"],
    )


def complete_time_series(
    df: pd.DataFrame,
    time_col: str,
    freq: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Return a complete, sorted weather covariate series with conservative fills."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.drop_duplicates(time_col).sort_values(time_col).set_index(time_col)
    full_index = pd.date_range(out.index.min(), out.index.max(), freq=freq, name=time_col)
    out = out.reindex(full_index)
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].interpolate(method="time").ffill().bfill()
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].ffill().bfill().fillna("missing")
    return out.reset_index()


def load_holidays() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "uk_bank_holidays.csv")
    df["Date"] = pd.to_datetime(df["Bank holidays"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    df = df.rename(columns={"Type": "holiday_type"})
    df["is_holiday"] = 1
    return df[["Date", "is_holiday", "holiday_type"]].drop_duplicates("Date")


def latest_clients(df: pd.DataFrame, time_col: str) -> dict[str, float]:
    latest = df.sort_values(time_col).groupby("Acorn").tail(1)
    return latest.set_index("Acorn")["nb_clients"].to_dict()


def prepare_half_hourly_frame(
    df: pd.DataFrame,
    weather_hourly: pd.DataFrame,
    temperatures: pd.DataFrame,
    holidays: pd.DataFrame,
    latest_client_counts: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    out["Date"] = out["DateTime"].dt.normalize()
    out["Acorn_grouped"] = out["Acorn"].map(ACORN_GROUPS)
    if "nb_clients" not in out.columns:
        out["nb_clients"] = out["Acorn"].map(latest_client_counts)
    out["nb_clients"] = pd.to_numeric(out["nb_clients"], errors="coerce")
    out["WeatherHour"] = out["DateTime"].dt.floor("h")
    out = out.merge(weather_hourly, on="WeatherHour", how="left")
    out = out.merge(temperatures, on="DateTime", how="left")
    out = out.merge(holidays[["Date", "is_holiday"]], on="Date", how="left")
    out["is_holiday"] = out["is_holiday"].fillna(0).astype(int)
    out["hour"] = out["DateTime"].dt.hour
    out["minute"] = out["DateTime"].dt.minute
    out["half_hour_slot"] = out["hour"] * 2 + (out["minute"] // 30)
    out["weekday"] = out["DateTime"].dt.weekday
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["month"] = out["DateTime"].dt.month
    out["dayofyear"] = out["DateTime"].dt.dayofyear
    out["weekofyear"] = out["DateTime"].dt.isocalendar().week.astype(int)
    return out.sort_values(["Acorn", "DateTime"]).reset_index(drop=True)


def prepare_daily_frame(
    df: pd.DataFrame,
    weather_daily: pd.DataFrame,
    holidays: pd.DataFrame,
    latest_client_counts: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    if "nb_clients" not in out.columns:
        out["nb_clients"] = out["Acorn"].map(latest_client_counts)
    out["nb_clients"] = pd.to_numeric(out["nb_clients"], errors="coerce")
    out = out.merge(weather_daily, on="Date", how="left")
    out = out.merge(holidays[["Date", "is_holiday"]], on="Date", how="left")
    out["is_holiday"] = out["is_holiday"].fillna(0).astype(int)
    out["weekday"] = out["Date"].dt.weekday
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["month"] = out["Date"].dt.month
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype(int)
    return out.sort_values(["Acorn", "Date"]).reset_index(drop=True)


def add_history_lags(df: pd.DataFrame, target_col: str, frequency: str) -> pd.DataFrame:
    out = df.sort_values(["Acorn", "DateTime" if frequency == "half_hourly" else "Date"]).copy()
    grouped = out.groupby("Acorn", group_keys=False)[target_col]
    if frequency == "half_hourly":
        out["lag_1"] = grouped.shift(1)
        out["lag_2"] = grouped.shift(2)
        out["lag_48"] = grouped.shift(48)
        out["lag_336"] = grouped.shift(336)
        out["rolling_48_mean"] = grouped.apply(lambda x: x.shift(1).rolling(48, min_periods=8).mean())
        out["rolling_336_mean"] = grouped.apply(lambda x: x.shift(1).rolling(336, min_periods=48).mean())
    else:
        out["lag_1"] = grouped.shift(1)
        out["lag_7"] = grouped.shift(7)
        out["lag_14"] = grouped.shift(14)
        out["rolling_7_mean"] = grouped.apply(lambda x: x.shift(1).rolling(7, min_periods=2).mean())
        out["rolling_28_mean"] = grouped.apply(lambda x: x.shift(1).rolling(28, min_periods=7).mean())
    return out.reset_index(drop=True)


def feature_columns(frequency: str) -> tuple[list[str], list[str]]:
    if frequency == "half_hourly":
        return HALF_NUMERIC_FEATURES, HALF_CATEGORICAL_FEATURES
    return DAILY_NUMERIC_FEATURES, DAILY_CATEGORICAL_FEATURES


def feature_matrix(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    numeric_cols, categorical_cols = feature_columns(frequency)
    out = df.copy()
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = np.nan
    for col in categorical_cols:
        if col not in out.columns:
            out[col] = "missing"
        out[col] = out[col].fillna("missing").astype(str)
    return out[numeric_cols + categorical_cols]


def make_preprocessor(frequency: str) -> ColumnTransformer:
    numeric_cols, categorical_cols = feature_columns(frequency)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def make_tree_model(frequency: str) -> Pipeline:
    return Pipeline(
        [
            ("features", make_preprocessor(frequency)),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    n_estimators=350,
                    learning_rate=0.04,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    tree_method="hist",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                    verbosity=0,
                ),
            ),
        ]
    )


def make_catboost_model(frequency: str) -> Pipeline:
    return Pipeline(
        [
            ("features", make_preprocessor(frequency)),
            (
                "model",
                CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    iterations=500,
                    learning_rate=0.04,
                    depth=6,
                    l2_leaf_reg=3.0,
                    random_seed=RANDOM_STATE,
                    allow_writing_files=False,
                    verbose=False,
                    thread_count=1,
                ),
            ),
        ]
    )


def make_lightgbm_model(frequency: str) -> Pipeline:
    return Pipeline(
        [
            ("features", make_preprocessor(frequency)),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    metric="rmse",
                    n_estimators=500,
                    learning_rate=0.04,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def make_linear_model(frequency: str) -> Pipeline:
    return Pipeline(
        [
            ("features", make_preprocessor(frequency)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


class AutoGluonTabularModel:
    def __init__(self, frequency: str, path: Path):
        self.frequency = frequency
        self.path = path
        self.predictor = None
        self.presets = os.environ.get("GROUP5_AUTOGLUON_PRESETS", "medium_quality")
        self.time_limit = autogluon_time_limit(frequency)
        self.num_gpus = int(os.environ.get("GROUP5_AUTOGLUON_NUM_GPUS", "0"))
        self.verbosity = int(os.environ.get("GROUP5_AUTOGLUON_VERBOSITY", "0"))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "AutoGluonTabularModel":
        TabularPredictor = autogluon_predictor_class()
        train_data = x.copy()
        train_data[AUTOGLUON_LABEL] = pd.Series(y).to_numpy()
        reset_path(self.path)
        predictor = TabularPredictor(
            label=AUTOGLUON_LABEL,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            path=str(self.path),
            verbosity=self.verbosity,
        )
        fit_kwargs: dict[str, Any] = {
            "train_data": train_data,
            "presets": self.presets,
            "num_gpus": self.num_gpus,
        }
        if self.time_limit > 0:
            fit_kwargs["time_limit"] = self.time_limit
        num_cpus = os.environ.get("GROUP5_AUTOGLUON_NUM_CPUS")
        if num_cpus:
            fit_kwargs["num_cpus"] = int(num_cpus)
        predictor.fit(**fit_kwargs)
        self.predictor = predictor
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.predictor is None:
            TabularPredictor = autogluon_predictor_class()
            self.predictor = TabularPredictor.load(str(self.path))
        return np.asarray(self.predictor.predict(x.copy()), dtype=float)

    def copy_to(self, destination: Path) -> None:
        if self.path.resolve() == destination.resolve():
            return
        reset_path(destination)
        shutil.copytree(self.path, destination)


class AutoGluonTimeSeriesModel:
    def __init__(self, frequency: str, path: Path, prediction_length: int):
        self.frequency = frequency
        self.path = path
        self.prediction_length = prediction_length
        self.predictor = None
        self.presets = os.environ.get("GROUP5_AUTOGLUON_TS_PRESETS", "fast_training")
        self.time_limit = autogluon_timeseries_time_limit(frequency)
        self.verbosity = int(os.environ.get("GROUP5_AUTOGLUON_TS_VERBOSITY", "0"))

    def fit_history(self, history: pd.DataFrame, target_col: str, time_col: str) -> "AutoGluonTimeSeriesModel":
        TimeSeriesPredictor, _ = autogluon_timeseries_classes()
        train_data = time_series_data_frame(history, self.frequency, target_col, time_col, include_target=True)
        reset_path(self.path)
        predictor = TimeSeriesPredictor(
            target=TIME_SERIES_TARGET,
            known_covariates_names=time_series_known_covariates(self.frequency),
            prediction_length=self.prediction_length,
            freq=time_series_frequency(self.frequency),
            eval_metric="RMSE",
            path=str(self.path),
            verbosity=self.verbosity,
        )
        fit_kwargs: dict[str, Any] = {
            "train_data": train_data,
            "presets": self.presets,
            "random_seed": RANDOM_STATE,
        }
        if self.time_limit > 0:
            fit_kwargs["time_limit"] = self.time_limit
        predictor.fit(**fit_kwargs)
        self.predictor = predictor
        return self

    def predict_horizon(
        self,
        history: pd.DataFrame,
        future: pd.DataFrame,
        target_col: str,
        time_col: str,
    ) -> np.ndarray:
        if self.predictor is None:
            TimeSeriesPredictor, _ = autogluon_timeseries_classes()
            self.predictor = TimeSeriesPredictor.load(str(self.path))
        history_data = time_series_data_frame(history, self.frequency, target_col, time_col, include_target=True)
        known_covariates = time_series_data_frame(future, self.frequency, target_col, time_col, include_target=False)
        predictions = self.predictor.predict(history_data, known_covariates=known_covariates)
        return align_time_series_predictions(predictions, future, time_col)

    def copy_to(self, destination: Path) -> None:
        if self.path.resolve() == destination.resolve():
            return
        reset_path(destination)
        shutil.copytree(self.path, destination)


def autogluon_time_limit(frequency: str) -> int:
    if frequency == "half_hourly":
        return int(os.environ.get("GROUP5_AUTOGLUON_HALF_HOURLY_TIME_LIMIT", "300"))
    if frequency == "daily":
        return int(os.environ.get("GROUP5_AUTOGLUON_DAILY_TIME_LIMIT", "120"))
    raise ValueError(f"Unsupported frequency: {frequency}")


def autogluon_timeseries_time_limit(frequency: str) -> int:
    if frequency == "half_hourly":
        return int(os.environ.get("GROUP5_AUTOGLUON_TS_HALF_HOURLY_TIME_LIMIT", "300"))
    if frequency == "daily":
        return int(os.environ.get("GROUP5_AUTOGLUON_TS_DAILY_TIME_LIMIT", "120"))
    raise ValueError(f"Unsupported frequency: {frequency}")


def autogluon_predictor_class() -> Any:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon is required for the 'autogluon' model. "
            "Install dependencies with `EI-climat/bin/pip install -r requirements.txt`."
        ) from exc
    return TabularPredictor


def autogluon_timeseries_classes() -> tuple[Any, Any]:
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon TimeSeries is required for the 'autogluon_timeseries' model. "
            "Install dependencies with `EI-climat/bin/pip install -r requirements.txt`."
        ) from exc
    return TimeSeriesPredictor, TimeSeriesDataFrame


def time_series_known_covariates(frequency: str) -> list[str]:
    if frequency == "half_hourly":
        return HALF_TIME_SERIES_KNOWN_COVARIATES
    if frequency == "daily":
        return DAILY_TIME_SERIES_KNOWN_COVARIATES
    raise ValueError(f"Unsupported frequency: {frequency}")


def time_series_frequency(frequency: str) -> str:
    if frequency == "half_hourly":
        return "30min"
    if frequency == "daily":
        return "D"
    raise ValueError(f"Unsupported frequency: {frequency}")


def time_series_data_frame(
    df: pd.DataFrame,
    frequency: str,
    target_col: str,
    time_col: str,
    include_target: bool,
) -> Any:
    _, TimeSeriesDataFrame = autogluon_timeseries_classes()
    known_covariates = time_series_known_covariates(frequency)
    out = df[["Acorn", time_col]].copy()
    for col in known_covariates:
        out[col] = df[col] if col in df.columns else np.nan
    if include_target:
        out[target_col] = df[target_col]
    out = out.rename(columns={"Acorn": TIME_SERIES_ID, time_col: TIME_SERIES_TIMESTAMP, target_col: TIME_SERIES_TARGET})
    out[TIME_SERIES_TIMESTAMP] = pd.to_datetime(out[TIME_SERIES_TIMESTAMP])
    if frequency == "daily":
        out[TIME_SERIES_TIMESTAMP] = out[TIME_SERIES_TIMESTAMP].dt.normalize()
    for col in known_covariates:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out.groupby(TIME_SERIES_ID)[col].transform(lambda series: series.ffill().bfill())
        median = out[col].median()
        out[col] = out[col].fillna(0.0 if pd.isna(median) else median)
    if include_target:
        out[TIME_SERIES_TARGET] = pd.to_numeric(out[TIME_SERIES_TARGET], errors="coerce")
    out = out.sort_values([TIME_SERIES_ID, TIME_SERIES_TIMESTAMP])
    return TimeSeriesDataFrame.from_data_frame(out, id_column=TIME_SERIES_ID, timestamp_column=TIME_SERIES_TIMESTAMP)


def align_time_series_predictions(predictions: Any, expected: pd.DataFrame, time_col: str) -> np.ndarray:
    prediction_df = predictions.reset_index().rename(
        columns={TIME_SERIES_ID: "Acorn", TIME_SERIES_TIMESTAMP: time_col, "mean": "prediction"}
    )
    prediction_df[time_col] = pd.to_datetime(prediction_df[time_col])
    expected_keys = expected[["Acorn", time_col]].copy()
    expected_keys[time_col] = pd.to_datetime(expected_keys[time_col])
    if time_col == "Date":
        prediction_df[time_col] = prediction_df[time_col].dt.normalize()
        expected_keys[time_col] = expected_keys[time_col].dt.normalize()
    merged = expected_keys.merge(prediction_df[["Acorn", time_col, "prediction"]], on=["Acorn", time_col], how="left")
    if merged["prediction"].isna().any():
        missing = int(merged["prediction"].isna().sum())
        raise ValueError(f"AutoGluon TimeSeries prediction alignment failed for {missing} rows.")
    return merged["prediction"].to_numpy()


def forecast_horizon_length(df: pd.DataFrame, time_col: str) -> int:
    counts = df.groupby("Acorn")[time_col].count()
    if counts.empty or counts.nunique() != 1:
        raise ValueError("Forecast horizon must contain the same number of timestamps for each ACORN segment.")
    return int(counts.iloc[0])


def make_model(
    model_name: str,
    frequency: str,
    model_path: Path | None = None,
    prediction_length: int | None = None,
) -> Any:
    if model_name == "xgboost":
        return make_tree_model(frequency)
    if model_name == CATBOOST_MODEL:
        return make_catboost_model(frequency)
    if model_name == LIGHTGBM_MODEL:
        return make_lightgbm_model(frequency)
    if model_name == "ridge":
        return make_linear_model(frequency)
    if model_name == AUTOGLUON_TABULAR_MODEL:
        return AutoGluonTabularModel(
            frequency=frequency,
            path=model_path or model_storage_path(frequency, model_name, selected=False),
        )
    if model_name == AUTOGLUON_TIMESERIES_MODEL:
        if prediction_length is None:
            raise ValueError("AutoGluon TimeSeries requires an explicit prediction_length.")
        return AutoGluonTimeSeriesModel(
            frequency=frequency,
            path=model_path or model_storage_path(frequency, model_name, selected=False),
            prediction_length=prediction_length,
        )
    raise ValueError(f"Unsupported trainable model: {model_name}")


def fit_and_evaluate(
    df: pd.DataFrame,
    frequency: str,
    target_col: str,
    time_col: str,
    validation_start: pd.Timestamp,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    train = df[df[time_col] < validation_start].copy()
    valid = df[df[time_col] >= validation_start].copy()
    if train.empty or valid.empty:
        raise ValueError(f"Invalid validation split for {frequency}.")

    valid_predictions = valid[["Acorn", time_col, target_col]].copy()
    valid_predictions = add_baseline_predictions(train, valid_predictions, valid, frequency, target_col)

    train_x = feature_matrix(train, frequency)
    valid_x = feature_matrix(valid, frequency)
    for model_name in active_trainable_model_names():
        if model_name == AUTOGLUON_TIMESERIES_MODEL:
            model = make_model(
                model_name,
                frequency,
                autogluon_validation_path(frequency, model_name),
                prediction_length=forecast_horizon_length(valid, time_col),
            )
            model.fit_history(train, target_col, time_col)
            valid_predictions[model_name] = clip_predictions(
                model.predict_horizon(train, valid, target_col, time_col)
            )
            continue
        model_path = autogluon_validation_path(frequency, model_name) if model_name == AUTOGLUON_TABULAR_MODEL else None
        model = make_model(model_name, frequency, model_path)
        model.fit(train_x, train[target_col])
        valid_predictions[model_name] = clip_predictions(model.predict(valid_x))

    valid_predictions = valid_predictions.rename(columns={target_col: "actual", time_col: "timestamp"})
    valid_predictions.insert(0, "frequency", frequency)

    metrics = validation_metrics(valid_predictions)
    selected_name = selected_trainable_model(metrics, frequency)["model"]
    return selected_name, metrics, valid_predictions


def add_baseline_predictions(
    train: pd.DataFrame,
    valid_predictions: pd.DataFrame,
    valid_features: pd.DataFrame,
    frequency: str,
    target_col: str,
) -> pd.DataFrame:
    out = valid_predictions.copy()
    if frequency == "half_hourly":
        out["previous_day"] = valid_features["lag_48"].to_numpy()
        out["previous_week"] = valid_features["lag_336"].to_numpy()
        seasonal_keys = ["Acorn", "weekday", "half_hour_slot"]
    else:
        out["previous_day"] = valid_features["lag_1"].to_numpy()
        out["previous_week"] = valid_features["lag_7"].to_numpy()
        seasonal_keys = ["Acorn", "weekday"]

    seasonal = train.groupby(seasonal_keys)[target_col].mean().rename("seasonal_mean")
    acorn_mean = train.groupby("Acorn")[target_col].mean()
    global_mean = float(train[target_col].mean())

    seasonal_values = []
    for _, row in valid_features.iterrows():
        key = tuple(row[col] for col in seasonal_keys)
        value = seasonal.get(key, np.nan)
        if pd.isna(value):
            value = acorn_mean.get(row["Acorn"], global_mean)
        seasonal_values.append(value)
    out["seasonal_mean"] = seasonal_values

    for col in ("previous_day", "previous_week", "seasonal_mean"):
        out[col] = out[col].fillna(out["Acorn"].map(acorn_mean)).fillna(global_mean)
    return out


def validation_metrics(valid_predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_cols = [*BASELINE_MODEL_NAMES, *active_trainable_model_names()]
    rows = []
    for model in prediction_cols:
        rows.append(metric_record(valid_predictions, model, "ALL"))
        for acorn in ACORNS:
            rows.append(metric_record(valid_predictions[valid_predictions["Acorn"] == acorn], model, acorn))
    return pd.DataFrame(rows)


def selected_trainable_model(metrics: pd.DataFrame, frequency: str) -> dict[str, object]:
    subset = metrics[
        (metrics["frequency"] == frequency)
        & (metrics["acorn"] == "ALL")
        & (metrics["model"].isin(active_trainable_model_names()))
    ].dropna(subset=["rmse"])
    if subset.empty:
        raise ValueError(f"No trainable model metrics found for {frequency}.")
    row = subset.sort_values("rmse").iloc[0]
    return {"model": row["model"], "rmse": float(row["rmse"]), "n": int(row["n"])}


def metric_record(df: pd.DataFrame, model: str, acorn: str) -> dict[str, object]:
    clean = df[["frequency", "actual", model]].dropna()
    if clean.empty:
        rmse = np.nan
        n = 0
        frequency = df["frequency"].iloc[0] if not df.empty else "unknown"
    else:
        rmse = float(np.sqrt(mean_squared_error(clean["actual"], clean[model])))
        n = int(len(clean))
        frequency = clean["frequency"].iloc[0]
    return {"frequency": frequency, "model": model, "acorn": acorn, "rmse": rmse, "n": n}


def clip_predictions(values: Iterable[float]) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def fit_final_models(
    frequency: str,
    history: pd.DataFrame,
    future: pd.DataFrame,
    target_col: str,
    time_col: str,
) -> dict[str, Any]:
    return {
        model_name: fit_final_model(
            model_name,
            frequency,
            history,
            future,
            target_col,
            time_col,
            model_storage_path(frequency, model_name, selected=False),
        )
        for model_name in active_trainable_model_names()
    }


def fit_final_model(
    model_name: str,
    frequency: str,
    history: pd.DataFrame,
    future: pd.DataFrame,
    target_col: str,
    time_col: str,
    model_path: Path | None = None,
) -> Any:
    storage_path = model_path or model_storage_path(frequency, model_name, selected=False)
    if model_name == AUTOGLUON_TIMESERIES_MODEL:
        model = make_model(
            model_name,
            frequency,
            storage_path,
            prediction_length=forecast_horizon_length(future, time_col),
        )
        model.fit_history(history, target_col, time_col)
        return model
    model = make_model(model_name, frequency, storage_path)
    model.fit(feature_matrix(history, frequency), history[target_col])
    return model


def forecast_with_model(
    model: Any,
    model_name: str,
    history: pd.DataFrame,
    future: pd.DataFrame,
    frequency: str,
    target_col: str,
    prediction_col: str,
    time_col: str,
) -> pd.DataFrame:
    if model_name == AUTOGLUON_TIMESERIES_MODEL:
        future_sorted = future.sort_values(["Acorn", time_col]).copy()
        out = future_sorted[["Acorn", time_col]].copy()
        out[prediction_col] = clip_predictions(model.predict_horizon(history, future_sorted, target_col, time_col))
        if frequency == "half_hourly":
            out["DateTime"] = pd.to_datetime(out["DateTime"])
        else:
            out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
        return out.reset_index(drop=True)
    return recursive_forecast(model, history, future, frequency, target_col, prediction_col, time_col)


def recursive_forecast(
    model: Any,
    history: pd.DataFrame,
    future: pd.DataFrame,
    frequency: str,
    target_col: str,
    prediction_col: str,
    time_col: str,
) -> pd.DataFrame:
    future = future.sort_values(["Acorn", time_col]).copy()
    predictions = []

    for acorn in ACORNS:
        hist_acorn = history[history["Acorn"] == acorn].sort_values(time_col)
        values = pd.Series(hist_acorn[target_col].to_numpy(), index=hist_acorn[time_col])
        values = values[~values.index.duplicated(keep="last")].sort_index()

        future_acorn = future[future["Acorn"] == acorn].sort_values(time_col)
        for _, base_row in future_acorn.iterrows():
            row = base_row.copy()
            timestamp = row[time_col]
            row = add_recursive_lags(row, values, timestamp, frequency)
            x_row = feature_matrix(pd.DataFrame([row]), frequency)
            prediction = float(clip_predictions(model.predict(x_row))[0])
            values.loc[timestamp] = prediction
            predictions.append({"Acorn": acorn, time_col: timestamp, prediction_col: prediction})

    out = pd.DataFrame(predictions).sort_values(["Acorn", time_col]).reset_index(drop=True)
    if frequency == "half_hourly":
        out["DateTime"] = pd.to_datetime(out["DateTime"])
    else:
        out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out


def add_recursive_lags(row: pd.Series, values: pd.Series, timestamp: pd.Timestamp, frequency: str) -> pd.Series:
    if frequency == "half_hourly":
        offsets = {
            "lag_1": pd.Timedelta(minutes=30),
            "lag_2": pd.Timedelta(minutes=60),
            "lag_48": pd.Timedelta(days=1),
            "lag_336": pd.Timedelta(days=7),
        }
        for col, offset in offsets.items():
            row[col] = values.get(timestamp - offset, np.nan)
        previous_values = values[values.index < timestamp]
        row["rolling_48_mean"] = previous_values.tail(48).mean()
        row["rolling_336_mean"] = previous_values.tail(336).mean()
    else:
        offsets = {"lag_1": 1, "lag_7": 7, "lag_14": 14}
        for col, days in offsets.items():
            row[col] = values.get(timestamp - pd.Timedelta(days=days), np.nan)
        previous_values = values[values.index < timestamp]
        row["rolling_7_mean"] = previous_values.tail(7).mean()
        row["rolling_28_mean"] = previous_values.tail(28).mean()
    return row


def write_core_outputs(
    half_predictions: pd.DataFrame,
    daily_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    half_valid_predictions: pd.DataFrame,
    daily_valid_predictions: pd.DataFrame,
    half_model_name: str,
    daily_model_name: str,
) -> None:
    half_csv = half_predictions.copy()
    half_csv["DateTime"] = half_csv["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    daily_csv = daily_predictions.copy()
    daily_csv["Date"] = daily_csv["Date"].dt.strftime("%Y-%m-%d")

    write_prediction_table(
        half_csv,
        half_predictions,
        PREDICTION_DIR / "group_5_half_hourly_predict.csv",
        PREDICTION_DIR / "group_5_half_hourly_predict.parquet",
    )
    write_prediction_table(
        daily_csv,
        daily_predictions,
        PREDICTION_DIR / "group_5_daily_predict.csv",
        PREDICTION_DIR / "group_5_daily_predict.parquet",
    )
    write_table(metrics, METRICS_DIR / "validation_metrics.csv")
    write_table(half_valid_predictions, METRICS_DIR / "validation_predictions_half_hourly.csv")
    write_table(daily_valid_predictions, METRICS_DIR / "validation_predictions_daily.csv")

    summary = {
        "half_hourly_rows": int(len(half_predictions)),
        "daily_rows": int(len(daily_predictions)),
        "parquet_available": parquet_available(),
        "selected_half_hourly_model": half_model_name,
        "selected_daily_model": daily_model_name,
        "active_trainable_models": list(active_trainable_model_names()),
        "interim_csv_dir": str(INTERIM_CSV_DIR),
        "interim_parquet_dir": str(INTERIM_PARQUET_DIR),
        "model_ready_csv_dir": str(MODEL_READY_CSV_DIR),
        "model_ready_parquet_dir": str(MODEL_READY_PARQUET_DIR),
        "short_term_model_path": str(model_storage_path("half_hourly", half_model_name, selected=True)),
        "medium_term_model_path": str(model_storage_path("daily", daily_model_name, selected=True)),
        "short_term_model_paths": final_model_paths("half_hourly"),
        "medium_term_model_paths": final_model_paths("daily"),
        "best_half_hourly_overall": best_model(metrics, "half_hourly"),
        "best_daily_overall": best_model(metrics, "daily"),
    }
    (METRICS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_prediction_table(
    csv_df: pd.DataFrame,
    parquet_df: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(csv_path, index=False)
    if parquet_available():
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_df.to_parquet(parquet_path, index=False)


def best_model(metrics: pd.DataFrame, frequency: str) -> dict[str, object]:
    subset = metrics[(metrics["frequency"] == frequency) & (metrics["acorn"] == "ALL")].dropna(subset=["rmse"])
    if subset.empty:
        return {}
    row = subset.sort_values("rmse").iloc[0]
    return {"model": row["model"], "rmse": float(row["rmse"]), "n": int(row["n"])}


def write_eda_tables(half_features: pd.DataFrame, daily_features: pd.DataFrame) -> None:
    half_profile = (
        half_features.groupby(["Acorn", "Acorn_grouped", "half_hour_slot", "hour", "minute"], as_index=False)[HALF_TARGET]
        .mean()
        .rename(columns={HALF_TARGET: "mean_conso_moy"})
    )
    weekly_profile = (
        daily_features.groupby(["Acorn", "weekday"], as_index=False)[DAILY_TARGET]
        .mean()
        .rename(columns={DAILY_TARGET: "mean_conso_kwh"})
    )
    daily_enriched = daily_features[
        [
            "Acorn",
            "Date",
            DAILY_TARGET,
            "temperatureMean",
            "temperatureMin",
            "temperatureMax",
            "weekday",
            "is_weekend",
            "is_holiday",
        ]
    ].copy()
    autocorr = daily_autocorrelation(daily_features)

    write_table(half_profile, METRICS_DIR / "eda_half_hour_profile.csv")
    write_table(weekly_profile, METRICS_DIR / "eda_weekly_profile.csv")
    write_table(daily_enriched, METRICS_DIR / "eda_daily_enriched.csv")
    write_table(autocorr, METRICS_DIR / "eda_daily_autocorrelation.csv")


def daily_autocorrelation(daily_features: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    rows = []
    for acorn, group in daily_features.sort_values("Date").groupby("Acorn"):
        series = group[DAILY_TARGET].reset_index(drop=True)
        for lag in range(1, max_lag + 1):
            rows.append({"Acorn": acorn, "lag_days": lag, "autocorrelation": float(series.autocorr(lag))})
    return pd.DataFrame(rows)


def generate_figures(
    half_features: pd.DataFrame,
    daily_features: pd.DataFrame,
    half_predictions: pd.DataFrame,
    daily_predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    palette = {"ACORN-E": "#2f6f73", "ACORN-F": "#7b5f39", "ACORN-Q": "#8f3f46"}

    fig, ax = plt.subplots(figsize=(11, 5))
    for acorn, group in daily_features.sort_values("Date").groupby("Acorn"):
        rolling = group.set_index("Date")[DAILY_TARGET].rolling(14, min_periods=1).mean()
        ax.plot(rolling.index, rolling.values, label=f"{acorn} ({ACORN_GROUPS[acorn]})", color=palette[acorn])
    ax.set_title("Daily electricity consumption, 14-day rolling mean")
    ax.set_ylabel("kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_daily_consumption_trend.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    profile = half_features.groupby(["Acorn", "half_hour_slot"], as_index=False)[HALF_TARGET].mean()
    profile["time_of_day"] = profile["half_hour_slot"] / 2
    sns.lineplot(data=profile, x="time_of_day", y=HALF_TARGET, hue="Acorn", palette=palette, ax=ax)
    ax.set_title("Typical half-hourly profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean consumption")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_half_hourly_profile.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    weekly = daily_features.groupby(["Acorn", "weekday"], as_index=False)[DAILY_TARGET].mean()
    sns.lineplot(data=weekly, x="weekday", y=DAILY_TARGET, hue="Acorn", marker="o", palette=palette, ax=ax)
    ax.set_title("Weekly daily-consumption pattern")
    ax.set_xlabel("Weekday (0=Monday)")
    ax.set_ylabel("Mean daily kWh")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_weekly_pattern.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=daily_features,
        x="temperatureMean",
        y=DAILY_TARGET,
        hue="Acorn",
        palette=palette,
        alpha=0.55,
        s=24,
        ax=ax,
    )
    ax.set_title("Daily consumption vs. mean temperature")
    ax.set_xlabel("Mean temperature")
    ax.set_ylabel("Daily kWh")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_weather_relationship.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    acf = daily_autocorrelation(daily_features)
    sns.lineplot(data=acf, x="lag_days", y="autocorrelation", hue="Acorn", palette=palette, ax=ax)
    ax.set_title("Daily autocorrelation by lag")
    ax.set_xlabel("Lag in days")
    ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_daily_autocorrelation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    metric_subset = metrics[metrics["acorn"] == "ALL"].copy()
    sns.barplot(data=metric_subset, x="model", y="rmse", hue="frequency", ax=ax)
    ax.set_title("Validation RMSE by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_validation_rmse.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for acorn, group in half_predictions.groupby("Acorn"):
        ax.plot(group["DateTime"], group["Conso_moy_predict"], label=acorn, color=palette[acorn])
    ax.set_title("Final 48-hour half-hourly forecast")
    ax.set_ylabel("Predicted mean consumption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_final_half_hourly_forecast.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for acorn, group in daily_predictions.groupby("Acorn"):
        ax.plot(group["Date"], group["Conso_kWh_predict"], label=acorn, color=palette[acorn])
    ax.set_title("Final daily forecast")
    ax.set_ylabel("Predicted daily kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_final_daily_forecast.pdf")
    plt.close(fig)


def write_report(half_features: pd.DataFrame, daily_features: pd.DataFrame, metrics: pd.DataFrame) -> None:
    daily_stats = markdown_table(
        daily_features.groupby("Acorn")[DAILY_TARGET].agg(["mean", "min", "max"]).round(3).reset_index()
    )
    half_stats = markdown_table(
        half_features.groupby("Acorn")[HALF_TARGET].agg(["mean", "min", "max"]).round(4).reset_index()
    )
    corr_rows = []
    for acorn, group in daily_features.groupby("Acorn"):
        corr_rows.append(
            {
                "Acorn": acorn,
                "temperature_consumption_corr": round(float(group["temperatureMean"].corr(group[DAILY_TARGET])), 3),
            }
        )
    corr_table = markdown_table(pd.DataFrame(corr_rows))

    metric_table = markdown_table(
        metrics[metrics["acorn"] == "ALL"].sort_values(["frequency", "rmse"]).round({"rmse": 5})
    )
    best_half = selected_trainable_model(metrics, "half_hourly")
    best_daily = selected_trainable_model(metrics, "daily")

    report = f"""# Group 5 Energy Forecasting Report

## Project Scope

Group 5 covers `ACORN-E` ({ACORN_GROUPS["ACORN-E"]}), `ACORN-F` ({ACORN_GROUPS["ACORN-F"]}), and `ACORN-Q` ({ACORN_GROUPS["ACORN-Q"]}). The historical consumption data runs up to `2014-01-12`, and the final outputs forecast:

- `2014-01-13 00:00` to `2014-01-14 23:30` at 30-minute resolution.
- `2014-01-13` to `2014-02-13` at daily resolution.

## Exploratory Findings

Daily consumption levels by ACORN:

{daily_stats}

Half-hourly consumption levels by ACORN:

{half_stats}

Correlation between daily mean temperature and daily consumption:

{corr_table}

The visual outputs in `outputs/group5/figures` show three useful dynamics: ACORN-E has the highest average consumption, all three ACORN segments have stable daily and weekly seasonality, and short lags plus weekly repetition are strong reference points for forecasting.

## Modeling Approach

The validation strategy is chronological: the models train only on dates before the validation window and are tested on later historical data. The half-hourly validation window starts at `{HALF_HOURLY_VALIDATION_START}`. The daily validation window starts at `{DAILY_VALIDATION_START}`.

The compared models are:

- `previous_day`: same ACORN value from the previous day.
- `previous_week`: same ACORN value from seven days earlier.
- `seasonal_mean`: historical mean by ACORN and calendar slot.
- `ridge`: regularized linear regression.
- `xgboost`: gradient-boosted tree regression using calendar, weather, holiday, lag, and rolling features.
- `catboost`: CatBoost gradient boosting regression on the same engineered feature table.
- `lightgbm`: LightGBM gradient boosting regression on the same engineered feature table.
- `autogluon`: AutoGluon TabularPredictor AutoML regression on the same engineered feature table.
- `autogluon_timeseries`: AutoGluon TimeSeriesPredictor using the target history plus known future calendar, holiday, and weather covariates.

Overall validation RMSE:

{metric_table}

Selected final trainable models:

- Half-hourly: `{best_half.get("model", "n/a")}` with RMSE `{best_half.get("rmse", float("nan")):.5f}`.
- Daily: `{best_daily.get("model", "n/a")}` with RMSE `{best_daily.get("rmse", float("nan")):.5f}`.

## Final Outputs

The filled forecast files are available in:

- `outputs/group5/predictions/group_5_half_hourly_predict.csv`
- `outputs/group5/predictions/group_5_daily_predict.csv`

Cleaned weather and joined intermediate frames are written to `data/01_interim/group5`. Model-ready feature files are written to `data/02_processed/group5_modeling`. The original client-provided files under `data/00_raw` and the existing `data/02_processed/csv` and `data/02_processed/parquet` template files are treated as read-only inputs.

Final fitted trainable models are saved by name under `models/short_term` and `models/medium_term`. The selected forecast model is also saved with the `selected` suffix for compatibility with downstream tools.

## Limits

The forecasts use real weather data for the forecast period, as allowed by the assignment. Future lag features are generated recursively, so later daily predictions depend partly on earlier model predictions. The models are practical and reproducible, but they are not calibrated probabilistic forecasts and do not estimate uncertainty intervals.
"""
    (REPORT_DIR / "group5_report.md").write_text(report, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: "" if pd.isna(value) else f"{value:.5g}")
        else:
            formatted[col] = formatted[col].astype(str)
    headers = [str(col) for col in formatted.columns]
    rows = formatted.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def validate_outputs(half_predictions: pd.DataFrame, daily_predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    if len(half_predictions) != 288:
        raise ValueError(f"Expected 288 half-hourly predictions, got {len(half_predictions)}.")
    if len(daily_predictions) != 96:
        raise ValueError(f"Expected 96 daily predictions, got {len(daily_predictions)}.")
    if half_predictions["Conso_moy_predict"].isna().any():
        raise ValueError("Half-hourly predictions contain missing values.")
    if daily_predictions["Conso_kWh_predict"].isna().any():
        raise ValueError("Daily predictions contain missing values.")
    if half_predictions.duplicated(["Acorn", "DateTime"]).any():
        raise ValueError("Half-hourly predictions contain duplicate ACORN timestamps.")
    if daily_predictions.duplicated(["Acorn", "Date"]).any():
        raise ValueError("Daily predictions contain duplicate ACORN dates.")

    half_start = pd.Timestamp(HALF_HOURLY_FORECAST_START)
    half_end = pd.Timestamp(HALF_HOURLY_FORECAST_END)
    daily_start = pd.Timestamp(DAILY_FORECAST_START)
    daily_end = pd.Timestamp(DAILY_FORECAST_END)
    if half_predictions["DateTime"].min() != half_start or half_predictions["DateTime"].max() != half_end:
        raise ValueError("Half-hourly prediction period does not match the assignment.")
    if daily_predictions["Date"].min() != daily_start or daily_predictions["Date"].max() != daily_end:
        raise ValueError("Daily prediction period does not match the assignment.")

    expected_frequencies = {"half_hourly", "daily"}
    if set(metrics["frequency"].unique()) != expected_frequencies:
        raise ValueError("Validation metrics are missing a frequency.")
    for model_name in active_trainable_model_names():
        if metrics[(metrics["acorn"] == "ALL") & (metrics["model"] == model_name)].empty:
            raise ValueError(f"Validation metrics are missing the {model_name} model.")
