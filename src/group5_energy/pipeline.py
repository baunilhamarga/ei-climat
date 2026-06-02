from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-group5")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
TRAINABLE_MODEL_NAMES = ("ridge", "gradient_boosting")


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

    half_model = make_model(half_model_name, "half_hourly")
    daily_model = make_model(daily_model_name, "daily")
    half_model.fit(feature_matrix(half_features, "half_hourly"), half_features[HALF_TARGET])
    daily_model.fit(feature_matrix(daily_features, "daily"), daily_features[DAILY_TARGET])
    save_models(half_model, daily_model, half_model_name, daily_model_name)

    half_predictions = recursive_forecast(
        half_model,
        half_features,
        half_future,
        frequency="half_hourly",
        target_col=HALF_TARGET,
        prediction_col="Conso_moy_predict",
        time_col="DateTime",
    )
    daily_predictions = recursive_forecast(
        daily_model,
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


def save_models(
    half_model: Pipeline,
    daily_model: Pipeline,
    half_model_name: str,
    daily_model_name: str,
) -> None:
    joblib.dump(half_model, SHORT_TERM_MODEL_DIR / "group5_half_hourly_selected.joblib")
    joblib.dump(half_model, SHORT_TERM_MODEL_DIR / f"group5_half_hourly_{half_model_name}.joblib")
    joblib.dump(daily_model, MEDIUM_TERM_MODEL_DIR / "group5_daily_selected.joblib")
    joblib.dump(daily_model, MEDIUM_TERM_MODEL_DIR / f"group5_daily_{daily_model_name}.joblib")


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
                HistGradientBoostingRegressor(
                    max_iter=220,
                    learning_rate=0.055,
                    max_leaf_nodes=31,
                    l2_regularization=0.03,
                    random_state=RANDOM_STATE,
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


def make_model(model_name: str, frequency: str) -> Pipeline:
    if model_name == "gradient_boosting":
        return make_tree_model(frequency)
    if model_name == "ridge":
        return make_linear_model(frequency)
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

    tree_model = make_tree_model(frequency)
    linear_model = make_linear_model(frequency)
    tree_model.fit(feature_matrix(train, frequency), train[target_col])
    linear_model.fit(feature_matrix(train, frequency), train[target_col])

    valid_predictions = valid[["Acorn", time_col, target_col]].copy()
    valid_predictions = add_baseline_predictions(train, valid_predictions, valid, frequency, target_col)
    valid_predictions["ridge"] = clip_predictions(
        linear_model.predict(feature_matrix(valid, frequency))
    )
    valid_predictions["gradient_boosting"] = clip_predictions(
        tree_model.predict(feature_matrix(valid, frequency))
    )
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
    prediction_cols = [
        "previous_day",
        "previous_week",
        "seasonal_mean",
        "ridge",
        "gradient_boosting",
    ]
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
        & (metrics["model"].isin(TRAINABLE_MODEL_NAMES))
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


def recursive_forecast(
    model: Pipeline,
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
        "interim_csv_dir": str(INTERIM_CSV_DIR),
        "interim_parquet_dir": str(INTERIM_PARQUET_DIR),
        "model_ready_csv_dir": str(MODEL_READY_CSV_DIR),
        "model_ready_parquet_dir": str(MODEL_READY_PARQUET_DIR),
        "short_term_model_path": str(SHORT_TERM_MODEL_DIR / "group5_half_hourly_selected.joblib"),
        "medium_term_model_path": str(MEDIUM_TERM_MODEL_DIR / "group5_daily_selected.joblib"),
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
    fig.savefig(FIGURES_DIR / "01_daily_consumption_trend.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    profile = half_features.groupby(["Acorn", "half_hour_slot"], as_index=False)[HALF_TARGET].mean()
    profile["time_of_day"] = profile["half_hour_slot"] / 2
    sns.lineplot(data=profile, x="time_of_day", y=HALF_TARGET, hue="Acorn", palette=palette, ax=ax)
    ax.set_title("Typical half-hourly profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean consumption")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_half_hourly_profile.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    weekly = daily_features.groupby(["Acorn", "weekday"], as_index=False)[DAILY_TARGET].mean()
    sns.lineplot(data=weekly, x="weekday", y=DAILY_TARGET, hue="Acorn", marker="o", palette=palette, ax=ax)
    ax.set_title("Weekly daily-consumption pattern")
    ax.set_xlabel("Weekday (0=Monday)")
    ax.set_ylabel("Mean daily kWh")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_weekly_pattern.png", dpi=160)
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
    fig.savefig(FIGURES_DIR / "04_weather_relationship.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    acf = daily_autocorrelation(daily_features)
    sns.lineplot(data=acf, x="lag_days", y="autocorrelation", hue="Acorn", palette=palette, ax=ax)
    ax.set_title("Daily autocorrelation by lag")
    ax.set_xlabel("Lag in days")
    ax.set_ylabel("Autocorrelation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_daily_autocorrelation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    metric_subset = metrics[metrics["acorn"] == "ALL"].copy()
    sns.barplot(data=metric_subset, x="model", y="rmse", hue="frequency", ax=ax)
    ax.set_title("Validation RMSE by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_validation_rmse.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for acorn, group in half_predictions.groupby("Acorn"):
        ax.plot(group["DateTime"], group["Conso_moy_predict"], label=acorn, color=palette[acorn])
    ax.set_title("Final 48-hour half-hourly forecast")
    ax.set_ylabel("Predicted mean consumption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_final_half_hourly_forecast.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for acorn, group in daily_predictions.groupby("Acorn"):
        ax.plot(group["Date"], group["Conso_kWh_predict"], label=acorn, color=palette[acorn])
    ax.set_title("Final daily forecast")
    ax.set_ylabel("Predicted daily kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_final_daily_forecast.png", dpi=160)
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
- `gradient_boosting`: tree-based regression using calendar, weather, holiday, lag, and rolling features.

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
    if metrics[(metrics["acorn"] == "ALL") & (metrics["model"] == "gradient_boosting")].empty:
        raise ValueError("Validation metrics are missing the main model.")
