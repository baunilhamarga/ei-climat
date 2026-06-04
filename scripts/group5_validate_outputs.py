from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group5_energy.pipeline import validate_model_prediction_outputs, validate_outputs


def main() -> None:
    half = pd.read_csv(ROOT / "outputs" / "group5" / "predictions" / "group_5_half_hourly_predict.csv")
    daily = pd.read_csv(ROOT / "outputs" / "group5" / "predictions" / "group_5_daily_predict.csv")
    half_all = pd.read_csv(ROOT / "outputs" / "group5" / "predictions" / "group_5_half_hourly_all_models_predict.csv")
    daily_all = pd.read_csv(ROOT / "outputs" / "group5" / "predictions" / "group_5_daily_all_models_predict.csv")
    metrics = pd.read_csv(ROOT / "outputs" / "group5" / "metrics" / "validation_metrics.csv")
    half["DateTime"] = pd.to_datetime(half["DateTime"])
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
    half_all["DateTime"] = pd.to_datetime(half_all["DateTime"])
    daily_all["Date"] = pd.to_datetime(daily_all["Date"]).dt.normalize()
    validate_outputs(half, daily, metrics)
    validate_model_prediction_outputs(half_all, daily_all)
    print("Group 5 output validation passed.")


if __name__ == "__main__":
    main()

