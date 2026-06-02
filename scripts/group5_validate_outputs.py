from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group5_energy.pipeline import validate_outputs


def main() -> None:
    half = pd.read_csv(ROOT / "outputs" / "group5" / "predictions" / "group_5_half_hourly_predict.csv")
    daily = pd.read_csv(ROOT / "outputs" / "group5" / "predictions" / "group_5_daily_predict.csv")
    metrics = pd.read_csv(ROOT / "outputs" / "group5" / "metrics" / "validation_metrics.csv")
    half["DateTime"] = pd.to_datetime(half["DateTime"])
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
    validate_outputs(half, daily, metrics)
    print("Group 5 output validation passed.")


if __name__ == "__main__":
    main()

