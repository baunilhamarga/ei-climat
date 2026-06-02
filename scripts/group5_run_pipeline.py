from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from group5_energy.pipeline import run_pipeline


def main() -> None:
    result = run_pipeline()
    print("Group 5 pipeline complete.")
    print(f"Half-hourly predictions: {len(result.half_hourly_predictions)} rows")
    print(f"Daily predictions: {len(result.daily_predictions)} rows")
    print("Outputs written to outputs/group5 and reports/group5_report.md")


if __name__ == "__main__":
    main()

