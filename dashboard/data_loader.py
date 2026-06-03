from pathlib import Path
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

class DataLoader:
    """Handles loading and caching data artifacts for the dashboard."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.prediction_dir = self.root_dir / "outputs" / "group5" / "predictions"
        self.metrics_dir = self.root_dir / "outputs" / "group5" / "metrics"

    def load_artifacts(self) -> dict[str, pd.DataFrame]:
        """Loads and caches all necessary datasets for the dashboard.
        
        Uses a helper method decorated with @st.cache_data to leverage Streamlit's caching mechanism.
        """
        return self._cached_load(
            prediction_dir=str(self.prediction_dir),
            metrics_dir=str(self.metrics_dir)
        )

    @staticmethod
    @st.cache_data
    def _cached_load(prediction_dir: str, metrics_dir: str) -> dict[str, pd.DataFrame]:
        pred_path = Path(prediction_dir)
        metrics_path = Path(metrics_dir)
        
        half_pred = pd.read_csv(pred_path / "group_5_half_hourly_predict.csv", parse_dates=["DateTime"])
        daily_pred = pd.read_csv(pred_path / "group_5_daily_predict.csv", parse_dates=["Date"])
        metrics = pd.read_csv(metrics_path / "validation_metrics.csv")
        half_profile = pd.read_csv(metrics_path / "eda_half_hour_profile.csv")
        weekly = pd.read_csv(metrics_path / "eda_weekly_profile.csv")
        daily_enriched = pd.read_csv(metrics_path / "eda_daily_enriched.csv", parse_dates=["Date"])
        autocorr = pd.read_csv(metrics_path / "eda_daily_autocorrelation.csv")
        half_valid = pd.read_csv(metrics_path / "validation_predictions_half_hourly.csv", parse_dates=["timestamp"])
        daily_valid = pd.read_csv(metrics_path / "validation_predictions_daily.csv", parse_dates=["timestamp"])
        
        return {
            "half_pred": half_pred,
            "daily_pred": daily_pred,
            "metrics": metrics,
            "half_profile": half_profile,
            "weekly": weekly,
            "daily_enriched": daily_enriched,
            "autocorr": autocorr,
            "half_valid": half_valid,
            "daily_valid": daily_valid,
        }
