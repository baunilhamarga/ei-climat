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
        self.processed_csv_dir = self.root_dir / "data" / "02_processed" / "csv"

    def load_artifacts(self) -> dict[str, pd.DataFrame]:
        """Loads and caches all necessary datasets for the dashboard.
        
        Uses a helper method decorated with @st.cache_data to leverage Streamlit's caching mechanism.
        """
        return self._cached_load(
            prediction_dir=str(self.prediction_dir),
            metrics_dir=str(self.metrics_dir),
            processed_csv_dir=str(self.processed_csv_dir),
            cache_token=self._artifact_cache_token(),
        )

    def _artifact_cache_token(self) -> tuple[tuple[str, int], ...]:
        paths = [
            *self.prediction_dir.glob("*.csv"),
            *self.prediction_dir.glob("*.parquet"),
            *self.metrics_dir.glob("*.csv"),
            self.metrics_dir / "run_summary.json",
            self.processed_csv_dir / "group_5_half_hourly_predict.csv",
            self.processed_csv_dir / "group_5_daily_predict.csv",
        ]
        token = []
        for path in sorted(set(paths)):
            if path.exists():
                token.append((str(path.relative_to(self.root_dir)), path.stat().st_mtime_ns))
        return tuple(token)

    @staticmethod
    @st.cache_data
    def _cached_load(
        prediction_dir: str,
        metrics_dir: str,
        processed_csv_dir: str,
        cache_token: tuple[tuple[str, int], ...],
    ) -> dict[str, pd.DataFrame]:
        pred_path = Path(prediction_dir)
        metrics_path = Path(metrics_dir)
        processed_path = Path(processed_csv_dir)
        
        half_pred = pd.read_csv(pred_path / "group_5_half_hourly_predict.csv", parse_dates=["DateTime"])
        daily_pred = pd.read_csv(pred_path / "group_5_daily_predict.csv", parse_dates=["Date"])
        half_model_pred = DataLoader._load_model_predictions(
            pred_path / "group_5_half_hourly_all_models_predict.csv",
            selected_predictions=half_pred,
            time_col="DateTime",
        )
        daily_model_pred = DataLoader._load_model_predictions(
            pred_path / "group_5_daily_all_models_predict.csv",
            selected_predictions=daily_pred,
            time_col="Date",
        )
        half_test_actual = DataLoader._load_optional_actuals(
            paths=[
                metrics_path / "test_actuals_half_hourly.csv",
                metrics_path / "final_actuals_half_hourly.csv",
                metrics_path / "test_ground_truth_half_hourly.csv",
                processed_path / "group_5_half_hourly_predict.csv",
            ],
            time_col="DateTime",
            target_col="Conso_moy",
            target_aliases=["Conso_moy_predict", "actual"],
        )
        daily_test_actual = DataLoader._load_optional_actuals(
            paths=[
                metrics_path / "test_actuals_daily.csv",
                metrics_path / "final_actuals_daily.csv",
                metrics_path / "test_ground_truth_daily.csv",
                processed_path / "group_5_daily_predict.csv",
            ],
            time_col="Date",
            target_col="Conso_kWh",
            target_aliases=["Conso_kWh_predict", "actual"],
        )
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
            "half_model_pred": half_model_pred,
            "daily_model_pred": daily_model_pred,
            "half_test_actual": half_test_actual,
            "daily_test_actual": daily_test_actual,
            "metrics": metrics,
            "half_profile": half_profile,
            "weekly": weekly,
            "daily_enriched": daily_enriched,
            "autocorr": autocorr,
            "half_valid": half_valid,
            "daily_valid": daily_valid,
        }

    @staticmethod
    def _load_model_predictions(
        path: Path,
        selected_predictions: pd.DataFrame,
        time_col: str,
    ) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path, parse_dates=[time_col])
        fallback = selected_predictions.copy()
        fallback.insert(0, "model", "selected")
        return fallback

    @staticmethod
    def _load_optional_actuals(
        paths: list[Path],
        time_col: str,
        target_col: str,
        target_aliases: list[str],
    ) -> pd.DataFrame:
        for path in paths:
            if not path.exists():
                continue
            actuals = pd.read_csv(path)
            if target_col not in actuals.columns:
                alias = next((col for col in target_aliases if col in actuals.columns), None)
                if alias is not None:
                    actuals = actuals.rename(columns={alias: target_col})
            required = {"Acorn", time_col, target_col}
            missing = sorted(required.difference(actuals.columns))
            if missing:
                raise ValueError(f"{path} is missing required columns: {missing}")
            actuals[time_col] = pd.to_datetime(actuals[time_col])
            return actuals[["Acorn", time_col, target_col]].copy()
        return pd.DataFrame(columns=["Acorn", time_col, target_col])

