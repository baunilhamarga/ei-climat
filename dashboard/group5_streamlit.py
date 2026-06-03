from pathlib import Path
import sys

# pyrefly: ignore [missing-import]
import streamlit as st

# Setup paths for resolving imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

# pyrefly: ignore [missing-import]
from group5_energy.config import ACORN_GROUPS
# pyrefly: ignore [missing-import]
from dashboard.data_loader import DataLoader
# pyrefly: ignore [missing-import]
from dashboard.styles import StyleManager
# pyrefly: ignore [missing-import]
from dashboard.tabs import OverviewTab, EDATab, ValidationTab, ForecastsTab, ComparisonTab

class ForecastDashboardApp:
    """Central app coordinating Streamlit setup, styling, caching, data-flow, and views."""
    
    def __init__(self):
        self.root_dir = ROOT
        # pyrefly: ignore [missing-import]
        self.data_loader = DataLoader(self.root_dir)
        
        # Instantiate the modular tab components
        self.tabs = {
            "Overview": OverviewTab(self.root_dir),
            "EDA": EDATab(),
            "Validation": ValidationTab(),
            "Forecasts": ForecastsTab(),
            "ACORN Comparison": ComparisonTab(),
        }

    def run(self) -> None:
        """Main execution flow of the Streamlit dashboard app."""
        st.set_page_config(page_title="Group 5 Energy Forecasts", layout="wide")
        
        # Inject custom styling & typography (Outfit Google Font, Glassmorphic KPI Cards)
        StyleManager.apply()
        
        st.title("Group 5 Energy Forecasts")

        # Load datasets safely and display instruction if run has not occurred
        try:
            data = self.data_loader.load_artifacts()
        except FileNotFoundError:
            st.error("Run `EI-climat/bin/python scripts/group5_run_pipeline.py` before opening the dashboard.")
            st.stop()

        # Sidebar multi-select filter for ACORN segments
        acorn_options = list(ACORN_GROUPS)
        selected_acorns = st.sidebar.multiselect(
            "ACORN segments",
            acorn_options,
            default=acorn_options,
            format_func=lambda value: f"{value} - {ACORN_GROUPS[value]}",
        )
        if not selected_acorns:
            selected_acorns = acorn_options

        # Initialize tabs layout via Streamlit
        tab_names = list(self.tabs.keys())
        st_tabs = st.tabs(tab_names)
        
        # Render each tab using its respective class instance
        for name, st_tab in zip(tab_names, st_tabs):
            with st_tab:
                self.tabs[name].render(data, selected_acorns)

if __name__ == "__main__":
    app = ForecastDashboardApp()
    app.run()
