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
from dashboard.tabs import AboutTab, OverviewTab, EDATab, ValidationTab, ForecastsTab, CarbonTab

class ForecastDashboardApp:
    """Central app coordinating Streamlit setup, styling, caching, data-flow, and views."""
    
    def __init__(self):
        self.root_dir = ROOT
        # pyrefly: ignore [missing-import]
        self.data_loader = DataLoader(self.root_dir)
        
        # Instantiate the modular tab components
        self.tabs = {
            "ℹ️ About": AboutTab(),
            "📌 Overview": OverviewTab(self.root_dir),
            "🔎 EDA": EDATab(),
            "✅ Validation": ValidationTab(),
            "📈 Forecasts": ForecastsTab(),
            "🌱 Carbon Footprint": CarbonTab(),
        }

    def run(self) -> None:
        """Main execution flow of the Streamlit dashboard app."""
        st.set_page_config(page_title="Group 5 Energy Forecasts", layout="wide")
        
        acorn_options = list(ACORN_GROUPS)

        # Enforce session state for ACORN selection to prevent empty states
        if "last_valid_acorns" not in st.session_state:
            st.session_state.last_valid_acorns = acorn_options
        if "acorn_filter" not in st.session_state:
            st.session_state.acorn_filter = acorn_options

        # Initialize or detect theme and viewport preferences from URL query parameters.
        # If any of them are missing, detect them via JavaScript and reload.
        if "theme" not in st.query_params or "viewport" not in st.query_params:
            st.html(
                """
                <script>
                const params = new URLSearchParams(window.location.search);
                let changed = false;
                if (!params.has('theme')) {
                    const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
                    params.set('theme', systemTheme);
                    changed = true;
                }
                if (!params.has('viewport')) {
                    const viewport = window.innerWidth <= 768 ? 'mobile' : 'desktop';
                    params.set('viewport', viewport);
                    changed = true;
                }
                if (changed) {
                    window.location.search = params.toString();
                }
                </script>
                """,
                unsafe_allow_javascript=True,
            )
            st.stop()

        # Retrieve and enforce theme parameter
        theme_param = st.query_params.get("theme")
        if theme_param not in ["light", "dark"]:
            theme_param = "dark"  # fallback default
            st.query_params["theme"] = theme_param

        if "theme_mode" not in st.session_state:
            st.session_state.theme_mode = theme_param

        # Retrieve and enforce viewport parameter
        viewport_param = st.query_params.get("viewport")
        if viewport_param not in ["mobile", "desktop"]:
            viewport_param = "desktop"  # fallback default
            st.query_params["viewport"] = viewport_param

        st.session_state.is_mobile = (viewport_param == "mobile")

        def on_acorn_change():
            if not st.session_state.acorn_filter:
                st.session_state.acorn_filter = st.session_state.last_valid_acorns
                st.session_state.show_acorn_warning = True
            else:
                st.session_state.last_valid_acorns = st.session_state.acorn_filter

        def toggle_theme():
            modes = ["light", "dark"]
            current_idx = modes.index(st.session_state.theme_mode)
            new_theme = modes[(current_idx + 1) % len(modes)]
            st.session_state.theme_mode = new_theme
            st.query_params["theme"] = new_theme

        # Inject custom styling & typography (Outfit Google Font, Glassmorphic KPI Cards)
        StyleManager.apply(st.session_state.theme_mode)
        
        # Inject Javascript to handle tab-specific filters and mobile chart layout adjustments
        st.html(
            """
            <script>
            const parentWin = window.parent || window;
            
            // 1. Register a debounced resize event listener to trigger a query-param reload on viewport crossover
            if (!parentWin.hasViewportResizeListener) {
                parentWin.hasViewportResizeListener = true;
                let resizeTimeout;
                parentWin.addEventListener('resize', () => {
                    clearTimeout(resizeTimeout);
                    resizeTimeout = setTimeout(() => {
                        const currentViewport = parentWin.innerWidth <= 768 ? 'mobile' : 'desktop';
                        const params = new URLSearchParams(parentWin.location.search);
                        if (params.get('viewport') !== currentViewport) {
                            params.set('viewport', currentViewport);
                            parentWin.location.search = params.toString();
                        }
                    }, 250);
                });
            }

            // 2. Interval loop to handle tab-specific filter hiding (About and Carbon Footprint tabs)
            if (parentWin.tabIntervalId) {
                parentWin.clearInterval(parentWin.tabIntervalId);
            }
            parentWin.tabIntervalId = parentWin.setInterval(() => {
                try {
                    const doc = parentWin.document;
                    const tabs = Array.from(doc.querySelectorAll('[data-baseweb="tab"], [role="tab"]'));
                    if (tabs.length > 0) {
                        const activeTab = tabs.find(tab => tab.getAttribute('aria-selected') === 'true');
                        if (activeTab) {
                            const activeText = activeTab.textContent || "";
                            const shouldHide = activeText.includes("About") || activeText.includes("Carbon");
                            
                            const filterEl = doc.querySelector('.st-key-acorn_filter');
                            if (filterEl) {
                                const filterCol = filterEl.closest('div[data-testid="column"]');
                                const target = filterCol || filterEl;
                                if (shouldHide) {
                                    if (target.style.display !== 'none') {
                                        target.style.setProperty('display', 'none', 'important');
                                    }
                                } else {
                                    if (target.style.display === 'none') {
                                        target.style.setProperty('display', 'flex', 'important');
                                    }
                                }
                            }
                        }
                    }
                } catch (e) {
                    // Fail silently inside interval loop
                }
            }, 250);
            </script>
            """,
            unsafe_allow_javascript=True,
        )
        
        # Load datasets safely and display instruction if run has not occurred
        try:
            data = self.data_loader.load_artifacts()
        except FileNotFoundError:
            st.error("Run `EI-climat/bin/python scripts/group5_run_pipeline.py` before opening the dashboard.")
            st.stop()

        # Header layout with Title and the Acorn selector in columns
        col1, col2 = st.columns([2, 3], vertical_alignment="center")
        with col1:
            st.title("UK Household Energy Forecasts - Group 5")
        with col2:
            col2_filter, col2_toggle = st.columns([8, 1], vertical_alignment="bottom")
            with col2_filter:
                selected_acorns = st.pills(
                    "Filter ACORN Segments",
                    acorn_options,
                    selection_mode="multi",
                    key="acorn_filter",
                    on_change=on_acorn_change,
                    format_func=lambda value: f"{value} - {ACORN_GROUPS[value]}",
                    help="Click a chip to toggle selection. At least one segment must remain selected.",
                    label_visibility="visible",
                )
            with col2_toggle:
                theme_icons = {"light": "☀️", "dark": "🌙"}
                st.button(
                    theme_icons.get(st.session_state.theme_mode, "🌙"),
                    key="theme_toggle",
                    on_click=toggle_theme,
                    help="Click to toggle between Light and Dark mode.",
                )

        if st.session_state.get("show_acorn_warning", False):
            st.toast("At least one ACORN segment must be selected!", icon="⚠️")
            st.session_state.show_acorn_warning = False

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
