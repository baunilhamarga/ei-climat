# pyrefly: ignore [missing-import]
import streamlit as st

class StyleManager:
    """Manages custom styling and premium aesthetics for the dashboard."""
    
    PALETTE = {
        "ACORN-E": "#2f6f73",  # Affluent - Sage green
        "ACORN-F": "#7b5f39",  # Comfortable - Warm brown
        "ACORN-Q": "#8f3f46",  # Adversity - Crimson/maroon
    }

    FREQUENCY_MAP = {
        "half_hourly": "Half-Hourly",
        "daily": "Daily"
    }

    MODEL_MAP = {
        "xgboost": "XGBoost",
        "ridge": "Ridge Regression",
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "autogluon": "AutoGluon Tabular",
        "autogluon_timeseries": "AutoGluon TimeSeries",
        "previous_day": "Baseline: Previous Day",
        "previous_week": "Baseline: Previous Week",
        "seasonal_mean": "Baseline: Seasonal Mean"
    }

    CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Typography & Background adjustments */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background-color: var(--theme-bg) !important;
        color: var(--theme-text) !important;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    /* Reduce main container margins to let contents extend closer to the screen edges */
    .block-container {
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-top: 3.5rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* Hide Streamlit default MainMenu, Header, and Footer */
    #MainMenu, footer, header, [data-testid="stHeader"] {
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Main Title customization */
    h1 {
        font-weight: 700 !important;
        background: linear-gradient(135deg, #4da4a9 0%, #d49c5e 50%, #c85a64 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        padding-bottom: 0.5rem;
    }

    /* Custom KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }

    .kpi-card {
        background: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #4da4a9, #d49c5e);
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }

    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        color: var(--kpi-label) !important;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--kpi-value) !important;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.75rem;
        color: var(--kpi-sub) !important;
        margin-top: 0.35rem;
    }
    
    /* Elegant Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: var(--tab-bg) !important;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid var(--tab-border) !important;
        border-bottom: none;
        color: var(--tab-text) !important;
        transition: all 0.2s ease-in-out;
        padding: 0 16px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--tab-text-hover) !important;
        background-color: var(--tab-bg-hover) !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--tab-selected-bg) !important;
        border-color: var(--tab-selected-border) !important;
        color: var(--tab-selected-text) !important;
        font-weight: 600 !important;
    }

    /* HeroUI style chips for ACORN selection pills */
    /* Force the Streamlit column holding the filter to align to the right */
    div[data-testid="column"]:has(.st-key-acorn_filter),
    div.element-container:has(.st-key-acorn_filter) {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        justify-content: flex-end !important;
        text-align: right !important;
        width: 100% !important;
    }

    .st-key-acorn_filter {
        margin-left: auto !important;
        margin-right: 0 !important;
        width: fit-content !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        justify-content: flex-end !important;
        text-align: right !important;
        padding-top: 0.5rem;
    }

    /* Force pills widget wrappers to layout label and buttons vertically */
    .st-key-acorn_filter div.stPills,
    .st-key-acorn_filter div.stButtonGroup {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        width: 100% !important;
    }

    /* Force button group itself to align buttons to the right and flow horizontally */
    .st-key-acorn_filter [role="group"] {
        margin-left: auto !important;
        margin-right: 0 !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }

    .st-key-acorn_filter [data-testid="stWidgetLabel"] {
        margin-bottom: 0.35rem !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        text-align: right !important;
        justify-content: flex-end !important;
    }

    .st-key-acorn_filter [data-testid="stWidgetLabel"] p {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.075em !important;
        color: #9ca3af !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }

    /* Common button properties for both active and inactive pills */
    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"],
    .st-key-acorn_filter [data-testid="stBaseButton-pills"] {
        border-radius: 9999px !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0.35rem 1rem !important;
        height: auto !important;
    }

    /* Hover effect for all buttons */
    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"]:hover,
    .st-key-acorn_filter [data-testid="stBaseButton-pills"]:hover {
        transform: translateY(-1px);
    }

    /* Base style for inactive button chips */
    .st-key-acorn_filter [data-testid="stBaseButton-pills"] {
        border: 1px solid var(--pill-inactive-border) !important;
        background-color: var(--pill-inactive-bg) !important;
        color: var(--pill-inactive-text) !important;
    }

    /* Style inactive buttons on hover */
    .st-key-acorn_filter [data-testid="stBaseButton-pills"]:hover {
        background-color: var(--pill-inactive-hover-bg) !important;
        color: var(--pill-inactive-hover-text) !important;
        border-color: var(--pill-inactive-hover-border) !important;
    }

    /* Add prefix symbols using CSS pseudo-elements with flex-shrink prevention */
    .st-key-acorn_filter [data-testid="stBaseButton-pills"]::before {
        content: "+  " !important;
        font-family: inherit !important;
        font-weight: 700 !important;
        display: inline-block !important;
        flex-shrink: 0 !important;
        margin-right: 0.35rem !important;
        color: inherit !important;
    }

    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"]::before {
        content: "✓  " !important;
        font-family: inherit !important;
        font-weight: 700 !important;
        display: inline-block !important;
        flex-shrink: 0 !important;
        margin-right: 0.35rem !important;
        color: #ffffff !important;
    }

    /* Selected state - Crimson Red for all active pills */
    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"] {
        background-color: #8f3f46 !important;
        border-color: #c85a64 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(143, 63, 70, 0.3) !important;
    }

    /* Hover glow for active chips */
    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"]:hover {
        filter: brightness(1.1);
    }

    /* Align filter and toggle button in the inner column row */
    div[data-testid="column"] div[data-testid="stHorizontalBlock"]:has(.st-key-theme_toggle) {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-end !important;
        align-items: flex-end !important;
        gap: 0px !important;
        width: 100% !important;
    }

    /* Force the filter column to grow and the toggle column to shrink to content */
    div[data-testid="column"] div[data-testid="stHorizontalBlock"]:has(.st-key-theme_toggle) > div[data-testid="column"]:nth-child(1) {
        width: auto !important;
        flex-grow: 1 !important;
        flex-shrink: 1 !important;
    }

    div[data-testid="column"] div[data-testid="stHorizontalBlock"]:has(.st-key-theme_toggle) > div[data-testid="column"]:nth-child(2) {
        width: auto !important;
        flex-grow: 0 !important;
        flex-shrink: 0 !important;
        margin-left: 2.5rem !important; /* Increased space between filter pills and toggle button */
    }

    .st-key-theme_toggle {
        margin: 0 !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: flex-end !important;
    }

    .st-key-theme_toggle button {
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        max-width: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 30px !important;
        font-size: 0.95rem !important;
        background-color: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        color: var(--theme-text) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
        margin-bottom: 0px !important;
    }

    .st-key-theme_toggle button:hover {
        background-color: var(--pill-inactive-hover-bg) !important;
        border-color: var(--pill-inactive-hover-border) !important;
        transform: scale(1.05) !important;
    }
    """

    @classmethod
    def apply(cls, theme_mode: str = "system") -> None:
        """Injects custom global CSS into the Streamlit app context based on theme mode."""
        # Define theme mode variables dynamically
        if theme_mode == "light":
            variables = """
            .stApp {
                --theme-bg: #ffffff;
                --theme-text: #1f2937;
                --kpi-bg: rgba(0, 0, 0, 0.02);
                --kpi-border: rgba(0, 0, 0, 0.06);
                --kpi-label: #4b5563;
                --kpi-value: #111827;
                --kpi-sub: #6b7280;
                --tab-bg: rgba(0, 0, 0, 0.01);
                --tab-border: rgba(0, 0, 0, 0.04);
                --tab-text: #4b5563;
                --tab-text-hover: #111827;
                --tab-bg-hover: rgba(0, 0, 0, 0.04);
                --tab-selected-bg: rgba(0, 0, 0, 0.06);
                --tab-selected-border: rgba(0, 0, 0, 0.12);
                --tab-selected-text: #111827;
                --pill-inactive-bg: rgba(0, 0, 0, 0.02);
                --pill-inactive-border: rgba(0, 0, 0, 0.06);
                --pill-inactive-text: #4b5563;
                --pill-inactive-hover-bg: rgba(0, 0, 0, 0.06);
                --pill-inactive-hover-border: rgba(0, 0, 0, 0.12);
                --pill-inactive-hover-text: #111827;
                
                /* Force Streamlit native variables */
                --background-color: #ffffff !important;
                --secondary-background-color: #f0f2f6 !important;
                --text-color: #1f2937 !important;
            }
            """
        elif theme_mode == "dark":
            variables = """
            .stApp {
                --theme-bg: #0e1117;
                --theme-text: #f3f4f6;
                --kpi-bg: rgba(255, 255, 255, 0.04);
                --kpi-border: rgba(255, 255, 255, 0.08);
                --kpi-label: #9ca3af;
                --kpi-value: #f3f4f6;
                --kpi-sub: #6b7280;
                --tab-bg: rgba(255, 255, 255, 0.02);
                --tab-border: rgba(255, 255, 255, 0.05);
                --tab-text: #9ca3af;
                --tab-text-hover: #f3f4f6;
                --tab-bg-hover: rgba(255, 255, 255, 0.06);
                --tab-selected-bg: rgba(255, 255, 255, 0.08);
                --tab-selected-border: rgba(255, 255, 255, 0.15);
                --tab-selected-text: #f3f4f6;
                --pill-inactive-bg: rgba(255, 255, 255, 0.03);
                --pill-inactive-border: rgba(255, 255, 255, 0.08);
                --pill-inactive-text: #9ca3af;
                --pill-inactive-hover-bg: rgba(255, 255, 255, 0.08);
                --pill-inactive-hover-border: rgba(255, 255, 255, 0.18);
                --pill-inactive-hover-text: #ffffff;
                
                /* Force Streamlit native variables */
                --background-color: #0e1117 !important;
                --secondary-background-color: #1a1c23 !important;
                --text-color: #f3f4f6 !important;
            }
            """
        else:  # system preference
            variables = """
            .stApp {
                --theme-bg: #0e1117;
                --theme-text: #f3f4f6;
                --kpi-bg: rgba(255, 255, 255, 0.04);
                --kpi-border: rgba(255, 255, 255, 0.08);
                --kpi-label: #9ca3af;
                --kpi-value: #f3f4f6;
                --kpi-sub: #6b7280;
                --tab-bg: rgba(255, 255, 255, 0.02);
                --tab-border: rgba(255, 255, 255, 0.05);
                --tab-text: #9ca3af;
                --tab-text-hover: #f3f4f6;
                --tab-bg-hover: rgba(255, 255, 255, 0.06);
                --tab-selected-bg: rgba(255, 255, 255, 0.08);
                --tab-selected-border: rgba(255, 255, 255, 0.15);
                --tab-selected-text: #f3f4f6;
                --pill-inactive-bg: rgba(255, 255, 255, 0.03);
                --pill-inactive-border: rgba(255, 255, 255, 0.08);
                --pill-inactive-text: #9ca3af;
                --pill-inactive-hover-bg: rgba(255, 255, 255, 0.08);
                --pill-inactive-hover-border: rgba(255, 255, 255, 0.18);
                --pill-inactive-hover-text: #ffffff;
            }
            @media (prefers-color-scheme: light) {
                .stApp {
                    --theme-bg: #ffffff;
                    --theme-text: #1f2937;
                    --kpi-bg: rgba(0, 0, 0, 0.02);
                    --kpi-border: rgba(0, 0, 0, 0.06);
                    --kpi-label: #4b5563;
                    --kpi-value: #111827;
                    --kpi-sub: #6b7280;
                    --tab-bg: rgba(0, 0, 0, 0.01);
                    --tab-border: rgba(0, 0, 0, 0.04);
                    --tab-text: #4b5563;
                    --tab-text-hover: #111827;
                    --tab-bg-hover: rgba(0, 0, 0, 0.04);
                    --tab-selected-bg: rgba(0, 0, 0, 0.06);
                    --tab-selected-border: rgba(0, 0, 0, 0.12);
                    --tab-selected-text: #111827;
                    --pill-inactive-bg: rgba(0, 0, 0, 0.02);
                    --pill-inactive-border: rgba(0, 0, 0, 0.06);
                    --pill-inactive-text: #4b5563;
                    --pill-inactive-hover-bg: rgba(0, 0, 0, 0.06);
                    --pill-inactive-hover-border: rgba(0, 0, 0, 0.12);
                    --pill-inactive-hover-text: #111827;
                    
                    /* Force Streamlit native variables */
                    --background-color: #ffffff !important;
                    --secondary-background-color: #f0f2f6 !important;
                    --text-color: #1f2937 !important;
                }
            }
            """
        st.markdown(f"<style>{variables}\n{cls.CSS}</style>", unsafe_allow_html=True)

    @staticmethod
    def render_kpi_card(label: str, value: str, subtext: str = "") -> str:
        """Generates the HTML representation of a premium KPI card."""
        sub_html = f'<div class="kpi-sub">{subtext}</div>' if subtext else ""
        return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """
