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
        "xgboost_by_acorn": "XGBoost by ACORN",
        "ridge": "Ridge Regression",
        "catboost": "CatBoost",
        "lightgbm": "LightGBM",
        "stack_regressor": "Stack Regressor",
        "autogluon": "AutoGluon Tabular",
        "autogluon_best": "AutoGluon Tabular Best",
        "autogluon_timeseries": "AutoGluon TimeSeries",
        "autogluon_timeseries_best": "AutoGluon TimeSeries Best",
        "previous_day": "Previous Day",
        "previous_week": "Previous Week",
        "seasonal_mean": "Seasonal Mean",
        "selected": "Selected Model"
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

    /* Force standard headers, labels, and text containers to inherit theme text and Outfit font */
    h1, h2, h3, h4, h5, h6, [data-testid="stWidgetLabel"] p, label, table, th, td {
        font-family: 'Outfit', sans-serif !important;
        color: var(--theme-text) !important;
        transition: color 0.3s ease;
    }

    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.02em !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Modern text styling for paragraphs and list items */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.02rem !important;
        line-height: 1.7 !important;
        font-weight: 300 !important;
        color: var(--kpi-sub) !important;
        letter-spacing: 0.01em !important;
        transition: color 0.3s ease;
        margin-bottom: 1.0rem !important;
    }

    [data-testid="stMarkdownContainer"] p strong,
    [data-testid="stMarkdownContainer"] li strong {
        font-weight: 600 !important;
        color: var(--theme-text) !important;
    }

    /* Premium Glassmorphic Intro Panels */
    .intro-panel {
        background: linear-gradient(135deg, rgba(77, 164, 169, 0.04) 0%, rgba(212, 156, 94, 0.04) 100%) !important;
        border: 1px solid var(--kpi-border) !important;
        border-left: 4px solid #4da4a9 !important;
        border-radius: 12px !important;
        padding: 1.25rem 1.5rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 1.75rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }

    .intro-panel:hover {
        border-color: rgba(77, 164, 169, 0.4) !important;
        border-left-color: #4da4a9 !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08) !important;
    }

    .intro-panel p {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        line-height: 1.75 !important;
        font-weight: 300 !important;
        color: var(--theme-text) !important;
        margin: 0 !important;
    }

    /* Keep Streamlit's built-in Material Symbols icons from rendering as raw text names. */
    span[data-testid="stIconMaterial"],
    span[class*="material-symbols"],
    span[class*="material-icons"],
    [data-testid="stExpander"] summary span[data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary span[class*="material"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        color: var(--theme-text) !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
        text-rendering: optimizeLegibility !important;
    }

    div[data-testid="stExpander"] {
        background-color: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] summary {
        background-color: transparent !important;
        color: var(--theme-text) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {
        margin: 0 !important;
        padding: 0 !important;
        line-height: inherit !important;
    }

    /* Reduce main container margins to let contents extend closer to the screen edges */
    .block-container {
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-top: 3.5rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* Force dropdown selectboxes, multi-selects, and list options to show pointer cursor on hover */
    div[data-testid="stSelectbox"],
    div[data-testid="stSelectbox"] *,
    div[data-testid="stMultiSelect"],
    div[data-testid="stMultiSelect"] *,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label *,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] li,
    [role="listbox"] li,
    [role="listbox"] [role="option"] {
        cursor: pointer !important;
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
        height: 48px;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        white-space: pre-wrap;
        background-color: var(--tab-bg) !important;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid var(--tab-border) !important;
        border-bottom: none;
        color: var(--tab-text) !important;
        transition: all 0.2s ease-in-out;
        padding: 0 20px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .stTabs [data-baseweb="tab"] * {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
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

    div[data-testid="column"]:has(.st-key-acorn_filter),
    div[data-testid="column"]:has(.st-key-theme_toggle),
    div[data-testid="column"]:has(.st-key-acorn_filter) div[data-testid="stVerticalBlock"],
    div[data-testid="column"]:has(.st-key-theme_toggle) div[data-testid="stVerticalBlock"],
    div.element-container:has(.st-key-acorn_filter),
    div.element-container:has(.st-key-theme_toggle) {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        justify-content: flex-end !important;
        text-align: right !important;
        width: 100% !important;
        margin: 0 !important;
        margin-bottom: 0px !important;
        padding: 0 !important;
        padding-bottom: 0px !important;
        gap: 0px !important;
    }

    .st-key-acorn_filter {
        margin: 0 !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        margin-bottom: 0px !important;
        width: fit-content !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-end !important;
        justify-content: flex-end !important;
        text-align: right !important;
        padding: 0 !important;
        padding-top: 0.5rem !important;
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
        margin-bottom: 0px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        flex-wrap: nowrap !important;
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
        font-size: 0.92rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 0 1.25rem !important;
        height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"] *,
    .st-key-acorn_filter [data-testid="stBaseButton-pills"] * {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
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

    /* Color for ACORN-E (Sage green) when active */
    .st-key-acorn_filter [role="group"] > div:nth-child(1) [data-testid="stBaseButton-pillsActive"],
    .st-key-acorn_filter [role="group"] > button:nth-child(1)[data-testid="stBaseButton-pillsActive"] {
        background-color: #2f6f73 !important;
        border-color: #4da4a9 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(47, 111, 115, 0.3) !important;
    }

    /* Color for ACORN-F (Warm brown) when active */
    .st-key-acorn_filter [role="group"] > div:nth-child(2) [data-testid="stBaseButton-pillsActive"],
    .st-key-acorn_filter [role="group"] > button:nth-child(2)[data-testid="stBaseButton-pillsActive"] {
        background-color: #7b5f39 !important;
        border-color: #d49c5e !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(123, 95, 57, 0.3) !important;
    }

    /* Color for ACORN-Q (Crimson red) when active */
    .st-key-acorn_filter [role="group"] > div:nth-child(3) [data-testid="stBaseButton-pillsActive"],
    .st-key-acorn_filter [role="group"] > button:nth-child(3)[data-testid="stBaseButton-pillsActive"] {
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

    /* Force the filter column to grow and the toggle column to shrink to content, and align them flush at the bottom */
    div[data-testid="column"] div[data-testid="stHorizontalBlock"]:has(.st-key-theme_toggle) > div[data-testid="column"]:nth-child(1) {
        width: auto !important;
        flex-grow: 1 !important;
        flex-shrink: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-end !important;
        align-items: flex-end !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }

    div[data-testid="column"] div[data-testid="stHorizontalBlock"]:has(.st-key-theme_toggle) > div[data-testid="column"]:nth-child(2) {
        width: auto !important;
        flex-grow: 0 !important;
        flex-shrink: 0 !important;
        margin-left: 2.5rem !important; /* Increased space between filter pills and toggle button */
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-end !important;
        align-items: flex-end !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }

    .st-key-theme_toggle {
        margin: 0 !important;
        margin-bottom: 0px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: flex-end !important;
    }

    .st-key-theme_toggle button {
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        max-width: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 36px !important;
        font-size: 1.05rem !important;
        background-color: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        color: var(--theme-text) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease-in-out !important;
        margin-bottom: 0px !important;
    }

    .st-key-theme_toggle button * {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    .st-key-theme_toggle button:hover {
        background-color: var(--pill-inactive-hover-bg) !important;
        border-color: var(--pill-inactive-hover-border) !important;
        transform: scale(1.05) !important;
    }

    /* Dropdowns / Selectboxes styling overrides */
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stMultiSelect"] [data-baseweb="select"] {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stSelectbox"] [data-baseweb="select"] *,
    div[data-testid="stMultiSelect"] [data-baseweb="select"] *:not([data-baseweb="tag"]):not([data-baseweb="tag"] *) {
        background-color: transparent !important;
        color: var(--theme-text) !important;
    }

    div[data-testid="stSelectbox"] *,
    div[data-testid="stMultiSelect"] * {
        color: var(--theme-text) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: var(--pill-inactive-bg) !important;
        border: 1px solid var(--kpi-border) !important;
    }

    div[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
        color: var(--theme-text) !important;
    }
    
    div[data-testid="stSelectbox"] svg,
    div[data-testid="stMultiSelect"] svg {
        fill: var(--theme-text) !important;
        color: var(--theme-text) !important;
    }
    
    div[data-baseweb="popover"],
    div[role="listbox"],
    ul[role="listbox"],
    [data-testid="stVirtualDropdown"] {
        background-color: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
    }
    
    div[data-baseweb="popover"] *,
    div[role="listbox"] *,
    ul[role="listbox"] *,
    [data-testid="stVirtualDropdown"] * {
        background-color: transparent !important;
        color: var(--theme-text) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    div[data-baseweb="popover"] [role="option"],
    div[role="listbox"] [role="option"],
    ul[role="listbox"] li,
    [data-testid="stVirtualDropdown"] [role="option"],
    [data-testid="stVirtualDropdown"] li {
        background-color: transparent !important;
        color: var(--theme-text) !important;
        font-family: 'Outfit', sans-serif !important;
        transition: background-color 0.2s ease !important;
    }
    
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"],
    div[role="listbox"] [role="option"]:hover,
    div[role="listbox"] [role="option"][aria-selected="true"],
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"],
    [data-testid="stVirtualDropdown"] [role="option"]:hover,
    [data-testid="stVirtualDropdown"] [role="option"][aria-selected="true"],
    [data-testid="stVirtualDropdown"] li:hover,
    [data-testid="stVirtualDropdown"] li[aria-selected="true"] {
        background-color: var(--dropdown-hover-bg) !important;
        color: var(--theme-text) !important;
    }

    /* Radio buttons layout spacing and typography */
    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex !important;
        gap: 1.5rem !important;
    }
    
    div[data-testid="stRadio"] [role="radiogroup"] label,
    div[data-testid="stRadio"] [role="radiogroup"] label * {
        font-family: 'Outfit', sans-serif !important;
        color: var(--theme-text) !important;
    }
    


    /* Tooltips and help popover styling overrides */
    div[data-testid="stTooltipIcon"],
    button[data-testid="stHelpButton"],
    [data-testid="stTooltipHoverTarget"] {
        color: var(--theme-text) !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Ensure the help button vectors use the theme stroke and have no solid fill */
    div[data-testid="stTooltipIcon"] svg *,
    button[data-testid="stHelpButton"] svg *,
    [data-testid="stTooltipHoverTarget"] svg * {
        stroke: var(--theme-text) !important;
        fill: none !important;
    }
    
    /* Popover / tooltips containers styling */
    div[role="tooltip"],
    div[data-baseweb="tooltip"],
    div[data-testid="stTooltipContent"] {
        background-color: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        border-radius: 6px !important;
    }

    div[role="tooltip"] *,
    div[data-baseweb="tooltip"] *,
    div[data-testid="stTooltipContent"] * {
        background-color: transparent !important;
        color: var(--theme-text) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* Active selected button font color override */
    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"] *,
    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"] span,
    .st-key-acorn_filter [data-testid="stBaseButton-pillsActive"] p {
        color: #ffffff !important;
    }

    /* Inactive unselected button font color override */
    .st-key-acorn_filter [data-testid="stBaseButton-pills"] *,
    .st-key-acorn_filter [data-testid="stBaseButton-pills"] span,
    .st-key-acorn_filter [data-testid="stBaseButton-pills"] p {
        color: var(--pill-inactive-text) !important;
    }

    .st-key-acorn_filter [data-testid="stBaseButton-pills"]:hover *,
    .st-key-acorn_filter [data-testid="stBaseButton-pills"]:hover span,
    .st-key-acorn_filter [data-testid="stBaseButton-pills"]:hover p {
        color: var(--pill-inactive-hover-text) !important;
    }

    /* Toast notifications styling */
    div[data-testid="stToast"],
    div[data-testid="stToast"] * {
        background-color: var(--kpi-bg) !important;
        color: var(--theme-text) !important;
        border-color: var(--kpi-border) !important;
        font-family: 'Outfit', sans-serif !important;
    }


    /* Tables (st.table and markdown tables) styling overrides */
    div[data-testid="stTable"],
    .scrollable-table-wrapper {
        max-height: 400px !important;
        overflow-y: auto !important;
        border: 1px solid var(--kpi-border) !important;
        border-radius: 8px !important;
        margin-top: 1.0rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* Custom premium scrollbar for table container */
    div[data-testid="stTable"]::-webkit-scrollbar,
    .scrollable-table-wrapper::-webkit-scrollbar {
        width: 6px !important;
        height: 6px !important;
    }
    div[data-testid="stTable"]::-webkit-scrollbar-track,
    .scrollable-table-wrapper::-webkit-scrollbar-track {
        background: transparent !important;
    }
    div[data-testid="stTable"]::-webkit-scrollbar-thumb,
    .scrollable-table-wrapper::-webkit-scrollbar-thumb {
        background: var(--kpi-border) !important;
        border-radius: 3px !important;
    }
    div[data-testid="stTable"]::-webkit-scrollbar-thumb:hover,
    .scrollable-table-wrapper::-webkit-scrollbar-thumb:hover {
        background: var(--theme-text) !important;
    }

    div[data-testid="stTable"] table,
    .scrollable-table-wrapper table,
    [data-testid="stMarkdownContainer"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        border: none !important;
        margin: 0 !important;
    }

    div[data-testid="stTable"] th,
    .scrollable-table-wrapper th,
    [data-testid="stMarkdownContainer"] th {
        background-color: var(--kpi-bg) !important;
        color: var(--theme-text) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--kpi-border) !important;
        padding: 10px 14px !important;
        text-align: left !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
        box-shadow: 0 1px 0 var(--kpi-border) !important;
    }

    div[data-testid="stTable"] td,
    .scrollable-table-wrapper td,
    [data-testid="stMarkdownContainer"] td {
        border-bottom: 1px solid var(--kpi-border) !important;
        padding: 10px 14px !important;
        color: var(--theme-text) !important;
        background-color: transparent !important;
    }

    div[data-testid="stTable"] tr:hover,
    .scrollable-table-wrapper tr:hover,
    [data-testid="stMarkdownContainer"] tr:hover {
        background-color: var(--table-hover-bg) !important;
    }

    /* Premium Member Cards & Team Layout */
    .member-grid {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.75rem !important;
        margin-top: 0.5rem !important;
        width: 100% !important;
    }

    .member-card {
        background: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .member-card:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(77, 164, 169, 0.4) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1) !important;
    }

    .member-avatar {
        width: 38px !important;
        height: 38px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #4da4a9 0%, #d49c5e 100%) !important;
        color: #ffffff !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        flex-shrink: 0 !important;
    }

    .member-info {
        display: flex !important;
        flex-direction: column !important;
        flex-grow: 1 !important;
        line-height: 1.2 !important;
    }

    .member-name {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: var(--theme-text) !important;
    }

    .member-detail {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 300 !important;
        font-size: 0.8rem !important;
        color: var(--kpi-sub) !important;
        margin-top: 0.15rem !important;
    }

    /* Premium Repository Badge */
    .repo-badge-container {
        margin-top: 0.75rem !important;
        margin-bottom: 1.25rem !important;
        width: fit-content !important;
    }

    .repo-badge {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.6rem !important;
        background: var(--kpi-bg) !important;
        border: 1px solid var(--kpi-border) !important;
        border-radius: 20px !important;
        padding: 0.45rem 1.1rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: var(--theme-text) !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }

    .repo-badge:hover {
        border-color: #4da4a9 !important;
        background: rgba(77, 164, 169, 0.05) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(77, 164, 169, 0.15) !important;
    }

    .repo-badge span.material-symbols-rounded {
        font-size: 1.1rem !important;
        color: #4da4a9 !important;
        line-height: 1 !important;
    }
    """

    @classmethod
    def apply(cls, theme_mode: str = "system") -> None:
        """Injects custom global CSS into the Streamlit app context based on theme mode."""
        # Define theme mode variables dynamically
        if theme_mode == "light":
            variables = """
            :root, .stApp {
                --theme-bg: #ffffff;
                --theme-text: #1f2937;
                --kpi-bg: #f9fafb;
                --kpi-border: #e5e7eb;
                --kpi-label: #4b5563;
                --kpi-value: #111827;
                --kpi-sub: #6b7280;
                --tab-bg: #f9fafb;
                --tab-border: #e5e7eb;
                --tab-text: #6b7280;
                --tab-text-hover: #374151;
                --tab-bg-hover: #f3f4f6;
                --tab-selected-bg: #ffffff;
                --tab-selected-border: #e5e7eb;
                --tab-selected-text: #111827;
                --pill-inactive-bg: #f3f4f6;
                --pill-inactive-border: #e5e7eb;
                --pill-inactive-text: #374151;
                --pill-inactive-hover-bg: #e5e7eb;
                --pill-inactive-hover-border: #d1d5db;
                --pill-inactive-hover-text: #111827;
                
                --table-hover-bg: #f3f4f6;
                --dropdown-hover-bg: #e5e7eb;
                --input-bg: #ffffff;
                --input-border: #d1d5db;
                
                /* Force Streamlit native variables */
                --primary-color: #2f6f73 !important;
                --background-color: #ffffff !important;
                --secondary-background-color: #f0f2f6 !important;
                --text-color: #1f2937 !important;
            }
            """
        elif theme_mode == "dark":
            variables = """
            :root, .stApp {
                --theme-bg: #0e1117;
                --theme-text: #f3f4f6;
                --kpi-bg: #1f2937;
                --kpi-border: #374151;
                --kpi-label: #9ca3af;
                --kpi-value: #ffffff;
                --kpi-sub: #9ca3af;
                --tab-bg: #1f2937;
                --tab-border: #374151;
                --tab-text: #9ca3af;
                --tab-text-hover: #f3f4f6;
                --tab-bg-hover: #374151;
                --tab-selected-bg: #0e1117;
                --tab-selected-border: #374151;
                --tab-selected-text: #ffffff;
                --pill-inactive-bg: #1f2937;
                --pill-inactive-border: #374151;
                --pill-inactive-text: #9ca3af;
                --pill-inactive-hover-bg: #374151;
                --pill-inactive-hover-border: #4b5563;
                --pill-inactive-hover-text: #ffffff;
                
                --table-hover-bg: #1f2937;
                --dropdown-hover-bg: #374151;
                --input-bg: #1f2937;
                --input-border: #374151;
                
                /* Force Streamlit native variables */
                --primary-color: #2f6f73 !important;
                --background-color: #0e1117 !important;
                --secondary-background-color: #1a1c23 !important;
                --text-color: #f3f4f6 !important;
            }
            """
        else:  # system preference
            variables = """
            :root, .stApp {
                --theme-bg: #0e1117;
                --theme-text: #f3f4f6;
                --kpi-bg: #1f2937;
                --kpi-border: #374151;
                --kpi-label: #9ca3af;
                --kpi-value: #ffffff;
                --kpi-sub: #9ca3af;
                --tab-bg: #1f2937;
                --tab-border: #374151;
                --tab-text: #9ca3af;
                --tab-text-hover: #f3f4f6;
                --tab-bg-hover: #374151;
                --tab-selected-bg: #0e1117;
                --tab-selected-border: #374151;
                --tab-selected-text: #ffffff;
                --pill-inactive-bg: #1f2937;
                --pill-inactive-border: #374151;
                --pill-inactive-text: #9ca3af;
                --pill-inactive-hover-bg: #374151;
                --pill-inactive-hover-border: #4b5563;
                --pill-inactive-hover-text: #ffffff;
                
                --table-hover-bg: #1f2937;
                --dropdown-hover-bg: #374151;
                --input-bg: #1f2937;
                --input-border: #374151;
            }
            @media (prefers-color-scheme: light) {
                :root, .stApp {
                    --theme-bg: #ffffff;
                    --theme-text: #1f2937;
                    --kpi-bg: #f9fafb;
                    --kpi-border: #e5e7eb;
                    --kpi-label: #4b5563;
                    --kpi-value: #111827;
                    --kpi-sub: #6b7280;
                    --tab-bg: #f9fafb;
                    --tab-border: #e5e7eb;
                    --tab-text: #6b7280;
                    --tab-text-hover: #374151;
                    --tab-bg-hover: #f3f4f6;
                    --tab-selected-bg: #ffffff;
                    --tab-selected-border: #e5e7eb;
                    --tab-selected-text: #111827;
                    --pill-inactive-bg: #f3f4f6;
                    --pill-inactive-border: #e5e7eb;
                    --pill-inactive-text: #374151;
                    --pill-inactive-hover-bg: #e5e7eb;
                    --pill-inactive-hover-border: #d1d5db;
                    --pill-inactive-hover-text: #111827;
                    
                    --table-hover-bg: #f3f4f6;
                    --dropdown-hover-bg: #e5e7eb;
                    --input-bg: #ffffff;
                    --input-border: #d1d5db;
                    
                    /* Force Streamlit native variables */
                    --primary-color: #2f6f73 !important;
                    --background-color: #ffffff !important;
                    --secondary-background-color: #f0f2f6 !important;
                    --text-color: #1f2937 !important;
                }
            }
            """
        st.markdown(f"<style>{variables}\n{cls.CSS}</style>", unsafe_allow_html=True)

    @classmethod
    def style_plotly_chart(cls, fig, theme_mode: str = "system") -> None:
        """Applies a premium, theme-consistent style to a Plotly figure."""
        is_light = (theme_mode == "light")
        
        bg_color = "rgba(0,0,0,0)"
        text_color = "#1f2937" if is_light else "#f3f4f6"
        grid_color = "rgba(0, 0, 0, 0.08)" if is_light else "rgba(255, 255, 255, 0.08)"
        zeroline_color = "rgba(0, 0, 0, 0.15)" if is_light else "rgba(255, 255, 255, 0.15)"
        
        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(
                family="'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                color=text_color
            ),
            legend=dict(
                font=dict(color=text_color),
                title=dict(font=dict(color=text_color)),
                bgcolor="rgba(0,0,0,0)"
            ),
            title=dict(
                font=dict(color=text_color, size=16)
            ),
            coloraxis_colorbar=dict(
                tickfont=dict(color=text_color),
                title=dict(font=dict(color=text_color))
            )
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor=grid_color,
            zeroline=True,
            zerolinecolor=zeroline_color,
            tickfont=dict(color=text_color),
            title_font=dict(color=text_color)
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=grid_color,
            zeroline=True,
            zerolinecolor=zeroline_color,
            tickfont=dict(color=text_color),
            title_font=dict(color=text_color)
        )
        
        # Dynamically adjust relative bar width based on number of active categories to keep them visually pleasing
        x_values = []
        for trace in fig.data:
            if getattr(trace, "type", None) == "bar" and hasattr(trace, "x") and trace.x is not None:
                try:
                    x_values.extend(list(trace.x))
                except Exception:
                    pass
        
        if x_values:
            num_categories = len(set(x_values))
            if num_categories == 1:
                bar_width = 0.15
            elif num_categories == 2:
                bar_width = 0.25
            elif num_categories == 3:
                bar_width = 0.35
            else:
                bar_width = 0.45
            fig.update_traces(width=bar_width, selector=dict(type="bar"))
        else:
            fig.update_traces(width=0.4, selector=dict(type="bar"))

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
