# pyrefly: ignore [missing-import]
import streamlit as st

class StyleManager:
    """Manages custom styling and premium aesthetics for the dashboard."""
    
    PALETTE = {
        "ACORN-E": "#2f6f73",  # Affluent - Sage green
        "ACORN-F": "#7b5f39",  # Comfortable - Warm brown
        "ACORN-Q": "#8f3f46",  # Adversity - Crimson/maroon
    }

    # CSS with Google Fonts (Outfit), glassmorphism, smooth animations, and custom cards.
    CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    /* Global Typography & Background adjustments */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
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
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
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
        color: #9ca3af;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f3f4f6;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.75rem;
        color: #6b7280;
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
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px 8px 0px 0px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: none;
        color: #9ca3af;
        transition: all 0.2s ease-in-out;
        padding: 0 16px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #f3f4f6;
        background-color: rgba(255, 255, 255, 0.06);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
        color: #f3f4f6 !important;
        font-weight: 600 !important;
    }
    """

    @classmethod
    def apply(cls) -> None:
        """Injects custom global CSS into the Streamlit app context."""
        st.markdown(f"<style>{cls.CSS}</style>", unsafe_allow_html=True)

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
