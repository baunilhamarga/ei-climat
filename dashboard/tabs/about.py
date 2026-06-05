from __future__ import annotations

# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
from dashboard.tabs.base import BaseTab
from dashboard.styles import StyleManager
from group5_energy.config import ACORN_GROUPS


class AboutTab(BaseTab):
    """Project context and scope for the Group 5 dashboard."""

    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        daily = data["daily_enriched"]
        kpi_html = f"""
        <div class="kpi-container">
            {StyleManager.render_kpi_card("Group Scope", "3 ACORNs", "E, F, and Q household segments")}
            {StyleManager.render_kpi_card("Short-Term", "48h", "30-minute forecasts")}
            {StyleManager.render_kpi_card("Medium-Term", "1 month", "Daily forecasts")}
            {StyleManager.render_kpi_card("Validation", "RMSE", "Chronological holdout")}
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.markdown(
            '<div class="intro-panel">'
            '<p>This dashboard summarizes the Group 5 electricity-consumption forecasting project for UK '
            'household ACORN segments. The goal is to understand historical consumption behaviour, '
            'quantify weather and calendar effects, compare forecasting models, and produce the two final '
            'assignment forecasts for the assigned segments.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        scope_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th>Task</th>'
            '<th>Target Period</th>'
            '<th>Resolution</th>'
            '<th>Required Rows</th>'
            '<th>Output Column & Unit</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            '<tr>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Short-Term Horizon</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Half-hourly predictions</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.9rem; color: var(--theme-text);">Jan 13 – Jan 14, 2014</div>'
            '<div style="font-size: 0.82rem; color: #4da4a9; font-weight: 500; margin-top: 2px;">48-Hour continuous window</div>'
            '</td>'
            '<td>'
            '<div style="font-size: 0.9rem; color: var(--theme-text);">30-minute intervals</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.9rem; color: var(--theme-text);">288 rows</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">96 steps × 3 ACORN segments</div>'
            '</td>'
            '<td>'
            '<code style="font-family: monospace; font-size: 0.82rem; background: var(--kpi-border); color: #4da4a9; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">Conso_moy_predict</code>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 4px;">Average active power (kW)</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Medium-Term Horizon</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Daily predictions</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.9rem; color: var(--theme-text);">Jan 13 – Feb 13, 2014</div>'
            '<div style="font-size: 0.82rem; color: #d49c5e; font-weight: 500; margin-top: 2px;">32-Day forecast window</div>'
            '</td>'
            '<td>'
            '<div style="font-size: 0.9rem; color: var(--theme-text);">Daily totals</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.9rem; color: var(--theme-text);">96 rows</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">32 days × 3 ACORN segments</div>'
            '</td>'
            '<td>'
            '<code style="font-family: monospace; font-size: 0.82rem; background: var(--kpi-border); color: #d49c5e; padding: 2px 6px; border-radius: 4px; font-weight: 600; display: inline-block;">Conso_kWh_predict</code>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 4px;">Total active energy (kWh)</div>'
            '</td>'
            '</tr>'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(scope_html, unsafe_allow_html=True)

        st.subheader("ACORN Segments")
        acorn_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th style="width: 25%;">ACORN Segment</th>'
            '<th style="width: 25%;">Classification</th>'
            '<th style="width: 50%;">Dashboard Role & Behavior</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            '<tr>'
            '<td>'
            '<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; background: rgba(77, 164, 169, 0.12); color: #4da4a9; border: 1px solid rgba(77, 164, 169, 0.25);">ACORN-E</span>'
            '</td>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Affluent</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Higher-income tier</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">High-consumption profile</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Used to model wealthier households with high baseline and heating/cooling sensitivity.</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            '<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; background: rgba(212, 156, 94, 0.12); color: #d49c5e; border: 1px solid rgba(212, 156, 94, 0.25);">ACORN-F</span>'
            '</td>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Comfortable</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Middle-income tier</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">Average-consumption profile</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Used to model average UK households showing typical baseline usage and holiday drops.</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            '<span style="display: inline-block; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; background: rgba(200, 90, 100, 0.12); color: #c85a64; border: 1px solid rgba(200, 90, 100, 0.25);">ACORN-Q</span>'
            '</td>'
            '<td>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Adversity</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Lower-income tier</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">Low-consumption profile</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Used to model economically constrained households with low overall electricity consumption.</div>'
            '</td>'
            '</tr>'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(acorn_html, unsafe_allow_html=True)

        st.subheader("Data Used")
        start = daily["Date"].min().date() if not daily.empty else "n/a"
        end = daily["Date"].max().date() if not daily.empty else "n/a"
        st.markdown(
            f"Historical consumption covers `{start}` to `{end}` for the three assigned ACORN "
            "segments. The dashboard also uses weather variables, public-holiday flags, client counts, "
            "and generated lag/rolling features for model comparison."
        )

        st.subheader("Dashboard Flow")
        flow_html = (
            '<div class="scrollable-table-wrapper">'
            '<table class="scope-table" style="width:100%; border-collapse:collapse;">'
            '<thead>'
            '<tr>'
            '<th style="width: 30%;">Dashboard Section</th>'
            '<th style="width: 70%;">Analysis Scope & Purpose</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            '<tr>'
            '<td>'
            '<div style="display: flex; align-items: center; gap: 8px;">'
            '<span style="font-size: 1.2rem;">📌</span>'
            '<div>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Overview</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Key metrics</div>'
            '</div>'
            '</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">High-level forecast summary & model selections</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Row validation counts, best models by RMSE, and quick configuration audit checks.</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            '<div style="display: flex; align-items: center; gap: 8px;">'
            '<span style="font-size: 1.2rem;">🔎</span>'
            '<div>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">EDA</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Exploration</div>'
            '</div>'
            '</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">Historical profiles, weather interactions & correlations</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Interactive views of seasonal trends, weather sensitivity curves, and autocorrelations.</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            '<div style="display: flex; align-items: center; gap: 8px;">'
            '<span style="font-size: 1.2rem;">✅</span>'
            '<div>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Validation</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Performance</div>'
            '</div>'
            '</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">Chronological holdout model evaluation</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">RMSE rankings and comparison charts for Ridge, XGBoost, LightGBM, CatBoost, and AutoGluon.</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            '<div style="display: flex; align-items: center; gap: 8px;">'
            '<span style="font-size: 1.2rem;">📈</span>'
            '<div>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Forecasts</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Predictions</div>'
            '</div>'
            '</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">Interactive forecast curves & raw outputs</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Visualize final deliverables, overlay predictions, and download CSV/Parquet files.</div>'
            '</td>'
            '</tr>'
            '<tr>'
            '<td>'
            '<div style="display: flex; align-items: center; gap: 8px;">'
            '<span style="font-size: 1.2rem;">🌱</span>'
            '<div>'
            '<div style="font-weight: 600; font-size: 0.95rem; color: var(--theme-text);">Carbon Footprint</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Eco impact</div>'
            '</div>'
            '</div>'
            '</td>'
            '<td>'
            '<div style="font-weight: 500; font-size: 0.9rem; color: var(--theme-text);">Carbon intensity scenarios & charging profiles</div>'
            '<div style="font-size: 0.82rem; color: var(--kpi-sub); margin-top: 2px;">Forecasted CO₂ equivalent emissions comparing peak hour load shifting scenarios.</div>'
            '</td>'
            '</tr>'
            '</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(flow_html, unsafe_allow_html=True)

        st.subheader("Credits and Sources")
        st.markdown(
            "This project was carried out as part of a CentraleSupélec and EDF partnership. "
            "The dashboard uses the assignment-provided `Households in the United Kingdom` dataset: "
            "preprocessed electricity consumption profiles by ACORN segment, weather data from the "
            "provided Dark Sky files, and UK public holiday data."
        )
        
        repo_html = """
        <div class="repo-badge-container">
            <a href="https://github.com/baunilhamarga/ei-climat" target="_blank" style="text-decoration: none;">
                <div class="repo-badge">
                    <span class="material-symbols-rounded">link</span>
                    github.com/baunilhamarga/ei-climat
                </div>
            </a>
        </div>
        """
        st.markdown(repo_html, unsafe_allow_html=True)

        students_data = [
            {"name": "Artur Bandeira Chan Jorge", "email": "artur.bandeira-chan-jorge@student-cs.fr"},
            {"name": "Pedro Lubaszewski Lima", "email": "pedro.lubaszewski-lima@student-cs.fr"},
            {"name": "Ignacio Lopez Acevedo", "email": "ignacio.lopez-acevedo@student-cs.fr"},
            {"name": "Heitor Gama", "email": "heitor.gama@student-cs.fr"},
        ]

        supervisors_data = [
            {"name": "Laurent Bozzi", "role": "Data scientist, EDF R&D"},
            {"name": "Théodore Cherriere", "role": "Enseignant-chercheur, CentraleSupélec - GeePs"},
            {"name": "Jing Dai", "role": "Enseignant-chercheur, CentraleSupélec - GeePs"},
            {"name": "Dung Lé", "role": "Enseignant-chercheur, CentraleSupélec - GeePs"},
        ]

        def get_initials(name: str) -> str:
            parts = name.split()
            if len(parts) >= 2:
                return (parts[0][0] + parts[-1][0]).upper()
            return name[:2].upper()

        students_html = '<div class="member-grid">'
        for student in students_data:
            initials = get_initials(student["name"])
            students_html += (
                f'<div class="member-card">'
                f'<div class="member-avatar">{initials}</div>'
                f'<div class="member-info">'
                f'<div class="member-name">{student["name"]}</div>'
                f'<div class="member-detail"><a href="mailto:{student["email"]}" style="color: inherit; text-decoration: none;">{student["email"]}</a></div>'
                f'</div>'
                f'</div>'
            )
        students_html += '</div>'

        supervisors_html = '<div class="member-grid">'
        for supervisor in supervisors_data:
            initials = get_initials(supervisor["name"])
            supervisors_html += (
                f'<div class="member-card">'
                f'<div class="member-avatar">{initials}</div>'
                f'<div class="member-info">'
                f'<div class="member-name">{supervisor["name"]}</div>'
                f'<div class="member-detail">{supervisor["role"]}</div>'
                f'</div>'
                f'</div>'
            )
        supervisors_html += '</div>'

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p style="font-weight: 600 !important; font-size: 1.05rem !important; margin-top: 1.25rem !important; margin-bottom: 0.5rem !important;">Student analysis team</p>', unsafe_allow_html=True)
            st.markdown(students_html, unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="mobile-only-spacer"></div>', unsafe_allow_html=True)
            st.markdown('<p style="font-weight: 600 !important; font-size: 1.05rem !important; margin-top: 1.25rem !important; margin-bottom: 0.5rem !important;">Supervision</p>', unsafe_allow_html=True)
            st.markdown(supervisors_html, unsafe_allow_html=True)
