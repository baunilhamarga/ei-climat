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

        st.subheader("Project Context")
        st.markdown(
            "This dashboard summarizes the Group 5 electricity-consumption forecasting project for UK "
            "household ACORN segments. The goal is to understand historical consumption behaviour, "
            "quantify weather and calendar effects, compare forecasting models, and produce the two final "
            "assignment forecasts for the assigned segments."
        )

        scope = pd.DataFrame(
            [
                {
                    "Task": "Short-term forecast",
                    "Period": "2014-01-13 00:00 to 2014-01-14 23:30",
                    "Resolution": "30 minutes",
                    "Rows": "288",
                    "Output column": "Conso_moy_predict",
                },
                {
                    "Task": "Medium-term forecast",
                    "Period": "2014-01-13 to 2014-02-13",
                    "Resolution": "Daily",
                    "Rows": "96",
                    "Output column": "Conso_kWh_predict",
                },
            ]
        )
        st.markdown(f'<div class="scrollable-table-wrapper">{scope.to_html(index=False)}</div>', unsafe_allow_html=True)

        st.subheader("ACORN Segments")
        acorn_rows = []
        segment_notes = {
            "ACORN-E": "Higher-consumption segment in the historical data, labelled Affluent in the assignment files.",
            "ACORN-F": "Middle-consumption segment, labelled Comfortable in the assignment files.",
            "ACORN-Q": "Lower-consumption segment, labelled Adversity in the assignment files.",
        }
        for acorn, label in ACORN_GROUPS.items():
            acorn_rows.append({"ACORN": acorn, "Segment label": label, "Dashboard role": segment_notes.get(acorn, "Assigned household segment")})
        acorn_df = pd.DataFrame(acorn_rows)
        st.markdown(f'<div class="scrollable-table-wrapper">{acorn_df.to_html(index=False)}</div>', unsafe_allow_html=True)

        st.subheader("Data Used")
        start = daily["Date"].min().date() if not daily.empty else "n/a"
        end = daily["Date"].max().date() if not daily.empty else "n/a"
        st.markdown(
            f"Historical consumption covers `{start}` to `{end}` for the three assigned ACORN "
            "segments. The dashboard also uses weather variables, public-holiday flags, client counts, "
            "and generated lag/rolling features for model comparison."
        )

        st.subheader("Dashboard Flow")
        flow = pd.DataFrame(
            [
                {"Section": "Overview", "Purpose": "Quick forecast row counts and best validation models."},
                {"Section": "EDA", "Purpose": "Historical profiles, weather relationship, autocorrelation, and ACORN comparison."},
                {"Section": "Validation", "Purpose": "Chronological holdout metrics by model and ACORN segment."},
                {"Section": "Forecasts", "Purpose": "Final forecasts plus saved-model prediction comparisons."},
            ]
        )
        st.markdown(f'<div class="scrollable-table-wrapper">{flow.to_html(index=False)}</div>', unsafe_allow_html=True)

        st.subheader("Credits and Sources")
        st.markdown(
            "This project was carried out as part of a CentraleSupélec and EDF partnership. "
            "The dashboard uses the assignment-provided `Households in the United Kingdom` dataset: "
            "preprocessed electricity consumption profiles by ACORN segment, weather data from the "
            "provided Dark Sky files, and UK public holiday data."
        )
        st.markdown("Repository: [github.com/baunilhamarga/ei-climat](https://github.com/baunilhamarga/ei-climat)")

        students = pd.DataFrame(
            [
                {"Student": "Artur Bandeira Chan Jorge", "Email": "artur.bandeira-chan-jorge@student-cs.fr"},
                {"Student": "Pedro Lubaszewski Lima", "Email": "pedro.lubaszewski-lima@student-cs.fr"},
                {"Student": "Ignacio Lopez Acevedo", "Email": "ignacio.lopez-acevedo@student-cs.fr"},
                {"Student": "Heitor Gama", "Email": "heitor.gama@student-cs.fr"},
            ]
        )
        supervisors = pd.DataFrame(
            [
                {"Supervisor": "Laurent BOZZI", "Affiliation": "Data scientist, EDF R&D"},
                {"Supervisor": "Théodore CHERRIERE", "Affiliation": "Enseignant-chercheur, CentraleSupélec - GeePs"},
                {"Supervisor": "Jing DAI", "Affiliation": "Enseignant-chercheur, CentraleSupélec - GeePs"},
                {"Supervisor": "Dung LÉ", "Affiliation": "Enseignant-chercheur, CentraleSupélec - GeePs"},
            ]
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Student analysis team")
            st.markdown(f'<div class="scrollable-table-wrapper">{students.to_html(index=False)}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("Supervision")
            st.markdown(f'<div class="scrollable-table-wrapper">{supervisors.to_html(index=False)}</div>', unsafe_allow_html=True)
