from __future__ import annotations

import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import streamlit as st

from dashboard.styles import StyleManager
from dashboard.tabs.base import BaseTab


class CarbonTab(BaseTab):
    """Renders estimated training carbon emissions for the AutoGluon best run."""

    GPU_POWER_KW = 0.300
    CPU_MEMORY_OVERHEAD_KW = 0.150
    PUE = 1.20
    BULGARIA_CARBON_INTENSITY_KG_PER_KWH = 0.3653
    EXPECTED_RUNTIME_HOURS = 12.0
    LOWER_RUNTIME_HOURS = 8.0
    TIME_USED_HOURS = 18.0

    def render(self, data: dict[str, pd.DataFrame], selected_acorns: list[str]) -> None:
        del data, selected_acorns

        expected = self._estimate(self.EXPECTED_RUNTIME_HOURS)
        lower = self._estimate(self.LOWER_RUNTIME_HOURS)
        upper = self._estimate(self.TIME_USED_HOURS)

        kpi_html = f"""
        <div class="kpi-container">
            {StyleManager.render_kpi_card("Expected CO2e", f"{expected['co2_kg']:.2f} kg", "12h planning estimate")}
            {StyleManager.render_kpi_card("Expected Energy", f"{expected['energy_kwh']:.1f} kWh", "Including PUE = 1.20")}
            {StyleManager.render_kpi_card("18h CO2e", f"{upper['co2_kg']:.2f} kg", "Time used estimate")}
            {StyleManager.render_kpi_card("Bulgaria Grid Factor", "365 g/kWh", "2025 flow-traced average")}
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        st.markdown(
            "This tab estimates the carbon emissions of the high-cost AutoGluon experiment before it is run. "
            "Most benchmark models trained quickly, but the final all-in AutoGluon calibration is intentionally "
            "the compute-heavy run. The goal is to make that cost visible in the same dashboard as model quality: "
            "training time, hardware power, data-center efficiency, and the Bulgarian electricity mix are combined "
            "into a reproducible CO2e estimate."
        )

        with st.expander("How do we calculate carbon emissions for ML?"):
            self._render_methodology(lower, expected, upper)

        st.subheader("Machine and Run Context")
        context = pd.DataFrame(
            [
                {
                    "Item": "Server location",
                    "Value": "Bulgaria",
                    "Status": "Known from GPU server location",
                },
                {
                    "Item": "GPU",
                    "Value": "NVIDIA GeForce RTX 3090, 24 GB VRAM, 350 W power limit",
                    "Status": "Observed with nvidia-smi",
                },
                {
                    "Item": "CPU host",
                    "Value": "AMD EPYC 7543 32-Core Processor, 2 sockets / 128 logical CPUs",
                    "Status": "Observed with lscpu",
                },
                {
                    "Item": "Usable CPU quota",
                    "Value": "About 6.47 CPU cores available to the container",
                    "Status": "Observed from cgroup cpu.max",
                },
                {
                    "Item": "Memory",
                    "Value": "503 GiB total system memory",
                    "Status": "Observed with free -h",
                },
                {
                    "Item": "Experiment scope",
                    "Value": "AutoGluon Tabular Best and AutoGluon TimeSeries Best, validation plus final refits for half-hourly and daily horizons",
                    "Status": "From scripts/group5_update_ag_best_models.py",
                },
            ]
        )
        st.markdown(f'<div class="scrollable-table-wrapper">{context.to_html(index=False)}</div>', unsafe_allow_html=True)

        st.subheader("Scenario Estimates")
        scenarios = pd.DataFrame(
            [
                self._scenario_row("18h time used estimate", self.TIME_USED_HOURS),
                self._scenario_row("Expected planning estimate", self.EXPECTED_RUNTIME_HOURS),
                self._scenario_row("Lower planning estimate", self.LOWER_RUNTIME_HOURS),
            ]
        )
        display = scenarios.copy()
        display["Runtime (h)"] = display["Runtime (h)"].map(lambda value: f"{value:.1f}")
        display["IT power (kW)"] = display["IT power (kW)"].map(lambda value: f"{value:.2f}")
        display["Facility energy (kWh)"] = display["Facility energy (kWh)"].map(lambda value: f"{value:.2f}")
        display["CO2e (kg)"] = display["CO2e (kg)"].map(lambda value: f"{value:.2f}")
        st.markdown(f'<div class="scrollable-table-wrapper">{display.to_html(index=False)}</div>', unsafe_allow_html=True)

        fig = px.bar(
            scenarios,
            x="Scenario",
            y="CO2e (kg)",
            color="Scenario",
            color_discrete_map={
                "Lower planning estimate": "#4da4a9",
                "Expected planning estimate": "#d49c5e",
                "18h time used estimate": "#c85a64",
            },
            title="Estimated Training Emissions by Runtime Scenario",
            labels={"CO2e (kg)": "Estimated CO2e (kg)"},
        )
        fig.update_layout(showlegend=False)
        StyleManager.style_plotly_chart(fig, st.session_state.theme_mode)
        st.plotly_chart(fig, width="stretch", theme=None)

    def _estimate(self, runtime_hours: float) -> dict[str, float]:
        it_power_kw = self.GPU_POWER_KW + self.CPU_MEMORY_OVERHEAD_KW
        energy_kwh = runtime_hours * it_power_kw * self.PUE
        co2_kg = energy_kwh * self.BULGARIA_CARBON_INTENSITY_KG_PER_KWH
        return {
            "runtime_hours": runtime_hours,
            "it_power_kw": it_power_kw,
            "energy_kwh": energy_kwh,
            "co2_kg": co2_kg,
        }

    def _scenario_row(self, scenario: str, runtime_hours: float) -> dict[str, float | str]:
        estimate = self._estimate(runtime_hours)
        return {
            "Scenario": scenario,
            "Runtime (h)": estimate["runtime_hours"],
            "IT power (kW)": estimate["it_power_kw"],
            "Facility energy (kWh)": estimate["energy_kwh"],
            "CO2e (kg)": estimate["co2_kg"],
        }

    def _render_methodology(
        self,
        lower: dict[str, float],
        expected: dict[str, float],
        upper: dict[str, float],
    ) -> None:
        st.markdown("The calculation follows this reference:")
        st.markdown(
            """```text
Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres.
Quantifying the Carbon Emissions of Machine Learning.
In NeurIPS, 2019.
arXiv: https://arxiv.org/abs/1910.09700
```"""
        )

        st.markdown(
            "We convert training electricity into emissions:"
        )
        st.code(
            """IT_power_kW = GPU_power_kW + CPU_memory_overhead_kW
facility_energy_kWh = runtime_hours * IT_power_kW * PUE
CO2e_kg = facility_energy_kWh * grid_carbon_intensity_kg_per_kWh""",
            language="text",
        )

        assumptions = pd.DataFrame(
            [
                {
                    "Variable": "GPU_power_kW",
                    "Value used": f"{self.GPU_POWER_KW:.3f}",
                    "How it was chosen": "Estimated average RTX 3090 draw during mixed AutoGluon training. The observed power limit is 350 W, but not all AutoGluon models fully saturate the GPU.",
                },
                {
                    "Variable": "CPU_memory_overhead_kW",
                    "Value used": f"{self.CPU_MEMORY_OVERHEAD_KW:.3f}",
                    "How it was chosen": "Estimated CPU, memory, storage, and host overhead attributable to this container. The host is large, but the container CPU quota is about 6.47 cores.",
                },
                {
                    "Variable": "PUE",
                    "Value used": f"{self.PUE:.2f}",
                    "How it was chosen": "Estimated data-center power usage effectiveness. The provider's exact PUE was not available in the project files.",
                },
                {
                    "Variable": "grid_carbon_intensity_kg_per_kWh",
                    "Value used": f"{self.BULGARIA_CARBON_INTENSITY_KG_PER_KWH:.4f}",
                    "How it was chosen": "Electricity Maps reports Bulgaria's 2025 flow-traced mean carbon intensity as 365.3 gCO2eq/kWh. No live grid API value was available locally.",
                },
                {
                    "Variable": "runtime_hours",
                    "Value used": f"{self.LOWER_RUNTIME_HOURS:.0f}-{self.EXPECTED_RUNTIME_HOURS:.0f} expected, {self.TIME_USED_HOURS:.0f} time used estimate",
                    "How it was chosen": "Estimated before running the best-quality experiment. The new command has no internal AutoGluon time cap, so the final duration is an assumption here.",
                },
            ]
        )
        st.markdown(f'<div class="scrollable-table-wrapper">{assumptions.to_html(index=False)}</div>', unsafe_allow_html=True)

        st.markdown(
            "Worked examples with the current assumptions:"
        )
        worked_example = f"""Expected scenario: {expected['runtime_hours']:.1f} h * {expected['it_power_kw']:.2f} kW * {self.PUE:.2f} PUE = {expected['energy_kwh']:.2f} kWh
{expected['energy_kwh']:.2f} kWh * {self.BULGARIA_CARBON_INTENSITY_KG_PER_KWH:.4f} kgCO2e/kWh = {expected['co2_kg']:.2f} kgCO2e

Lower scenario: {lower['co2_kg']:.2f} kgCO2e
18h time used estimate: {upper['co2_kg']:.2f} kgCO2e"""
        st.code(worked_example, language="text")

        st.markdown(
            "References: Lacoste, Luccioni, Schmidt, and Dandres, `Quantifying the carbon emissions of machine learning`, "
            "NeurIPS 2019 / [arXiv:1910.09700](https://arxiv.org/abs/1910.09700); "
            "[Electricity Maps Grid Review 2025 for Bulgaria](https://www.electricitymaps.com/grid-in-review-2025/bulgaria); "
            "[Electricity Maps methodology](https://www.electricitymaps.com/data/methodology) for flow-traced consumption-based carbon intensity."
        )
