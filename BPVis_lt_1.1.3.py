"""
BPVis LT — Optimized (forms + caching)  v1.1.3
This version keeps all features/appearance but reduces full-page reruns on every keystroke:
- Sidebar inputs are wrapped in st.form (Apply buttons).
- Heavy data steps are cached with @st.cache_data.
- Widgets write to st.session_state; charts read from it after "Apply".
"""

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
from plotly import graph_objects as go
import streamlit as st

# =========================
# Page setup & constants
# =========================
st.set_page_config(
    page_title="WSGT_BPvis LT 1.1.3 (Optimized)",
    page_icon="📊",
    layout="wide",
)

# Fixed categorical orders
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
END_USE_ORDER = [
    "Heating", "Cooling", "Ventilation", "Lighting",
    "Equipment", "HotWater", "Pumps", "Other", "PV_Generation",
]
ENERGY_SOURCE_ORDER = ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"]

# Color maps
color_map = {
    "Heating": "#c02419",
    "Cooling": "#5a73a5",
    "Ventilation": "#42b38d",
    "Lighting": "#d3b402",
    "Equipment": "#833fd1",
    "HotWater": "#ff9a0a",
    "Pumps": "#06b6d1",
    "Other": "#d0448c",
    "PV_Generation": "#a9c724",
}
color_map_sources = {
    "Electricity": "#42b360",
    "Green Electricity": "#64c423",
    "Gas": "#c9d302",
    "District Heating": "#ec6939",
    "District Cooling": "#5a5ea5",
}

# =========================
# Sidebar — template download & file upload
# =========================
st.sidebar.write("## BPVis LT")
st.sidebar.write("Version 1.1.3 (Optimized)")

st.sidebar.markdown("### Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type="xlsx")

st.sidebar.markdown("### Project Information")
with st.sidebar.expander("Benchmark Settings"):
    st.write("Default file: templates/benchmark_template.xlsx (sheet per Building Use). You can override for this session:")
    bench_upload = st.file_uploader("Upload Benchmark File (optional)", type=["xlsx"], key="bench_upload")

# =========================
# Data loaders (CACHED)
# =========================
@st.cache_data(show_spinner=False)
def energy_balance_sheet(file_bytes: bytes) -> pd.DataFrame:
    """Load 'Energy_Balance' and strip '_kWh' suffix."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df_ = pd.read_excel(xls, sheet_name="Energy_Balance")
    df_.columns = df_.columns.str.replace("_kWh", "", regex=False)
    return df_

@st.cache_data(show_spinner=False)
def loads_balance_sheet(file_bytes: bytes) -> pd.DataFrame:
    """Load 'Loads_Balance' and strip '_load' suffix."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df_ = pd.read_excel(xls, sheet_name="Loads_Balance")
    df_.columns = [c.removesuffix("_load") for c in df_.columns]
    return df_

@st.cache_data(show_spinner=False)
def melted_energy_balance(file_bytes: bytes) -> pd.DataFrame:
    """Cached long-format energy balance (wide→long)."""
    df = energy_balance_sheet(file_bytes)
    return df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

# =========================
# Configuration I/O helpers (Save/Load Project settings)
# =========================
SHEET_PROJECT = "Project_Data"
SHEET_FACTORS = "Emission_Factors"
SHEET_TARIFFS = "Energy_Tariffs"
SHEET_MAPPING = "EndUse_to_Source"

@st.cache_data(show_spinner=False)
def read_config_from_excel(file_bytes: bytes) -> Dict[str, Optional[pd.DataFrame]]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
    return {
        "project": sheets.get(SHEET_PROJECT),
        "factors": sheets.get(SHEET_FACTORS),
        "tariffs": sheets.get(SHEET_TARIFFS),
        "mapping": sheets.get(SHEET_MAPPING),
        "all_sheets": sheets,
    }

def parse_project_df_with_building_use(
    df: Optional[pd.DataFrame],
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str], Optional[float], Optional[float]]:
    """Parse Project_Data sheet (name, area, currency, building use, latitude, longitude)."""
    if df is None or not {"Key", "Value"}.issubset(df.columns):
        return None, None, None, None, None, None

    kv = dict(zip(df["Key"].astype(str), df["Value"]))

    name = kv.get("Project_Name")
    currency = kv.get("Currency")
    building_use = kv.get("Building_Use")

    def _to_float(x):
        try:
            return float(x) if x is not None and str(x).strip() != "" else None
        except Exception:
            return None

    area = _to_float(kv.get("Project_Area"))
    latitude_saved = _to_float(kv.get("Project_Latitude"))
    longitude_saved = _to_float(kv.get("Project_Longitude"))

    return name, area, currency, building_use, latitude_saved, longitude_saved

def parse_factors_df(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    out = {}
    if df is not None and {"Energy_Source", "Factor_kgCO2_per_kWh"}.issubset(df.columns):
        for _, row in df.iterrows():
            src = str(row["Energy_Source"])
            try:
                out[src] = float(row["Factor_kgCO2_per_kWh"])
            except Exception:
                pass
    return out

def parse_tariffs_df(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    out = {}
    if df is not None and {"Energy_Source", "Tariff_per_kWh"}.issubset(df.columns):
        for _, row in df.iterrows():
            src = str(row["Energy_Source"])
            try:
                out[src] = float(row["Tariff_per_kWh"])
            except Exception:
                pass
    return out

def parse_mapping_df(df: Optional[pd.DataFrame]) -> Dict[str, str]:
    out = {}
    if df is not None and {"End_Use", "Energy_Source"}.issubset(df.columns):
        for _, row in df.iterrows():
            eu = str(row["End_Use"])
            es = str(row["Energy_Source"])
            out[eu] = es
    return out

def build_project_df_with_building_use(
    project_name: str,
    project_area: float,
    currency_symbol: str,
    building_use: str,
    latitude: Optional[float],
    longitude: Optional[float],
) -> pd.DataFrame:
    """Build the Project_Data sheet including lat/long."""
    return pd.DataFrame(
        {
            "Key": [
                "Project_Name",
                "Project_Area",
                "Currency",
                "Building_Use",
                "Project_Latitude",
                "Project_Longitude",
            ],
            "Value": [
                project_name,
                project_area,
                currency_symbol,
                building_use,
                latitude,
                longitude,
            ],
        }
    )

def build_factors_df(co2_elec: float, co2_green: float, co2_dh: float, co2_dc: float, co2_gas: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"],
            "Factor_kgCO2_per_kWh": [co2_elec, co2_green, co2_gas, co2_dh, co2_dc],
        }
    )

def build_tariffs_df(cost_elec: float, cost_green: float, cost_dh: float, cost_dc: float, cost_gas: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"],
            "Tariff_per_kWh": [cost_elec, cost_green, cost_gas, cost_dh, cost_dc],
        }
    )

def build_mapping_df(end_uses) -> pd.DataFrame:
    rows = []
    for use in end_uses:
        rows.append({"End_Use": use, "Energy_Source": st.session_state.get(f"source_{use}", "Electricity")})
    return pd.DataFrame(rows)

def write_config_to_excel(
    original_bytes: bytes,
    project_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    tariffs_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> bytes:
    cfg = read_config_from_excel(original_bytes)
    sheets = cfg["all_sheets"]

    sheets[SHEET_PROJECT] = project_df
    sheets[SHEET_FACTORS] = factors_df
    sheets[SHEET_TARIFFS] = tariffs_df
    sheets[SHEET_MAPPING] = mapping_df

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return buf.getvalue()

# =========================
# Preload saved configuration (if Excel is uploaded)
# =========================
preloaded = None
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    cfg_saved = read_config_from_excel(file_bytes)

    saved_name, saved_area, saved_currency, saved_building_use, saved_lat, saved_lon = parse_project_df_with_building_use(
        cfg_saved["project"]
    )
    saved_factors = parse_factors_df(cfg_saved["factors"])
    saved_tariffs = parse_tariffs_df(cfg_saved["tariffs"])
    saved_mapping_df = cfg_saved["mapping"]

    preloaded = {
        "name": saved_name,
        "area": saved_area,
        "currency": saved_currency,
        "building_use": saved_building_use,
        "lat": saved_lat,
        "lon": saved_lon,
        "factors": saved_factors,
        "tariffs": saved_tariffs,
        "mapping_df": saved_mapping_df,
        "file_bytes": file_bytes,
    }

# Seed defaults in session_state (only if not set yet)
def seed_state(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

seed_state("project_name", (preloaded and preloaded["name"]) or "Example Building 1")
seed_state("project_area", float((preloaded and preloaded["area"]) or 1000.0))
seed_state("building_use", (preloaded and preloaded["building_use"]) or "Office")
seed_state("project_latitude", str((preloaded and preloaded["lat"]) if (preloaded and preloaded["lat"] is not None) else 53.54955))
seed_state("project_longitude", str((preloaded and preloaded["lon"]) if (preloaded and preloaded["lon"] is not None) else 9.9936))
seed_state("currency_symbol", (preloaded and preloaded["currency"]) if (preloaded and preloaded["currency"] in ["€", "$", "£"]) else "€")

# =========================
# Header
# =========================
st.title(st.session_state["project_name"])

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Energy Balance", "CO2 Emissions", "Energy Cost", "Loads Analysis", "Benchmark"]
)

# =========================
# Tab 1 — Energy Balance
# =========================
with tab1:
    if uploaded_file:
        # LONG df (cached)
        df_melted = melted_energy_balance(uploaded_file.getvalue())

        # --- Sidebar forms ---
        # Project Data
        with st.sidebar.expander("Project Data"):
            with st.form("project_data_form", clear_on_submit=False):
                st.write("Enter Project's Basic Informations")
                st.text_input("Project Name", key="project_name")
                st.number_input("Project Area", 0.00, value=float(st.session_state["project_area"]), key="project_area")
                st.text_input("Project Latitude", key="project_latitude")
                st.text_input("Project Longitude", key="project_longitude")

                building_use_options = [
                    "Office", "Hospitality", "Retail", "Residential", "Industrial", "Education", "Leisure", "Healthcare"
                ]
                current_use = st.session_state.get("building_use", "Office")
                st.selectbox(
                    "Building Use (for benchmarks)",
                    building_use_options,
                    index=building_use_options.index(current_use) if current_use in building_use_options else 0,
                    key="building_use",
                )
                st.form_submit_button("Apply Project Data")

        # Emission Factors
        with st.sidebar.expander("Emission Factors"):
            with st.form("emission_factors_form", clear_on_submit=False):
                st.write("Assign Emission Factors")
                def_f = preloaded["factors"] if preloaded else {}
                st.number_input("CO2 Factor Electricity", 0.000, 1.000, float(def_f.get("Electricity", 0.300)),
                                format="%0.3f", key="co2_elec")
                st.number_input("CO2 Factor Green Electricity", 0.000, 1.000, float(def_f.get("Green Electricity", 0.000)),
                                format="%0.3f", key="co2_green")
                st.number_input("CO2 Factor District Heating", 0.000, 1.000, float(def_f.get("District Heating", 0.260)),
                                format="%0.3f", key="co2_dh")
                st.number_input("CO2 Factor District Cooling", 0.000, 1.000, float(def_f.get("District Cooling", 0.280)),
                                format="%0.3f", key="co2_dc")
                st.number_input("CO2 Factor Gas", 0.000, 1.000, float(def_f.get("Gas", 0.180)),
                                format="%0.3f", key="co2_gas")
                st.form_submit_button("Apply Emission Factors")

        # Energy Tariffs
        with st.sidebar.expander("Energy Tariffs"):
            with st.form("tariffs_form", clear_on_submit=False):
                st.write("Assign energy cost per source (per kWh)")
                st.selectbox("Currency", ["€", "$", "£"],
                             index=["€", "$", "£"].index(st.session_state["currency_symbol"]),
                             key="currency_symbol")

                def_t = preloaded["tariffs"] if preloaded else {}
                st.number_input(
                    f"Cost Electricity ({st.session_state['currency_symbol']}/kWh)",
                    0.00, 100.00, float(def_t.get("Electricity", 0.35)), step=0.01, format="%.2f", key="cost_elec"
                )
                st.number_input(
                    f"Cost Green Electricity ({st.session_state['currency_symbol']}/kWh)",
                    0.00, 100.00, float(def_t.get("Green Electricity", 0.40)), step=0.01, format="%.2f", key="cost_green"
                )
                st.number_input(
                    f"Cost District Heating ({st.session_state['currency_symbol']}/kWh)",
                    0.00, 100.00, float(def_t.get("District Heating", 0.16)), step=0.01, format="%.2f", key="cost_dh"
                )
                st.number_input(
                    f"Cost District Cooling ({st.session_state['currency_symbol']}/kWh)",
                    0.00, 100.00, float(def_t.get("District Cooling", 0.16)), step=0.01, format="%.2f", key="cost_dc"
                )
                st.number_input(
                    f"Cost Gas ({st.session_state['currency_symbol']}/kWh)",
                    0.00, 100.00, float(def_t.get("Gas", 0.12)), step=0.01, format="%.2f", key="cost_gas"
                )
                st.form_submit_button("Apply Energy Tariffs")

        # Assign Energy Sources
        with st.sidebar.expander("Assign Energy Sources"):
            with st.form("source_mapping_form", clear_on_submit=False):
                st.write("Assign Energy Sources")
                end_uses = df_melted["End_Use"].unique().tolist()
                saved_mapping = parse_mapping_df(preloaded["mapping_df"]) if (preloaded and preloaded["mapping_df"] is not None) else {}
                for use in end_uses:
                    default_source = saved_mapping.get(use, "Electricity")
                    idx = ENERGY_SOURCE_ORDER.index(default_source) if default_source in ENERGY_SOURCE_ORDER else 0
                    st.selectbox(
                        f"{use}", ENERGY_SOURCE_ORDER, index=idx, key=f"source_{use}",
                    )
                st.form_submit_button("Apply Energy Sources")

        # Helper to coerce string lat/lon
        def _to_float_safe(s):
            try:
                return float(s)
            except Exception:
                return None

        # Read values (now final, after Apply)
        project_name = st.session_state["project_name"]
        project_area = float(st.session_state["project_area"])
        building_use = st.session_state["building_use"]
        currency_symbol = st.session_state["currency_symbol"]
        latitude = _to_float_safe(st.session_state["project_latitude"])
        longitude = _to_float_safe(st.session_state["project_longitude"])

        co2_Emissions_Electricity = float(st.session_state["co2_elec"])
        co2_Emissions_Green_Electricity = float(st.session_state["co2_green"])
        co2_emissions_dh = float(st.session_state["co2_dh"])
        co2_emissions_dc = float(st.session_state["co2_dc"])
        co2_emissions_gas = float(st.session_state["co2_gas"])

        cost_electricity = float(st.session_state["cost_elec"])
        cost_green_electricity = float(st.session_state["cost_green"])
        cost_dh = float(st.session_state["cost_dh"])
        cost_dc = float(st.session_state["cost_dc"])
        cost_gas = float(st.session_state["cost_gas"])

        # Save Project (unchanged behavior)
        with st.sidebar:
            st.markdown("---")
            if preloaded and st.button("Save Project", use_container_width=True):
                project_df = build_project_df_with_building_use(
                    project_name, project_area, currency_symbol, building_use, latitude, longitude
                )
                factors_df = build_factors_df(
                    co2_Emissions_Electricity, co2_Emissions_Green_Electricity, co2_emissions_dh, co2_emissions_dc, co2_emissions_gas
                )
                tariffs_df = build_tariffs_df(cost_electricity, cost_green_electricity, cost_dh, cost_dc, cost_gas)
                mapping_df = build_mapping_df(end_uses)

                updated_bytes = write_config_to_excel(
                    preloaded["file_bytes"], project_df, factors_df, tariffs_df, mapping_df
                )
                st.success("Project settings saved to workbook.")
                st.download_button(
                    label="Download Updated Workbook",
                    data=updated_bytes,
                    file_name=uploaded_file.name.replace(".xlsx", "_with_project.xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        # Apply mapping to df_melted
        df_melted["Energy_Source"] = df_melted["End_Use"].map({k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted["End_Use"].unique()})

        # Monthly totals for net overlay
        monthly_totals = (
            df_melted.groupby("Month", as_index=False)["kWh"].sum()
            .assign(Month=lambda d: pd.Categorical(d["Month"], categories=MONTH_ORDER, ordered=True))
            .sort_values("Month", kind="stable")
            .reset_index(drop=True)
        )

        # Monthly per End Use + net line
        monthly_chart = px.bar(
            df_melted,
            x="Month",
            y="kWh",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"Month": MONTH_ORDER},
            text_auto=".0f",
        )
        monthly_chart.update_traces(textfont_size=14, textfont_color="white")
        line_monthly_net = px.line(monthly_totals, x="Month", y="kWh", markers=True, labels={"kWh": "Net total"})
        for tr in line_monthly_net.data:
            tr.name = "Net total"
            tr.line.width = 5
            tr.line.color = "black"
            tr.line.dash = "dash"
            tr.marker.size = 12
            monthly_chart.add_trace(tr)
        monthly_chart.update_layout(showlegend=False)

        # Monthly per Energy Source
        monthly_by_source = df_melted.groupby(["Month", "Energy_Source"], as_index=False)["kWh"].sum()
        monthly_by_source["Month"] = pd.Categorical(monthly_by_source["Month"], categories=MONTH_ORDER, ordered=True)
        monthly_chart_source = px.bar(
            monthly_by_source,
            x="Month",
            y="kWh",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Month": MONTH_ORDER},
            text_auto=".0f",
        )
        monthly_chart_source.update_layout(showlegend=False)
        monthly_chart_source.update_traces(textfont_size=14, textfont_color="white")

        st.write("## Energy Balance (per Energy Use)")

        # Annual totals & intensities
        totals = df_melted.groupby("End_Use", as_index=False)["kWh"].sum()
        totals["Per Use"] = "Total"
        totals["kWh_per_m2"] = (totals["kWh"] / project_area).round(1)

        eui = totals.loc[totals["kWh_per_m2"] > 0, "kWh_per_m2"].sum()
        net_energy = totals["kWh"].sum()
        net_eui = totals["kWh_per_m2"].sum()

        totals_per_source = df_melted.groupby("Energy_Source", as_index=False)["kWh"].sum()
        totals_per_source["Per Source"] = "total_per_source"
        totals_per_source["kWh_per_m2_per_source"] = (totals_per_source["kWh"] / project_area).round(1)

        annual_chart = px.bar(
            totals,
            x="Per Use",
            y="kWh",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
            text_auto=".0f",
        )
        annual_chart.add_hline(y=net_energy, line_width=4, line_dash="dash", line_color="black")
        annual_chart.add_annotation(
            x=0.5, xref="paper",
            y=net_energy, yref="y",
            text=f"{net_energy:,.0f} kWh",
            showarrow=False, yshift=12,
            font=dict(size=16, color="white"),
        )
        annual_chart.update_traces(textfont_size=14, textfont_color="white")

        annual_chart_per_source = px.bar(
            totals_per_source,
            x="Per Source",
            y="kWh",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
            text_auto=".0f",
        )
        annual_chart_per_source.update_traces(textfont_size=14, textfont_color="white")

        # Donuts (EUI shares)
        energy_intensity_chart = px.pie(
            totals,
            names="End_Use",
            values="kWh_per_m2",
            color="End_Use",
            color_discrete_map=color_map,
            hole=0.5,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
        )
        energy_intensity_chart.update_layout(
            annotations=[dict(
                text=f"{eui:,.1f}<br>kWh/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=40, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart.update_traces(textinfo="value+percent", textfont_size=16, textfont_color="white")

        energy_intensity_chart_per_source = px.pie(
            totals_per_source,
            names="Energy_Source",
            values="kWh_per_m2_per_source",
            color="Energy_Source",
            color_discrete_map=color_map_sources,
            hole=0.5,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
        )
        energy_intensity_chart_per_source.update_layout(
            annotations=[dict(
                text=f"{eui:,.1f}<br>kWh/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=40, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart_per_source.update_traces(textinfo="value+percent", textfont_size=16, textfont_color="white")

        # KPIs
        totals_indexed = totals.set_index("End_Use")
        pv_value = totals_indexed.loc["PV_Generation", "kWh_per_m2"] if "PV_Generation" in totals_indexed.index else 0.0
        pv_coverage = abs((pv_value / eui) * 100) if eui != 0 else 0.0

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy Demand")
            st.plotly_chart(monthly_chart, use_container_width=True)
        with col2:
            st.subheader("Annual Energy Demand")
            st.plotly_chart(annual_chart, use_container_width=True)

        monthly_avr = (totals["kWh"].sum()) / 12
        net_total = totals["kWh"].sum()
        total_energy = totals.loc[totals["kWh"] > 0, "kWh"].sum()
        pv_total = abs(df_melted.groupby("End_Use")["kWh"].sum().get("PV_Generation", 0.0))

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st.plotly_chart(energy_intensity_chart, use_container_width=True)
        with col2:
            st.subheader("Energy KPI's")
            st.metric(label="Monthly Average Energy Consumption", value=f"{monthly_avr:,.0f} kWh")
            st.metric(label="Total Annual Energy Consumption", value=f"{total_energy:,.0f} kWh")
            st.metric(label="Net Annual Energy Consumption", value=f"{net_total:,.0f} kWh")
            st.metric(label="EUI", value=f"{eui:,.1f} kWh/m2.a")
            st.metric(label="Net EUI", value=f"{net_eui:,.1f} kWh/m2.a")
            st.metric(label="PV Production", value=f"{pv_total:,.1f} kWh")
            st.metric(label="PV Coverage", value=f"{pv_coverage:,.1f} %")

        st.write("## Energy Balance (per Energy Source)")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy Demand")
            st.plotly_chart(monthly_chart_source, use_container_width=True)
        with col2:
            st.subheader("Annual Energy Demand")
            st.plotly_chart(annual_chart_per_source, use_container_width=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st.plotly_chart(energy_intensity_chart_per_source, use_container_width=True)
        with col2:
            st.subheader("Energy KPI's")
            for _, row in totals_per_source.iterrows():
                st.metric(
                    label=f"EUI - {row['Energy_Source']}",
                    value=f"{row['kWh_per_m2_per_source']:,.1f} kWh/m².a",
                )

    else:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 2 — CO₂ Emissions
# =========================
with tab2:
    if uploaded_file:
        df_melted = melted_energy_balance(uploaded_file.getvalue())
        df_melted["Energy_Source"] = df_melted["End_Use"].map({k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted["End_Use"].unique()})

        factor_map = {
            "Electricity": float(st.session_state["co2_elec"]),
            "Green Electricity": float(st.session_state["co2_green"]),
            "Gas": float(st.session_state["co2_gas"]),
            "District Heating": float(st.session_state["co2_dh"]),
            "District Cooling": float(st.session_state["co2_dc"]),
        }

        df_co2 = df_melted.copy()
        df_co2["CO2_factor_kg_per_kWh"] = df_co2["Energy_Source"].map(factor_map).fillna(0.0)
        df_co2["kgCO2"] = df_co2["kWh"] * df_co2["CO2_factor_kg_per_kWh"]

        monthly_totals_co2 = (
            df_co2.groupby("Month", as_index=False)["kgCO2"].sum()
            .assign(Month=lambda d: pd.Categorical(d["Month"], categories=MONTH_ORDER, ordered=True))
            .sort_values("Month", kind="stable").reset_index(drop=True)
        )

        monthly_chart_co2_use = px.bar(
            df_co2,
            x="Month", y="kgCO2",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"Month": MONTH_ORDER},
            text_auto=".0f",
        )
        monthly_chart_co2_use.update_traces(textfont_size=14, textfont_color="white")

        line_monthly_net_co2 = px.line(monthly_totals_co2, x="Month", y="kgCO2", markers=True, labels={"kgCO2": "Net total"})
        for tr in line_monthly_net_co2.data:
            tr.name = "Net total"
            tr.line.width = 5
            tr.line.color = "black"
            tr.line.dash = "dash"
            tr.marker.size = 12
            monthly_chart_co2_use.add_trace(tr)
        monthly_chart_co2_use.update_layout(showlegend=False)

        monthly_co2_by_source = df_co2.groupby(["Month", "Energy_Source"], as_index=False)["kgCO2"].sum()
        monthly_co2_by_source["Month"] = pd.Categorical(monthly_co2_by_source["Month"], categories=MONTH_ORDER, ordered=True)
        monthly_chart_co2_source = px.bar(
            monthly_co2_by_source,
            x="Month", y="kgCO2",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Month": MONTH_ORDER, "Energy_Source": ENERGY_SOURCE_ORDER},
            text_auto=".0f",
        )
        monthly_chart_co2_source.update_layout(showlegend=False)
        monthly_chart_co2_source.update_traces(textfont_size=14, textfont_color="white")

        totals_co2_use = df_co2.groupby("End_Use", as_index=False)["kgCO2"].sum()
        totals_co2_use["Per Use"] = "Total"
        totals_co2_use["kgCO2_per_m2"] = (totals_co2_use["kgCO2"] / float(st.session_state["project_area"])).round(1)
        net_co2 = totals_co2_use["kgCO2"].sum()

        totals_co2_source = df_co2.groupby("Energy_Source", as_index=False)["kgCO2"].sum()
        totals_co2_source["Per Source"] = "total_per_source"
        totals_co2_source["kgCO2_per_m2_per_source"] = (totals_co2_source["kgCO2"] / float(st.session_state["project_area"])).round(1)

        annual_chart_co2_use = px.bar(
            totals_co2_use,
            x="Per Use", y="kgCO2",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
            text_auto=".0f",
        )
        annual_chart_co2_use.update_traces(textfont_size=14, textfont_color="white")
        annual_chart_co2_use.add_hline(y=net_co2, line_width=4, line_dash="dash", line_color="black")
        annual_chart_co2_use.add_annotation(
            x=0.5, xref="paper",
            y=net_co2, yref="y",
            text=f"{net_co2:,.0f} kgCO2",
            showarrow=False, yshift=12,
            font=dict(size=16, color="white"),
        )

        annual_chart_co2_source = px.bar(
            totals_co2_source,
            x="Per Source", y="kgCO2",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
        )

        co2_intensity_pie_use = px.pie(
            totals_co2_use,
            names="End_Use",
            values="kgCO2_per_m2",
            color="End_Use",
            color_discrete_map=color_map,
            hole=0.5,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
        )
        co2_intensity_pie_use.update_layout(showlegend=True)
        co2_intensity_pie_use.update_traces(textinfo="value+percent", textfont_size=16, textfont_color="white")

        co2_intensity_pie_source = px.pie(
            totals_co2_source,
            names="Energy_Source",
            values="kgCO2_per_m2_per_source",
            color="Energy_Source",
            color_discrete_map=color_map_sources,
            hole=0.5,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
        )
        co2_intensity_pie_source.update_layout(showlegend=True)
        co2_intensity_pie_source.update_traces(textinfo="value+percent", textfont_size=16, textfont_color="white")

        monthly_avg_co2 = monthly_totals_co2["kgCO2"].mean()
        annual_total_co2 = totals_co2_use["kgCO2"].sum()
        co2_intensity_total = totals_co2_use["kgCO2_per_m2"].sum()

        co2_intensity_pie_use.update_layout(
            annotations=[dict(
                text=f"{co2_intensity_total:,.1f}<br>kgCO₂/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=40, color="black"),
            )]
        )
        co2_intensity_pie_source.update_layout(
            annotations=[dict(
                text=f"{co2_intensity_total:,.1f}<br>kgCO₂/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=40, color="black"),
            )]
        )

        st.write("## CO₂ Emissions (per End Use)")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Monthly CO₂")
            st.plotly_chart(monthly_chart_co2_use, use_container_width=True)
        with c2:
            st.subheader("Annual CO₂")
            st.plotly_chart(annual_chart_co2_use, use_container_width=True)

        c3, c4 = st.columns([3, 1])
        with c3:
            st.subheader("CO₂ Intensity (kgCO₂/m²·a)")
            st.plotly_chart(co2_intensity_pie_use, use_container_width=True)
        with c4:
            st.subheader("CO₂ KPI's")
            st.metric("Monthly Average CO₂", f"{monthly_avg_co2:,.0f} kgCO₂")
            st.metric("Total Annual CO₂", f"{annual_total_co2:,.0f} kgCO₂")
            st.metric("CO₂ Intensity", f"{co2_intensity_total:,.1f} kgCO₂/m²·a")

        st.write("## CO₂ Emissions (per Energy Source)")
        c5, c6 = st.columns([3, 1])
        with c5:
            st.subheader("Monthly CO₂")
            st.plotly_chart(monthly_chart_co2_source, use_container_width=True)
        with c6:
            st.subheader("Annual CO₂")
            st.plotly_chart(annual_chart_co2_source, use_container_width=True)

        c7, c8 = st.columns([3, 1])
        with c7:
            st.subheader("CO₂ Intensity (kgCO₂/m²·a)")
            st.plotly_chart(co2_intensity_pie_source, use_container_width=True)
        with c8:
            st.subheader("CO₂ KPI's")
            for _, row in totals_co2_source.iterrows():
                st.metric(
                    label=f"CO₂ Intensity - {row['Energy_Source']}",
                    value=f"{row['kgCO2_per_m2_per_source']:,.1f} kgCO₂/m²·a",
                )

    else:
        st.write("### ← Please upload data on side bar")

# =========================
# Tab 3 — Energy Cost
# =========================
with tab3:
    if uploaded_file:
        df_melted_cost = melted_energy_balance(uploaded_file.getvalue())
        df_melted_cost["Energy_Source"] = df_melted_cost["End_Use"].map({k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted_cost["End_Use"].unique()})

        cost_map = {
            "Electricity": float(st.session_state["cost_elec"]),
            "Green Electricity": float(st.session_state["cost_green"]),
            "Gas": float(st.session_state["cost_gas"]),
            "District Heating": float(st.session_state["cost_dh"]),
            "District Cooling": float(st.session_state["cost_dc"]),
        }

        df_cost = df_melted_cost.copy()
        df_cost["cost_per_kWh"] = df_cost["Energy_Source"].map(cost_map).fillna(0.0)
        df_cost["cost"] = df_cost["kWh"] * df_cost["cost_per_kWh"]

        monthly_totals_cost = (
            df_cost.groupby("Month", as_index=False)["cost"].sum()
            .assign(Month=lambda d: pd.Categorical(d["Month"], categories=MONTH_ORDER, ordered=True))
            .sort_values("Month").reset_index(drop=True)
        )

        monthly_chart_cost_use = px.bar(
            df_cost,
            x="Month", y="cost",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"Month": MONTH_ORDER},
        )
        line_monthly_net_cost = px.line(monthly_totals_cost, x="Month", y="cost", markers=True, labels={"cost": "Net total"})
        for tr in line_monthly_net_cost.data:
            tr.name = "Net total"
            tr.line.width = 5
            tr.line.color = "black"
            tr.line.dash = "dash"
            tr.marker.size = 12
            monthly_chart_cost_use.add_trace(tr)
        monthly_chart_cost_use.update_layout(showlegend=False)

        monthly_cost_by_source = df_cost.groupby(["Month", "Energy_Source"], as_index=False)["cost"].sum()
        monthly_cost_by_source["Month"] = pd.Categorical(monthly_cost_by_source["Month"], categories=MONTH_ORDER, ordered=True)
        monthly_chart_cost_source = px.bar(
            monthly_cost_by_source,
            x="Month", y="cost",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Month": MONTH_ORDER},
        )
        monthly_chart_cost_source.update_layout(showlegend=False)

        totals_cost_use = df_cost.groupby("End_Use", as_index=False)["cost"].sum()
        totals_cost_use["Per Use"] = "Total"
        totals_cost_use["cost_per_m2"] = (totals_cost_use["cost"] / float(st.session_state["project_area"])).round(2)

        totals_cost_source = df_cost.groupby("Energy_Source", as_index=False)["cost"].sum()
        totals_cost_source["Per Source"] = "total_per_source"
        totals_cost_source["cost_per_m2_per_source"] = (totals_cost_source["cost"] / float(st.session_state["project_area"])).round(2)

        annual_chart_cost_use = px.bar(
            totals_cost_use,
            x="Per Use", y="cost",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
        )
        net_cost = totals_cost_use["cost"].sum()
        annual_chart_cost_use.add_hline(y=net_cost, line_width=4, line_dash="dash", line_color="black")
        annual_chart_cost_use.add_annotation(
            x=0.5, xref="paper", y=net_cost, yref="y",
            text=f"{st.session_state['currency_symbol']} {net_cost:,.0f}",
            showarrow=False, yshift=10, font=dict(size=16, color="white"),
        )

        annual_chart_cost_source = px.bar(
            totals_cost_source,
            x="Per Source", y="cost",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
        )

        cost_intensity_pie_use = px.pie(
            totals_cost_use,
            names="End_Use",
            values="cost_per_m2",
            color="End_Use",
            color_discrete_map=color_map,
            hole=0.5,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
        )
        cost_intensity_pie_use.update_traces(textinfo="value+percent", textfont_size=16, textfont_color="white")

        cost_intensity_pie_source = px.pie(
            totals_cost_source,
            names="Energy_Source",
            values="cost_per_m2_per_source",
            color="Energy_Source",
            color_discrete_map=color_map_sources,
            hole=0.5,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
        )
        cost_intensity_pie_source.update_traces(textinfo="value+percent", textfont_size=16, textfont_color="white")

        cost_intensity_total = totals_cost_use["cost_per_m2"].sum()
        cost_intensity_pie_use.update_layout(
            showlegend=True,
            annotations=[dict(
                text=f"{st.session_state['currency_symbol']} {cost_intensity_total:,.2f}<br>per m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=40, color="black"),
            )]
        )
        cost_intensity_pie_source.update_layout(
            showlegend=True,
            annotations=[dict(
                text=f"{st.session_state['currency_symbol']} {cost_intensity_total:,.2f}<br>per m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=40, color="black"),
            )]
        )

        monthly_avg_cost = monthly_totals_cost["cost"].mean()
        annual_total_cost = totals_cost_use["cost"].sum()

        st.write(f"## Energy Cost {st.session_state['currency_symbol']} (per End Use)")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Monthly Cost")
            st.plotly_chart(monthly_chart_cost_use, use_container_width=True)
        with c2:
            st.subheader("Annual Cost")
            st.plotly_chart(annual_chart_cost_use, use_container_width=True)

        c3, c4 = st.columns([3, 1])
        with c3:
            st.subheader(f"Cost Intensity ( {st.session_state['currency_symbol']}/m²·a )")
            st.plotly_chart(cost_intensity_pie_use, use_container_width=True)
        with c4:
            st.subheader("Cost KPI's")
            st.metric("Monthly Average Cost", f"{st.session_state['currency_symbol']} {monthly_avg_cost:,.0f}")
            st.metric("Total Annual Cost", f"{st.session_state['currency_symbol']} {annual_total_cost:,.0f}")
            st.metric("Cost Intensity (Total)", f"{st.session_state['currency_symbol']} {cost_intensity_total:,.2f} /m²·a")

        st.write(f"## Energy Cost {st.session_state['currency_symbol']} (per Energy Source)")
        c5, c6 = st.columns([3, 1])
        with c5:
            st.subheader("Monthly Cost")
            st.plotly_chart(monthly_chart_cost_source, use_container_width=True)
        with c6:
            st.subheader("Annual Cost")
            st.plotly_chart(annual_chart_cost_source, use_container_width=True)

        c7, c8 = st.columns([3, 1])
        with c7:
            st.subheader(f"Cost Intensity ( {st.session_state['currency_symbol']}/m²·a )")
            st.plotly_chart(cost_intensity_pie_source, use_container_width=True)
        with c8:
            st.subheader("Cost KPI's")
            for _, row in totals_cost_source.iterrows():
                st.metric(
                    label=f"Cost Intensity - {row['Energy_Source']}",
                    value=f"{st.session_state['currency_symbol']} {row['cost_per_m2_per_source']:,.2f} /m²·a",
                )

    else:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 4 — Loads Analysis
# =========================
with tab4:
    if uploaded_file:
        df_loads = loads_balance_sheet(uploaded_file.getvalue())
        load_cols = [c for c in df_loads.columns if c not in ["hoy", "doy", "day", "month", "weekday", "hour"]]
        df_loads["doy"] = pd.to_numeric(df_loads["doy"], errors="coerce")
        df_loads["hour"] = pd.to_numeric(df_loads["hour"], errors="coerce")

        st.subheader("Load Analysis")
        selected_load = st.selectbox("Select Load", load_cols, index=0)

        load_heatmap = px.density_heatmap(
            df_loads, x="doy", y="hour", z=selected_load,
            nbinsx=365, nbinsy=24, color_continuous_scale="thermal",
        )
        load_heatmap.update_layout(
            xaxis_title="Day of Year (doy)", yaxis_title="Hour of Day",
            coloraxis_colorbar=dict(title=selected_load), height=700,
        )

        sum_load = pd.to_numeric(df_loads[selected_load], errors="coerce")
        total_load_selected = sum_load.sum()
        max_load_selected = sum_load.max()
        min_load_selected = sum_load.min()

        project_area = float(st.session_state["project_area"])
        specific_load = (sum_load / project_area) * 1000
        max_specific_load = (max_load_selected / project_area) * 1000
        min_specific_load = (min_load_selected / project_area) * 1000
        p95_specific_load = np.percentile(specific_load.dropna(), 95)
        p80_specific_load = np.percentile(specific_load.dropna(), 80)

        totals_by_month = df_loads.groupby("month", as_index=False)[selected_load].sum()
        totals_by_month["month"] = pd.Categorical(totals_by_month["month"], ordered=True)
        totals_by_month = totals_by_month.sort_values("month")

        monthly_total_load_bar = px.bar(
            totals_by_month, x="month", y=selected_load,
            labels={"month": "Month", selected_load: "kWh"},
            text_auto=".0f", height=700,
        )
        key = selected_load.replace("_load", "")
        bar_color = color_map.get(key, "#c02419")
        monthly_total_load_bar.update_traces(textfont_size=14, textfont_color="white", marker_color=bar_color, name=selected_load, showlegend=True)
        monthly_total_load_bar.update_layout(showlegend=True, legend=dict(title=""))

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Monthly Load — {selected_load} (kWh)")
            st.plotly_chart(monthly_total_load_bar, use_container_width=True)
        with col2:
            st.subheader("Load KPI's")
            st.metric("Total Load", f"{total_load_selected:,.0f} kWh")
            st.metric("Maximum Load", f"{max_load_selected:,.1f} kW")
            st.metric("Minimum Load", f"{min_load_selected:,.1f} kW")
            st.metric("Maximum Specific Load", f"{max_specific_load:,.1f} W/m2")
            st.metric("Minimum Specific Load", f"{min_specific_load:,.1f} W/m2")
            st.metric("95th Percentile Specific Load", f"{p95_specific_load:,.1f} W/m2")
            st.metric("80th Percentile Specific Load", f"{p80_specific_load:,.1f} W/m2")

        st.subheader(f"Hourly Load Heatmap — {selected_load} (kW)")
        st.plotly_chart(load_heatmap, use_container_width=True)

        peak_load = max_load_selected
        st.subheader(f"Hours Above Threshold — {selected_load}")
        thr = st.number_input("Heatmap threshold (kW)", value=float(round(0.8 * peak_load, 1)), key="thr_heatmap")
        df_bool = df_loads.copy()
        df_bool["exceed"] = (pd.to_numeric(df_bool[selected_load], errors="coerce") > thr).astype(int)
        total_exceedance = df_bool["exceed"].sum()

        exceed_heatmap = px.density_heatmap(
            df_bool, x="doy", y="hour", z="exceed",
            histfunc="sum", nbinsx=365, nbinsy=24,
            color_continuous_scale="Reds",
            title=f"Exceedance Count Heatmap — {selected_load} > {thr:g} kW",
        )
        exceed_heatmap.update_layout(
            xaxis_title="Day of Year (doy)", yaxis_title="Hour of Day",
            coloraxis_colorbar=dict(title="Exceed"), height=700,
        )
        st.plotly_chart(exceed_heatmap, use_container_width=True)
        st.caption(f"Total Exceeded Hours {total_exceedance:,.1f}")

        peaks = (
            df_loads.loc[:, ["month", "day", "weekday", "hour", selected_load]]
            .sort_values(selected_load, ascending=False).head(5)
        )
        st.subheader(f"Top 5 Peak Loads — {selected_load} (kW)")
        st.dataframe(peaks.style.format({selected_load: "{:,.1f} kW"}), use_container_width=True)

        s = pd.to_numeric(df_loads[selected_load], errors="coerce")
        daily = df_loads.assign(_val=s).groupby("doy", as_index=False)["_val"].sum()
        peak_idx = daily["_val"].abs().idxmax()
        peak_doy = int(daily.loc[peak_idx, "doy"])
        peak_total = float(daily.loc[peak_idx, "_val"])

        date_label = f"DOY {peak_doy}"
        if {"month", "day"}.issubset(df_loads.columns):
            month_val = df_loads.loc[df_loads["doy"] == peak_doy, "month"].iloc[0]
            day_val = df_loads.loc[df_loads["doy"] == peak_doy, "day"].iloc[0]
            if str(month_val).isdigit():
                month_map = dict(enumerate(MONTH_ORDER, start=1))
                month_val = month_map.get(int(month_val), month_val)
            date_label = f"{month_val} {int(day_val)} (DOY {peak_doy})"

        day_profile = df_loads.loc[df_loads["doy"] == peak_doy, ["hour", selected_load]].copy()
        day_profile["hour"] = pd.to_numeric(day_profile["hour"], errors="coerce")
        day_profile[selected_load] = pd.to_numeric(day_profile[selected_load], errors="coerce")
        day_profile = day_profile.sort_values("hour")

        peak_day_fig = px.line(
            day_profile, x="hour", y=selected_load, markers=True,
            title=f"Peak Day Profile — {selected_load} | {date_label}",
        )
        r, g, b = pc.hex_to_rgb(bar_color)
        peak_day_fig.update_traces(line=dict(width=6, color=bar_color), marker=dict(size=12),
                                   fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.25)")
        peak_day_fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title=f"{selected_load} (kW)",
            xaxis=dict(dtick=1),
            height=700,
            showlegend=False,
        )

        st.subheader(f"Peak Day — {selected_load}")
        st.plotly_chart(peak_day_fig, use_container_width=True)
        st.caption(f"Daily Total on {date_label}: {peak_total:,.1f}")

    else:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 5 — Benchmark
# =========================
@st.cache_data(show_spinner=False)
def load_benchmark_sheet(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    """Load a building-use sheet from a benchmark workbook.
    Expected columns: KPI, Unit, Excellent_Max, Good_Max, Poor_Max
    """
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    dfb = pd.read_excel(xls, sheet_name=sheet_name)
    required = {"KPI", "Unit", "Excellent_Max", "Good_Max", "Poor_Max"}
    if not required.issubset(dfb.columns):
        raise ValueError(f"Benchmark sheet '{sheet_name}' must have columns: {sorted(required)}")
    return dfb

def classify_kpi(value: float, exc: float, good: float, poor: float) -> str:
    if value <= exc:
        return "Excellent"
    elif value <= good:
        return "Good"
    elif value <= poor:
        return "Poor"
    else:
        return "Very Poor"

def class_color(label: str) -> str:
    return {
        "Excellent": "#22c55e",
        "Good": "#eab308",
        "Poor": "#ef4444",
        "Very Poor": "#991b1b",
    }.get(label, "#6b7280")

with tab5:
    if uploaded_file:
        df_m = melted_energy_balance(uploaded_file.getvalue())
        df_m["Energy_Source"] = df_m["End_Use"].map({k: st.session_state.get(f"source_{k}", "Electricity") for k in df_m["End_Use"].unique()})

        # KPI trio
        totals_use = df_m.groupby("End_Use", as_index=False)["kWh"].sum()
        project_area = float(st.session_state["project_area"])
        totals_use["kWh_per_m2"] = (totals_use["kWh"] / project_area).round(4)
        eui_val = totals_use.loc[totals_use["kWh_per_m2"] > 0, "kWh_per_m2"].sum()

        factor_map = {
            "Electricity": float(st.session_state["co2_elec"]),
            "Green Electricity": float(st.session_state["co2_green"]),
            "Gas": float(st.session_state["co2_gas"]),
            "District Heating": float(st.session_state["co2_dh"]),
            "District Cooling": float(st.session_state["co2_dc"]),
        }
        df_co2 = df_m.copy()
        df_co2["kgCO2"] = df_co2["kWh"] * df_co2["Energy_Source"].map(factor_map).fillna(0.0)
        co2_intensity_val = (df_co2["kgCO2"].sum() / project_area)

        cost_map = {
            "Electricity": float(st.session_state["cost_elec"]),
            "Green Electricity": float(st.session_state["cost_green"]),
            "Gas": float(st.session_state["cost_gas"]),
            "District Heating": float(st.session_state["cost_dh"]),
            "District Cooling": float(st.session_state["cost_dc"]),
        }
        df_cost = df_m.copy()
        df_cost["cost"] = df_cost["kWh"] * df_cost["Energy_Source"].map(cost_map).fillna(0.0)
        cost_intensity_val = (df_cost["cost"].sum() / project_area)

        # Load benchmark file
        bench_bytes = None
        if st.session_state.get("bench_upload"):
            bench_bytes = st.session_state["bench_upload"].getvalue()
            bench_source_label = "Uploaded benchmark (session)"
        else:
            default_path = Path("templates/benchmark_template.xlsx")
            if default_path.exists():
                bench_bytes = default_path.read_bytes()
                bench_source_label = "templates/benchmark_template.xlsx"
            else:
                bench_source_label = "No benchmark file found"

        if bench_bytes is None:
            st.error("Benchmark template not found. Please add 'templates/benchmark_template.xlsx' or upload a file in 'Benchmark Settings'.")
            st.stop()

        sel_use = st.session_state.get("building_use", "Office")
        try:
            bench_df = load_benchmark_sheet(bench_bytes, sel_use)
        except Exception as e:
            st.error(f"Could not load sheet '{sel_use}' from benchmark file. {e}")
            st.stop()

        def get_limits(kpi_name: str):
            row = bench_df.loc[bench_df["KPI"] == kpi_name]
            if row.empty:
                return None
            r = row.iloc[0]
            return float(r["Excellent_Max"]), float(r["Good_Max"]), float(r["Poor_Max"]), str(r.get("Unit", ""))

        eui_lim = get_limits("Energy_Density")
        co2_lim = get_limits("CO2_Intensity")
        cost_lim = get_limits("Cost_Intensity")
        if not all([eui_lim, co2_lim, cost_lim]):
            st.error("Benchmark sheet is missing: 'Energy_Density', 'CO2_Intensity', 'Cost_Intensity'.")
            st.stop()

        exc_eui, good_eui, poor_eui, unit_eui = eui_lim
        exc_co2, good_co2, poor_co2, unit_co2 = co2_lim
        exc_cost, good_cost, poor_cost, unit_cost = cost_lim

        cat_eui = classify_kpi(eui_val, exc_eui, good_eui, poor_eui)
        cat_co2 = classify_kpi(co2_intensity_val, exc_co2, good_co2, poor_co2)
        cat_cost = classify_kpi(cost_intensity_val, exc_cost, good_cost, poor_cost)

        def build_gauge(value, exc, good, poor, title, unit_text):
            axis_max = max(poor * 1.2, value * 1.1, exc * 1.2)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': title},
                number={'valueformat': ".1f"},
                gauge={
                    'axis': {'range': [0, axis_max]},
                    'bar': {'color': '#111827'},
                    'steps': [
                        {'range': [0, exc], 'color': class_color("Excellent")},
                        {'range': [exc, good], 'color': class_color("Good")},
                        {'range': [good, poor], 'color': class_color("Poor")},
                    ],
                    'threshold': {'line': {'color': '#111827', 'width': 4}, 'thickness': 0.75, 'value': value}
                }
            ))
            fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=20))
            return fig

        def build_vertical_band(value, exc, good, poor, title, unit_text):
            ymax = max(poor * 1.2, value * 1.1, exc * 1.2)
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=0, y1=exc, yref="y",
                          fillcolor=class_color("Excellent"), opacity=0.35, line_width=0)
            fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=exc, y1=good, yref="y",
                          fillcolor=class_color("Good"), opacity=0.35, line_width=0)
            fig.add_shape(type="rect", x0=0, x1=1, xref="paper", y0=good, y1=poor, yref="y",
                          fillcolor=class_color("Poor"), opacity=0.35, line_width=0)
            cat = classify_kpi(value, exc, good, poor)
            fig.add_trace(go.Scatter(
                x=[0.5], y=[value], mode="markers+text",
                marker=dict(size=16, color=class_color(cat), symbol="diamond"),
                text=[f"{value:,.1f} {unit_text} — {cat}"],
                textposition="top center",
                hovertemplate=f"Project: {value:,.1f} {unit_text}<extra></extra>",
            ))
            fig.update_layout(
                title=title,
                xaxis=dict(visible=False, range=[0, 1]),
                yaxis=dict(range=[0, ymax], title=unit_text),
                height=320, margin=dict(l=40, r=40, t=60, b=30),
                showlegend=False,
            )
            return fig

        st.write(f"### Benchmark — {sel_use}  \n_Source: {bench_source_label}_")

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(build_gauge(eui_val, exc_eui, good_eui, poor_eui, "Energy Density", "kWh/m²·a"), use_container_width=True)
        with g2:
            st.plotly_chart(build_gauge(co2_intensity_val, exc_co2, good_co2, poor_co2, "CO₂ Intensity", "kgCO₂/m²·a"), use_container_width=True)
        with g3:
            sym = st.session_state["currency_symbol"]
            st.plotly_chart(build_gauge(cost_intensity_val, exc_cost, good_cost, poor_cost, f"Cost Intensity ({sym})", f"{sym}/m²·a"), use_container_width=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.plotly_chart(build_vertical_band(eui_val, exc_eui, good_eui, poor_eui, "Energy Label — EUI", "kWh/m²·a"), use_container_width=True)
        with b2:
            st.plotly_chart(build_vertical_band(co2_intensity_val, exc_co2, good_co2, poor_co2, "Energy Label — CO₂", "kgCO₂/m²·a"), use_container_width=True)
        with b3:
            sym = st.session_state["currency_symbol"]
            st.plotly_chart(build_vertical_band(cost_intensity_val, exc_cost, good_cost, poor_cost, f"Energy Label — Cost ({sym})", f"{sym}/m²·a"), use_container_width=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.write("#### Thresholds")
            thresh = pd.DataFrame({
                "KPI": ["Energy_Density", "CO2_Intensity", "Cost_Intensity"],
                "Excellent ≤": [exc_eui, exc_co2, exc_cost],
                "Good ≤": [good_eui, good_co2, good_cost],
                "Poor ≤": [poor_eui, poor_co2, poor_cost],
            })
            st.dataframe(thresh, use_container_width=True)
        with c2:
            sym = st.session_state["currency_symbol"]
            st.subheader("Project KPI's")
            st.metric("Energy Density", f"{eui_val:,.1f} kWh/m²·a", help=f"Class: {cat_eui}")
            st.metric("CO₂ Intensity", f"{co2_intensity_val:,.1f} kgCO₂/m²·a", help=f"Class: {cat_co2}")
            st.metric("Cost Intensity", f"{sym} {cost_intensity_val:,.2f} /m²·a", help=f"Class: {cat_cost}")

    else:
        st.write("### ← Please upload data on sidebar")
