import io
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Robust numeric input helpers (dot/comma tolerant, no spinner behavior)
def _seed_default(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def _parse_float_locale(s, default):
    try:
        if isinstance(s, (int, float)):
            return float(s)
        if s is None:
            return float(default)
        ss = str(s).strip().replace(",", ".")
        v = float(ss)
        return v
    except Exception:
        return float(default)

def numeric_input(label, default, key, min_value=None, max_value=None, fmt=None, help=None):
    txt_key = f"{key}_txt"
    if txt_key not in st.session_state:
        st.session_state[txt_key] = (fmt.format(default) if fmt else str(default)) if hasattr(fmt, "format") else (fmt or str(default))
    val = st.text_input(label, key=txt_key, help=help)
    v = _parse_float_locale(val, default)
    if (min_value is not None) and (v < min_value):
        v = min_value
    if (max_value is not None) and (v > max_value):
        v = max_value
    st.session_state[key] = v
    return v

import numpy as np
import plotly.colors as pc
from typing import Optional, Tuple, Dict

### Werner Sobek Green Technologies GmbH. All rights reserved.###
### Author: Rodrigo Carvalho ###


# =========================
# Page setup & constants
# =========================
st.set_page_config(
    page_title="WSGT_BPvis LT 1.0.0",
    page_icon="Pamo_Icon_White.png",
    layout="wide"
)

# Centralized categorical orders (used across charts)
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
END_USE_ORDER = [
    "Heating", "Cooling", "Ventilation", "Lighting",
    "Equipment", "HotWater", "Pumps", "Other", "PV_Generation"
]
ENERGY_SOURCE_ORDER = ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"]

# Color maps (keep appearance identical to your current version)
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
    "Electricity": "#42b360",
    "Green Electricity": "#64c423",
    "Gas": "#c9d302",
    "District Heating": "#ec6939",
    "District Cooling": "#5a5ea5",
    # negative values will still be this color
}
color_map_sources = {
    "Electricity": "#42b360",
    "Green Electricity": "#64c423",
    "Gas": "#c9d302",
    "District Heating": "#ec6939",
    "District Cooling": "#5a5ea5",
}

# --- NEW: ensure a default project name exists before rendering the title
if "project_name" not in st.session_state:
    st.session_state["project_name"] = "Building Performance Dashboard"

# =========================
# Sidebar — template download & file upload
# =========================
st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## BPVis LT")
st.sidebar.write("Version 1.1.3")

st.sidebar.markdown("### Download Template")
template_path = Path("templates/energy_database_complete_template.xlsx")
if template_path.exists():
    with open(template_path, "rb") as file:
        st.sidebar.download_button(
            label="Download Excel Template",
            data=file.read(),
            file_name="energy_database_complete_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.sidebar.markdown("---")
st.sidebar.write("### Upload Data")

# Upload Excel File (xlsx)
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type="xlsx")

st.sidebar.markdown("---")
st.sidebar.write("### Project Information")


# =========================
# Small helper — cached loader
# (Speeds up reruns while you tweak sidebar inputs)
# =========================
@st.cache_data(show_spinner=False)
def energy_balance_sheet(file_bytes: bytes) -> pd.DataFrame:
    """Load 'Energy_Balance' sheet and strip '_kWh' suffix from columns."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df_ = pd.read_excel(xls, sheet_name="Energy_Balance")
    df_.columns = df_.columns.str.replace("_kWh", "", regex=False)
    return df_


def loads_balace_sheet(file_bytes: bytes) -> pd.DataFrame:
    """Load 'Loads_Balance' sheet and strip '_load' suffix from columns."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df_loads = pd.read_excel(xls, sheet_name="Loads_Balance")
    df_loads.columns = [c.removesuffix("_load") for c in df_loads.columns]
    return df_loads


# =========================
# NEW — Configuration I/O helpers (Save/Load Project settings)
# =========================
SHEET_PROJECT = "Project_Data"
SHEET_FACTORS = "Emission_Factors"
SHEET_TARIFFS = "Energy_Tariffs"
SHEET_MAPPING = "EndUse_to_Source"


def read_config_from_excel(file_bytes: bytes) -> Dict[str, Optional[pd.DataFrame]]:
    """Read known config sheets if present; return dict of dataframes (or None)."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
    return {
        "project": sheets.get(SHEET_PROJECT),
        "factors": sheets.get(SHEET_FACTORS),
        "tariffs": sheets.get(SHEET_TARIFFS),
        "mapping": sheets.get(SHEET_MAPPING),
        "all_sheets": sheets,  # keep to preserve everything when writing back
    }


def parse_project_df(df: Optional[pd.DataFrame]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    if df is None or not {"Key", "Value"}.issubset(df.columns):
        return None, None, None
    kv = dict(zip(df["Key"].astype(str), df["Value"]))
    name = kv.get("Project_Name")
    area = None
    try:
        if kv.get("Project_Area") is not None:
            area = float(kv.get("Project_Area"))
    except Exception:
        area = None
    currency = kv.get("Currency")
    return name, area, currency


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


def build_project_df(project_name: str, project_area: float, currency_symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"Key": ["Project_Name", "Project_Area", "Currency"],
         "Value": [project_name, project_area, currency_symbol]}
    )


def build_factors_df(co2_elec: float, co2_green: float, co2_dh: float, co2_dc: float, co2_gas: float) -> pd.DataFrame:
    # Keep a clear mapping independent of ENERGY_SOURCE_ORDER variable ordering
    return pd.DataFrame({
        "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"],
        "Factor_kgCO2_per_kWh": [co2_elec, co2_green, co2_gas, co2_dh, co2_dc]
    })


def build_tariffs_df(cost_elec: float, cost_green: float, cost_dh: float, cost_dc: float,
                     cost_gas: float) -> pd.DataFrame:
    return pd.DataFrame({
        "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"],
        "Tariff_per_kWh": [cost_elec, cost_green, cost_gas, cost_dh, cost_dc]
    })


def build_mapping_df(end_uses) -> pd.DataFrame:
    rows = []
    for use in end_uses:
        rows.append({"End_Use": use, "Energy_Source": st.session_state.get(f"source_{use}", "Electricity")})
    return pd.DataFrame(rows)


# =========================
# Benchmark Functions
# =========================

def parse_project_df_with_building_use(
    df: Optional[pd.DataFrame]
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[str], Optional[float], Optional[float]]:
    """Parse Project_Data sheet (name, area, currency, building use, latitude, longitude)."""
    if df is None or not {"Key", "Value"}.issubset(df.columns):
        return None, None, None, None, None, None

    kv = dict(zip(df["Key"].astype(str), df["Value"]))

    name = kv.get("Project_Name")
    currency = kv.get("Currency")
    building_use = kv.get("Building_Use")

    # coerce to floats where possible
    def _to_float(x):
        try:
            return float(x) if x is not None and str(x).strip() != "" else None
        except Exception:
            return None

    area = _to_float(kv.get("Project_Area"))
    latitude_saved = _to_float(kv.get("Project_Latitude"))
    longitude_saved = _to_float(kv.get("Project_Longitude"))

    return name, area, currency, building_use, latitude_saved, longitude_saved


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


@st.cache_data(show_spinner=False)
def load_benchmark_data(building_use: str) -> Optional[pd.DataFrame]:
    """Load benchmark data for the specified building use"""
    try:
        benchmark_path = Path("templates/benchmark_template.xlsx")
        if not benchmark_path.exists():
            return None

        df = pd.read_excel(benchmark_path, sheet_name=building_use)
        return df
    except Exception:
        return None


def get_benchmark_category(value: float, good_threshold: float, excellent_threshold: float) -> str:
    """Determine benchmark category based on value and thresholds"""
    if value <= excellent_threshold:
        return "Excellent"
    elif value <= good_threshold:
        return "Good"
    else:
        return "Poor"


def get_benchmark_color(category: str) -> str:
    """Get color for benchmark category using existing color scheme"""
    color_mapping = {
        "Excellent": "#a9c724",  # Green from existing scheme
        "Good": "#d3b402",  # Yellow from existing scheme
        "Poor": "#c02419"  # Red from existing scheme
    }
    return color_mapping.get(category, "#666666")


def create_gauge_chart(value: float, good_threshold: float, excellent_threshold: float,
                       title: str, unit: str) -> go.Figure:
    """Create a gauge/speedometer chart for benchmark visualization"""
    category = get_benchmark_category(value, good_threshold, excellent_threshold)
    color = get_benchmark_color(category)

    # Set gauge range - extend beyond thresholds for better visualization
    max_range = max(value * 1.2, good_threshold * 1.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br><span style='font-size:1.2em'>{category}</span>"},
        number={
            'font': {'size': 70},  # big centered value
            'suffix': ""  # leave empty, we'll add unit below
        },
        gauge={
            'axis': {'range': [None, max_range]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, excellent_threshold], 'color': "#a9c724"},
                {'range': [excellent_threshold, good_threshold], 'color': "#d3b402"},
                {'range': [good_threshold, max_range], 'color': "#c02419"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 12},
                'thickness': 0.9,
                'value': value
            }
        }
    ))

    # Add unit below, without shifting the number
    fig.add_annotation(
        x=0.5, y=0.01,  # just below the number (0.44 works well with 0.5 center)
        text=f"<span style='font-size:20px'>{unit}</span>",
        showarrow=False,
        font=dict(size=20, color="black"),
        xanchor="center",
        yanchor="top"  # stick to the top so number remains centered
    )

    fig.update_layout(
        height=400,
        font={'color': "black", 'family': "Arial", 'size': 12},
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_benchmark_bar_chart(values_dict: Dict[str, float], thresholds_dict: Dict[str, Dict[str, float]],
                               title: str, unit: str) -> go.Figure:
    """Create vertical bar chart with benchmark zones"""

    kpis = list(values_dict.keys())
    values = list(values_dict.values())
    colors = []

    # Determine colors based on benchmark categories
    for kpi in kpis:
        value = values_dict[kpi]
        good_thresh = thresholds_dict[kpi]["Good_Threshold"]
        excellent_thresh = thresholds_dict[kpi]["Excellent_Threshold"]
        category = get_benchmark_category(value, good_thresh, excellent_thresh)
        colors.append(get_benchmark_color(category))

    fig = go.Figure(data=[
        go.Bar(
            x=kpis,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition='auto',
            textfont=dict(size=14, color="white")
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="KPI",
        yaxis_title=unit,
        height=400,
        showlegend=False,
        font={'color': "black", 'family': "Arial"},
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def write_config_to_excel(original_bytes: bytes,
                          project_df: pd.DataFrame,
                          factors_df: pd.DataFrame,
                          tariffs_df: pd.DataFrame,
                          mapping_df: pd.DataFrame) -> bytes:
    """Return a new workbook (bytes) with all original sheets + updated config sheets."""
    cfg = read_config_from_excel(original_bytes)
    sheets = cfg["all_sheets"]  # dict[name] -> df

    # overwrite/create the config sheets
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
# Preload any saved configuration (if an Excel is uploaded)
# =========================
preloaded = None
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    cfg_saved = read_config_from_excel(file_bytes)

    saved_name, saved_area, saved_currency, saved_building_use, saved_lat, saved_lon = \
        parse_project_df_with_building_use(cfg_saved["project"])

    saved_factors = parse_factors_df(cfg_saved["factors"])
    saved_tariffs = parse_tariffs_df(cfg_saved["tariffs"])
    saved_mapping_df = cfg_saved["mapping"]
    has_any_saved = any([
        saved_name, saved_area, saved_currency, saved_building_use, bool(saved_factors), bool(saved_tariffs),
        saved_mapping_df is not None
    ])
    if has_any_saved:
        st.sidebar.success("Saved project settings found in this workbook; preloading values.")
    else:
        st.sidebar.info("No saved project settings found; using defaults. You can save them for next time.")

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

    # --- NEW: seed the title from file if available (only once)
    if preloaded.get("name") and not st.session_state.get("_project_name_set_from_file"):
        st.session_state["project_name"] = preloaded["name"]
        st.session_state["_project_name_set_from_file"] = True

# =========================
# Header (moved here so it can use preloaded name)
# =========================
col1, col2 = st.columns(2)
with col2:
    logo_path = Path("WS_Logo.jpg")
    if logo_path.exists():
        st.image(str(logo_path), width=900)

# --- UPDATED: title now reflects Project Name
st.title(st.session_state["project_name"])

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Energy Balance", "CO2 Emissions", "Energy Cost", "Loads Analysis", "Benchmark"])

# =========================
# Tab 1 — Energy Balance (Energy Balance Tab)
# =========================
with tab1:
    if uploaded_file:
        # ---- Load data
        df = energy_balance_sheet(uploaded_file.getvalue())

        # ---- Wide->Long transform for plotting and grouping
        df_melted = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

        # ---- Sidebar: project info (prefill from saved if available)
        with st.sidebar.expander("Project Data"):
            st.write("Enter Project's Basic Informations")

            default_name = preloaded["name"] if (preloaded and preloaded["name"]) else "Example Building 1"
            default_area = preloaded["area"] if (preloaded and preloaded["area"] is not None) else 1000.00
            default_building_use = preloaded["building_use"] if (preloaded and preloaded["building_use"]) else "Office"

            # NEW: defaults for lat/lon (fallback to your previous hard-coded values)
            default_lat = preloaded["lat"] if (preloaded and preloaded["lat"] is not None) else 53.54955
            default_lon = preloaded["lon"] if (preloaded and preloaded["lon"] is not None) else 9.9936

            # keep title reactive via session_state
            project_name = st.text_input("Project Name", value=default_name, key="project_name")
            project_area = numeric_input("Project Area", float(default_area), key="project_area", min_value=0.0)

            # FIXED LABEL + use defaults from file if present
            latitude = st.text_input("Project Latitude", value=str(default_lat), key="project_latitude")
            longitude = st.text_input("Project Longitude", value=str(default_lon), key="project_longitude")

            # building use dropdown unchanged...
            building_use_options = ["Office", "Hospitality", "Retail", "Residential", "Industrial", "Education",
                                    "Leisure", "Healthcare"]
            building_use_index = building_use_options.index(
                default_building_use) if default_building_use in building_use_options else 0
            building_use = st.selectbox("Building Use", building_use_options, index=building_use_index)

        # ---- Sidebar: emission factors (used in Tab 2, but defined once)
        with st.sidebar.expander("Emission Factors"):
            st.write("Assign Emission Factors")
            def_f = preloaded["factors"] if preloaded else {}
            co2_Emissions_Electricity = numeric_input("CO2 Factor Electricity", float(def_f.get("Electricity", 0.300)), key="co2_Emissions_Electricity", min_value=0.0, max_value=1.0, fmt="{:.3f}")
            co2_Emissions_Green_Electricity = numeric_input("CO2 Factor Green Electricity", float(def_f.get("Green Electricity", 0.000)), key="co2_Emissions_Green_Electricity", min_value=0.0, max_value=1.0, fmt="{:.3f}")
            co2_emissions_dh = numeric_input("CO2 Factor District Heating", float(def_f.get("District Heating", 0.260)), key="co2_emissions_dh", min_value=0.0, max_value=1.0, fmt="{:.3f}")
            co2_emissions_dc = numeric_input("CO2 Factor District Cooling", float(def_f.get("District Cooling", 0.280)), key="co2_emissions_dc", min_value=0.0, max_value=1.0, fmt="{:.3f}")
            co2_emissions_gas = numeric_input("CO2 Factor Gas", float(def_f.get("Gas", 0.180)), key="co2_emissions_gas", min_value=0.0, max_value=1.0, fmt="{:.3f}")


        # --- Energy Cost (€/kWh) ---
        with st.sidebar.expander("Energy Tariffs"):
            st.write("Assign energy cost per source (per kWh)")
            default_currency = preloaded["currency"] if (
                        preloaded and preloaded["currency"] in ["€", "$", "£"]) else "€"
            currency_symbol = st.selectbox("Currency", ["€", "$", "£"], index=["€", "$", "£"].index(default_currency))

            def_t = preloaded["tariffs"] if preloaded else {}
            cost_electricity = numeric_input(f"Cost Electricity ({currency_symbol}/kWh)", float(def_t.get("Electricity", 0.35)), key="cost_electricity", min_value=0.0, max_value=100.0, fmt="{:.2f}")
            cost_green_electricity = numeric_input(f"Cost Green Electricity ({currency_symbol}/kWh)", float(def_t.get("Green Electricity", 0.40)), key="cost_green_electricity", min_value=0.0, max_value=100.0, fmt="{:.2f}")
            cost_dh = numeric_input(f"Cost District Heating ({currency_symbol}/kWh)", float(def_t.get("District Heating", 0.16)), key="cost_dh", min_value=0.0, max_value=100.0, fmt="{:.2f}")
            cost_dc = numeric_input(f"Cost District Cooling ({currency_symbol}/kWh)", float(def_t.get("District Cooling", 0.16)), key="cost_dc", min_value=0.0, max_value=100.0, fmt="{:.2f}")
            cost_gas = numeric_input(f"Cost Gas ({currency_symbol}/kWh)", float(def_t.get("Gas", 0.12)), key="cost_gas", min_value=0.0, max_value=100.0, fmt="{:.2f}")


        # ---- Sidebar: map End_Use -> Energy_Source (user-controlled)
        with st.sidebar.expander("Assign Energy Sources"):
            st.write("Assign Energy Sources")
            end_uses = df_melted["End_Use"].unique().tolist()

            # If we have a saved mapping sheet, parse it to set defaults:
            saved_mapping = parse_mapping_df(preloaded["mapping_df"]) if (
                        preloaded and preloaded["mapping_df"] is not None) else {}

            mapping_dict = {}
            st.sidebar.markdown("---")
            for use in end_uses:
                default_source = saved_mapping.get(use, "Electricity")
                idx = ENERGY_SOURCE_ORDER.index(default_source) if default_source in ENERGY_SOURCE_ORDER else 0
                source = st.selectbox(
                    f"{use}",
                    ENERGY_SOURCE_ORDER,
                    index=idx,
                    key=f"source_{use}",  # distinct widget keys
                )
                mapping_dict[use] = source

        # ---- Save Project button (exports current inputs into the workbook)
        with st.sidebar:
            if st.button("Save Project", use_container_width=True):
                # coerce UI strings to floats when possible
                def _to_float_safe(s):
                    try:
                        return float(s)
                    except Exception:
                        return None


                lat_val = _to_float_safe(latitude)
                lon_val = _to_float_safe(longitude)

                project_df = build_project_df_with_building_use(
                    st.session_state.get("project_name", project_name),
                    project_area,
                    currency_symbol,
                    building_use,
                    lat_val,
                    lon_val,
                )

                factors_df = build_factors_df(co2_Emissions_Electricity, co2_Emissions_Green_Electricity,
                                              co2_emissions_dh, co2_emissions_dc, co2_emissions_gas)
                tariffs_df = build_tariffs_df(cost_electricity, cost_green_electricity, cost_dh, cost_dc, cost_gas)
                mapping_df = build_mapping_df(end_uses)

                updated_bytes = write_config_to_excel(preloaded["file_bytes"], project_df, factors_df, tariffs_df,
                                                      mapping_df)

                st.success("Project settings saved to workbook.")
                st.download_button(
                    label="Download Updated Workbook",
                    data=updated_bytes,
                    file_name=uploaded_file.name.replace(".xlsx", "_with_project.xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            st.markdown("---")

            st.caption("*A product of*")
            st.image("WS_Logo.png", width=300)
            st.caption("Werner Sobek Green Technologies GmbH")
            st.caption("Fachgruppe Simulation")
            st.markdown("---")
            st.caption("*Coded by*")
            st.caption("Rodrigo Carvalho")
            st.caption("*Need help? Contact me under:*")
            st.caption("*email:* rodrigo.carvalho@wernersobek.com")
            st.caption("*Tel* +49.40.6963863-14")
            st.caption("*Mob* +49.171.964.7850")

        # ---- Apply mapping to create Energy_Source column
        df_melted["Energy_Source"] = df_melted["End_Use"].map(mapping_dict)

        # ---- Monthly net totals (used for overlay line)
        monthly_totals = (
            df_melted.groupby("Month", as_index=False)["kWh"].sum()
            .assign(Month=lambda d: pd.Categorical(d["Month"], categories=MONTH_ORDER, ordered=True))
            .sort_values("Month", kind="stable")
            .reset_index(drop=True)
        )

        # ---- Monthly bar per End_Use (stacked, pos/neg relative) + net line overlay
        monthly_chart = px.bar(
            df_melted,
            x="Month",
            y="kWh",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"Month": MONTH_ORDER},  # ensure bars align with the line
            text_auto=".0f",  # value labels on bars,
        )

        monthly_chart.update_traces(textfont_size=14, textfont_color="white")

        line_monthly_net = px.line(
            monthly_totals, x="Month", y="kWh", markers=True, labels={"kWh": "Net total"}
        )
        for tr in line_monthly_net.data:
            tr.name = "Net total"
            tr.line.width = 5
            tr.line.color = "black"
            tr.line.dash = "dash"
            tr.marker.size = 12
            monthly_chart.add_trace(tr)
        monthly_chart.update_layout(showlegend=False)

        # ---- Monthly bar per Energy_Source (aggregate first for correct hover totals)
        monthly_by_source = (
            df_melted.groupby(["Month", "Energy_Source"], as_index=False)["kWh"].sum()
        )
        monthly_by_source["Month"] = pd.Categorical(
            monthly_by_source["Month"], categories=MONTH_ORDER, ordered=True
        )
        monthly_chart_source = px.bar(
            monthly_by_source,
            x="Month",
            y="kWh",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Month": MONTH_ORDER},
            text_auto=".0f",  # value labels on bars

        )
        monthly_chart_source.update_layout(showlegend=False)
        monthly_chart_source.update_traces(textfont_size=14, textfont_color="white")


        st.write("## Energy Balance (per End Use)")

        # ---- Annual totals per End_Use and per Energy_Source (+ intensities)
        totals = df_melted.groupby("End_Use", as_index=False)["kWh"].sum()
        totals["Per Use"] = "Total"
        totals["kWh_per_m2"] = (totals["kWh"] / project_area).round(1)

        # KPI helpers
        eui = totals.loc[totals["kWh_per_m2"] > 0, "kWh_per_m2"].sum()  # consumption-only intensity
        net_energy = totals["kWh"].sum()  # net kWh (PV included)
        net_eui = totals["kWh_per_m2"].sum()  # net intensity

        totals_per_source = df_melted.groupby("Energy_Source", as_index=False)["kWh"].sum()
        totals_per_source["Per Source"] = "total_per_source"
        totals_per_source["kWh_per_m2_per_source"] = (totals_per_source["kWh"] / project_area).round(1)

        # ---- Annual stacked bars (per End_Use + reference line)
        annual_chart = px.bar(
            totals,
            x="Per Use",
            y="kWh",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
            text_auto=".0f",  # value labels on bars
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

        # ---- Annual stacked bars (per Energy_Source)
        annual_chart_per_source = px.bar(
            totals_per_source,
            x="Per Source",
            y="kWh",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
            text_auto=".0f",  # value labels on bars
        )
        annual_chart_per_source.update_traces(textfont_size=14, textfont_color="white")

        # ---- Donuts (EUI shares)
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
                font=dict(size=50, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")

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
                font=dict(size=50, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart_per_source.update_traces(textinfo="value+percent", textfont_size=18,
                                                        textfont_color="white")

        # ---- PV coverage (share of PV vs consumption-only EUI)
        totals_indexed = totals.set_index("End_Use")
        pv_value = totals_indexed.loc["PV_Generation", "kWh_per_m2"] if "PV_Generation" in totals_indexed.index else 0.0
        pv_coverage = abs((pv_value / eui) * 100) if eui != 0 else 0.0

        # ---- Layout: charts and KPIs (kept identical)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy Demand")
            st.plotly_chart(monthly_chart, use_container_width=True)
        with col2:
            st.subheader("Annual Energy Demand")
            st.plotly_chart(annual_chart, use_container_width=True)

        # KPI calculations (kept identical logic)
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

        st.markdown("---")
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

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 2 — CO₂ Emissions (CO2 Emissions Tab)
# =========================
with tab2:
    if uploaded_file:
        # Ensure Energy_Source exists (same mapping as Tab 1)
        df = energy_balance_sheet(uploaded_file.getvalue())
        df_melted = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")
        df_melted["Energy_Source"] = df_melted["End_Use"].map(
            {k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted["End_Use"].unique()})

        # Factor map from sidebar inputs (declared in Tab 1)
        factor_map = {
            "Electricity": co2_Emissions_Electricity,
            "Green Electricity": co2_Emissions_Green_Electricity,
            "Gas": co2_emissions_gas,
            "District Heating": co2_emissions_dh,
            "District Cooling": co2_emissions_dc,
        }

        # Compute emissions per row
        df_co2 = df_melted.copy()
        df_co2["CO2_factor_kg_per_kWh"] = df_co2["Energy_Source"].map(factor_map).fillna(0.0)
        df_co2["kgCO2"] = df_co2["kWh"] * df_co2["CO2_factor_kg_per_kWh"]

        # Monthly net CO₂ totals (line overlay)
        monthly_totals_co2 = (
            df_co2.groupby("Month", as_index=False)["kgCO2"].sum()
            .assign(Month=lambda d: pd.Categorical(d["Month"], categories=MONTH_ORDER, ordered=True))
            .sort_values("Month", kind="stable")
            .reset_index(drop=True)
        )

        # Monthly CO₂ per End_Use + net line overlay
        monthly_chart_co2_use = px.bar(
            df_co2,
            x="Month",
            y="kgCO2",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"Month": MONTH_ORDER},
            text_auto=".0f",  # value labels on bars
        )
        monthly_chart_co2_use.update_traces(textfont_size=14, textfont_color="white")

        line_monthly_net_co2 = px.line(
            monthly_totals_co2, x="Month", y="kgCO2", markers=True, labels={"kgCO2": "Net total"}
        )
        for tr in line_monthly_net_co2.data:
            tr.name = "Net total"
            tr.line.width = 5
            tr.line.color = "black"
            tr.line.dash = "dash"
            tr.marker.size = 12
            monthly_chart_co2_use.add_trace(tr)
        monthly_chart_co2_use.update_layout(showlegend=False)

        # Monthly CO₂ per Energy_Source (aggregate first)
        monthly_co2_by_source = df_co2.groupby(["Month", "Energy_Source"], as_index=False)["kgCO2"].sum()
        monthly_co2_by_source["Month"] = pd.Categorical(
            monthly_co2_by_source["Month"], categories=MONTH_ORDER, ordered=True
        )
        monthly_chart_co2_source = px.bar(
            monthly_co2_by_source,
            x="Month",
            y="kgCO2",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Month": MONTH_ORDER, "Energy_Source": ENERGY_SOURCE_ORDER},
            text_auto=".0f",  # value labels on bars
        )
        monthly_chart_co2_source.update_layout(showlegend=False)
        monthly_chart_co2_source.update_traces(textfont_size=14, textfont_color="white")

        # Annual CO₂ totals (per End_Use and per Energy_Source)
        totals_co2_use = df_co2.groupby("End_Use", as_index=False)["kgCO2"].sum()
        totals_co2_use["Per Use"] = "Total"
        totals_co2_use["kgCO2_per_m2"] = (totals_co2_use["kgCO2"] / project_area).round(1)
        net_co2 = totals_co2_use["kgCO2"].sum()

        totals_co2_source = df_co2.groupby("Energy_Source", as_index=False)["kgCO2"].sum()
        totals_co2_source["Per Source"] = "total_per_source"
        totals_co2_source["kgCO2_per_m2_per_source"] = (totals_co2_source["kgCO2"] / project_area).round(1)

        # Annual stacked bars + net line (End_Use)
        annual_chart_co2_use = px.bar(
            totals_co2_use,
            x="Per Use",
            y="kgCO2",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
            text_auto=".0f",  # value labels on bars
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

        # Annual stacked bars (Energy_Source)
        annual_chart_co2_source = px.bar(
            totals_co2_source,
            x="Per Source",
            y="kgCO2",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
            text_auto=".0f",  # value labels on bars
        )
        annual_chart_co2_source.update_traces(textfont_size=14, textfont_color="white")

        # Donuts: CO₂ intensity shares
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
        co2_intensity_pie_use.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")

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
        co2_intensity_pie_source.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")

        # KPIs
        monthly_avg_co2 = monthly_totals_co2["kgCO2"].mean()
        annual_total_co2 = totals_co2_use["kgCO2"].sum()
        co2_intensity_total = totals_co2_use["kgCO2_per_m2"].sum()

        # Center annotations (show total intensity in donut centers)
        co2_intensity_pie_use.update_layout(
            annotations=[dict(
                text=f"{co2_intensity_total:,.1f}<br>kgCO₂/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=50, color="black"),
            )]
        )
        co2_intensity_pie_source.update_layout(
            annotations=[dict(
                text=f"{co2_intensity_total:,.1f}<br>kgCO₂/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=50, color="black"),
            )]
        )

        # Layout (kept identical)

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

        st.markdown("---")
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

    if not uploaded_file:
        st.write("### ← Please upload data on side bar")

# =========================
# Tab 3 — Energy Cost (Energy Cost Tab)
# =========================
with tab3:
    if uploaded_file:
        # Ensure we have the same melted data + mapping used in other tabs
        xls = pd.ExcelFile(uploaded_file)
        df_cost_base = pd.read_excel(xls, sheet_name="Energy_Balance")
        df_cost_base.columns = df_cost_base.columns.str.replace("_kWh", "", regex=False)
        df_melted_cost = df_cost_base.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

        # Reuse the user's End_Use -> Energy_Source mapping from the sidebar
        end_uses_here = df_melted_cost["End_Use"].unique()
        mapping_dict_cost = {use: st.session_state.get(f"source_{use}", "Electricity") for use in end_uses_here}
        df_melted_cost["Energy_Source"] = df_melted_cost["End_Use"].map(mapping_dict_cost)

        # Build the cost map from sidebar inputs
        cost_map = {
            "Electricity": cost_electricity,
            "Gas": cost_gas,
            "District Heating": cost_dh,
            "District Cooling": cost_dc,
            "Green Electricity": cost_green_electricity,
        }

        # Compute row-level cost
        df_cost = df_melted_cost.copy()
        df_cost["cost_per_kWh"] = df_cost["Energy_Source"].map(cost_map).fillna(0.0)
        df_cost["cost"] = df_cost["kWh"] * df_cost["cost_per_kWh"]  # negative PV -> negative cost (saves money)

        # ---------- Monthly charts ----------
        month_order = MONTH_ORDER

        monthly_totals_cost = (
            df_cost.groupby("Month", as_index=False)["cost"].sum()
        )
        monthly_totals_cost["Month"] = pd.Categorical(monthly_totals_cost["Month"], categories=month_order,
                                                      ordered=True)
        monthly_totals_cost = monthly_totals_cost.sort_values("Month").reset_index(drop=True)

        # (A) by End Use + overlay line
        monthly_chart_cost_use = px.bar(
            df_cost,
            x="Month", y="cost",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"Month": month_order},
            text_auto=".0f",
        )
        monthly_chart_cost_use.update_traces(textfont_size=14, textfont_color="white")

        line_monthly_net_cost = px.line(
            monthly_totals_cost, x="Month", y="cost", markers=True, labels={"cost": "Net total"}
        )
        for tr in line_monthly_net_cost.data:
            tr.name = "Net total"
            tr.line.width = 5
            tr.line.color = "black"
            tr.line.dash = "dash"
            tr.marker.size = 12
            monthly_chart_cost_use.add_trace(tr)
        monthly_chart_cost_use.update_layout(showlegend=False)

        # (B) by Energy Source (aggregate first for clean hovers)
        monthly_cost_by_source = df_cost.groupby(["Month", "Energy_Source"], as_index=False)["cost"].sum()
        monthly_cost_by_source["Month"] = pd.Categorical(monthly_cost_by_source["Month"], categories=month_order,
                                                         ordered=True)
        monthly_chart_cost_source = px.bar(
            monthly_cost_by_source,
            x="Month", y="cost",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Month": month_order,
                             "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating",
                                               "District Cooling"]},
            text_auto=".0f",  # value labels on bars,
        )

        monthly_chart_cost_source.update_layout(showlegend=False)
        monthly_chart_cost_source.update_traces(textfont_size=14, textfont_color="white")

        # ---------- Annual totals & intensities ----------
        # (A) By End Use
        totals_cost_use = df_cost.groupby("End_Use", as_index=False)["cost"].sum()
        totals_cost_use["Per Use"] = "Total"
        totals_cost_use["cost_per_m2"] = (totals_cost_use["cost"] / project_area).round(2)

        # (B) By Energy Source
        totals_cost_source = df_cost.groupby("Energy_Source", as_index=False)["cost"].sum()
        totals_cost_source["Per Source"] = "total_per_source"
        totals_cost_source["cost_per_m2_per_source"] = (totals_cost_source["cost"] / project_area).round(2)

        # ---------- Annual stacked bars ----------
        # End Use + net horizontal line
        annual_chart_cost_use = px.bar(
            totals_cost_use,
            x="Per Use", y="cost",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={
                "End_Use": ["Heating", "Cooling", "Ventilation", "Lighting", "Equipment", "HotWater", "Pumps", "Other",
                            "PV_Generation"]},
            text_auto=".0f",
        )
        net_cost = totals_cost_use["cost"].sum()
        annual_chart_cost_use.add_hline(y=net_cost, line_width=4, line_dash="dash", line_color="black")
        annual_chart_cost_use.add_annotation(
            x=0.5, xref="paper",
            y=net_cost, yref="y",
            text=f"{currency_symbol} {net_cost:,.0f}",
            showarrow=False, yshift=10, font=dict(size=16, color="white")
        )
        annual_chart_cost_use.update_traces(textfont_size=14, textfont_color="white")

        # By Energy Source
        annual_chart_cost_source = px.bar(
            totals_cost_source,
            x="Per Source", y="cost",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={
                "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"]},
            text_auto=".0f",
        )

        annual_chart_cost_source.update_traces(textfont_size=14, textfont_color="white")

        # ---------- Donuts: Cost intensity (currency/m²·a) ----------
        cost_intensity_pie_use = px.pie(
            totals_cost_use,
            names="End_Use",
            values="cost_per_m2",
            color="End_Use",
            color_discrete_map=color_map,
            hole=0.5,
            height=800,
            category_orders={
                "End_Use": ["Heating", "Cooling", "Ventilation", "Lighting", "Equipment", "HotWater", "Pumps", "Other",
                            "PV_Generation"]}
        )
        cost_intensity_pie_use.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")

        cost_intensity_pie_source = px.pie(
            totals_cost_source,
            names="Energy_Source",
            values="cost_per_m2_per_source",
            color="Energy_Source",
            color_discrete_map=color_map_sources,
            hole=0.5,
            height=800,
            category_orders={
                "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"]},
        )
        cost_intensity_pie_source.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")

        # Center totals (sum of intensities)
        cost_intensity_total = totals_cost_use["cost_per_m2"].sum()
        cost_intensity_pie_use.update_layout(
            showlegend=True,
            annotations=[dict(
                text=f"{currency_symbol} {cost_intensity_total:,.2f}<br>per m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=50, color="black"),
            )]
        )
        cost_intensity_pie_source.update_layout(
            showlegend=True,
            annotations=[dict(
                text=f"{currency_symbol} {cost_intensity_total:,.2f}<br>per m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=50, color="black")
            )]
        )

        # ---------- KPIs ----------
        monthly_avg_cost = monthly_totals_cost["cost"].mean()
        annual_total_cost = totals_cost_use["cost"].sum()

        # ---------- Layout (mirrors other tabs) ----------
        st.write(f"## Energy Cost {currency_symbol} (per End Use)")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Monthly Cost")
            st.plotly_chart(monthly_chart_cost_use, use_container_width=True)
        with c2:
            st.subheader("Annual Cost")
            st.plotly_chart(annual_chart_cost_use, use_container_width=True)

        c3, c4 = st.columns([3, 1])
        with c3:
            st.subheader(f"Cost Intensity ( {currency_symbol}/m²·a )")
            st.plotly_chart(cost_intensity_pie_use, use_container_width=True)
        with c4:
            st.subheader("Cost KPI's")
            st.metric("Monthly Average Cost", f"{currency_symbol} {monthly_avg_cost:,.0f}")
            st.metric("Total Annual Cost", f"{currency_symbol} {annual_total_cost:,.0f}")
            st.metric("Cost Intensity (Total)", f"{currency_symbol} {cost_intensity_total:,.2f} /m²·a")

        st.markdown("---")
        st.write(f"## Energy Cost {currency_symbol} (per Energy Source)")
        c5, c6 = st.columns([3, 1])
        with c5:
            st.subheader("Monthly Cost")
            st.plotly_chart(monthly_chart_cost_source, use_container_width=True)
        with c6:
            st.subheader("Annual Cost")
            st.plotly_chart(annual_chart_cost_source, use_container_width=True)

        c7, c8 = st.columns([3, 1])
        with c7:
            st.subheader(f"Cost Intensity ( {currency_symbol}/m²·a )")
            st.plotly_chart(cost_intensity_pie_source, use_container_width=True)
        with c8:
            st.subheader("Cost KPI's")
            for _, row in totals_cost_source.iterrows():
                st.metric(
                    label=f"Cost Intensity - {row['Energy_Source']}",
                    value=f"{currency_symbol} {row['cost_per_m2_per_source']:,.2f} /m²·a"
                )

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 4 — Loads Analysis (Loads Analysis Tab)
# =========================
with tab4:
    if uploaded_file:
        # ---- Load data
        df_loads = loads_balace_sheet(uploaded_file.getvalue())

        # columns that are load metrics
        load_cols = [c for c in df_loads.columns if c not in ["hoy", "doy", "day", "month", "weekday", "hour"]]

        # (optional) ensure doy/hour are numeric
        df_loads["doy"] = pd.to_numeric(df_loads["doy"], errors="coerce")
        df_loads["hour"] = pd.to_numeric(df_loads["hour"], errors="coerce")

        st.write("## Load Analysis")
        selected_load = st.selectbox("Select Load", load_cols, index=0)

        load_heatmap = px.density_heatmap(
            df_loads,
            x="doy",
            y="hour",
            z=selected_load,
            nbinsx=365,  # bin per day-of-year
            nbinsy=24,  # bin per hour
            color_continuous_scale="thermal",
        )

        # cosmetics (tick steps, colorbar title, etc.)
        load_heatmap.update_layout(
            xaxis_title="Day of Year (doy)",
            yaxis_title="Hour of Day",
            coloraxis_colorbar=dict(title=selected_load),
            height=700,
        )

        sum_load = pd.to_numeric(df_loads[selected_load], errors="coerce")  # ensure numeric
        total_load_selected = sum_load.sum()
        max_load_selected = sum_load.max()
        min_load_selected = sum_load.min()
        specific_load = (sum_load / project_area) * 1000
        max_specific_load = ((max_load_selected / project_area) * 1000)
        min_specific_load = ((min_load_selected / project_area) * 1000)
        p95_specific_load = np.percentile(specific_load.dropna(), 95)
        p80_specific_load = np.percentile(specific_load.dropna(), 80)
        totals_by_month = df_loads.groupby("month", as_index=False)[selected_load].sum()
        totals_by_month["month"] = pd.Categorical(
            totals_by_month["month"], ordered=True
        )
        totals_by_month = totals_by_month.sort_values("month")

        monthly_total_load_bar = px.bar(
            totals_by_month,
            x="month",
            y=selected_load,  # the column you summed
            labels={"month": "Month", selected_load: "kWh"},
            text_auto=".0f",  # value labels on bars
            height=700
        )

        key = selected_load.replace("_load", "")  # in case the name still has the suffix
        bar_color = color_map.get(key, "#c02419")  # fallback color

        monthly_total_load_bar.update_traces(textfont_size=14, textfont_color="white")

        monthly_total_load_bar.update_traces(marker_color=bar_color, name=selected_load, showlegend=True)
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

        # exceed threshold heat map
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
            title=f"Exceedance Count Heatmap — {selected_load} > {thr:g} kW"
        )
        exceed_heatmap.update_layout(
            xaxis_title="Day of Year (doy)", yaxis_title="Hour of Day",
            coloraxis_colorbar=dict(title="Exceed"),
            height=700
        )
        st.plotly_chart(exceed_heatmap, use_container_width=True)

        st.caption(f"Total Exceeded Hours {total_exceedance:,.1f}")

        # find 5 peaks
        peaks = (df_loads.loc[:, ["month", "day", "weekday", "hour", selected_load]]
                 .sort_values(selected_load, ascending=False)
                 .head(5))

        st.subheader(f"Top 5 Peak Loads — {selected_load} (kW)")
        st.dataframe(
            peaks.style.format({selected_load: "{:,.1f} kW"}),
            use_container_width=True
        )

        # --- Peak day (by daily sum of the selected load) ---
        # Ensure numeric
        s = pd.to_numeric(df_loads[selected_load], errors="coerce")

        # Daily totals (sum over 24 hours)
        daily = (df_loads.assign(_val=s)
                 .groupby("doy", as_index=False)["_val"].sum())

        # Day-of-year with the highest total
        peak_idx = daily["_val"].abs().idxmax()
        peak_doy = int(daily.loc[peak_idx, "doy"])
        peak_total = float(daily.loc[peak_idx, "_val"])

        # Optional: nice label using month/day if available
        date_label = f"DOY {peak_doy}"
        if {"month", "day"}.issubset(df_loads.columns):
            month_val = df_loads.loc[df_loads["doy"] == peak_doy, "month"].iloc[0]
            day_val = df_loads.loc[df_loads["doy"] == peak_doy, "day"].iloc[0]

            # If month is numeric (1–12), map to names
            if pd.api.types.is_numeric_dtype(type(month_val)) or str(month_val).isdigit():
                month_order = ["January", "February", "March", "April", "May", "June",
                               "July", "August", "September", "October", "November", "December"]
                month_map = dict(enumerate(month_order, start=1))
                try:
                    month_val = month_map[int(month_val)]
                except Exception:
                    pass

            date_label = f"{month_val} {int(day_val)} (DOY {peak_doy})"

        # --- Hourly profile for that peak day ---
        day_profile = (df_loads.loc[df_loads["doy"] == peak_doy, ["hour", selected_load]]
                       .copy())
        day_profile["hour"] = pd.to_numeric(day_profile["hour"], errors="coerce")
        day_profile[selected_load] = pd.to_numeric(day_profile[selected_load], errors="coerce")
        day_profile = day_profile.sort_values("hour")

        # --- Plot: x=hour, y=load (line + markers) ---
        peak_day_fig = px.line(
            day_profile,
            x="hour",
            y=selected_load,
            markers=True,
            title=f"Peak Day Profile — {selected_load} | {date_label}"
        )
        peak_day_fig.update_traces(line=dict(width=6, color=bar_color), marker=dict(size=12))
        peak_day_fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title=f"{selected_load} (kW)",
            xaxis=dict(dtick=1),
            height=700,
            showlegend=False
        )

        r, g, b = pc.hex_to_rgb(bar_color)
        peak_day_fig.update_traces(marker_color=bar_color, fill="tozeroy",
                                   fillcolor=f"rgba({r},{g},{b},0.25)")

        st.subheader(f"Peak Day — {selected_load}")
        st.plotly_chart(peak_day_fig, use_container_width=True)
        st.caption(f"Daily Total on {date_label}: {peak_total:,.1f}")

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 5 — Benchmark (Benchmark Tab)
# =========================
with tab5:
    if uploaded_file:
        # Load benchmark data for the selected building use
        benchmark_df = load_benchmark_data(building_use)

        if benchmark_df is not None:
            # Calculate project KPIs (reuse calculations from other tabs)
            df = energy_balance_sheet(uploaded_file.getvalue())
            df_melted = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")
            df_melted["Energy_Source"] = df_melted["End_Use"].map(
                {k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted["End_Use"].unique()})

            # Energy calculations
            totals = df_melted.groupby("End_Use", as_index=False)["kWh"].sum()
            totals["kWh_per_m2"] = (totals["kWh"] / project_area).round(1)

            # Net and Gross EUI calculations
            eui_gross = totals.loc[totals["kWh_per_m2"] > 0, "kWh_per_m2"].sum()  # Consumption only (gross)
            eui_net = totals["kWh_per_m2"].sum()  # Including PV (net)

            # CO2 calculations
            factor_map = {
                "Electricity": co2_Emissions_Electricity,
                "Green Electricity": co2_Emissions_Green_Electricity,
                "Gas": co2_emissions_gas,
                "District Heating": co2_emissions_dh,
                "District Cooling": co2_emissions_dc,
            }
            df_co2 = df_melted.copy()
            df_co2["CO2_factor_kg_per_kWh"] = df_co2["Energy_Source"].map(factor_map).fillna(0.0)
            df_co2["kgCO2"] = df_co2["kWh"] * df_co2["CO2_factor_kg_per_kWh"]
            totals_co2 = df_co2.groupby("End_Use", as_index=False)["kgCO2"].sum()
            totals_co2["kgCO2_per_m2"] = (totals_co2["kgCO2"] / project_area).round(1)

            co2_intensity_gross = totals_co2.loc[totals_co2["kgCO2_per_m2"] > 0, "kgCO2_per_m2"].sum()  # Gross
            co2_intensity_net = totals_co2["kgCO2_per_m2"].sum()  # Net

            # Cost calculations
            cost_map = {
                "Electricity": cost_electricity,
                "Gas": cost_gas,
                "District Heating": cost_dh,
                "District Cooling": cost_dc,
                "Green Electricity": cost_green_electricity,
            }
            df_cost = df_melted.copy()
            df_cost["cost_per_kWh"] = df_cost["Energy_Source"].map(cost_map).fillna(0.0)
            df_cost["cost"] = df_cost["kWh"] * df_cost["cost_per_kWh"]
            totals_cost = df_cost.groupby("End_Use", as_index=False)["cost"].sum()
            totals_cost["cost_per_m2"] = (totals_cost["cost"] / project_area).round(2)

            cost_intensity_gross = totals_cost.loc[totals_cost["cost_per_m2"] > 0, "cost_per_m2"].sum()  # Gross
            cost_intensity_net = totals_cost["cost_per_m2"].sum()  # Net

            # Extract benchmark thresholds
            benchmark_dict = {}
            for _, row in benchmark_df.iterrows():
                kpi_name = row["KPI_Name"]
                benchmark_dict[kpi_name] = {
                    "Good_Threshold": float(row["Good_Threshold"]),
                    "Excellent_Threshold": float(row["Excellent_Threshold"])
                }

            # Project values (using net values for benchmarking)
            project_values = {
                "Energy_Density": eui_net,
                "CO2_Emissions": co2_intensity_net,
                "Energy_Cost": cost_intensity_net
            }

            # Project values (gross - without PV)
            project_values_gross = {
                "Energy_Density": eui_gross,
                "CO2_Emissions": co2_intensity_gross,
                "Energy_Cost": cost_intensity_gross
            }

            st.write(f"## Benchmark Analysis")


            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(f"Project Name:", project_name, help='User Input on Sidebar')
            with col2:
                st.metric(f"Building Use:", building_use, help='User Input on Sidebar')
            with col3:
                st.metric(f"Building Area:", f"{project_area:,.0f} m²", help='User Input on Sidebar')
            with col4:

                latitude_map = float(latitude)
                longitude_map = float(longitude)
                df = pd.DataFrame(
                    {
                        "col1": [latitude_map],
                        "col2": [longitude_map],
                        "label": [f"project_name"],
                    }
                )
                st.metric("Project Location:","", help='User Input on Sidebar')
                st.map(data=df, latitude="col1", longitude="col2", height=200, size=500, zoom=10)

            st.markdown("---")

            # Create gauge charts for each KPI
            st.subheader("Energy Density")

            # Create 3:1 column layout
            col1, col2 = st.columns([3, 1])

            with col1:

                if "Energy_Density" in benchmark_dict:
                    good_thresh = benchmark_dict["Energy_Density"]["Good_Threshold"]
                    excellent_thresh = benchmark_dict["Energy_Density"]["Excellent_Threshold"]
                    gauge_fig = create_gauge_chart(
                        project_values["Energy_Density"],
                        good_thresh,
                        excellent_thresh,
                        "",
                        "kWh/m²·a"
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)

            with col2:


                # Display textual results (Net values)
                st.write("**Gross Values (without PV):**")
                for kpi_name, value in project_values_gross.items():
                    if kpi_name in benchmark_dict:
                        good_thresh = benchmark_dict[kpi_name]["Good_Threshold"]
                        excellent_thresh = benchmark_dict[kpi_name]["Excellent_Threshold"]
                        category = get_benchmark_category(value, good_thresh, excellent_thresh)

                        if kpi_name == "Energy_Density":
                            st.metric(f"EUI Gross ({category})", f"{value:.1f} kWh/m²·a",
                                      help="Gross energy before accounting for on-site generation")


                st.write("**Net Values (with PV):**")
                for kpi_name, value in project_values.items():
                    if kpi_name in benchmark_dict:
                        good_thresh = benchmark_dict[kpi_name]["Good_Threshold"]
                        excellent_thresh = benchmark_dict[kpi_name]["Excellent_Threshold"]
                        category = get_benchmark_category(value, good_thresh, excellent_thresh)

                        if kpi_name == "Energy_Density":
                            st.metric(f"EUI ({category})", f"{value:.1f} kWh/m²·a",
                                      help="Net energy after accounting for on-site generation")

                            st.write("**WS Benchmark:**")

                            if category == "Excellent":
                                st.image("Pamo_Icon_Platin.png", width=100)
                                st.write("**Platin**")
                            if category == "Good":
                                st.image("Pamo_Icon_Green.png", width=100)
                                st.write("**Green**")
                            if category == "Poor":
                                st.image("Pamo_Icon_Gray.png", width=100)
                                st.write("*not Benchmarked")

            st.markdown("---")
            st.subheader("CO2 Emissions")

            col1, col2 = st.columns([3, 1])

            with col1:

                if "CO2_Emissions" in benchmark_dict:
                    good_thresh = benchmark_dict["CO2_Emissions"]["Good_Threshold"]
                    excellent_thresh = benchmark_dict["CO2_Emissions"]["Excellent_Threshold"]
                    gauge_fig = create_gauge_chart(
                        project_values["CO2_Emissions"],
                        good_thresh,
                        excellent_thresh,
                        "",
                        "kgCO₂/m²·a"
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)

            with col2:

                # Display textual results (Net values)
                st.write("**Gross Values (without PV):**")
                for kpi_name, value in project_values_gross.items():
                    if kpi_name in benchmark_dict:
                        good_thresh = benchmark_dict[kpi_name]["Good_Threshold"]
                        excellent_thresh = benchmark_dict[kpi_name]["Excellent_Threshold"]
                        category = get_benchmark_category(value, good_thresh, excellent_thresh)

                        if kpi_name == "CO2_Emissions":
                            st.metric(f"CO₂ Gross ({category})", f"{value:.1f} kgCO₂/m²·a"
                                      ,
                                      help="Gross emissions before accounting for on-site generation")


                st.write("**Net Values (with PV):**")
                for kpi_name, value in project_values.items():
                    if kpi_name in benchmark_dict:
                        good_thresh = benchmark_dict[kpi_name]["Good_Threshold"]
                        excellent_thresh = benchmark_dict[kpi_name]["Excellent_Threshold"]
                        category = get_benchmark_category(value, good_thresh, excellent_thresh)

                        if kpi_name == "CO2_Emissions":
                            st.metric(f"CO₂ Intensity ({category})", f"{value:.1f} kgCO₂/m²·a",
                                      help="Net emissions after accounting for on-site generation")

                            st.write("**WS Benchmark:**")

                            if category == "Excellent":
                                st.image("Pamo_Icon_Platin.png", width=100)
                                st.write("**Platin**")
                            if category == "Good":
                                st.image("Pamo_Icon_Green.png", width=100)
                                st.write("**Green**")
                            if category == "Poor":
                                st.image("Pamo_Icon_Gray.png", width=100)
                                st.write("*not Benchmarked")

            st.markdown("---")
            st.subheader("Energy Cost")

            col1, col2 = st.columns([3, 1])

            with col1:

                if "Energy_Cost" in benchmark_dict:
                    good_thresh = benchmark_dict["Energy_Cost"]["Good_Threshold"]
                    excellent_thresh = benchmark_dict["Energy_Cost"]["Excellent_Threshold"]
                    gauge_fig = create_gauge_chart(
                        project_values["Energy_Cost"],
                        good_thresh,
                        excellent_thresh,
                        "",
                        f"{currency_symbol}/m²·a"
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)


            with col2:

                # Display textual results (Net values)

                st.write("**Gross Values (without PV):**")
                for kpi_name, value in project_values_gross.items():
                    if kpi_name in benchmark_dict:
                        good_thresh = benchmark_dict[kpi_name]["Good_Threshold"]
                        excellent_thresh = benchmark_dict[kpi_name]["Excellent_Threshold"]
                        category = get_benchmark_category(value, good_thresh, excellent_thresh)


                        if kpi_name == "Energy_Cost":
                            st.metric(f"Cost Gross ({category})", f"{currency_symbol} {value:.2f}/m²·a",
                                      help="Gross energy cost before accounting for on-site generation")

                st.write("**Net Values (with PV):**")
                for kpi_name, value in project_values.items():
                    if kpi_name in benchmark_dict:
                        good_thresh = benchmark_dict[kpi_name]["Good_Threshold"]
                        excellent_thresh = benchmark_dict[kpi_name]["Excellent_Threshold"]
                        category = get_benchmark_category(value, good_thresh, excellent_thresh)

                        if kpi_name == "Energy_Cost":
                            st.metric(f"Cost Intensity ({category})", f"{currency_symbol} {value:.2f}/m²·a",
                                      help="Net energy cost after accounting for on-site generation")

                            st.write("**WS Benchmark:**")

                            if category == "Excellent":
                                st.image("Pamo_Icon_Platin.png", width=100)
                                st.write("**Platin**")
                            if category == "Good":
                                st.image("Pamo_Icon_Green.png", width=100)
                                st.write("**Green**")
                            if category == "Poor":
                                st.image("Pamo_Icon_Gray.png", width=100)
                                st.write("-not Benchmarked-")



        else:
            st.error(f"Benchmark data not found for building use: {building_use}")
            st.write("Please ensure the benchmark template file exists in the templates folder.")

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

