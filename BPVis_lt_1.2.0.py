import io
from pathlib import Path
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
from copy import deepcopy

# --- CRREM chart colors (synced across CRREM diagrams)
CRREM_COLOR_LIMIT = "#c02419"  # light red
CRREM_COLOR_BASELINE = "#5a73a5"  # light blue
CRREM_COLOR_MEASURES = "#a9c724"  # light green

# --- Scenario palette (used in Scenarios tab Net KPI bar charts)
SCENARIO_COLOR_PALETTE = ["#c02419", "#5a73a5", "#a9c724", "#42b360", "#833fd1", "#42b38d"]


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
        st.session_state[txt_key] = (fmt.format(default) if fmt else str(default)) if hasattr(fmt, "format") else (
                fmt or str(default))
    val = st.text_input(label, key=txt_key, help=help)
    v = _parse_float_locale(val, default)
    if (min_value is not None) and (v < min_value):
        v = min_value
    if (max_value is not None) and (v > max_value):
        v = max_value
    st.session_state[key] = v
    return v


import numpy as np
import plotly.colors as pcolors
from typing import Optional, Tuple, Dict

### Werner Sobek Green Technologies GmbH. All rights reserved.###
### Author: Rodrigo Carvalho ###


# =========================
# Page setup & constants
# =========================
st.set_page_config(
    page_title="WSGT_BPVis_ENE 1.3.0",
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
ENERGY_SOURCE_ORDER = ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling", "Biomass"]

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
    "Biomass": "#8b5a2b",
    # negative values will still be this color
}
color_map_sources = {
    "Electricity": "#42b360",
    "Green Electricity": "#64c423",
    "Gas": "#c9d302",
    "District Heating": "#ec6939",
    "District Cooling": "#5a5ea5",
    "Biomass": "#8b5a2b",
}

# --- NEW: ensure a default project name exists before rendering the title
if "project_name" not in st.session_state:
    st.session_state["project_name"] = "Building Performance Dashboard"

# =========================
# Sidebar — template download & file upload
# =========================
st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## BPVis ENE")
st.sidebar.write("Version 1.3.0")

st.sidebar.markdown("### Download Template")
template_path = Path("templates/energy_database_complete_template.xlsx")
if template_path.exists():
    with open(template_path, "rb") as file:
        st.sidebar.download_button(
            label="Download Excel Template",
            data=file.read(),
            file_name="../../Downloads/energy_database_complete_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.sidebar.markdown("---")
st.sidebar.write("### Upload Data")

# Upload Excel File (xlsx)
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type="xlsx")

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Information")


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
SHEET_EFFICIENCY = "Efficiency_Factors"
SHEET_SCENARIOS = "Scenarios"


def read_config_from_excel(file_bytes: bytes) -> Dict[str, Optional[pd.DataFrame]]:
    """Read known config sheets if present; return dict of dataframes (or None)."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}
    return {
        "project": sheets.get(SHEET_PROJECT),
        "factors": sheets.get(SHEET_FACTORS),
        "tariffs": sheets.get(SHEET_TARIFFS),
        "mapping": sheets.get(SHEET_MAPPING),
        "efficiency": sheets.get(SHEET_EFFICIENCY),
        "scenarios": sheets.get(SHEET_SCENARIOS),
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


def parse_efficiency_df(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    out = {}
    if df is not None and {"End_Use", "Efficiency_Factor"}.issubset(df.columns):
        for _, row in df.iterrows():
            eu = str(row["End_Use"])
            try:
                out[eu] = float(row["Efficiency_Factor"])
            except Exception:
                pass
    return out


# =========================
# NEW — Scenario Manager helpers
# =========================

def parse_scenarios_sheet(df: Optional[pd.DataFrame]) -> Tuple[Dict[str, dict], Optional[str]]:
    """Parse Scenarios sheet into dict[name] -> payload and return active scenario name if present."""
    scenarios: Dict[str, dict] = {}
    active_name: Optional[str] = None
    if df is None or df.empty:
        return scenarios, active_name
    if "Scenario" not in df.columns:
        return scenarios, active_name

    has_payload = "PayloadJSON" in df.columns
    has_active = "Active" in df.columns

    for _, row in df.iterrows():
        name = str(row.get("Scenario", "")).strip()
        if not name:
            continue
        payload = {}
        if has_payload:
            raw = row.get("PayloadJSON", "")
            try:
                if pd.notna(raw) and str(raw).strip():
                    payload = json.loads(str(raw))
            except Exception:
                payload = {}
        scenarios[name] = payload

        if has_active and active_name is None:
            try:
                # accept 1/0, True/False
                if bool(int(row.get("Active", 0))):
                    active_name = name
            except Exception:
                try:
                    if bool(row.get("Active", False)):
                        active_name = name
                except Exception:
                    pass

    if active_name is None and scenarios:
        active_name = list(scenarios.keys())[0]
    return scenarios, active_name


def build_scenarios_sheet(scenarios: Dict[str, dict], active_name: Optional[str]) -> pd.DataFrame:
    rows = []
    for name, payload in scenarios.items():
        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
        except Exception:
            payload_json = "{}"
        rows.append({
            "Scenario": name,
            "Active": 1 if (active_name is not None and name == active_name) else 0,
            "PayloadJSON": payload_json,
        })
    return pd.DataFrame(rows)


def _measures_df_to_records(df) -> list:
    """Convert a CRREM measures dataframe to JSON-serializable records."""
    if df is None:
        return []
    if isinstance(df, list):
        # already records
        return df
    try:
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return []
            cols = ["Parameter", "Year", "New Value"]
            for c in cols:
                if c not in df.columns:
                    df[c] = ""
            out = []
            for _, r in df[cols].iterrows():
                rec = {
                    "Parameter": "" if pd.isna(r["Parameter"]) else str(r["Parameter"]),
                    "Year": None if pd.isna(r["Year"]) or str(r["Year"]).strip() == "" else int(float(r["Year"])),
                    "New Value": "" if pd.isna(r["New Value"]) else str(r["New Value"]),
                }
                out.append(rec)
            return out
    except Exception:
        return []
    return []


def _measures_records_to_df(records) -> pd.DataFrame:
    """Convert saved records to a measures dataframe with stable columns."""
    try:
        if records is None:
            return pd.DataFrame(columns=["Parameter", "Year", "New Value"])
        if isinstance(records, pd.DataFrame):
            df = records.copy()
        else:
            df = pd.DataFrame(list(records))
        if df.empty:
            return pd.DataFrame(columns=["Parameter", "Year", "New Value"])
        for c in ["Parameter", "Year", "New Value"]:
            if c not in df.columns:
                df[c] = ""
        df = df[["Parameter", "Year", "New Value"]].copy()
        return df
    except Exception:
        return pd.DataFrame(columns=["Parameter", "Year", "New Value"])


def _mixed_use_df_to_records(df) -> list:
    """Convert a CRREM mixed-use dataframe to JSON-serializable records."""
    if df is None:
        return []
    if isinstance(df, list):
        return df
    try:
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return []
            cols = ["Use Type", "Area Share %"]
            for c in cols:
                if c not in df.columns:
                    return []
            out = []
            for _, row in df.iterrows():
                use = row.get("Use Type")
                share = row.get("Area Share %")
                if use is None or str(use).strip() == "":
                    continue
                try:
                    share_f = float(str(share).replace(",", ".")) if share is not None and str(
                        share).strip() != "" else 0.0
                except Exception:
                    share_f = 0.0
                out.append({"Use Type": str(use), "Area Share %": share_f})
            return out
    except Exception:
        return []
    return []


def _mixed_use_records_to_df(records) -> pd.DataFrame:
    """Convert saved mixed-use records to a dataframe with stable columns."""
    cols = ["Use Type", "Area Share %"]
    try:
        if records is None:
            return pd.DataFrame(columns=cols)
        if isinstance(records, pd.DataFrame):
            df = records.copy()
        elif isinstance(records, list):
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(columns=cols)

        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols].copy()

        # Coerce share to float
        def _to_f(x):
            try:
                return float(str(x).replace(",", ".")) if x is not None and str(x).strip() != "" else 0.0
            except Exception:
                return 0.0

        df["Area Share %"] = df["Area Share %"].apply(_to_f)
        df["Use Type"] = df["Use Type"].astype(str)

        # drop empty
        df = df[df["Use Type"].str.strip() != ""]
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=cols)


def default_scenario_payload(end_uses: list, preloaded_cfg: Optional[dict]) -> dict:
    """Backwards compatible defaults (single-config sheets -> Base scenario)."""
    def_f = (preloaded_cfg.get("factors") if preloaded_cfg else {}) or {}
    def_t = (preloaded_cfg.get("tariffs") if preloaded_cfg else {}) or {}
    saved_mapping = parse_mapping_df(preloaded_cfg.get("mapping_df")) if (
            preloaded_cfg and preloaded_cfg.get("mapping_df") is not None) else {}
    def_eff = (preloaded_cfg.get("efficiency") if preloaded_cfg else {}) or {}

    return {
        "factors": {
            "Electricity": float(def_f.get("Electricity", 0.300)),
            "Green Electricity": float(def_f.get("Green Electricity", 0.000)),
            "Gas": float(def_f.get("Gas", 0.180)),
            "District Heating": float(def_f.get("District Heating", 0.260)),
            "District Cooling": float(def_f.get("District Cooling", 0.280)),
            "Biomass": float(def_f.get("Biomass", 0.000)),
        },
        "tariffs": {
            "Electricity": float(def_t.get("Electricity", 0.35)),
            "Green Electricity": float(def_t.get("Green Electricity", 0.40)),
            "Gas": float(def_t.get("Gas", 0.12)),
            "District Heating": float(def_t.get("District Heating", 0.16)),
            "District Cooling": float(def_t.get("District Cooling", 0.16)),
            "Biomass": float(def_t.get("Biomass", 0.10)),
        },
        "mapping": {use: str(saved_mapping.get(use, "Electricity")) for use in end_uses},
        "efficiency": {use: float(def_eff.get(use, 1.0)) for use in end_uses},
        "pv": {"enabled": False, "scale": 1.0},
        "crrem_measures": [],
        "crrem_use_type": "Office",
        "crrem_mixed_use": [
            {"Use Type": "Office", "Area Share %": 50.0},
            {"Use Type": "Retail, High Street", "Area Share %": 50.0},
        ],
    }


def capture_scenario_from_widgets(end_uses: list) -> dict:
    """Capture current sidebar/widget values into a scenario payload."""
    payload = {
        "factors": {
            "Electricity": float(st.session_state.get("co2_Emissions_Electricity", 0.300)),
            "Green Electricity": float(st.session_state.get("co2_Emissions_Green_Electricity", 0.000)),
            "Gas": float(st.session_state.get("co2_emissions_gas", 0.180)),
            "District Heating": float(st.session_state.get("co2_emissions_dh", 0.260)),
            "District Cooling": float(st.session_state.get("co2_emissions_dc", 0.280)),
            "Biomass": float(st.session_state.get("co2_emissions_biomass", 0.000)),
        },
        "tariffs": {
            "Electricity": float(st.session_state.get("cost_electricity", 0.35)),
            "Green Electricity": float(st.session_state.get("cost_green_electricity", 0.40)),
            "Gas": float(st.session_state.get("cost_gas", 0.12)),
            "District Heating": float(st.session_state.get("cost_dh", 0.16)),
            "District Cooling": float(st.session_state.get("cost_dc", 0.16)),
            "Biomass": float(st.session_state.get("cost_biomass", 0.10)),
        },
        "mapping": {use: str(st.session_state.get(f"source_{use}", "Electricity")) for use in end_uses},
        "efficiency": {use: float(st.session_state.get(f"eff_{use}", 1.0)) for use in end_uses},
        "pv": {
            "enabled": bool(st.session_state.get("pv_sc_enabled", False)),
            "scale": float(st.session_state.get("pv_scale", 1.0)),
        },
        "crrem_measures": _measures_df_to_records(st.session_state.get("crrem_measures_df")),
        "crrem_use_type": str(st.session_state.get("crrem_use_type", "Office")),
        "crrem_mixed_use": _mixed_use_df_to_records(st.session_state.get("crrem_mixed_use_df")),
    }
    return payload


def load_scenario_into_widgets(payload: dict, end_uses: list) -> None:
    """Seed Streamlit widget state from a scenario payload.

    This must run before widgets are created (or be followed by st.rerun).
    """

    def _set_num(key: str, value: float, fmt: str):
        st.session_state[key] = float(value)
        st.session_state[f"{key}_txt"] = fmt.format(float(value))

    f = payload.get("factors", {})
    t = payload.get("tariffs", {})
    m = payload.get("mapping", {})
    e = payload.get("efficiency", {})
    pv = payload.get("pv", {})

    _set_num("co2_Emissions_Electricity", float(f.get("Electricity", 0.300)), "{:.3f}")
    _set_num("co2_Emissions_Green_Electricity", float(f.get("Green Electricity", 0.000)), "{:.3f}")
    _set_num("co2_emissions_dh", float(f.get("District Heating", 0.260)), "{:.3f}")
    _set_num("co2_emissions_dc", float(f.get("District Cooling", 0.280)), "{:.3f}")
    _set_num("co2_emissions_gas", float(f.get("Gas", 0.180)), "{:.3f}")
    _set_num("co2_emissions_biomass", float(f.get("Biomass", 0.000)), "{:.3f}")

    _set_num("cost_electricity", float(t.get("Electricity", 0.35)), "{:.2f}")
    _set_num("cost_green_electricity", float(t.get("Green Electricity", 0.40)), "{:.2f}")
    _set_num("cost_dh", float(t.get("District Heating", 0.16)), "{:.2f}")
    _set_num("cost_dc", float(t.get("District Cooling", 0.16)), "{:.2f}")
    _set_num("cost_gas", float(t.get("Gas", 0.12)), "{:.2f}")
    _set_num("cost_biomass", float(t.get("Biomass", 0.10)), "{:.2f}")

    for use in end_uses:
        st.session_state[f"source_{use}"] = str(m.get(use, "Electricity"))
        _set_num(f"eff_{use}", float(e.get(use, 1.0)), "{:.3f}")

    _set_num("pv_scale", float(pv.get("scale", 1.0)), "{:.3f}")
    st.session_state["pv_sc_enabled"] = bool(pv.get("enabled", False))

    # CRREM measures (scenario-specific)
    st.session_state["crrem_measures_df"] = _measures_records_to_df(payload.get("crrem_measures", []))

    # CRREM use settings (scenario-specific; defaults to Office if absent)
    try:
        st.session_state["crrem_use_type"] = str(payload.get("crrem_use_type", "Office") or "Office")
    except Exception:
        st.session_state["crrem_use_type"] = "Office"

    mixed_records = payload.get("crrem_mixed_use", None)
    mixed_df = _mixed_use_records_to_df(mixed_records)
    if mixed_df is None or mixed_df.empty:
        mixed_df = pd.DataFrame({
            "Use Type": ["Office", "Retail, High Street"],
            "Area Share %": [50.0, 50.0],
        })
    st.session_state["crrem_mixed_use_df"] = mixed_df


def build_efficiency_df(end_uses) -> pd.DataFrame:
    rows = []
    for use in end_uses:
        rows.append({"End_Use": use, "Efficiency_Factor": st.session_state.get(f"eff_{use}", 1.0)})
    return pd.DataFrame(rows)


def build_project_df(project_name: str, project_area: float, currency_symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"Key": ["Project_Name", "Project_Area", "Currency"],
         "Value": [project_name, project_area, currency_symbol]}
    )


def build_factors_df(
        co2_elec: float,
        co2_green: float,
        co2_dh: float,
        co2_dc: float,
        co2_gas: float,
        co2_biomass: float,
) -> pd.DataFrame:
    """Build Emission_Factors sheet."""
    # Keep a clear mapping independent of ENERGY_SOURCE_ORDER variable ordering
    return pd.DataFrame({
        "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling", "Biomass"],
        "Factor_kgCO2_per_kWh": [co2_elec, co2_green, co2_gas, co2_dh, co2_dc, co2_biomass],
    })


def build_tariffs_df(
        cost_elec: float,
        cost_green: float,
        cost_dh: float,
        cost_dc: float,
        cost_gas: float,
        cost_biomass: float,
) -> pd.DataFrame:
    """Build Energy_Tariffs sheet."""
    return pd.DataFrame({
        "Energy_Source": ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling", "Biomass"],
        "Tariff_per_kWh": [cost_elec, cost_green, cost_gas, cost_dh, cost_dc, cost_biomass],
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
) -> Tuple[
    Optional[str], Optional[float], Optional[str], Optional[str], Optional[float], Optional[float], Optional[int]]:
    """Parse Project_Data sheet (name, area, currency, building use, country, latitude, longitude, year).

    Backwards compatible:
      - accepts missing Year
      - accepts either 'Year' or 'Project_Year'
    """
    if df is None or not {"Key", "Value"}.issubset(df.columns):
        return None, None, None, None, None, None, None

    kv = dict(zip(df["Key"].astype(str), df["Value"]))

    name = kv.get("Project_Name")
    currency = kv.get("Currency")
    building_use = kv.get("Building_Use")
    country = kv.get("Country")

    def _to_float(x):
        try:
            if x is None or str(x).strip() == "":
                return None
            s = str(x).strip().replace(" ", "").replace(",", ".")
            return float(s)
        except Exception:
            return None

    def _to_int(x):
        try:
            if x is None or str(x).strip() == "":
                return None
            return int(float(str(x).replace(",", ".")))
        except Exception:
            return None

    area = _to_float(kv.get("Project_Area"))
    latitude_saved = _to_float(kv.get("Project_Latitude"))
    longitude_saved = _to_float(kv.get("Project_Longitude"))
    year_saved = _to_int(kv.get("Year"))
    if year_saved is None:
        year_saved = _to_int(kv.get("Project_Year"))

    return name, area, currency, building_use, country, latitude_saved, longitude_saved, year_saved


def build_project_df_with_building_use(
        project_name: str,
        project_area: float,
        currency_symbol: str,
        building_use: str,
        country: str,
        latitude: Optional[float],
        longitude: Optional[float],
        year: Optional[int],
) -> pd.DataFrame:
    """Build the Project_Data sheet including lat/long and year."""
    return pd.DataFrame(
        {
            "Key": [
                "Project_Name",
                "Project_Area",
                "Currency",
                "Building_Use",
                "Country",
                "Project_Latitude",
                "Project_Longitude",
                "Year",
            ],
            "Value": [
                project_name,
                project_area,
                currency_symbol,
                building_use,
                country,
                latitude,
                longitude,
                year,
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
                          mapping_df: pd.DataFrame,
                          efficiency_df: pd.DataFrame,
                          scenarios_df: Optional[pd.DataFrame] = None) -> bytes:
    """Return a new workbook (bytes) with all original sheets + updated config sheets."""
    cfg = read_config_from_excel(original_bytes)
    sheets = cfg["all_sheets"]  # dict[name] -> df

    # overwrite/create the config sheets
    sheets[SHEET_PROJECT] = project_df
    sheets[SHEET_FACTORS] = factors_df
    sheets[SHEET_TARIFFS] = tariffs_df
    sheets[SHEET_MAPPING] = mapping_df
    sheets[SHEET_EFFICIENCY] = efficiency_df
    if scenarios_df is not None:
        sheets[SHEET_SCENARIOS] = scenarios_df

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return buf.getvalue()


# =========================
# CRREM (Germany) — data loader & helpers
# =========================

# =========================
# CRREM (EU multi-country) — data loader & helpers
# =========================

CRREM_EU_EXTRACT_FILENAME = "CRREM_EU_Data_Extract_v2_07_1p5_2C.xlsx"
CRREM_DE_EXTRACT_FILENAME = "CRREM_DE_Data_Extract_v2_07_1p5_2C.xlsx"

APP_DIR = Path(__file__).resolve().parent

CRREM_DATA_CANDIDATES = [
    # Prefer paths relative to this script (robust for Streamlit deployments)
    APP_DIR / CRREM_EU_EXTRACT_FILENAME,
    APP_DIR / "templates" / CRREM_EU_EXTRACT_FILENAME,
    APP_DIR / "data" / CRREM_EU_EXTRACT_FILENAME,
    APP_DIR / CRREM_DE_EXTRACT_FILENAME,
    APP_DIR / "templates" / CRREM_DE_EXTRACT_FILENAME,
    APP_DIR / "data" / CRREM_DE_EXTRACT_FILENAME,

    # Fallback to current working directory (legacy behavior)
    Path(CRREM_EU_EXTRACT_FILENAME),
    Path("templates") / CRREM_EU_EXTRACT_FILENAME,
    Path("data") / CRREM_EU_EXTRACT_FILENAME,
    Path(CRREM_DE_EXTRACT_FILENAME),
    Path("templates") / CRREM_DE_EXTRACT_FILENAME,
    Path("data") / CRREM_DE_EXTRACT_FILENAME,
]


@st.cache_data(show_spinner=False)
def load_crrem_meta() -> Optional[dict]:
    """Load CRREM extract workbook metadata (countries + property types).

    Supports:
      - EU multi-country extract (with a 'COUNTRIES' sheet)
      - legacy DE-only extract (no 'COUNTRIES' sheet; assumes Germany)

    Returns None if no dataset is found.
    """
    path = None
    for p in CRREM_DATA_CANDIDATES:
        try:
            if p.exists():
                path = p
                break
        except Exception:
            continue

    if path is None:
        return None

    try:
        xls = pd.ExcelFile(path)
        sheet_names = set(xls.sheet_names)

        property_types = pd.read_excel(xls, sheet_name="PROPERTY_TYPES")

        if "COUNTRIES" in sheet_names:
            countries = pd.read_excel(xls, sheet_name="COUNTRIES")
            # normalize
            if not {"country_name", "country_code"}.issubset(set(countries.columns)):
                # fallback if template differs
                countries = countries.rename(
                    columns={countries.columns[0]: "country_name", countries.columns[1]: "country_code"})
            countries["country_name"] = countries["country_name"].astype(str).str.strip()
            countries["country_code"] = countries["country_code"].astype(str).str.strip()
            is_eu = True
        else:
            countries = pd.DataFrame([{"country_name": "Germany", "country_code": "DE"}])
            is_eu = False

    except Exception:
        return None

    return {
        "path": str(path),
        "is_eu": is_eu,
        "countries": countries,
        "property_types": property_types,
    }


def get_crrem_country_options() -> list:
    """Return list of country *names* available in the CRREM extract.

    Always includes 'Germany' as a safe default.
    """
    meta = load_crrem_meta()
    if meta is None:
        return ["Germany"]
    countries = meta.get("countries")
    if countries is None or countries.empty:
        return ["Germany"]
    opts = sorted([c for c in countries["country_name"].dropna().astype(str).unique().tolist() if c.strip()])
    if "Germany" not in opts:
        opts = ["Germany"] + opts
    return opts


@st.cache_data(show_spinner=False)
def load_crrem_dataset(country_name: str) -> Optional[dict]:
    """Load CRREM pathways and grid electricity EF series for the given country name."""
    meta = load_crrem_meta()
    if meta is None:
        return None

    path = Path(meta["path"])
    is_eu = bool(meta.get("is_eu"))
    countries = meta.get("countries", pd.DataFrame())
    property_types = meta.get("property_types", pd.DataFrame()).copy()

    # Resolve country code (ISO2). Default Germany.
    resolved_country_name = "Germany"
    country_code = "DE"

    if is_eu and (countries is not None) and (not countries.empty):
        cn = str(country_name).strip() if country_name else "Germany"
        hit = countries.loc[countries["country_name"].astype(str).str.strip() == cn]
        if hit.empty:
            hit = countries.loc[countries["country_name"].astype(str).str.strip() == "Germany"]
        if not hit.empty:
            resolved_country_name = str(hit.iloc[0]["country_name"]).strip()
            country_code = str(hit.iloc[0]["country_code"]).strip().upper()
        else:
            resolved_country_name = "Germany"
            country_code = "DE"

    # Load country-specific sheets (EU) or DE-only sheets (legacy)
    code = country_code if is_eu else "DE"
    try:
        xls = pd.ExcelFile(path)
        pathways_carbon = pd.read_excel(xls, sheet_name=f"PATHWAYS_CARBON_{code}")
        pathways_eui = pd.read_excel(xls, sheet_name=f"PATHWAYS_EUI_{code}")
        ef = pd.read_excel(xls, sheet_name=f"EMISSION_FACTORS_{code}")
    except Exception:
        return None

    ef_grid = (
        ef.loc[ef["energy_carrier"].astype(str) == "grid_electricity", ["year", "kgco2e_per_kwh"]]
        .dropna()
        .astype({"year": int, "kgco2e_per_kwh": float})
        .set_index("year")["kgco2e_per_kwh"]
        .sort_index()
    )
    if ef_grid.empty:
        return None

    return {
        "path": str(path),
        "is_eu": is_eu,
        "country_name": resolved_country_name,
        "country_code": code,
        "property_types": property_types,
        "pathways_carbon": pathways_carbon,
        "pathways_eui": pathways_eui,
        "ef_grid": ef_grid,
    }


def _clamp_year_to_series(year: int, s: pd.Series) -> int:
    y_min, y_max = int(s.index.min()), int(s.index.max())
    return int(max(y_min, min(y_max, int(year))))


def compute_decarb_multiplier(ef_grid: pd.Series, base_year: int, years: list, ref_year: int = 2020) -> pd.Series:
    """
    Return multiplier m(y) = DF(y) / DF(base_year), where DF(y) = EF_grid(y) / EF_grid(ref_year).
    This matches CRREM-style decarbonization factors and ensures m(base_year) = 1.
    """
    ref_year_c = _clamp_year_to_series(ref_year, ef_grid)
    base_year_c = _clamp_year_to_series(base_year, ef_grid)

    ef_ref = float(ef_grid.loc[ref_year_c])
    if ef_ref == 0:
        return pd.Series({int(y): 1.0 for y in years})

    df_series = ef_grid.astype(float) / ef_ref
    df_base = float(df_series.loc[base_year_c])
    if df_base == 0:
        return pd.Series({int(y): 1.0 for y in years})

    idx_years = [int(y) for y in years]
    out = df_series.reindex(idx_years).astype(float)
    if out.isna().any():
        out = out.interpolate(method="linear", limit_direction="both")
    return out / df_base


def find_stranding_year(asset: pd.Series, limit: pd.Series) -> Optional[int]:
    """First year where asset exceeds limit (strictly >)."""
    df = pd.DataFrame({"asset": asset, "limit": limit}).dropna()
    if df.empty:
        return None
    over = df["asset"] > df["limit"]
    if not over.any():
        return None
    return int(df.index[over].min())


# =========================
# Preload any saved configuration (if an Excel is uploaded)
# =========================
preloaded = None
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    cfg_saved = read_config_from_excel(file_bytes)

    saved_name, saved_area, saved_currency, saved_building_use, saved_country, saved_lat, saved_lon, saved_year = \
        parse_project_df_with_building_use(cfg_saved["project"])

    saved_factors = parse_factors_df(cfg_saved["factors"])
    saved_tariffs = parse_tariffs_df(cfg_saved["tariffs"])
    saved_mapping_df = cfg_saved["mapping"]
    saved_efficiency = parse_efficiency_df(cfg_saved.get("efficiency"))
    has_any_saved = any([
        saved_name, saved_area, saved_currency, saved_building_use, saved_country, bool(saved_factors),
        bool(saved_tariffs),
        bool(saved_efficiency), saved_mapping_df is not None
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
        "country": saved_country,
        "lat": saved_lat,
        "lon": saved_lon,
        "year": saved_year,
        "factors": saved_factors,
        "tariffs": saved_tariffs,
        "mapping_df": saved_mapping_df,
        "efficiency": saved_efficiency,
        "scenarios_df": cfg_saved.get("scenarios"),
        "file_bytes": file_bytes,
    }

    # --- Seed Project Data from file on each new upload (token-based)
    #     This keeps Project Data global (not scenario-dependent) and ensures it reloads correctly from the workbook.
    wb_token = f"{uploaded_file.name}|{hashlib.md5(file_bytes).hexdigest()}"
    if st.session_state.get("_loaded_workbook_token") != wb_token:
        if preloaded.get("name"):
            st.session_state["project_name"] = str(preloaded["name"])

        if preloaded.get("area") is not None:
            try:
                st.session_state["project_area"] = float(preloaded["area"])
                st.session_state["project_area_txt"] = str(float(preloaded["area"]))
            except Exception:
                pass

        if preloaded.get("lat") is not None:
            try:
                st.session_state["project_latitude"] = float(preloaded["lat"])
                st.session_state["project_latitude_txt"] = f"{float(preloaded['lat']):.6f}"
            except Exception:
                pass

        if preloaded.get("lon") is not None:
            try:
                st.session_state["project_longitude"] = float(preloaded["lon"])
                st.session_state["project_longitude_txt"] = f"{float(preloaded['lon']):.6f}"
            except Exception:
                pass

        if preloaded.get("year") is not None:
            try:
                st.session_state["project_year"] = int(float(preloaded["year"]))
                st.session_state["project_year_txt"] = str(int(float(preloaded["year"])))
            except Exception:
                pass

        if preloaded.get("year") is not None:
            try:
                st.session_state["project_year"] = int(float(preloaded["year"]))
                st.session_state["project_year_txt"] = str(int(float(preloaded["year"])))
            except Exception:
                pass

        if preloaded.get("building_use"):
            st.session_state["building_use"] = str(preloaded["building_use"])

        if preloaded.get("country"):
            st.session_state["project_country"] = str(preloaded["country"])

        if preloaded.get("currency") in ["€", "$", "£"]:
            st.session_state["currency_symbol"] = str(preloaded["currency"])

        st.session_state["_loaded_workbook_token"] = wb_token

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
tab1, tab1_factors, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Energy Balance", "Energy Balance with Factors", "CO2 Emissions", "Energy Cost", "Loads Analysis", "Benchmark",
     "CRREM-Analysis", "Scenarios"])

# =========================
# Tab 1 — Energy Balance (Energy Balance Tab)
# =========================
with tab1:
    if uploaded_file:
        # ---- Load data
        df = energy_balance_sheet(uploaded_file.getvalue())

        # ---- Wide->Long transform for plotting and grouping
        df_melted = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

        # ---- Scenario Manager initialization (backwards compatible)
        end_uses = df_melted["End_Use"].unique().tolist()

        if "scenarios" not in st.session_state:
            scenarios_from_file, active_from_file = parse_scenarios_sheet(
                preloaded.get("scenarios_df") if preloaded else None
            )
            if scenarios_from_file:
                st.session_state["scenarios"] = scenarios_from_file
                st.session_state["active_scenario"] = active_from_file or list(scenarios_from_file.keys())[0]
            else:
                st.session_state["scenarios"] = {"Base": default_scenario_payload(end_uses, preloaded)}
                st.session_state["active_scenario"] = "Base"
            st.session_state["_prev_active_scenario"] = st.session_state["active_scenario"]
            load_scenario_into_widgets(st.session_state["scenarios"][st.session_state["active_scenario"]], end_uses)

        # ---- Sidebar: scenario manager UI
        with st.sidebar.expander("Scenario Manager", expanded=True):
            scenarios = st.session_state.get("scenarios", {})
            scenario_names = list(scenarios.keys()) if scenarios else ["Base"]

            # active scenario selector
            active_idx = scenario_names.index(
                st.session_state.get("active_scenario", scenario_names[0])) if st.session_state.get(
                "active_scenario") in scenario_names else 0
            active_selected = st.selectbox("Active Scenario", scenario_names, index=active_idx, key="active_scenario")

            prev = st.session_state.get("_prev_active_scenario")
            if prev is None:
                st.session_state["_prev_active_scenario"] = active_selected
                prev = active_selected

            if active_selected != prev:
                # persist current widgets into previous scenario, then load new scenario into widgets
                if prev in scenarios:
                    scenarios[prev] = capture_scenario_from_widgets(end_uses)
                if active_selected in scenarios:
                    load_scenario_into_widgets(scenarios[active_selected], end_uses)
                st.session_state["_prev_active_scenario"] = active_selected
                st.session_state["scenarios"] = scenarios
                st.rerun()

            # Scenario actions (stacked vertically for clarity)
            if st.button("New", use_container_width=True, key="scenario_btn_new"):
                # Save current first
                scenarios[active_selected] = capture_scenario_from_widgets(end_uses)
                base_name = "Scenario"
                n = 1
                new_name = f"{base_name} {n}"
                while new_name in scenarios:
                    n += 1
                    new_name = f"{base_name} {n}"
                scenarios[new_name] = default_scenario_payload(end_uses, preloaded)
                st.session_state["scenarios"] = scenarios
                st.session_state["active_scenario"] = new_name
                st.session_state["_prev_active_scenario"] = new_name
                load_scenario_into_widgets(scenarios[new_name], end_uses)
                st.rerun()

            if st.button("Duplicate", use_container_width=True, key="scenario_btn_duplicate"):
                scenarios[active_selected] = capture_scenario_from_widgets(end_uses)
                base_name = f"{active_selected} Copy"
                new_name = base_name
                i = 2
                while new_name in scenarios:
                    new_name = f"{base_name} {i}"
                    i += 1
                scenarios[new_name] = deepcopy(scenarios[active_selected])
                st.session_state["scenarios"] = scenarios
                st.session_state["active_scenario"] = new_name
                st.session_state["_prev_active_scenario"] = new_name
                load_scenario_into_widgets(scenarios[new_name], end_uses)
                st.rerun()

            rename_to = st.text_input("Rename to", value="", key="scenario_rename_to")
            if st.button("Rename", use_container_width=True, key="scenario_btn_rename"):
                rename_to_clean = str(rename_to).strip()
                if rename_to_clean and rename_to_clean not in scenarios:
                    scenarios[active_selected] = capture_scenario_from_widgets(end_uses)
                    scenarios[rename_to_clean] = scenarios.pop(active_selected)
                    st.session_state["scenarios"] = scenarios
                    st.session_state["active_scenario"] = rename_to_clean
                    st.session_state["_prev_active_scenario"] = rename_to_clean
                    load_scenario_into_widgets(scenarios[rename_to_clean], end_uses)
                    st.rerun()

            if st.button("Delete", use_container_width=True, key="scenario_btn_delete"):
                if len(scenarios) > 1 and active_selected in scenarios:
                    scenarios[active_selected] = capture_scenario_from_widgets(end_uses)
                    scenarios.pop(active_selected, None)
                    new_active = list(scenarios.keys())[0]
                    st.session_state["scenarios"] = scenarios
                    st.session_state["active_scenario"] = new_active
                    st.session_state["_prev_active_scenario"] = new_active
                    load_scenario_into_widgets(scenarios[new_active], end_uses)
                    st.rerun()
            st.caption("Scenarios store CO₂ factors, tariffs, source mapping, efficiency factors and PV settings.")

        # ---- Sidebar: project info (prefill from saved if available)
        with st.sidebar.expander("Project Data"):
            st.write("Enter Project's Basic Informations")

            # Prefer current session values (so Project Data stays global across scenarios)
            default_name = st.session_state.get("project_name")
            if not default_name:
                default_name = preloaded["name"] if (preloaded and preloaded["name"]) else "Example Building 1"

            default_area = st.session_state.get("project_area")
            if default_area is None:
                default_area = preloaded["area"] if (preloaded and preloaded["area"] is not None) else 1000.00

            default_building_use = st.session_state.get("building_use")
            if not default_building_use:
                default_building_use = preloaded["building_use"] if (
                        preloaded and preloaded["building_use"]) else "Office"

            # Defaults for lat/lon (fallback to previous hard-coded values)
            default_lat = st.session_state.get("project_latitude")
            if default_lat is None:
                default_lat = preloaded["lat"] if (preloaded and preloaded["lat"] is not None) else 53.54955

            default_lon = st.session_state.get("project_longitude")
            if default_lon is None:
                default_lon = preloaded["lon"] if (preloaded and preloaded["lon"] is not None) else 9.9936

            # keep title reactive via session_state
            project_name = st.text_input("Project Name", value=str(default_name), key="project_name")
            project_area = numeric_input("Project Area", float(default_area), key="project_area", min_value=0.0)

            default_year = st.session_state.get("project_year")
            if default_year is None:
                default_year = preloaded.get("year") if (preloaded and preloaded.get("year") is not None) else 2025

            # Year must be an integer. Use number_input (not the custom text-based numeric_input)
            # to avoid modifying a widget-bound *_txt key after instantiation.
            project_year = st.number_input(
                "Year",
                value=int(default_year),
                min_value=2020,
                max_value=2050,
                step=1,
                format="%d",
                key="project_year",
            )

            # Country (CRREM-aligned). Stored as full name. Default: Germany.
            country_options = get_crrem_country_options()
            default_country = st.session_state.get("project_country")
            if not default_country:
                default_country = preloaded.get("country") if (preloaded and preloaded.get("country")) else "Germany"
            if (not country_options) or (default_country not in country_options):
                default_country = "Germany" if (country_options and "Germany" in country_options) else (
                    country_options[0] if country_options else "Germany"
                )

            st.selectbox(
                "Country",
                options=country_options if country_options else ["Germany"],
                index=(country_options.index(default_country) if (
                            country_options and default_country in country_options) else 0),
                key="project_country",
            )

            latitude = numeric_input(
                "Project Latitude",
                float(default_lat),
                key="project_latitude",
                min_value=-90.0,
                max_value=90.0,
                fmt="{:.6f}",
            )
            longitude = numeric_input(
                "Project Longitude",
                float(default_lon),
                key="project_longitude",
                min_value=-180.0,
                max_value=180.0,
                fmt="{:.6f}",
            )

            # building use dropdown unchanged...
            building_use_options = ["Office", "Hospitality", "Retail", "Residential", "Industrial", "Education",
                                    "Leisure", "Healthcare"]
            building_use_index = building_use_options.index(
                default_building_use) if default_building_use in building_use_options else 0
            building_use = st.selectbox("Building Use", building_use_options, index=building_use_index,
                                        key="building_use")

        # ---- Sidebar: emission factors (used in Tab 2, but defined once)
        with st.sidebar.expander("Emission Factors"):
            st.write("Assign Emission Factors")
            def_f = preloaded["factors"] if preloaded else {}
            co2_Emissions_Electricity = numeric_input("CO2 Factor Electricity", float(def_f.get("Electricity", 0.300)),
                                                      key="co2_Emissions_Electricity", min_value=0.0, max_value=1.0,
                                                      fmt="{:.3f}")
            co2_Emissions_Green_Electricity = numeric_input("CO2 Factor Green Electricity",
                                                            float(def_f.get("Green Electricity", 0.000)),
                                                            key="co2_Emissions_Green_Electricity", min_value=0.0,
                                                            max_value=1.0, fmt="{:.3f}")
            co2_emissions_dh = numeric_input("CO2 Factor District Heating", float(def_f.get("District Heating", 0.260)),
                                             key="co2_emissions_dh", min_value=0.0, max_value=1.0, fmt="{:.3f}")
            co2_emissions_dc = numeric_input("CO2 Factor District Cooling", float(def_f.get("District Cooling", 0.280)),
                                             key="co2_emissions_dc", min_value=0.0, max_value=1.0, fmt="{:.3f}")
            co2_emissions_gas = numeric_input("CO2 Factor Gas", float(def_f.get("Gas", 0.180)), key="co2_emissions_gas",
                                              min_value=0.0, max_value=1.0, fmt="{:.3f}")
            co2_emissions_biomass = numeric_input("CO2 Factor Biomass", float(def_f.get("Biomass", 0.000)),
                                                  key="co2_emissions_biomass",
                                                  min_value=0.0, max_value=5.0, fmt="{:.3f}")

        # --- Energy Cost (€/kWh) ---
        with st.sidebar.expander("Energy Tariffs"):
            st.write("Assign energy cost per source (per kWh)")
            default_currency = preloaded["currency"] if (
                    preloaded and preloaded["currency"] in ["€", "$", "£"]) else "€"
            currency_symbol = st.selectbox("Currency", ["€", "$", "£"], index=["€", "$", "£"].index(default_currency),
                                           key="currency_symbol")

            def_t = preloaded["tariffs"] if preloaded else {}
            cost_electricity = numeric_input(f"Cost Electricity ({currency_symbol}/kWh)",
                                             float(def_t.get("Electricity", 0.35)), key="cost_electricity",
                                             min_value=0.0, max_value=100.0, fmt="{:.2f}")
            cost_green_electricity = numeric_input(f"Cost Green Electricity ({currency_symbol}/kWh)",
                                                   float(def_t.get("Green Electricity", 0.40)),
                                                   key="cost_green_electricity", min_value=0.0, max_value=100.0,
                                                   fmt="{:.2f}")
            cost_dh = numeric_input(f"Cost District Heating ({currency_symbol}/kWh)",
                                    float(def_t.get("District Heating", 0.16)), key="cost_dh", min_value=0.0,
                                    max_value=100.0, fmt="{:.2f}")
            cost_dc = numeric_input(f"Cost District Cooling ({currency_symbol}/kWh)",
                                    float(def_t.get("District Cooling", 0.16)), key="cost_dc", min_value=0.0,
                                    max_value=100.0, fmt="{:.2f}")
            cost_gas = numeric_input(f"Cost Gas ({currency_symbol}/kWh)", float(def_t.get("Gas", 0.12)), key="cost_gas",
                                     min_value=0.0, max_value=100.0, fmt="{:.2f}")
            cost_biomass = numeric_input(f"Cost Biomass ({currency_symbol}/kWh)", float(def_t.get("Biomass", 0.10)),
                                         key="cost_biomass",
                                         min_value=0.0, max_value=100.0, fmt="{:.2f}")

        # ---- Sidebar: efficiency factors per End_Use (used in 'Energy Balance with Factors' tab)
        with st.sidebar.expander("Efficiency Factors"):
            st.write("Assign efficiency factors per End Use (dimensionless; kWh is divided by factor)")
            def_eff = preloaded["efficiency"] if (preloaded and preloaded.get("efficiency")) else {}
            for use in df_melted["End_Use"].unique().tolist():
                numeric_input(
                    f"Efficiency Factor {use}",
                    float(def_eff.get(use, 1.0)),
                    key=f"eff_{use}",
                    min_value=0.001,
                    max_value=1000.0,
                    fmt="{:.3f}",
                )

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

        # ---- Persist current widget values back into the active scenario (for switching/comparison/save)
        if "scenarios" in st.session_state and st.session_state.get("active_scenario") in st.session_state["scenarios"]:
            st.session_state["scenarios"][st.session_state["active_scenario"]] = capture_scenario_from_widgets(end_uses)

        # ---- Save Project button (exports current inputs into the workbook)
        with st.sidebar:
            if st.button("Save Project", use_container_width=True):
                # Ensure the active scenario payload is up-to-date (including CRREM measures)
                try:
                    if "scenarios" in st.session_state and st.session_state.get("active_scenario") in st.session_state[
                        "scenarios"]:
                        st.session_state["scenarios"][
                            st.session_state["active_scenario"]] = capture_scenario_from_widgets(end_uses)
                except Exception:
                    pass


                # coerce UI strings to floats when possible
                def _to_float_safe(s):
                    try:
                        return float(str(s).replace(",", "."))
                    except Exception:
                        return None


                lat_val = _to_float_safe(latitude)
                lon_val = _to_float_safe(longitude)

                project_df = build_project_df_with_building_use(
                    st.session_state.get("project_name", project_name),
                    float(st.session_state.get("project_area", project_area) or 0.0),
                    currency_symbol,
                    building_use,
                    st.session_state.get("project_country", "Germany"),
                    lat_val,
                    lon_val,
                    int(st.session_state.get("project_year", 2025)),
                )

                factors_df = build_factors_df(
                    co2_Emissions_Electricity,
                    co2_Emissions_Green_Electricity,
                    co2_emissions_dh,
                    co2_emissions_dc,
                    co2_emissions_gas,
                    co2_emissions_biomass,
                )
                tariffs_df = build_tariffs_df(
                    cost_electricity,
                    cost_green_electricity,
                    cost_dh,
                    cost_dc,
                    cost_gas,
                    cost_biomass,
                )
                mapping_df = build_mapping_df(end_uses)
                efficiency_df = build_efficiency_df(end_uses)

                # Scenarios sheet (stores all scenarios; active scenario marked)
                scenarios_df = None
                if "scenarios" in st.session_state:
                    scenarios_df = build_scenarios_sheet(
                        st.session_state.get("scenarios", {}),
                        st.session_state.get("active_scenario")
                    )

                updated_bytes = write_config_to_excel(preloaded["file_bytes"], project_df, factors_df, tariffs_df,
                                                      mapping_df, efficiency_df, scenarios_df=scenarios_df)

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

        totals_clean = totals[
            (totals["End_Use"] != "PV_Generation")]

        # ---- Donuts (EUI shares)
        energy_intensity_chart = px.pie(
            totals_clean,
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
            st.subheader("Monthly Energy")
            st.plotly_chart(monthly_chart, use_container_width=True)
        with col2:
            st.subheader("Annual Energy")
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
# Tab 6 — Scenarios (Scenario Manager comparison)
# =========================

# =========================
# Tab 6 — CRREM-Analysis
# =========================
with tab6:
    if uploaded_file:
        st.write(f"## CRREM-Analysis ({st.session_state.get('project_country', 'Germany')})")

        crrem = load_crrem_dataset(st.session_state.get("project_country", "Germany"))
        if crrem is None:
            st.warning(
                "CRREM dataset not found. Place 'CRREM_EU_Data_Extract_v2_07_1p5_2C.xlsx' (preferred) or 'CRREM_DE_Data_Extract_v2_07_1p5_2C.xlsx' in the app root, 'templates/' or 'data/' folder."
            )
        else:
            # --- Controls
            target_label = st.selectbox(
                "Target (temperature pathway)",
                ["1.5°C", "2°C"],
                index=0,
                key="crrem_target_select",
            )
            target_id = "1.5C" if target_label.startswith("1.5") else "2C"

            pt_df = crrem["property_types"].copy()
            use_options = pt_df["app_use"].dropna().astype(str).tolist()
            # keep Mixed Use last (if present)
            if "Mixed Use" in use_options:
                use_options = [u for u in use_options if u != "Mixed Use"] + ["Mixed Use"]
            # Default CRREM use: Office if not available / invalid (backwards compatible)
            if "crrem_use_type" not in st.session_state or st.session_state.get("crrem_use_type") not in use_options:
                st.session_state["crrem_use_type"] = "Office" if "Office" in use_options else (
                    use_options[0] if use_options else "Office")

            crrem_use = st.selectbox(
                "CRREM Use Type",
                use_options,
                index=use_options.index(st.session_state["crrem_use_type"]) if st.session_state[
                                                                                   "crrem_use_type"] in use_options else 0,
                key="crrem_use_type",
            )

            mixed_components = None
            if crrem_use == "Mixed Use":
                st.caption("Define area shares per use-type (must sum to 100%).")
                if "crrem_mixed_use_df" not in st.session_state:
                    st.session_state["crrem_mixed_use_df"] = pd.DataFrame({
                        "Use Type": ["Office", "Retail, High Street"],
                        "Area Share %": [50.0, 50.0],
                    })
                editor_kwargs = {
                    "num_rows": "dynamic",
                    "use_container_width": True,
                    "key": "crrem_mixed_use_editor",
                }
                if hasattr(st, "column_config"):
                    editor_kwargs["column_config"] = {
                        "Use Type": st.column_config.SelectboxColumn(
                            "Use Type",
                            options=[u for u in use_options if u != "Mixed Use"],
                            required=True,
                        ),
                        "Area Share %": st.column_config.NumberColumn(
                            "Area Share %",
                            min_value=0.0,
                            max_value=100.0,
                            step=1.0,
                            format="%.1f",
                        ),
                    }

                mixed_df = st.data_editor(
                    st.session_state["crrem_mixed_use_df"],
                    **editor_kwargs,
                )
                st.session_state["crrem_mixed_use_df"] = mixed_df

                total_share = float(mixed_df["Area Share %"].fillna(0.0).sum()) if not mixed_df.empty else 0.0
                if abs(total_share - 100.0) > 0.5:
                    st.warning(f"Mixed use shares sum to {total_share:.1f}%. Adjust to 100% for CRREM blending.")
                # build components list (exclude empty/zero)
                mixed_components = [
                    (str(r["Use Type"]), float(r["Area Share %"]))
                    for _, r in mixed_df.iterrows()
                    if str(r.get("Use Type", "")).strip() and float(r.get("Area Share %", 0.0) or 0.0) > 0.0
                ]

            # --- Project and scenario inputs
            project_area_val = float(st.session_state.get("project_area", 0.0) or 0.0)
            if project_area_val <= 0:
                st.error("Project Area must be greater than 0 to run CRREM analysis.")
            else:
                project_year_val = int(st.session_state.get("project_year", 2025))
                # Use annual energy from the uploaded Energy_Balance sheet, adjusted by the active scenario:
                df_crrem = energy_balance_sheet(uploaded_file.getvalue())
                df_crrem_m = df_crrem.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

                # Apply efficiency factors (scenario-specific)
                eff_map_crrem = {use: st.session_state.get(f"eff_{use}", 1.0) for use in df_crrem_m["End_Use"].unique()}
                df_crrem_m["Efficiency_Factor"] = df_crrem_m["End_Use"].map(eff_map_crrem).fillna(1.0)
                df_crrem_m["kWh_adj"] = df_crrem_m["kWh"] / df_crrem_m["Efficiency_Factor"]

                # Apply PV scaling (scenario-specific). For CRREM carbon, PV always offsets Electricity (EF=0).
                pv_apply_scale = bool(st.session_state.get("pv_sc_enabled", False))
                pv_scale = float(st.session_state.get("pv_scale", 1.0))
                pv_mask = df_crrem_m["End_Use"].astype(str) == "PV_Generation"
                if pv_mask.any():
                    scale = pv_scale if pv_apply_scale else 1.0
                    df_crrem_m.loc[pv_mask, "kWh_adj"] = df_crrem_m.loc[pv_mask, "kWh_adj"] * scale
                    # Enforce PV as an electricity offset (negative)
                    df_crrem_m.loc[pv_mask, "kWh_adj"] = -df_crrem_m.loc[pv_mask, "kWh_adj"].abs()

                # Energy source mapping (scenario-specific)
                src_map_crrem = {u: st.session_state.get(f"source_{u}", "Electricity") for u in
                                 df_crrem_m["End_Use"].unique()}
                df_crrem_m["Energy_Source"] = df_crrem_m["End_Use"].map(src_map_crrem).fillna("Electricity")
                # normalize unknown sources
                df_crrem_m.loc[~df_crrem_m["Energy_Source"].isin(ENERGY_SOURCE_ORDER), "Energy_Source"] = "Electricity"
                # PV always offsets Electricity
                df_crrem_m.loc[pv_mask, "Energy_Source"] = "Electricity"

                # Annual kWh per source (net, PV included as negative electricity)
                annual_kwh_by_source = df_crrem_m.groupby("Energy_Source", as_index=True)["kWh_adj"].sum()

                # Clamp net electricity to >= 0 (PV offsets electricity up to demand; no export credit)
                if "Electricity" in annual_kwh_by_source.index:
                    annual_kwh_by_source.loc["Electricity"] = max(float(annual_kwh_by_source.loc["Electricity"]), 0.0)

                # CRREM EUI is consumption-only (exclude PV_Generation)
                annual_consumption_kwh = df_crrem_m.loc[~pv_mask, "kWh_adj"].sum()
                eui_asset = float(annual_consumption_kwh) / project_area_val

                # Base (user) emission factors at project_year
                base_factors = {
                    "Electricity": float(st.session_state.get("co2_Emissions_Electricity", 0.0)),
                    "Green Electricity": 0.0,  # forced to 0 per project rule
                    "Gas": float(st.session_state.get("co2_emissions_gas", 0.0)),
                    "District Heating": float(st.session_state.get("co2_emissions_dh", 0.0)),
                    "District Cooling": float(st.session_state.get("co2_emissions_dc", 0.0)),
                    "Biomass": float(st.session_state.get("co2_emissions_biomass", 0.0)),
                }

                # Decarbonization multiplier based on CRREM DE grid electricity EF series
                ef_grid = crrem["ef_grid"]
                # analysis horizon (scenario: start at project year; cap at CRREM data horizon)
                min_year = int(max(ef_grid.index.min(), 2020))
                max_year = int(min(ef_grid.index.max(), 2050))
                start_year = max(int(project_year_val), min_year)
                years = list(range(start_year, max_year + 1))

                m = compute_decarb_multiplier(ef_grid, int(project_year_val), years)

                # Net annual emissions (kgCO2e) in the base year, excluding Green Electricity (EF=0)
                emissions_base = 0.0
                for src, kwh in annual_kwh_by_source.items():
                    if str(src) == "Green Electricity":
                        continue
                    emissions_base += float(kwh) * float(base_factors.get(str(src), 0.0))

                emissions_series = pd.Series({y: float(emissions_base) * float(m.loc[y]) for y in years})
                carbon_asset = emissions_series / project_area_val  # kgCO2e/m²·yr
                eui_asset_series = pd.Series({y: float(eui_asset) for y in years})

                # --- CRREM limits (pathways)
                pc = crrem["pathways_carbon"].copy()
                pe = crrem["pathways_eui"].copy()

                pc_t = pc.loc[pc["target"].astype(str) == target_id]
                pe_t = pe.loc[pe["target"].astype(str) == target_id]

                carbon_pivot = pc_t.pivot_table(index="year", columns="property_type_code", values="kgco2e_per_m2_yr")
                eui_pivot = pe_t.pivot_table(index="year", columns="property_type_code", values="kwh_per_m2_yr")

                # Restrict to available years and analysis horizon
                years_avail = [y for y in years if (y in carbon_pivot.index and y in eui_pivot.index)]
                carbon_asset = carbon_asset.reindex(years_avail)
                eui_asset_series = eui_asset_series.reindex(years_avail)

                if crrem_use != "Mixed Use":
                    code_row = pt_df.loc[pt_df["app_use"].astype(str) == str(crrem_use)]
                    if code_row.empty:
                        st.error("Selected CRREM use-type not found in dataset.")
                    else:
                        p_code = str(code_row.iloc[0]["crrem_code"])
                        carbon_limit = carbon_pivot[p_code].reindex(years_avail)
                        eui_limit = eui_pivot[p_code].reindex(years_avail)
                else:
                    if not mixed_components:
                        st.error("Define at least one mixed-use component with a positive area share.")
                        carbon_limit = pd.Series(index=years_avail, dtype=float)
                        eui_limit = pd.Series(index=years_avail, dtype=float)
                    else:
                        # normalize weights
                        tot = sum(w for _, w in mixed_components)
                        weights = [(u, w / tot) for u, w in mixed_components if tot > 0]
                        # map use->code
                        use_to_code = dict(zip(pt_df["app_use"].astype(str), pt_df["crrem_code"].astype(str)))
                        carbon_limit = pd.Series(0.0, index=years_avail)
                        eui_limit = pd.Series(0.0, index=years_avail)
                        missing = []
                        for u, w in weights:
                            c = use_to_code.get(str(u))
                            if not c or c not in carbon_pivot.columns:
                                missing.append(str(u))
                                continue
                            carbon_limit = carbon_limit + w * carbon_pivot[c].reindex(years_avail).astype(float)
                            eui_limit = eui_limit + w * eui_pivot[c].reindex(years_avail).astype(float)
                        if missing:
                            st.warning(
                                f"Mixed-use components missing in dataset and ignored: {', '.join(sorted(set(missing)))}")

                # --- Stranding years
                stranding_carbon = find_stranding_year(carbon_asset, carbon_limit)
                stranding_eui = find_stranding_year(eui_asset_series, eui_limit)

                # --- Helper: additional CRREM charts (totals & cumulative)
                def _render_crrem_totals_and_cumulative(
                        years_list,
                        carbon_project_s: pd.Series,
                        carbon_limit_s: pd.Series,
                        eui_project_s: pd.Series,
                        eui_limit_s: pd.Series,
                        area_m2: float,
                        project_label: str,
                        project_color: str,
                        overlay_baseline: Optional[Tuple[pd.Series, pd.Series]] = None,
                ) -> None:
                    """Render total and cumulative charts for emissions (tCO2e) and energy (MWh)."""
                    if not years_list:
                        st.info("No overlapping years available for totals/cumulative charts.")
                        return

                    # Align series
                    carbon_project_s = carbon_project_s.reindex(years_list).astype(float)
                    carbon_limit_s = carbon_limit_s.reindex(years_list).astype(float)
                    eui_project_s = eui_project_s.reindex(years_list).astype(float)
                    eui_limit_s = eui_limit_s.reindex(years_list).astype(float)

                    # Totals (convert kg/m²·a -> t/a; kWh/m²·a -> MWh/a)
                    total_emis_t = (carbon_project_s * float(area_m2)) / 1000.0
                    total_emis_limit_t = (carbon_limit_s * float(area_m2)) / 1000.0
                    total_energy_mwh = (eui_project_s * float(area_m2)) / 1000.0
                    total_energy_limit_mwh = (eui_limit_s * float(area_m2)) / 1000.0

                    # Cumulative
                    cum_emis_t = total_emis_t.cumsum()
                    cum_emis_limit_t = total_emis_limit_t.cumsum()
                    cum_energy_mwh = total_energy_mwh.cumsum()
                    cum_energy_limit_mwh = total_energy_limit_mwh.cumsum()

                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("#### Total annual emissions")
                        fig_tot = go.Figure()
                        fig_tot.add_trace(go.Scatter(
                            x=years_list, y=total_emis_limit_t.values,
                            mode="lines+markers",
                            name="CRREM limit",
                            line=dict(color=CRREM_COLOR_LIMIT),
                            marker=dict(color=CRREM_COLOR_LIMIT),
                        ))
                        if overlay_baseline is not None:
                            base_carbon_s, _ = overlay_baseline
                            base_total_t = (base_carbon_s.reindex(years_list).astype(float) * float(area_m2)) / 1000.0
                            fig_tot.add_trace(go.Scatter(
                                x=years_list, y=base_total_t.values,
                                mode="lines+markers",
                                name="Baseline project",
                                line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                marker=dict(color=CRREM_COLOR_BASELINE),
                            ))
                        fig_tot.add_trace(go.Scatter(
                            x=years_list, y=total_emis_t.values,
                            mode="lines+markers",
                            name=project_label,
                            line=dict(color=project_color),
                            marker=dict(color=project_color),
                        ))
                        fig_tot.update_layout(height=420, yaxis_title="tCO₂e/a", legend_title="",
                                              legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                                              margin=dict(l=40, r=20, t=50, b=85))
                        fig_tot.update_yaxes(rangemode="tozero")
                        st.plotly_chart(fig_tot, use_container_width=True, key=f"crrem_tot_emis_{project_label}")

                        st.write("#### Cumulative emissions")
                        fig_cum = go.Figure()
                        fig_cum.add_trace(go.Scatter(
                            x=years_list, y=cum_emis_limit_t.values,
                            mode="lines+markers",
                            name="CRREM cumulative limit",
                            line=dict(color=CRREM_COLOR_LIMIT),
                            marker=dict(color=CRREM_COLOR_LIMIT),
                        ))
                        if overlay_baseline is not None:
                            base_carbon_s, _ = overlay_baseline
                            base_total_t = (base_carbon_s.reindex(years_list).astype(float) * float(area_m2)) / 1000.0
                            fig_cum.add_trace(go.Scatter(
                                x=years_list, y=base_total_t.cumsum().values,
                                mode="lines+markers",
                                name="Baseline cumulative",
                                line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                marker=dict(color=CRREM_COLOR_BASELINE),
                            ))
                        fig_cum.add_trace(go.Scatter(
                            x=years_list, y=cum_emis_t.values,
                            mode="lines+markers",
                            name=f"{project_label} cumulative",
                            line=dict(color=project_color),
                            marker=dict(color=project_color),
                        ))
                        fig_cum.update_layout(height=420, yaxis_title="tCO₂e", legend_title="",
                                              legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                                              margin=dict(l=40, r=20, t=50, b=85))
                        fig_cum.update_yaxes(rangemode="tozero")
                        st.plotly_chart(fig_cum, use_container_width=True, key=f"crrem_cum_emis_{project_label}")

                    with c2:
                        st.write("#### Total annual site energy")
                        fig_e_tot = go.Figure()
                        fig_e_tot.add_trace(go.Scatter(
                            x=years_list, y=total_energy_limit_mwh.values,
                            mode="lines+markers",
                            name="CRREM limit",
                            line=dict(color=CRREM_COLOR_LIMIT),
                            marker=dict(color=CRREM_COLOR_LIMIT),
                        ))
                        if overlay_baseline is not None:
                            _, base_eui_s = overlay_baseline
                            base_total_mwh = (base_eui_s.reindex(years_list).astype(float) * float(area_m2)) / 1000.0
                            fig_e_tot.add_trace(go.Scatter(
                                x=years_list, y=base_total_mwh.values,
                                mode="lines+markers",
                                name="Baseline project",
                                line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                marker=dict(color=CRREM_COLOR_BASELINE),
                            ))
                        fig_e_tot.add_trace(go.Scatter(
                            x=years_list, y=total_energy_mwh.values,
                            mode="lines+markers",
                            name=project_label,
                            line=dict(color=project_color),
                            marker=dict(color=project_color),
                        ))
                        fig_e_tot.update_layout(height=420, yaxis_title="MWh/a", legend_title="",
                                                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                                                margin=dict(l=40, r=20, t=50, b=85))
                        fig_e_tot.update_yaxes(rangemode="tozero")
                        st.plotly_chart(fig_e_tot, use_container_width=True, key=f"crrem_tot_energy_{project_label}")

                        st.write("#### Cumulative site energy")
                        fig_e_cum = go.Figure()
                        fig_e_cum.add_trace(go.Scatter(
                            x=years_list, y=cum_energy_limit_mwh.values,
                            mode="lines+markers",
                            name="CRREM cumulative limit",
                            line=dict(color=CRREM_COLOR_LIMIT),
                            marker=dict(color=CRREM_COLOR_LIMIT),
                        ))
                        if overlay_baseline is not None:
                            _, base_eui_s = overlay_baseline
                            base_total_mwh = (base_eui_s.reindex(years_list).astype(float) * float(area_m2)) / 1000.0
                            fig_e_cum.add_trace(go.Scatter(
                                x=years_list, y=base_total_mwh.cumsum().values,
                                mode="lines+markers",
                                name="Baseline cumulative",
                                line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                marker=dict(color=CRREM_COLOR_BASELINE),
                            ))
                        fig_e_cum.add_trace(go.Scatter(
                            x=years_list, y=cum_energy_mwh.values,
                            mode="lines+markers",
                            name=f"{project_label} cumulative",
                            line=dict(color=project_color),
                            marker=dict(color=project_color),
                        ))
                        fig_e_cum.update_layout(height=420, yaxis_title="MWh", legend_title="",
                                                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                                                margin=dict(l=40, r=20, t=50, b=85))
                        fig_e_cum.update_yaxes(rangemode="tozero")
                        st.plotly_chart(fig_e_cum, use_container_width=True, key=f"crrem_cum_energy_{project_label}")

                    # Cumulative exceedance (project − CRREM limit) — totals only
                    ex1, ex2 = st.columns(2)

                    # Exceedance is defined as max(project − limit, 0) in absolute units (tCO₂e and MWh)
                    exc_emis_t = (total_emis_t - total_emis_limit_t).clip(lower=0.0)
                    exc_energy_mwh = (total_energy_mwh - total_energy_limit_mwh).clip(lower=0.0)

                    cum_exc_emis_t = exc_emis_t.cumsum()
                    cum_exc_energy_mwh = exc_energy_mwh.cumsum()

                    with ex1:
                        st.write("#### Cumulative exceedance — Carbon (project − CRREM limit)")
                        fig_exc_c = go.Figure()

                        # Optional baseline overlay (also exceedance vs the same limit)
                        if overlay_baseline is not None:
                            base_carbon_s, _ = overlay_baseline
                            base_total_t = (base_carbon_s.reindex(years_list).astype(float) * float(area_m2)) / 1000.0
                            base_exc = (base_total_t - total_emis_limit_t).clip(lower=0.0)
                            fig_exc_c.add_trace(go.Scatter(
                                x=years_list, y=base_exc.cumsum().values,
                                mode="lines+markers",
                                name="Baseline cumulative exceedance",
                                line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                marker=dict(color=CRREM_COLOR_BASELINE),
                                fill="tozeroy",
                            ))

                        fig_exc_c.add_trace(go.Scatter(
                            x=years_list, y=cum_exc_emis_t.values,
                            mode="lines+markers",
                            name=f"{project_label} cumulative exceedance",
                            line=dict(color=project_color),
                            marker=dict(color=project_color),
                            fill="tozeroy",
                        ))
                        fig_exc_c.update_layout(
                            height=420, yaxis_title="tCO₂e", legend_title="",
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                            margin=dict(l=40, r=20, t=50, b=85),
                        )
                        fig_exc_c.update_yaxes(rangemode="tozero")
                        st.plotly_chart(fig_exc_c, use_container_width=True, key=f"crrem_cum_exceed_carbon_{project_label}")

                    with ex2:
                        st.write("#### Cumulative exceedance — Energy (project − CRREM limit)")
                        fig_exc_e = go.Figure()

                        if overlay_baseline is not None:
                            _, base_eui_s = overlay_baseline
                            base_total_mwh = (base_eui_s.reindex(years_list).astype(float) * float(area_m2)) / 1000.0
                            base_exc_e = (base_total_mwh - total_energy_limit_mwh).clip(lower=0.0)
                            fig_exc_e.add_trace(go.Scatter(
                                x=years_list, y=base_exc_e.cumsum().values,
                                mode="lines+markers",
                                name="Baseline cumulative exceedance",
                                line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                marker=dict(color=CRREM_COLOR_BASELINE),
                                fill="tozeroy",
                            ))

                        fig_exc_e.add_trace(go.Scatter(
                            x=years_list, y=cum_exc_energy_mwh.values,
                            mode="lines+markers",
                            name=f"{project_label} cumulative exceedance",
                            line=dict(color=project_color),
                            marker=dict(color=project_color),
                            fill="tozeroy",
                        ))
                        fig_exc_e.update_layout(
                            height=420, yaxis_title="MWh", legend_title="",
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                            margin=dict(l=40, r=20, t=50, b=85),
                        )
                        fig_exc_e.update_yaxes(rangemode="tozero")
                        st.plotly_chart(fig_exc_e, use_container_width=True, key=f"crrem_cum_exceed_energy_{project_label}")

                    # Optional: headroom (limit - project)
                    show_headroom = st.checkbox(
                        "Show headroom (limit − project) charts",
                        value=False,
                        key=f"crrem_show_headroom_{project_label}",
                        help="Positive values indicate compliance; negative values indicate exceedance.",
                    )
                    if show_headroom:
                        h1, h2 = st.columns(2)
                        with h1:
                            headroom_c = (carbon_limit_s - carbon_project_s).astype(float)
                            bar_colors = [CRREM_COLOR_MEASURES if v >= 0 else CRREM_COLOR_LIMIT for v in headroom_c.values]
                            fig_hc = go.Figure(go.Bar(x=years_list, y=headroom_c.values, marker_color=bar_colors, name="Headroom"))
                            fig_hc.update_layout(height=420, yaxis_title="kgCO₂e/m²·a", title="Carbon headroom",
                                                 margin=dict(l=40, r=20, t=45, b=45))
                            st.plotly_chart(fig_hc, use_container_width=True, key=f"crrem_headroom_carbon_{project_label}")
                        with h2:
                            headroom_e = (eui_limit_s - eui_project_s).astype(float)
                            bar_colors = [CRREM_COLOR_MEASURES if v >= 0 else CRREM_COLOR_LIMIT for v in headroom_e.values]
                            fig_he = go.Figure(go.Bar(x=years_list, y=headroom_e.values, marker_color=bar_colors, name="Headroom"))
                            fig_he.update_layout(height=420, yaxis_title="kWh/m²·a", title="EUI headroom",
                                                 margin=dict(l=40, r=20, t=45, b=45))
                            st.plotly_chart(fig_he, use_container_width=True, key=f"crrem_headroom_energy_{project_label}")

                st.write("## Prognose without measures")
                st.metric(label="Selected Scenario",value=f"{st.session_state.get('active_scenario')}")
                # --- Display
                kpi1, kpi2, kpi3 = st.columns(3)
                with kpi1:
                    st.metric("Baseline year", f"{project_year_val}")
                with kpi2:
                    st.metric("Stranding year (Carbon)",
                              "Not stranded" if stranding_carbon is None else str(stranding_carbon))
                with kpi3:
                    st.metric("Stranding year (EUI)",
                              "Not stranded" if stranding_eui is None else str(stranding_eui))

                ccol, ecol = st.columns(2)

                with ccol:
                    st.write("#### Carbon intensity vs CRREM pathway")
                    df_plot = pd.DataFrame({
                        "year": years_avail,
                        "Project": carbon_asset.values,
                        "CRREM limit": carbon_limit.values,
                    })
                    fig = px.line(df_plot, x="year", y=["Project", "CRREM limit"])
                    fig.update_layout(height=520, yaxis_title="kgCO₂e/m²·a", legend_title="",
                                      legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                                      margin=dict(l=40, r=20, t=50, b=85))
                    fig.update_traces(mode="lines+markers")
                    fig.update_yaxes(rangemode="tozero")
                    # Enforce consistent colors across baseline and measures charts
                    for tr in fig.data:
                        if tr.name == "Project":
                            tr.update(line=dict(color=CRREM_COLOR_BASELINE), marker=dict(color=CRREM_COLOR_BASELINE))
                        elif tr.name == "CRREM limit":
                            tr.update(line=dict(color=CRREM_COLOR_LIMIT), marker=dict(color=CRREM_COLOR_LIMIT))
                    if stranding_carbon is not None:
                        fig.add_vline(x=stranding_carbon, line_width=3, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)

                with ecol:
                    st.write("#### EUI vs CRREM pathway")
                    df_plot2 = pd.DataFrame({
                        "year": years_avail,
                        "Project": eui_asset_series.values,
                        "CRREM limit": eui_limit.values,
                    })
                    fig2 = px.line(df_plot2, x="year", y=["Project", "CRREM limit"])
                    fig2.update_layout(height=520, yaxis_title="kWh/m²·a", legend_title="",
                                       legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                                       margin=dict(l=40, r=20, t=50, b=85))
                    fig2.update_traces(mode="lines+markers")
                    fig2.update_yaxes(rangemode="tozero")
                    # Enforce consistent colors across baseline and measures charts
                    for tr in fig2.data:
                        if tr.name == "Project":
                            tr.update(line=dict(color=CRREM_COLOR_BASELINE), marker=dict(color=CRREM_COLOR_BASELINE))
                        elif tr.name == "CRREM limit":
                            tr.update(line=dict(color=CRREM_COLOR_LIMIT), marker=dict(color=CRREM_COLOR_LIMIT))
                    if stranding_eui is not None:
                        fig2.add_vline(x=stranding_eui, line_width=3, line_dash="dash", line_color="black")
                    st.plotly_chart(fig2, use_container_width=True)

                with st.expander("Additional CRREM diagrams — Baseline", expanded=False):
                    st.caption("Totals and cumulative charts use your Project Area (m²) and the same CRREM pathway years as the intensity plots.")
                    _render_crrem_totals_and_cumulative(
                        years_avail,
                        carbon_asset,
                        carbon_limit,
                        eui_asset_series,
                        eui_limit,
                        project_area_val,
                        project_label="Project",
                        project_color=CRREM_COLOR_BASELINE,
                        overlay_baseline=None,
                    )

                st.divider()

                # =========================
                # Measures (scenario-specific)
                # =========================

                with st.expander("Decarbonization Path Analysis", expanded=False):
                    st.write("## Decarbonization Path Analysis")
                    show_overlay = st.checkbox(
                        "## Show baseline vs with measures",
                        value=True,
                        key="crrem_show_baseline_overlay",
                        help="Overlay baseline and with-measures trajectories in the measures charts below.",
                    )

                    # Build a parameter registry (dropdown options) from the existing sidebar parameters
                    end_uses_all = sorted(df_crrem_m["End_Use"].astype(str).unique().tolist())
                    end_uses_no_pv = [u for u in end_uses_all if u != "PV_Generation"]

                    param_specs = {}
                    param_options = []


                    def _add_param(label: str, spec: dict):
                        param_options.append(label)
                        param_specs[label] = spec


                    # Emission Factors (numeric)
                    _add_param("Emission Factors → Electricity", {"kind": "ef", "source": "Electricity", "dtype": "float"})
                    _add_param("Emission Factors → Green Electricity",
                               {"kind": "ef", "source": "Green Electricity", "dtype": "float"})
                    _add_param("Emission Factors → Gas", {"kind": "ef", "source": "Gas", "dtype": "float"})
                    _add_param("Emission Factors → District Heating",
                               {"kind": "ef", "source": "District Heating", "dtype": "float"})
                    _add_param("Emission Factors → District Cooling",
                               {"kind": "ef", "source": "District Cooling", "dtype": "float"})
                    _add_param("Emission Factors → Biomass", {"kind": "ef", "source": "Biomass", "dtype": "float"})

                    # Energy Tariffs (numeric; stored for future extensions)
                    _add_param("Energy Tariffs → Electricity",
                               {"kind": "tariff", "source": "Electricity", "dtype": "float"})
                    _add_param("Energy Tariffs → Green Electricity",
                               {"kind": "tariff", "source": "Green Electricity", "dtype": "float"})
                    _add_param("Energy Tariffs → Gas", {"kind": "tariff", "source": "Gas", "dtype": "float"})
                    _add_param("Energy Tariffs → District Heating",
                               {"kind": "tariff", "source": "District Heating", "dtype": "float"})
                    _add_param("Energy Tariffs → District Cooling",
                               {"kind": "tariff", "source": "District Cooling", "dtype": "float"})
                    _add_param("Energy Tariffs → Biomass", {"kind": "tariff", "source": "Biomass", "dtype": "float"})
                    # PV (numeric; affects CRREM by offsetting Electricity)
                    _add_param("PV_Generation → PV Annual Production (kWh/a)", {"kind": "pv", "dtype": "float"})

                    # Efficiency Factors (numeric)
                    for u in end_uses_no_pv:
                        _add_param(f"Efficiency Factors → {u}", {"kind": "eff", "end_use": u, "dtype": "float"})

                    # Assign Energy Sources (categorical)
                    for u in end_uses_no_pv:
                        _add_param(f"Assign Energy Sources → {u}", {"kind": "src", "end_use": u, "dtype": "source"})

                    # Measures editor (scenario-specific storage)
                    with st.expander("Measures (scenario-specific)", expanded=False):
                        st.write(
                            "Each row is one measure. From the selected year onwards, the parameter takes the new value. "
                            "Multiple measures for the same parameter in different years are allowed."
                        )

                        if "crrem_measures_df" not in st.session_state or not isinstance(
                                st.session_state.get("crrem_measures_df"), pd.DataFrame):
                            st.session_state["crrem_measures_df"] = pd.DataFrame(columns=["Parameter", "Year", "New Value"])

                        if st.button("Add measure", key="crrem_add_measure_btn", use_container_width=False):
                            df_tmp = st.session_state["crrem_measures_df"].copy()
                            default_param = param_options[0] if param_options else ""
                            df_tmp = pd.concat(
                                [
                                    df_tmp,
                                    pd.DataFrame(
                                        [{"Parameter": default_param, "Year": int(project_year_val), "New Value": ""}]),
                                ],
                                ignore_index=True,
                            )
                            st.session_state["crrem_measures_df"] = df_tmp

                        editor_kwargs = {
                            "num_rows": "dynamic",
                            "use_container_width": True,
                            "key": "crrem_measures_editor",
                        }
                        if hasattr(st, "column_config"):
                            editor_kwargs["column_config"] = {
                                "Parameter": st.column_config.SelectboxColumn("Parameter", options=param_options,
                                                                              required=True),
                                "Year": st.column_config.NumberColumn(
                                    "Year",
                                    min_value=int(start_year),
                                    max_value=int(max_year),
                                    step=1,
                                    format="%d",
                                    required=True,
                                ),
                                "New Value": st.column_config.TextColumn("New Value", required=True),
                            }

                        measures_df = st.data_editor(st.session_state["crrem_measures_df"], **editor_kwargs)
                        st.session_state["crrem_measures_df"] = measures_df
                        # Deleting measures (explicit controls to support Streamlit versions where row-delete UI is not exposed)
                        if not measures_df.empty:

                            def _fmt_measure_idx(i):
                                try:
                                    p = str(measures_df.loc[i, "Parameter"]) if pd.notna(
                                        measures_df.loc[i, "Parameter"]) else ""
                                except Exception:
                                    p = ""
                                try:
                                    yv = measures_df.loc[i, "Year"]
                                    y = str(int(float(yv))) if pd.notna(yv) and str(yv).strip() != "" else ""
                                except Exception:
                                    y = ""
                                return f"{i + 1}: {p} @ {y}"


                            def _crrem_delete_selected_measures():
                                sel = st.session_state.get("crrem_measures_delete_idx", [])
                                if sel:
                                    df = st.session_state.get("crrem_measures_df", pd.DataFrame()).copy()
                                    try:
                                        df = df.drop(sel).reset_index(drop=True)
                                    except Exception:
                                        df = df.iloc[[j for j in range(len(df)) if j not in set(sel)]].reset_index(
                                            drop=True)
                                    st.session_state["crrem_measures_df"] = df
                                # Clear selection (safe to mutate in callback)
                                st.session_state["crrem_measures_delete_idx"] = []


                            st.multiselect(
                                "Select measure rows to delete",
                                options=list(measures_df.index),
                                format_func=_fmt_measure_idx,
                                key="crrem_measures_delete_idx",
                            )
                            st.button(
                                "Delete selected measures",
                                key="crrem_delete_measures_btn",
                                on_click=_crrem_delete_selected_measures,
                                use_container_width=False,
                            )

                        # Persist measures into the active scenario payload (saved in Scenarios sheet)
                        try:
                            _sc = st.session_state.get("scenarios", {})
                            _act = st.session_state.get("active_scenario")
                            if _act in _sc:
                                _sc[_act]["crrem_measures"] = _measures_df_to_records(measures_df)
                                st.session_state["scenarios"] = _sc
                        except Exception:
                            pass

                        st.caption("Note: tariff measures are stored but do not affect the CRREM Carbon/EUI charts yet.")

                    # --- Compute trajectories WITH measures (step changes)
                    measures_df = st.session_state.get("crrem_measures_df")
                    measures_records = _measures_df_to_records(measures_df)

                    # Parse and validate measures
                    ef_measures = {s: [] for s in ENERGY_SOURCE_ORDER}  # by energy source
                    tariff_measures = {s: [] for s in ENERGY_SOURCE_ORDER}
                    eff_measures = []  # (year, end_use, value)
                    src_measures = []  # (year, end_use, source)
                    pv_measures = []  # (year, pv_annual_production_kwh_per_a)
                    parse_errors = []


                    def _to_int_year(x):
                        try:
                            return int(float(x))
                        except Exception:
                            return None


                    def _to_float(x):
                        try:
                            return float(str(x).replace(",", "."))
                        except Exception:
                            return None


                    def _norm_source(s):
                        s = str(s).strip()
                        # allow case-insensitive matching
                        for opt in ENERGY_SOURCE_ORDER:
                            if str(opt).lower() == s.lower():
                                return str(opt)
                        return None


                    for i, rec in enumerate(measures_records, start=1):
                        p = str(rec.get("Parameter", "")).strip()
                        y = _to_int_year(rec.get("Year"))
                        v = rec.get("New Value", "")
                        if not p or p not in param_specs:
                            continue
                        if y is None:
                            parse_errors.append(f"Row {i}: invalid Year.")
                            continue
                        if y < int(start_year) or y > int(max_year):
                            # ignore out-of-horizon rows
                            continue

                        spec = param_specs[p]
                        kind = spec.get("kind")

                        if spec.get("dtype") == "float":
                            fv = _to_float(v)
                            if fv is None:
                                parse_errors.append(f"Row {i}: '{p}' expects a numeric New Value.")
                                continue
                            if kind == "ef":
                                ef_measures[str(spec["source"])].append((int(y), float(fv)))
                            elif kind == "tariff":
                                tariff_measures[str(spec["source"])].append((int(y), float(fv)))
                            elif kind == "eff":
                                eff_measures.append((int(y), str(spec["end_use"]), float(fv)))
                            elif kind == "pv":
                                pv_measures.append((int(y), float(fv)))
                        elif spec.get("dtype") == "source":
                            sv = _norm_source(v)
                            if sv is None:
                                parse_errors.append(f"Row {i}: '{p}' expects one of: {', '.join(ENERGY_SOURCE_ORDER)}.")
                                continue
                            src_measures.append((int(y), str(spec["end_use"]), str(sv)))

                    if parse_errors:
                        st.warning("Some measures were ignored due to invalid inputs:\n- " + "\n- ".join(parse_errors))

                    has_any_measures = any([
                        any(v for v in ef_measures.values()),
                        any(v for v in tariff_measures.values()),
                        len(eff_measures) > 0,
                        len(src_measures) > 0,
                        len(pv_measures) > 0,
                    ])

                    if has_any_measures:
                        # Sort measures by year
                        for k in ef_measures:
                            ef_measures[k] = sorted(ef_measures[k], key=lambda t: t[0])
                        for k in tariff_measures:
                            tariff_measures[k] = sorted(tariff_measures[k], key=lambda t: t[0])
                        eff_measures = sorted(eff_measures, key=lambda t: t[0])
                        src_measures = sorted(src_measures, key=lambda t: t[0])
                        pv_measures = sorted(pv_measures, key=lambda t: t[0])

                        # Baseline maps (from current scenario state)
                        eff_base = {u: float(st.session_state.get(f"eff_{u}", 1.0)) for u in end_uses_all}
                        src_base = {u: str(st.session_state.get(f"source_{u}", "Electricity")) for u in end_uses_all}
                        src_base["PV_Generation"] = "Electricity"

                        base_tariffs = {
                            "Electricity": float(st.session_state.get("cost_electricity", 0.0)),
                            "Green Electricity": float(st.session_state.get("cost_green_electricity", 0.0)),
                            "Gas": float(st.session_state.get("cost_gas", 0.0)),
                            "District Heating": float(st.session_state.get("cost_dh", 0.0)),
                            "District Cooling": float(st.session_state.get("cost_dc", 0.0)),
                            "Biomass": float(st.session_state.get("cost_biomass", 0.0)),
                        }

                        # Annual kWh by end use from uploaded sheet (raw, before efficiency/source assignment)
                        annual_by_enduse = df_crrem_m.groupby("End_Use", as_index=True)["kWh"].sum()

                        # PV scaling (scenario-specific). PV is always included; if PV scaling is disabled, scale=1.0.
                        pv_apply_scale = bool(st.session_state.get("pv_sc_enabled", False))
                        pv_scale = float(st.session_state.get("pv_scale", 1.0))
                        pv_scale_eff = pv_scale if pv_apply_scale else 1.0


                        def _ef_at_year(src: str, year: int, inclusive: bool = True) -> float:
                            # Green Electricity is always zero by project rule
                            if str(src) == "Green Electricity":
                                return 0.0
                            # Find the most recent EF-setting (project baseline year or EF measure)
                            y0 = int(project_year_val)
                            v0 = float(base_factors.get(str(src), 0.0))
                            for ym, vm in ef_measures.get(str(src), []):
                                if ((int(ym) <= int(year)) if inclusive else (int(ym) < int(year))) and int(ym) >= int(y0):
                                    y0 = int(ym)
                                    v0 = float(vm)

                            # Apply electricity-based decarbonization ratio EF_grid(year)/EF_grid(y0)
                            y0_c = _clamp_year_to_series(int(y0), ef_grid)
                            y_c = _clamp_year_to_series(int(year), ef_grid)
                            denom = float(ef_grid.loc[y0_c]) if float(ef_grid.loc[y0_c]) != 0 else None
                            if denom is None:
                                return float(v0)
                            return float(v0) * float(ef_grid.loc[y_c]) / denom


                        # Trajectories
                        carbon_meas = {}
                        eui_meas = {}
                        carbon_pre = {}
                        eui_pre = {}

                        # Years where the measures curve should have a vertical step (only for parameters that affect these charts)
                        step_years = set()
                        step_years.update([int(ym) for ym, _, _ in eff_measures])
                        step_years.update([int(ym) for ym, _, _ in src_measures])
                        for _src, _lst in ef_measures.items():
                            step_years.update([int(ym) for ym, _ in _lst])
                        step_years.update([int(ym) for ym, _ in pv_measures])
                        step_years = sorted([yy for yy in step_years if int(start_year) <= int(yy) <= int(max_year)])


                        def _compute_for_year(y: int, include_year_measures: bool) -> Tuple[float, float]:
                            # Build year-specific parameter sets (piecewise constant; only measure years should create vertical steps in plots)
                            eff_y = dict(eff_base)
                            src_y = dict(src_base)
                            tariffs_y = dict(base_tariffs)

                            # PV annual production override (kWh/a). If set via measures, overrides baseline PV (and sidebar PV scale).
                            pv_annual_override_y = None

                            def _cmp(ym: int) -> bool:
                                return int(ym) <= int(y) if include_year_measures else int(ym) < int(y)

                            # Apply efficiency measures (< y for 'pre', <= y for 'post')
                            for ym, eu, val in eff_measures:
                                if _cmp(ym):
                                    eff_y[str(eu)] = float(val)

                            # Apply source assignment measures
                            for ym, eu, val in src_measures:
                                if _cmp(ym):
                                    src_y[str(eu)] = str(val)

                            # Tariffs measures (stored; not used in these charts yet)
                            for s in tariffs_y.keys():
                                for ym, val in tariff_measures.get(str(s), []):
                                    if _cmp(ym):
                                        tariffs_y[str(s)] = float(val)

                            # PV measures (affect these charts)
                            for ym, val in pv_measures:
                                if _cmp(ym):
                                    pv_annual_override_y = float(val)
                            # Compute annual kWh by energy source for year y
                            kwh_by_source_y = {}
                            consumption_kwh_y = 0.0

                            for eu, kwh in annual_by_enduse.items():
                                eu = str(eu)
                                effv = float(eff_y.get(eu, 1.0) or 1.0)
                                if effv == 0:
                                    effv = 1.0
                                kwh_adj = float(kwh) / effv

                                if eu == "PV_Generation":
                                    # PV always offsets electricity; enforce negative generation
                                    if pv_annual_override_y is not None:
                                        # Absolute annual PV production (kWh/a) provided by measures
                                        kwh_adj = -abs(float(pv_annual_override_y))
                                    else:
                                        # Baseline PV (from uploaded data) scaled by sidebar PV factor (if enabled)
                                        kwh_adj = -abs(kwh_adj) * float(pv_scale_eff)
                                    src = "Electricity"
                                else:
                                    consumption_kwh_y += kwh_adj
                                    src = str(src_y.get(eu, "Electricity"))
                                    if src not in ENERGY_SOURCE_ORDER:
                                        src = "Electricity"

                                kwh_by_source_y[src] = float(kwh_by_source_y.get(src, 0.0)) + float(kwh_adj)

                            # Clamp net electricity to >= 0 (no export credit)
                            if "Electricity" in kwh_by_source_y:
                                kwh_by_source_y["Electricity"] = max(float(kwh_by_source_y["Electricity"]), 0.0)

                            # Compute emissions intensity for year y
                            emis_y = 0.0
                            for src, kwhv in kwh_by_source_y.items():
                                ef_y = _ef_at_year(str(src), int(y), inclusive=include_year_measures)
                                emis_y += float(kwhv) * float(ef_y)

                            carbon_int = float(emis_y) / project_area_val
                            eui_int = float(consumption_kwh_y) / project_area_val
                            return carbon_int, eui_int

                        with st.expander("Measures timeline", expanded=True):
                            if measures_records:
                                df_meas_tl = pd.DataFrame(measures_records)
                                df_meas_tl = df_meas_tl.dropna(subset=["Year"])
                                if not df_meas_tl.empty:
                                    df_meas_tl["Category"] = df_meas_tl["Parameter"].astype(str).str.split("→").str[0].str.strip()
                                    df_meas_tl["Parameter"] = df_meas_tl["Parameter"].astype(str).str.strip()
                                    df_meas_tl = df_meas_tl.sort_values(by="Year", ascending=False)

                                    fig_tl = px.scatter(
                                        df_meas_tl,
                                        x="Year",
                                        y="Parameter",
                                        color="Category",
                                        hover_data={"New Value": True, "Year": True, "Category": True,
                                                    "Parameter": True},
                                    )
                                    fig_tl.update_layout(height=420, xaxis_title="Year", yaxis_title="", legend_title="",
                                                         margin=dict(l=20, r=20, t=50, b=30))
                                    fig_tl.update_traces(marker=dict(size=14, symbol="square"))
                                    st.plotly_chart(fig_tl, use_container_width=True)
                                else:
                                    st.info("No valid measures to plot in the timeline.")
                            else:
                                st.info("No measures defined yet.")

                        for y in years_avail:
                            y_int = int(y)

                            # Pre-step point at the SAME calendar year using parameters strictly before year y
                            if (y_int in step_years) and (y_int != int(years_avail[0])):
                                cpre, epre = _compute_for_year(y_int, include_year_measures=False)
                                carbon_pre[y_int] = float(cpre)
                                eui_pre[y_int] = float(epre)

                            cpost, epost = _compute_for_year(y_int, include_year_measures=True)
                            carbon_meas[y_int] = float(cpost)
                            eui_meas[y_int] = float(epost)

                        carbon_meas_s = pd.Series(carbon_meas).reindex(years_avail)
                        eui_meas_s = pd.Series(eui_meas).reindex(years_avail)


                        # Plot series for with-measures: one value per year (no duplicate x-values).
                        # Plotly will draw straight line segments between consecutive years.
                        carbon_meas_x = years_avail
                        carbon_meas_y = carbon_meas_s.astype(float).values.tolist()
                        eui_meas_x = years_avail
                        eui_meas_y = eui_meas_s.astype(float).values.tolist()
                        stranding_carbon_meas = find_stranding_year(carbon_meas_s, carbon_limit)
                        stranding_eui_meas = find_stranding_year(eui_meas_s, eui_limit)

                        st.write("## Prognose with measures")
                        st.metric(label="Selected Scenario", value=f"{st.session_state.get('active_scenario')}")
                        mk1, mk2, mk3 = st.columns(3)
                        with mk1:
                            st.metric("Measures defined", str(len(measures_records)))
                        with mk2:
                            st.metric("Stranding year (Carbon, with measures)",
                                      "Not stranded" if stranding_carbon_meas is None else str(
                                          stranding_carbon_meas))
                        with mk3:
                            st.metric("Stranding year (EUI, with measures)",
                                      "Not stranded" if stranding_eui_meas is None else str(stranding_eui_meas))

                        mcol, ecol2 = st.columns(2)

                        with mcol:
                            st.write("#### Carbon intensity vs CRREM pathway")
                            figm = go.Figure()
                            # CRREM limit
                            figm.add_trace(go.Scatter(
                                x=years_avail, y=carbon_limit.values,
                                mode="lines+markers",
                                name="CRREM limit",
                                line=dict(color=CRREM_COLOR_LIMIT),
                                marker=dict(color=CRREM_COLOR_LIMIT),
                            ))
                            # baseline
                            if show_overlay:
                                figm.add_trace(go.Scatter(
                                    x=years_avail, y=carbon_asset.values,
                                    mode="lines+markers",
                                    name="Baseline project",
                                    line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                    marker=dict(color=CRREM_COLOR_BASELINE),
                                ))
                            # with measures (step)
                            figm.add_trace(go.Scatter(
                                x=carbon_meas_x, y=carbon_meas_y,
                                mode="lines+markers",
                                name="Project (with measures)",
                                line=dict(color=CRREM_COLOR_MEASURES),
                                marker=dict(color=CRREM_COLOR_MEASURES),
                            ))
                            figm.update_layout(height=520, yaxis_title="kgCO₂e/m²·a", legend_title="",
                                               legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center",
                                                           x=0.5), margin=dict(l=40, r=20, t=50, b=85))
                            figm.update_yaxes(rangemode="tozero")
                            if stranding_carbon_meas is not None:
                                figm.add_vline(x=stranding_carbon_meas, line_width=3, line_dash="dash", line_color="black")
                            st.plotly_chart(figm, use_container_width=True, key="crrem_carbon_measures_chart")

                        with ecol2:
                            st.write("#### EUI vs CRREM pathway")
                            fige = go.Figure()
                            fige.add_trace(go.Scatter(
                                x=years_avail, y=eui_limit.values,
                                mode="lines+markers",
                                name="CRREM limit",
                                line=dict(color=CRREM_COLOR_LIMIT),
                                marker=dict(color=CRREM_COLOR_LIMIT),
                            ))
                            if show_overlay:
                                fige.add_trace(go.Scatter(
                                    x=years_avail, y=eui_asset_series.values,
                                    mode="lines+markers",
                                    name="Baseline project",
                                    line=dict(dash="dash", color=CRREM_COLOR_BASELINE),
                                    marker=dict(color=CRREM_COLOR_BASELINE),
                                ))
                            fige.add_trace(go.Scatter(
                                x=eui_meas_x, y=eui_meas_y,
                                mode="lines+markers",
                                name="Project (with measures)",
                                line=dict(color=CRREM_COLOR_MEASURES),
                                marker=dict(color=CRREM_COLOR_MEASURES),
                            ))
                            fige.update_layout(height=520, yaxis_title="kWh/m²·a", legend_title="",
                                               legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center",
                                                           x=0.5), margin=dict(l=40, r=20, t=50, b=85))
                            fige.update_yaxes(rangemode="tozero")
                            if stranding_eui_meas is not None:
                                fige.add_vline(x=stranding_eui_meas, line_width=3, line_dash="dash", line_color="black")
                            st.plotly_chart(fige, use_container_width=True, key="crrem_eui_measures_chart")

                        with st.expander("Additional CRREM diagrams — With measures", expanded=False):
                            st.caption("Totals and cumulative charts are computed from the with-measures trajectories shown above.")
                            _render_crrem_totals_and_cumulative(
                                years_avail,
                                carbon_meas_s,
                                carbon_limit,
                                eui_meas_s,
                                eui_limit,
                                project_area_val,
                                project_label="Project (with measures)",
                                project_color=CRREM_COLOR_MEASURES,
                                overlay_baseline=(carbon_asset, eui_asset_series) if show_overlay else None,
                            )



                    st.caption(
                        "Notes: Green Electricity and PV offset are treated with EF=0. PV offsets Electricity consumption (no export credit).")

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

with tab7:
    if uploaded_file:
        st.write("## Scenario Comparison")

        # Use the same currency the user selected in the sidebar (fallback to preloaded or €)
        _curr = None
        try:
            _curr = currency_symbol  # set in sidebar 'Energy Tariffs'
        except Exception:
            _curr = preloaded.get("currency") if preloaded else None
        if not _curr:
            _curr = "€"

        _area = float(st.session_state.get("project_area", 0.0)) if st.session_state.get(
            "project_area") is not None else 0.0
        if _area <= 0:
            try:
                _area = float(project_area)
            except Exception:
                _area = 0.0

        scenarios = st.session_state.get("scenarios", {})
        if not scenarios:
            st.info("No scenarios found. Use the Scenario Manager in the sidebar to create scenarios.")
        else:
            # Base data (monthly energy balance)
            df = energy_balance_sheet(uploaded_file.getvalue())
            df_base = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

            rows = []
            energy_rows = []  # per-source, per-scenario (factored) end energy intensity
            cost_rows = []  # per-source, per-scenario (factored) cost intensity
            emissions_rows = []  # per-source, per-scenario (factored) emissions intensity

            for name, payload in scenarios.items():
                payload = payload or {}
                eff = (payload.get("efficiency") or {})
                mapping = (payload.get("mapping") or {})
                factors = (payload.get("factors") or {})
                tariffs = (payload.get("tariffs") or {})

                df_s = df_base.copy()
                df_s["Efficiency_Factor"] = df_s["End_Use"].map(lambda u: float(eff.get(u, 1.0))).fillna(1.0)

                # Apply efficiency factors (kWh is divided by factor)
                df_s["kWh_factored"] = df_s["kWh"] / df_s["Efficiency_Factor"]

                # Apply per-scenario PV scale (PV_Generation only)
                # In Scenarios tab net KPIs, PV is always considered as an offset.
                # To model a "no PV" scenario, set PV scale to 0.0.
                pv_cfg = (payload.get("pv") or {}) if isinstance(payload, dict) else {}
                pv_scale = float(pv_cfg.get("scale", 1.0))
                pv_mask = df_s["End_Use"] == "PV_Generation"
                if pv_mask.any():
                    df_s.loc[pv_mask, "kWh_factored"] = df_s.loc[pv_mask, "kWh_factored"] * pv_scale

                # Enforce sign convention for net calculations:
                # - PV_Generation is always treated as a negative credit (generation)
                # - All other end uses are treated as consumption only (clip negatives to 0)
                pv_mask = df_s["End_Use"] == "PV_Generation"
                df_s["kWh_signed"] = df_s["kWh_factored"]
                df_s.loc[pv_mask, "kWh_signed"] = -df_s.loc[pv_mask, "kWh_factored"].abs()
                df_s.loc[~pv_mask, "kWh_signed"] = df_s.loc[~pv_mask, "kWh_factored"].clip(lower=0.0)

                df_s["Energy_Source"] = df_s["End_Use"].map(lambda u: str(mapping.get(u, "Electricity")))

                # Annual energy (net includes PV as a negative contribution)
                totals_use = df_s.groupby("End_Use", as_index=False)["kWh_signed"].sum()
                net_kwh = float(totals_use["kWh_signed"].sum())
                gross_kwh = float(
                    totals_use.loc[
                        (totals_use["End_Use"] != "PV_Generation") & (totals_use["kWh_signed"] > 0),
                        "kWh_signed"
                    ].sum()
                )
                pv_kwh = float(abs(totals_use.loc[totals_use["End_Use"] == "PV_Generation", "kWh_signed"].sum()))

                # Net CO2 and net cost (including PV credit as signed kWh)
                df_net = df_s.copy()
                df_net["co2_factor"] = df_net["Energy_Source"].map(lambda s: float(factors.get(s, 0.0))).fillna(0.0)
                df_net["tariff"] = df_net["Energy_Source"].map(lambda s: float(tariffs.get(s, 0.0))).fillna(0.0)

                co2_kg = float((df_net["kWh_signed"] * df_net["co2_factor"]).sum())
                cost_val = float((df_net["kWh_signed"] * df_net["tariff"]).sum())
                # Gross CO2 and gross cost (excluding PV_Generation)
                df_gross = df_s.loc[df_s["End_Use"] != "PV_Generation"].copy()
                df_gross["kWh_pos"] = df_gross["kWh_factored"].clip(lower=0.0)
                df_gross["co2_factor"] = df_gross["Energy_Source"].map(lambda s: float(factors.get(s, 0.0))).fillna(0.0)
                df_gross["tariff"] = df_gross["Energy_Source"].map(lambda s: float(tariffs.get(s, 0.0))).fillna(0.0)
                gross_co2_kg = float((df_gross["kWh_pos"] * df_gross["co2_factor"]).sum())
                gross_cost_val = float((df_gross["kWh_pos"] * df_gross["tariff"]).sum())

                # Per-source breakdown (net, including PV) for scenario comparison charts (intensities)
                if _area and _area > 0:
                    df_src = df_net.copy()
                    df_src["cost"] = df_src["kWh_signed"] * df_src["tariff"]
                    df_src["co2_kg"] = df_src["kWh_signed"] * df_src["co2_factor"]

                    grp = df_src.groupby("Energy_Source", as_index=False).agg(
                        kWh=("kWh_signed", "sum"),
                        cost=("cost", "sum"),
                        co2_kg=("co2_kg", "sum"),
                    )

                    for _, r in grp.iterrows():
                        src = r["Energy_Source"]
                        energy_rows.append({
                            "Scenario": str(name),
                            "Energy_Source": src,
                            "End Energy (kWh/m²·a)": float(r["kWh"]) / _area,
                        })
                        cost_rows.append({
                            "Scenario": str(name),
                            "Energy_Source": src,
                            f"Cost ({_curr}/m²·a)": float(r["cost"]) / _area,
                        })
                        emissions_rows.append({
                            "Scenario": str(name),
                            "Energy_Source": src,
                            "Emissions (kgCO₂e/m²·a)": float(r["co2_kg"]) / _area,
                        })

                rows.append({
                    "Scenario": str(name),
                    "Net Energy (kWh/a)": net_kwh,
                    "Gross Consumption (kWh/a)": gross_kwh,
                    "PV Generation (kWh/a)": pv_kwh,
                    "Net CO2 (t/a)": co2_kg / 1000.0,
                    f"Net Cost ({_curr}/a)": cost_val,
                    # Hidden (used for Gross KPI charts)
                    "Gross CO2 (t/a)": gross_co2_kg / 1000.0,
                    f"Gross Cost ({_curr}/a)": gross_cost_val,
                    "Net EUI (kWh/m²·a)": (net_kwh / _area) if _area else np.nan,
                    "Gross EUI (kWh/m²·a)": (gross_kwh / _area) if _area else np.nan,
                })

            scenario_order = [str(s) for s in scenarios.keys()]
            df_cmp = pd.DataFrame(rows)
            df_cmp["Scenario"] = df_cmp["Scenario"].astype(str)
            df_cmp["Scenario"] = pd.Categorical(df_cmp["Scenario"], categories=scenario_order, ordered=True)
            df_cmp = df_cmp.sort_values("Scenario", kind="stable").reset_index(drop=True)
            df_cmp_display = df_cmp[[
                "Scenario",
                "Net Energy (kWh/a)",
                "Gross Consumption (kWh/a)",
                "PV Generation (kWh/a)",
                "Net CO2 (t/a)",
                f"Net Cost ({_curr}/a)",
                "Net EUI (kWh/m²·a)",
                "Gross EUI (kWh/m²·a)",
            ]].copy()
            st.dataframe(df_cmp_display, use_container_width=True)

            # Net KPI charts (incl. PV_Generation) — values printed on bars
            if _area and _area > 0:
                df_kpi = df_cmp.copy()
                df_kpi["Scenario"] = df_kpi["Scenario"].astype(str)
                df_kpi["Net Emissions (kgCO₂e/m²·a)"] = (df_kpi["Net CO2 (t/a)"] * 1000.0) / _area

                net_cost_col_a = f"Net Cost ({_curr}/a)"
                net_cost_col_m2 = f"Net Cost ({_curr}/m²·a)"
                if net_cost_col_a in df_kpi.columns:
                    df_kpi[net_cost_col_m2] = df_kpi[net_cost_col_a] / _area

                st.markdown("### Net KPI comparison (incl. PV)")

                scenario_color_map = {s: SCENARIO_COLOR_PALETTE[i % len(SCENARIO_COLOR_PALETTE)] for i, s in
                                      enumerate(scenario_order)}
                k1, k2, k3 = st.columns(3)

                with k1:
                    fig_net_eui = px.bar(
                        df_kpi,
                        x="Scenario",
                        y="Net EUI (kWh/m²·a)",
                        color="Scenario",
                        color_discrete_map=scenario_color_map,
                        category_orders={"Scenario": scenario_order},
                        text_auto=".1f",
                        title="Net EUI (kWh/m²·a)",
                    )
                    fig_net_eui.update_xaxes(type="category")
                    fig_net_eui.update_yaxes(rangemode="tozero")
                    fig_net_eui.update_layout(
                        xaxis_title="",
                        yaxis_title="kWh/m²·a",
                        legend_title_text="Scenario",
                        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                        margin=dict(b=90),
                    )
                    st.plotly_chart(fig_net_eui, use_container_width=True, key="scenario_net_eui")

                with k2:
                    fig_net_emis = px.bar(
                        df_kpi,
                        x="Scenario",
                        y="Net Emissions (kgCO₂e/m²·a)",
                        color="Scenario",
                        color_discrete_map=scenario_color_map,
                        category_orders={"Scenario": scenario_order},
                        text_auto=".1f",
                        title="Net Emissions (kgCO₂e/m²·a)",
                    )
                    fig_net_emis.update_xaxes(type="category")
                    fig_net_emis.update_yaxes(rangemode="tozero")
                    fig_net_emis.update_layout(
                        xaxis_title="",
                        yaxis_title="kgCO₂e/m²·a",
                        legend_title_text="Scenario",
                        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                        margin=dict(b=90),
                    )
                    st.plotly_chart(fig_net_emis, use_container_width=True, key="scenario_net_emissions")

                with k3:
                    if net_cost_col_m2 in df_kpi.columns:
                        fig_net_cost = px.bar(
                            df_kpi,
                            x="Scenario",
                            y=net_cost_col_m2,
                            color="Scenario",
                            color_discrete_map=scenario_color_map,
                            category_orders={"Scenario": scenario_order},
                            text_auto=".2f",
                            title=f"Net Cost ({_curr}/m²·a)",
                        )
                        fig_net_cost.update_xaxes(type="category")
                        fig_net_cost.update_yaxes(rangemode="tozero")
                        fig_net_cost.update_layout(
                            xaxis_title="",
                            yaxis_title=f"{_curr}/m²·a",
                            legend_title_text="Scenario",
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                            margin=dict(b=90),
                        )
                        st.plotly_chart(fig_net_cost, use_container_width=True, key="scenario_net_cost")

                # Gross KPI charts (excl. PV_Generation)
                df_kpi["Gross Emissions (kgCO₂e/m²·a)"] = (df_kpi["Gross CO2 (t/a)"] * 1000.0) / _area

                gross_cost_col_a = f"Gross Cost ({_curr}/a)"
                gross_cost_col_m2 = f"Gross Cost ({_curr}/m²·a)"
                if gross_cost_col_a in df_kpi.columns:
                    df_kpi[gross_cost_col_m2] = df_kpi[gross_cost_col_a] / _area

                st.markdown("### Gross KPI comparison (excl. PV)")

                g1, g2, g3 = st.columns(3)

                with g1:
                    fig_gross_eui = px.bar(
                        df_kpi,
                        x="Scenario",
                        y="Gross EUI (kWh/m²·a)",
                        color="Scenario",
                        color_discrete_map=scenario_color_map,
                        category_orders={"Scenario": scenario_order},
                        text_auto=".1f",
                        title="Gross EUI (kWh/m²·a)",
                    )
                    fig_gross_eui.update_xaxes(type="category")
                    fig_gross_eui.update_yaxes(rangemode="tozero")
                    fig_gross_eui.update_layout(
                        xaxis_title="",
                        yaxis_title="kWh/m²·a",
                        legend_title_text="Scenario",
                        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                        margin=dict(b=90),
                    )
                    st.plotly_chart(fig_gross_eui, use_container_width=True, key="scenario_gross_eui")

                with g2:
                    fig_gross_emis = px.bar(
                        df_kpi,
                        x="Scenario",
                        y="Gross Emissions (kgCO₂e/m²·a)",
                        color="Scenario",
                        color_discrete_map=scenario_color_map,
                        category_orders={"Scenario": scenario_order},
                        text_auto=".1f",
                        title="Gross Emissions (kgCO₂e/m²·a)",
                    )
                    fig_gross_emis.update_xaxes(type="category")
                    fig_gross_emis.update_yaxes(rangemode="tozero")
                    fig_gross_emis.update_layout(
                        xaxis_title="",
                        yaxis_title="kgCO₂e/m²·a",
                        legend_title_text="Scenario",
                        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                        margin=dict(b=90),
                    )
                    st.plotly_chart(fig_gross_emis, use_container_width=True, key="scenario_gross_emissions")

                with g3:
                    if gross_cost_col_m2 in df_kpi.columns:
                        fig_gross_cost = px.bar(
                            df_kpi,
                            x="Scenario",
                            y=gross_cost_col_m2,
                            color="Scenario",
                            color_discrete_map=scenario_color_map,
                            category_orders={"Scenario": scenario_order},
                            text_auto=".2f",
                            title=f"Gross Cost ({_curr}/m²·a)",
                        )
                        fig_gross_cost.update_xaxes(type="category")
                        fig_gross_cost.update_yaxes(rangemode="tozero")
                        fig_gross_cost.update_layout(
                            xaxis_title="",
                            yaxis_title=f"{_curr}/m²·a",
                            legend_title_text="Scenario",
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                            margin=dict(b=90),
                        )
                        st.plotly_chart(fig_gross_cost, use_container_width=True, key="scenario_gross_cost")
                    else:
                        st.info("Gross cost not available for this project.")

            else:
                st.info("Project Area must be greater than 0 to show per m² net KPI charts.")

            # Scenario comparison charts (factored values, stacked by Energy Source)
            if not _area or _area <= 0:
                st.warning("Project Area must be greater than 0 to show per m² scenario charts.")
            else:
                # scenario_order is defined above from scenarios (categorical x-axis)

                # 1) End Energy /m² (factored) by energy source
                df_energy_src = pd.DataFrame(energy_rows)
                if not df_energy_src.empty:
                    df_energy_src["Scenario"] = df_energy_src["Scenario"].astype(str)
                    fig_end_energy = px.bar(
                        df_energy_src,
                        x="Scenario",
                        y="End Energy (kWh/m²·a)",
                        color="Energy_Source",
                        barmode="relative",
                        title="End Energy /m² (factored) by Energy Source and Scenario",
                        category_orders={"Scenario": scenario_order},
                        color_discrete_map=color_map_sources,
                        text_auto=".1f",
                    )
                    fig_end_energy.update_layout(
                        xaxis_title="Scenario",
                        yaxis_title="kWh/m²·a",
                        legend_title_text="Energy Source",
                    )
                    fig_end_energy.update_traces(textfont_size=14, textfont_color="white")
                    fig_end_energy.update_xaxes(type="category")
                    st.plotly_chart(fig_end_energy, use_container_width=True, key="scenario_end_energy_m2_by_source")

                # 2) Energy Cost /m² (factored) by energy source
                cost_col = f"Cost ({_curr}/m²·a)"
                df_cost_src = pd.DataFrame(cost_rows)
                if not df_cost_src.empty and cost_col in df_cost_src.columns:
                    df_cost_src["Scenario"] = df_cost_src["Scenario"].astype(str)
                    fig_cost = px.bar(
                        df_cost_src,
                        x="Scenario",
                        y=cost_col,
                        color="Energy_Source",
                        barmode="relative",
                        title=f"Energy Cost /m² (factored) by Energy Source and Scenario [{_curr}]",
                        category_orders={"Scenario": scenario_order},
                        color_discrete_map=color_map_sources,
                        text_auto=".2f",
                    )
                    fig_cost.update_layout(
                        xaxis_title="Scenario",
                        yaxis_title=f"{_curr}/m²·a",
                        legend_title_text="Energy Source",
                    )
                    fig_cost.update_traces(textfont_size=14, textfont_color="white")
                    fig_cost.update_xaxes(type="category")
                    st.plotly_chart(fig_cost, use_container_width=True, key="scenario_cost_m2_by_source")

                # 3) Energy Emissions /m² (factored) by energy source
                df_emis_src = pd.DataFrame(emissions_rows)
                if not df_emis_src.empty:
                    df_emis_src["Scenario"] = df_emis_src["Scenario"].astype(str)
                    fig_emis = px.bar(
                        df_emis_src,
                        x="Scenario",
                        y="Emissions (kgCO₂e/m²·a)",
                        color="Energy_Source",
                        barmode="relative",
                        title="Energy Emissions /m² (factored) by Energy Source and Scenario",
                        category_orders={"Scenario": scenario_order},
                        color_discrete_map=color_map_sources,
                        text_auto=".2f",
                    )
                    fig_emis.update_layout(
                        xaxis_title="Scenario",
                        yaxis_title="kgCO₂e/m²·a",
                        legend_title_text="Energy Source",
                    )
                    fig_emis.update_traces(textfont_size=14, textfont_color="white")
                    fig_emis.update_xaxes(type="category")
                    st.plotly_chart(fig_emis, use_container_width=True, key="scenario_emissions_m2_by_source")

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 1b — Energy Balance with Factors (Energy Balance with Factors Tab)
# =========================
with tab1_factors:
    if uploaded_file:
        # ---- Load data
        df_eff = energy_balance_sheet(uploaded_file.getvalue())

        # ---- Wide->Long transform for plotting and grouping
        df_melted_eff = df_eff.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

        # ---- Apply per-End_Use efficiency factors (kWh is divided by factor)
        eff_map = {use: st.session_state.get(f"eff_{use}", 1.0) for use in df_melted_eff["End_Use"].unique()}
        df_melted_eff["Efficiency_Factor"] = df_melted_eff["End_Use"].map(eff_map).fillna(1.0)
        df_melted_eff["kWh"] = df_melted_eff["kWh"] / df_melted_eff["Efficiency_Factor"]

        # ---- Ensure Energy_Source exists (same mapping as Tab 1)
        df_melted_eff["Energy_Source"] = df_melted_eff["End_Use"].map(
            {k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted_eff["End_Use"].unique()}
        )

        project_area_eff = float(st.session_state.get("project_area", 1000.0))

        # ---- Monthly net totals (used for overlay line)
        monthly_totals_eff = (
            df_melted_eff.groupby("Month", as_index=False)["kWh"].sum()
            .assign(Month=lambda d: pd.Categorical(d["Month"], categories=MONTH_ORDER, ordered=True))
            .sort_values("Month", kind="stable")
            .reset_index(drop=True)
        )

        # ---- Monthly bar per End_Use (stacked, pos/neg relative) + net line overlay
        monthly_chart_eff = px.bar(
            df_melted_eff,
            x="Month",
            y="kWh",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"Month": MONTH_ORDER},
            text_auto=".0f",
        )
        monthly_chart_eff.update_traces(textfont_size=14, textfont_color="white")

        line_monthly_net_eff = px.line(
            monthly_totals_eff, x="Month", y="kWh", markers=True, labels={"kWh": "Net total"}
        )
        for tr in line_monthly_net_eff.data:
            tr.name = "Net total"
            tr.line.width = 5
            tr.line.color = "black"
            tr.line.dash = "dash"
            tr.marker.size = 12
            monthly_chart_eff.add_trace(tr)
        monthly_chart_eff.update_layout(showlegend=False)

        # ---- Monthly bar per Energy_Source (aggregate first for correct hover totals)
        monthly_by_source_eff = (
            df_melted_eff.groupby(["Month", "Energy_Source"], as_index=False)["kWh"].sum()
        )
        monthly_by_source_eff["Month"] = pd.Categorical(
            monthly_by_source_eff["Month"], categories=MONTH_ORDER, ordered=True
        )
        monthly_chart_source_eff = px.bar(
            monthly_by_source_eff,
            x="Month",
            y="kWh",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Month": MONTH_ORDER},
            text_auto=".0f",
        )
        monthly_chart_source_eff.update_layout(showlegend=False)
        monthly_chart_source_eff.update_traces(textfont_size=14, textfont_color="white")

        st.write("## Energy Balance with Factors (per End Use)")

        # ---- Annual totals per End_Use and per Energy_Source (+ intensities)
        totals_eff = df_melted_eff.groupby("End_Use", as_index=False)["kWh"].sum()
        totals_eff["Per Use"] = "Total"
        totals_eff["kWh_per_m2"] = (totals_eff["kWh"] / project_area_eff).round(1)

        # KPI helpers
        eui_eff = totals_eff.loc[totals_eff["kWh_per_m2"] > 0, "kWh_per_m2"].sum()
        net_energy_eff = totals_eff["kWh"].sum()
        net_eui_eff = totals_eff["kWh_per_m2"].sum()

        totals_per_source_eff = df_melted_eff.groupby("Energy_Source", as_index=False)["kWh"].sum()
        totals_per_source_eff["Per Source"] = "total_per_source"
        totals_per_source_eff["kWh_per_m2_per_source"] = (totals_per_source_eff["kWh"] / project_area_eff).round(1)

        # ---- Annual stacked bars (per End_Use + reference line)
        annual_chart_eff = px.bar(
            totals_eff,
            x="Per Use",
            y="kWh",
            color="End_Use",
            barmode="relative",
            color_discrete_map=color_map,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
            text_auto=".0f",
        )
        annual_chart_eff.add_hline(y=net_energy_eff, line_width=4, line_dash="dash", line_color="black")
        annual_chart_eff.add_annotation(
            x=0.5, xref="paper",
            y=net_energy_eff, yref="y",
            text=f"{net_energy_eff:,.0f} kWh",
            showarrow=False, yshift=12,
            font=dict(size=16, color="white"),
        )
        annual_chart_eff.update_traces(textfont_size=14, textfont_color="white")

        # ---- Annual stacked bars (per Energy_Source)
        annual_chart_per_source_eff = px.bar(
            totals_per_source_eff,
            x="Per Source",
            y="kWh",
            color="Energy_Source",
            barmode="relative",
            color_discrete_map=color_map_sources,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
            text_auto=".0f",
        )
        annual_chart_per_source_eff.update_traces(textfont_size=14, textfont_color="white")

        totals_eff_clean = totals_eff[(totals_eff["End_Use"] != "PV_Generation")]

        # ---- Donuts (EUI shares)
        energy_intensity_chart_eff = px.pie(
            totals_eff_clean,
            names="End_Use",
            values="kWh_per_m2",
            color="End_Use",
            color_discrete_map=color_map,
            hole=0.5,
            height=800,
            category_orders={"End_Use": END_USE_ORDER},
        )
        energy_intensity_chart_eff.update_layout(
            annotations=[dict(
                text=f"{eui_eff:,.1f}<br>kWh/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=50, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart_eff.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")

        energy_intensity_chart_per_source_eff = px.pie(
            totals_per_source_eff,
            names="Energy_Source",
            values="kWh_per_m2_per_source",
            color="Energy_Source",
            color_discrete_map=color_map_sources,
            hole=0.5,
            height=800,
            category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
        )
        energy_intensity_chart_per_source_eff.update_layout(
            annotations=[dict(
                text=f"{eui_eff:,.1f}<br>kWh/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=50, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart_per_source_eff.update_traces(textinfo="value+percent", textfont_size=18,
                                                            textfont_color="white")

        # ---- PV coverage (share of PV vs consumption-only EUI)
        totals_indexed_eff = totals_eff.set_index("End_Use")
        pv_value_eff = totals_indexed_eff.loc[
            "PV_Generation", "kWh_per_m2"] if "PV_Generation" in totals_indexed_eff.index else 0.0
        pv_coverage_eff = abs((pv_value_eff / eui_eff) * 100) if eui_eff != 0 else 0.0

        # ---- Layout: charts and KPIs
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy")
            st.plotly_chart(monthly_chart_eff, use_container_width=True, key="ebf_monthly_enduse")
        with col2:
            st.subheader("Annual Energy")
            st.plotly_chart(annual_chart_eff, use_container_width=True, key="ebf_annual_enduse")

        # KPI calculations (kept identical logic)
        monthly_avr_eff = (totals_eff["kWh"].sum()) / 12
        net_total_eff = totals_eff["kWh"].sum()
        total_energy_eff = totals_eff.loc[totals_eff["kWh"] > 0, "kWh"].sum()
        pv_total_eff = abs(df_melted_eff.groupby("End_Use")["kWh"].sum().get("PV_Generation", 0.0))

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st.plotly_chart(energy_intensity_chart_eff, use_container_width=True, key="ebf_eui_enduse")
        with col2:
            st.subheader("Energy KPI's")
            st.metric(label="Monthly Average Energy Consumption", value=f"{monthly_avr_eff:,.0f} kWh")
            st.metric(label="Total Annual Energy Consumption", value=f"{total_energy_eff:,.0f} kWh")
            st.metric(label="Net Annual Energy Consumption", value=f"{net_total_eff:,.0f} kWh")
            st.metric(label="EUI", value=f"{eui_eff:,.1f} kWh/m2.a")
            st.metric(label="Net EUI", value=f"{net_eui_eff:,.1f} kWh/m2.a")
            st.metric(label="PV Production", value=f"{pv_total_eff:,.1f} kWh")
            st.metric(label="PV Coverage", value=f"{pv_coverage_eff:,.1f} %")

        st.markdown("---")
        st.write("## Energy Balance with Factors (per Energy Source)")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy Demand")
            st.plotly_chart(monthly_chart_source_eff, use_container_width=True, key="ebf_monthly_source")
        with col2:
            st.subheader("Annual Energy Demand")
            st.plotly_chart(annual_chart_per_source_eff, use_container_width=True, key="ebf_annual_source")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st.plotly_chart(energy_intensity_chart_per_source_eff, use_container_width=True, key="ebf_eui_source")
        with col2:
            st.subheader("Energy KPI's")
            for _, row in totals_per_source_eff.iterrows():
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
        # ---- Apply per-End_Use efficiency factors (align with 'Energy Balance with Factors')
        eff_map = {use: st.session_state.get(f"eff_{use}", 1.0) for use in df_melted["End_Use"].unique()}
        df_melted["Efficiency_Factor"] = df_melted["End_Use"].map(eff_map).fillna(1.0)
        df_melted["kWh"] = df_melted["kWh"] / df_melted["Efficiency_Factor"]

        df_melted["Energy_Source"] = df_melted["End_Use"].map(
            {k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted["End_Use"].unique()})

        # Factor map from sidebar inputs (declared in Tab 1)
        factor_map = {
            "Electricity": co2_Emissions_Electricity,
            "Green Electricity": co2_Emissions_Green_Electricity,
            "Gas": co2_emissions_gas,
            "District Heating": co2_emissions_dh,
            "District Cooling": co2_emissions_dc,
            "Biomass": co2_emissions_biomass,
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

        totals_co2_use_clean = totals_co2_use[
            (totals_co2_use["End_Use"] != "PV_Generation")]

        # Donuts: CO₂ intensity shares
        co2_intensity_pie_use = px.pie(
            totals_co2_use_clean,
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
        # ---- Apply per-End_Use efficiency factors (align with 'Energy Balance with Factors')
        eff_map_cost = {use: st.session_state.get(f"eff_{use}", 1.0) for use in df_melted_cost["End_Use"].unique()}
        df_melted_cost["Efficiency_Factor"] = df_melted_cost["End_Use"].map(eff_map_cost).fillna(1.0)
        df_melted_cost["kWh"] = df_melted_cost["kWh"] / df_melted_cost["Efficiency_Factor"]

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
            "Biomass": cost_biomass,
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

        totals_cost_use_clean = totals_cost_use[
            (totals_cost_use["End_Use"] != "PV_Generation")]

        cost_intensity_pie_use = px.pie(
            totals_cost_use_clean,
            names="End_Use",
            values="cost_per_m2",
            color="End_Use",
            color_discrete_map=color_map,
            hole=0.5,
            height=800,
            category_orders={
                "End_Use": ["Heating", "Cooling", "Ventilation", "Lighting", "Equipment", "HotWater", "Pumps", "Other"]}
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

        r, g, b = pcolors.hex_to_rgb(bar_color)
        peak_day_fig.update_traces(marker_color=bar_color, fill="tozeroy",
                                   fillcolor=f"rgba({r},{g},{b},0.25)")

        st.subheader(f"Peak Day — {selected_load}")
        st.plotly_chart(peak_day_fig, use_container_width=True)
        st.caption(f"Daily Total on {date_label}: {peak_total:,.1f}")

        # --- Load Duration Curve (percentage of hours vs load) ---
        # Ensure numeric and drop NaNs
        ldc_vals = pd.to_numeric(df_loads[selected_load], errors="coerce").dropna()

        # Sort descending (exceedance)
        ldc_sorted = ldc_vals.sort_values(ascending=False).reset_index(drop=True)

        # Percentage of hours (0–100%)
        ldc_pct = (np.arange(1, len(ldc_sorted) + 1) / len(ldc_sorted)) * 100

        ldc_df = pd.DataFrame({
            "Percentage of Hours (%)": ldc_pct,
            f"{selected_load} (kW)": ldc_sorted.values
        })

        ldc_fig = px.line(
            ldc_df,
            x="Percentage of Hours (%)",
            y=f"{selected_load} (kW)",
            title=f"Load Duration Curve — {selected_load}"
        )
        ldc_fig.update_traces(line=dict(width=6, color=bar_color))
        r, g, b = pcolors.hex_to_rgb(bar_color)
        ldc_fig.update_traces(fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.25)")
        ldc_fig.update_layout(
            xaxis_title="Percentage of Hours (%)",
            yaxis_title=f"{selected_load} (kW)",
            xaxis=dict(range=[0, 100], dtick=10, ticksuffix="%"),
            height=700,
            showlegend=False
        )

        st.subheader(f"Load Duration Curve — {selected_load}")
        st.plotly_chart(ldc_fig, use_container_width=True)

        # -------------------------
        # PV Self-Consumption (hourly) — uses PV_Generation from Loads_Balance
        # -------------------------
        st.subheader("PV Self-Consumption (hourly) — PV_Generation")
        pv_col = "PV_Generation" if "PV_Generation" in df_loads.columns else None
        if pv_col is None:
            st.warning(
                "No hourly PV generation column 'PV_Generation' found in Loads_Balance. Add it to enable PV self-consumption.")
        else:
            pv_enabled = st.checkbox(
                "Enable PV self-consumption using PV_Generation",
                value=bool(st.session_state.get("pv_sc_enabled", False)),
                key="pv_sc_enabled",
            )
            pv_scale = numeric_input(
                "PV scale factor (dimensionless)",
                float(st.session_state.get("pv_scale", 1.0)),
                key="pv_scale",
                min_value=0.0,
                max_value=1000.0,
                fmt="{:.3f}",
                help="Scales the PV_Generation profile (e.g., 0.5 = half size, 2.0 = double size)."
            )

            # Persist PV settings into the active scenario (without touching other scenario fields)
            if "scenarios" in st.session_state and st.session_state.get("active_scenario") in st.session_state[
                "scenarios"]:
                _act = st.session_state.get("active_scenario")
                _payload = st.session_state["scenarios"].get(_act, {}) or {}
                if not isinstance(_payload.get("pv"), dict):
                    _payload["pv"] = {}
                _payload["pv"]["enabled"] = bool(pv_enabled)
                _payload["pv"]["scale"] = float(pv_scale)
                st.session_state["scenarios"][_act] = _payload

            if pv_enabled:
                load_series = pd.to_numeric(df_loads[selected_load], errors="coerce").fillna(0.0).clip(lower=0.0)
                pv_series = pd.to_numeric(df_loads[pv_col], errors="coerce").fillna(0.0).clip(lower=0.0) * float(
                    pv_scale)

                export = np.maximum(pv_series - load_series, 0.0)
                self_consumed = pv_series - export  # == min(load, pv)
                grid_import = np.maximum(load_series - pv_series, 0.0)

                pv_total = float(pv_series.sum())
                load_total = float(load_series.sum())
                self_total = float(self_consumed.sum())
                export_total = float(export.sum())
                import_total = float(grid_import.sum())

                sc_ratio = (self_total / pv_total) if pv_total > 0 else 0.0
                coverage_ratio = (self_total / load_total) if load_total > 0 else 0.0

                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("PV generation", f"{pv_total:,.0f} kWh")
                k2.metric("Self-consumed PV", f"{self_total:,.0f} kWh")
                k3.metric("PV export", f"{export_total:,.0f} kWh")
                k4.metric("Grid import after PV", f"{import_total:,.0f} kWh")
                k5.metric("Self-consumption ratio", f"{sc_ratio * 100:,.1f} %")
                k6.metric("PV coverage of load", f"{coverage_ratio * 100:,.1f} %")

                # Peak-day overlay: load vs PV vs net import
                pv_day = (df_loads.loc[df_loads["doy"] == peak_doy, ["hour", pv_col]]
                          .copy())
                pv_day["hour"] = pd.to_numeric(pv_day["hour"], errors="coerce")
                pv_day[pv_col] = pd.to_numeric(pv_day[pv_col], errors="coerce").fillna(0.0).clip(lower=0.0) * float(
                    pv_scale)
                pv_day = pv_day.sort_values("hour")

                net_day = np.maximum(
                    pd.to_numeric(day_profile[selected_load], errors="coerce").fillna(0.0).clip(lower=0.0).values
                    - pv_day[pv_col].values,
                    0.0
                )

                fig_pv = go.Figure()
                fig_pv.add_trace(go.Scatter(x=day_profile["hour"], y=day_profile[selected_load], mode="lines+markers",
                                            name="Load", line=dict(color=bar_color, width=6)))
                fig_pv.add_trace(go.Scatter(x=pv_day["hour"], y=pv_day[pv_col], mode="lines+markers",
                                            name="PV",
                                            line=dict(color=color_map.get("PV_Generation", "#a9c724"), width=5,
                                                      dash="dash")))
                fig_pv.add_trace(go.Scatter(x=day_profile["hour"], y=net_day, mode="lines",
                                            name="Net import", line=dict(color="black", width=5)))
                fig_pv.update_layout(
                    title=f"PV Matching on Peak Day — {selected_load} | {date_label}",
                    xaxis_title="Hour of Day",
                    yaxis_title="kW",
                    xaxis=dict(dtick=1),
                    height=700,
                )
                st.plotly_chart(fig_pv, use_container_width=True,
                                key=f"pv_match_peak_{st.session_state.get('active_scenario', '')}_{selected_load}")

                # Annual split: load covered by PV vs grid import
                split_df = pd.DataFrame({
                    "Component": ["Covered by PV (self-consumed)", "Grid import"],
                    "kWh": [self_total, import_total]
                })
                split_fig = px.pie(split_df, names="Component", values="kWh",
                                   title="Annual Electricity Supply Split (selected load)")
                st.plotly_chart(split_fig, use_container_width=True,
                                key=f"pv_split_{st.session_state.get('active_scenario', '')}_{selected_load}")

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 5 — Benchmark (Benchmark Tab)
# =========================
with tab5:
    if uploaded_file:
        st.write("## Benchmark")

        # -------------------------
        # Load benchmark thresholds
        # -------------------------
        benchmark_df = load_benchmark_data(building_use)
        if benchmark_df is None:
            st.error(f"Benchmark data not found for building use: {building_use}")
            st.write("Please ensure the benchmark template file exists in the templates folder.")
        else:
            # -------------------------
            # Recompute project KPIs (aligned with other tabs)
            # -------------------------
            df_energy = energy_balance_sheet(uploaded_file.getvalue())
            df_melted = df_energy.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

            # Apply per-End_Use efficiency factors (align with 'Energy Balance with Factors')
            eff_map_bm = {use: st.session_state.get(f"eff_{use}", 1.0) for use in df_melted["End_Use"].unique()}
            df_melted["Efficiency_Factor"] = df_melted["End_Use"].map(eff_map_bm).fillna(1.0)
            df_melted["kWh"] = df_melted["kWh"] / df_melted["Efficiency_Factor"]

            # Map to energy sources (align with user mappings in the sidebar)
            df_melted["Energy_Source"] = df_melted["End_Use"].map(
                {k: st.session_state.get(f"source_{k}", "Electricity") for k in df_melted["End_Use"].unique()}
            )

            # Totals by end use (kWh and intensity)
            totals = df_melted.groupby("End_Use", as_index=False)["kWh"].sum()
            totals["kWh_per_m2"] = (totals["kWh"] / project_area).round(2)

            # Gross vs net (gross = consumption only, net includes on-site generation like PV as negative)
            eui_gross = float(totals.loc[totals["kWh_per_m2"] > 0, "kWh_per_m2"].sum())
            eui_net = float(totals["kWh_per_m2"].sum())

            # CO2 calculations (net accounting)
            factor_map = {
                "Electricity": co2_Emissions_Electricity,
                "Green Electricity": co2_Emissions_Green_Electricity,
                "Gas": co2_emissions_gas,
                "District Heating": co2_emissions_dh,
                "District Cooling": co2_emissions_dc,
                "Biomass": co2_emissions_biomass,
            }
            df_co2 = df_melted.copy()
            df_co2["CO2_factor_kg_per_kWh"] = df_co2["Energy_Source"].map(factor_map).fillna(0.0)
            df_co2["kgCO2"] = df_co2["kWh"] * df_co2["CO2_factor_kg_per_kWh"]
            totals_co2 = df_co2.groupby("End_Use", as_index=False)["kgCO2"].sum()
            totals_co2["kgCO2_per_m2"] = (totals_co2["kgCO2"] / project_area).round(2)

            co2_intensity_gross = float(totals_co2.loc[totals_co2["kgCO2_per_m2"] > 0, "kgCO2_per_m2"].sum())
            co2_intensity_net = float(totals_co2["kgCO2_per_m2"].sum())

            # Cost calculations (net accounting)
            cost_map = {
                "Electricity": cost_electricity,
                "Gas": cost_gas,
                "District Heating": cost_dh,
                "District Cooling": cost_dc,
                "Green Electricity": cost_green_electricity,
                "Biomass": cost_biomass,
            }
            df_cost = df_melted.copy()
            df_cost["cost_per_kWh"] = df_cost["Energy_Source"].map(cost_map).fillna(0.0)
            df_cost["cost"] = df_cost["kWh"] * df_cost["cost_per_kWh"]
            totals_cost = df_cost.groupby("End_Use", as_index=False)["cost"].sum()
            totals_cost["cost_per_m2"] = (totals_cost["cost"] / project_area).round(2)

            cost_intensity_gross = float(totals_cost.loc[totals_cost["cost_per_m2"] > 0, "cost_per_m2"].sum())
            cost_intensity_net = float(totals_cost["cost_per_m2"].sum())

            # -------------------------
            # Benchmark thresholds dict
            # -------------------------
            benchmark_dict = {}
            for _, row in benchmark_df.iterrows():
                kpi_name = row.get("KPI_Name")
                if pd.isna(kpi_name):
                    continue
                benchmark_dict[str(kpi_name)] = {
                    "Good_Threshold": float(row.get("Good_Threshold", float("nan"))),
                    "Excellent_Threshold": float(row.get("Excellent_Threshold", float("nan"))),
                }

            # Use same currency the user selected (fallback to preloaded or €)
            _curr = None
            try:
                _curr = currency_symbol
            except Exception:
                _curr = preloaded.get("currency") if preloaded else None
            if not _curr:
                _curr = "€"

            # -------------------------
            # Header metrics
            # -------------------------
            total_consumption_kwh = float(df_melted.loc[df_melted["kWh"] > 0, "kWh"].sum())
            total_generation_kwh = float(-df_melted.loc[df_melted["kWh"] < 0, "kWh"].sum())
            pv_coverage = (total_generation_kwh / total_consumption_kwh) if total_consumption_kwh > 0 else 0.0
            a1, a2 = st.columns([3,1])
            with a1:
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.metric("Building Use", building_use, help="User input (sidebar)")
                with b2:
                    st.metric("Building Area", f"{project_area:,.0f} m²", help="User input (sidebar)")
                with b3:
                    st.metric("On-site generation share", f"{pv_coverage * 100:.0f} %",
                              help="Derived from negative energy balance entries (e.g., PV)")
                with b3:
                    st.metric("EUI (Net)", f"{eui_net:.1f} kWh/m²·a")
                with b2:
                    st.metric("Energy Cost Intensity (Net)", f"{cost_intensity_net:.1f} €/m²·a")
                with b1:
                    st.metric("CO₂ Intensity (Net)", f"{co2_intensity_net:.1f} kgCO₂/m²·a")

            with a2:
                try:
                    latitude_map = float(latitude)
                    longitude_map = float(longitude)
                    df_map = pd.DataFrame({"lat": [latitude_map], "lon": [longitude_map]})
                    st.metric("Project Location", "", help="User input (sidebar)")
                    st.map(data=df_map, latitude="lat", longitude="lon", height=300, zoom=9)
                except Exception:
                    st.metric("Project Location", "–")
                    st.caption("Latitude/Longitude not available.")

            st.markdown("---")

            # -------------------------
            # KPI benchmark visuals (no more speedometers)
            # -------------------------
            def _benchmark_band_chart(
                title: str,
                unit: str,
                value_net: float,
                value_gross: float,
                good_thr: float,
                excellent_thr: float,
            ) -> go.Figure:
                # Range: extend beyond good threshold for readability
                candidates = [v for v in [value_net, value_gross, good_thr, excellent_thr] if pd.notna(v)]
                xmax = max(candidates) if candidates else max(value_net, value_gross, 1.0)
                xmax = xmax * 1.20 if xmax > 0 else 1.0

                fig = go.Figure()

                # Background bands (Excellent -> Good -> Poor)
                if pd.notna(excellent_thr) and pd.notna(good_thr):
                    fig.add_shape(
                        type="rect", x0=0, x1=excellent_thr, y0=0, y1=1,
                        fillcolor=get_benchmark_color("Excellent"), opacity=0.12, line_width=0
                    )
                    fig.add_shape(
                        type="rect", x0=excellent_thr, x1=good_thr, y0=0, y1=1,
                        fillcolor=get_benchmark_color("Good"), opacity=0.12, line_width=0
                    )
                    fig.add_shape(
                        type="rect", x0=good_thr, x1=xmax, y0=0, y1=1,
                        fillcolor=get_benchmark_color("Poor"), opacity=0.12, line_width=0
                    )
                    # Threshold lines
                    fig.add_vline(x=excellent_thr, line_width=2, line_dash="dot", line_color=get_benchmark_color("Excellent"))
                    fig.add_vline(x=good_thr, line_width=2, line_dash="dot", line_color=get_benchmark_color("Poor"))

                if value_net < excellent_thr:
                    MARKER_NET_COLOR = get_benchmark_color("Excellent")
                elif value_net < good_thr:
                    MARKER_NET_COLOR = get_benchmark_color("Good")
                else:
                    MARKER_NET_COLOR = get_benchmark_color("Poor")

                if value_gross < excellent_thr:
                    MARKER_GROSS_COLOR = get_benchmark_color("Excellent")
                elif value_gross < good_thr:
                    MARKER_GROSS_COLOR = get_benchmark_color("Good")
                else:
                    MARKER_GROSS_COLOR = get_benchmark_color("Poor")

                # Markers for gross / net
                fig.add_trace(go.Scatter(
                    x=[value_gross], y=[0.3],
                    mode="markers",
                    marker=dict(size=40, symbol="square-open", color=MARKER_GROSS_COLOR, line=dict(width=2, color=MARKER_GROSS_COLOR)),
                    name="Gross",
                    hovertemplate=f"Gross: %{{x:.2f}} {unit}<extra></extra>",
                ))

                fig.add_trace(go.Scatter(
                    x=[value_net], y=[0.7],
                    mode="markers",
                    marker=dict(size=40, symbol="square", color=MARKER_NET_COLOR, line=dict(width=2, color=MARKER_NET_COLOR)),
                    name="Net",
                    hovertemplate=f"Net: %{{x:.2f}} {unit}<extra></extra>",
                ))

                fig.update_yaxes(visible=False, range=[0, 1])
                fig.update_xaxes(range=[0, xmax], title_text=unit, zeroline=False)
                fig.update_layout(
                    title=title,
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=10),
                    legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
                )
                return fig

            st.write("## Core benchmark KPIs")

            kpi_specs = [
                dict(
                    template_key="Energy_Density",
                    title="Energy Density (EUI) vs Benchmark",
                    unit="kWh/m²·a",
                    net=eui_net,
                    gross=eui_gross,
                    metric_net_fmt="{:.1f} kWh/m²·a",
                    metric_gross_fmt="{:.1f} kWh/m²·a",
                ),
                dict(
                    template_key="CO2_Emissions",
                    title="Carbon Intensity vs Benchmark",
                    unit="kgCO₂/m²·a",
                    net=co2_intensity_net,
                    gross=co2_intensity_gross,
                    metric_net_fmt="{:.1f} kgCO₂/m²·a",
                    metric_gross_fmt="{:.1f} kgCO₂/m²·a",
                ),
                dict(
                    template_key="Energy_Cost",
                    title="Energy Cost vs Benchmark",
                    unit=f"{_curr}/m²·a",
                    net=cost_intensity_net,
                    gross=cost_intensity_gross,
                    metric_net_fmt=_curr + " {:.2f}/m²·a",
                    metric_gross_fmt=_curr + " {:.2f}/m²·a",
                ),
            ]

            for spec in kpi_specs:
                tkey = spec["template_key"]
                good_thr = benchmark_dict.get(tkey, {}).get("Good_Threshold", float("nan"))
                excellent_thr = benchmark_dict.get(tkey, {}).get("Excellent_Threshold", float("nan"))

                c1, c2 = st.columns([3, 1], gap="large")

                with c1:
                    fig_band = _benchmark_band_chart(
                        title=spec["title"],
                        unit=spec["unit"],
                        value_net=float(spec["net"]),
                        value_gross=float(spec["gross"]),
                        good_thr=good_thr,
                        excellent_thr=excellent_thr,
                    )
                    st.plotly_chart(fig_band, use_container_width=True, key=f"bm_band_{tkey}")

                with c2:
                    if pd.notna(good_thr) and pd.notna(excellent_thr):
                        category = get_benchmark_category(float(spec["net"]), float(good_thr), float(excellent_thr))
                        st.metric("Net", spec["metric_net_fmt"].format(float(spec["net"])))
                        st.metric("Gross", spec["metric_gross_fmt"].format(float(spec["gross"])))
                        st.write("**WS Benchmark**")
                        if category == "Excellent":
                            st.image("Pamo_Icon_Platin.png", width=90)
                            st.write("**Platin**")
                        elif category == "Good":
                            st.image("Pamo_Icon_Green.png", width=90)
                            st.write("**Green**")
                        else:
                            st.image("Pamo_Icon_Gray.png", width=90)
                            st.write("*not Benchmarked*")
                    else:
                        st.metric("Net", spec["metric_net_fmt"].format(float(spec["net"])))
                        st.metric("Gross", spec["metric_gross_fmt"].format(float(spec["gross"])))
                        st.caption("No benchmark thresholds available for this KPI.")

            st.markdown("---")

            # -------------------------
            # Drivers / breakdowns (aligned with other tabs' chart style)
            # -------------------------
            with st.expander(label="Validation Diagrams (under development)",expanded=False):
                st.subheader("Drivers and breakdowns")

                # Energy waterfall: Gross -> On-site generation -> Net
                gen_intensity = eui_net - eui_gross  # negative when generation exists
                fig_water = go.Figure(
                    go.Waterfall(
                        x=["Gross consumption", "On-site generation", "Net (site)"],
                        y=[eui_gross, gen_intensity, eui_net],
                        measure=["relative", "relative", "total"],
                        text=[f"{eui_gross:.1f}", f"{gen_intensity:.1f}", f"{eui_net:.1f}"],
                        textposition="outside",
                    )
                )
                fig_water.update_layout(
                    title="EUI accounting (Gross → Net)",
                    xaxis_title="",
                    yaxis_title="kWh/m²·a",
                    height=380,
                    margin=dict(l=20, r=20, t=60, b=40),
                    showlegend=False,
                )

                # End-use breakdown (kWh/m²·a)
                df_end_use = totals.copy()
                df_end_use = df_end_use.sort_values("kWh_per_m2", ascending=True)

                _enduse_order = df_end_use["End_Use"].tolist()
                _enduse_cmap = {eu: color_map.get(eu, "#999999") for eu in df_end_use["End_Use"].unique()}

                fig_end_use = px.bar(
                    df_end_use,
                    x="kWh_per_m2",
                    y="End_Use",
                    color="End_Use",
                    orientation="h",
                    title="Energy intensity by end use (Net accounting)",
                    color_discrete_map=_enduse_cmap,
                    category_orders={"End_Use": _enduse_order},
                    text_auto=".1f",
                )
                fig_end_use.update_layout(
                    xaxis_title="kWh/m²·a",
                    yaxis_title="",
                    legend_title_text="",
                    height=380,
                    margin=dict(l=10, r=10, t=60, b=40),
                    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                )
                fig_end_use.add_vline(x=0, line_width=1, line_color="#666666")

                a1, a2 = st.columns(2, gap="large")
                with a1:
                    st.plotly_chart(fig_water, use_container_width=True, key="bm_waterfall_eui")
                with a2:
                    st.plotly_chart(fig_end_use, use_container_width=True, key="bm_enduse_energy")

                # Source split (energy & CO2) — handle negative entries explicitly as on-site generation
                df_src = df_melted.copy()
                df_src["Energy_Source_BM"] = df_src.apply(
                    lambda r: "On-site generation" if r["kWh"] < 0 else r["Energy_Source"],
                    axis=1,
                )

                _src_labels = list(pd.unique(df_src["Energy_Source_BM"]))
                _src_cmap = {s: color_map_sources.get(s, color_map.get(s, "#999999")) for s in _src_labels}
                if "On-site generation" in _src_cmap:
                    _src_cmap["On-site generation"] = color_map.get("PV_Generation", CRREM_COLOR_MEASURES)

                src_energy = df_src.groupby("Energy_Source_BM", as_index=False)["kWh"].sum()
                src_energy["kWh_per_m2"] = (src_energy["kWh"] / project_area).round(2)
                src_energy = src_energy.sort_values("kWh_per_m2", ascending=True)

                fig_src_energy = px.bar(
                    src_energy,
                    x="kWh_per_m2",
                    y="Energy_Source_BM",
                    color="Energy_Source_BM",
                    orientation="h",
                    title="Energy intensity by energy source (Net accounting)",
                    color_discrete_map=_src_cmap,
                    category_orders={"Energy_Source_BM": src_energy["Energy_Source_BM"].tolist()},
                    text_auto=".1f",
                )
                fig_src_energy.update_layout(
                    xaxis_title="kWh/m²·a",
                    yaxis_title="",
                    legend_title_text="",
                    height=360,
                    margin=dict(l=10, r=10, t=60, b=40),
                    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                )
                fig_src_energy.add_vline(x=0, line_width=1, line_color="#666666")

                df_src_co2 = df_co2.copy()
                df_src_co2["Energy_Source_BM"] = df_src["Energy_Source_BM"].values
                src_co2 = df_src_co2.groupby("Energy_Source_BM", as_index=False)["kgCO2"].sum()
                src_co2["kgCO2_per_m2"] = (src_co2["kgCO2"] / project_area).round(2)
                src_co2 = src_co2.sort_values("kgCO2_per_m2", ascending=True)

                fig_src_co2 = px.bar(
                    src_co2,
                    x="kgCO2_per_m2",
                    y="Energy_Source_BM",
                    color="Energy_Source_BM",
                    orientation="h",
                    title="CO₂ intensity by energy source (Net accounting)",
                    color_discrete_map=_src_cmap,
                    category_orders={"Energy_Source_BM": src_co2["Energy_Source_BM"].tolist()},
                    text_auto=".1f",
                )
                fig_src_co2.update_layout(
                    xaxis_title="kgCO₂/m²·a",
                    yaxis_title="",
                    legend_title_text="",
                    height=360,
                    margin=dict(l=10, r=10, t=60, b=40),
                    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                )
                fig_src_co2.add_vline(x=0, line_width=1, line_color="#666666")

                b1, b2 = st.columns(2, gap="large")
                with b1:
                    st.plotly_chart(fig_src_energy, use_container_width=True, key="bm_source_energy")
                with b2:
                    st.plotly_chart(fig_src_co2, use_container_width=True, key="bm_source_co2")

                with st.expander("Cost breakdown (Net accounting)", expanded=False):
                    df_src_cost = df_cost.copy()
                    df_src_cost["Energy_Source_BM"] = df_src["Energy_Source_BM"].values
                    src_cost = df_src_cost.groupby("Energy_Source_BM", as_index=False)["cost"].sum()
                    src_cost["cost_per_m2"] = (src_cost["cost"] / project_area).round(2)
                    src_cost = src_cost.sort_values("cost_per_m2", ascending=True)

                    fig_src_cost = px.bar(
                        src_cost,
                        x="cost_per_m2",
                        y="Energy_Source_BM",
                        color="Energy_Source_BM",
                        orientation="h",
                        title="Energy cost by energy source (Net accounting)",
                        color_discrete_map=_src_cmap,
                        category_orders={"Energy_Source_BM": src_cost["Energy_Source_BM"].tolist()},
                        text_auto=".2f",
                    )
                    fig_src_cost.update_layout(
                        xaxis_title=f"{_curr}/m²·a",
                        yaxis_title="",
                        legend_title_text="",
                        height=360,
                        margin=dict(l=10, r=10, t=60, b=40),
                        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                    )
                    fig_src_cost.add_vline(x=0, line_width=1, line_color="#666666")
                    st.plotly_chart(fig_src_cost, use_container_width=True, key="bm_source_cost")

                    # Optional: show raw numbers for transparency
                    st.dataframe(
                        src_cost.rename(columns={"Energy_Source_BM": "Energy Source", "cost_per_m2": f"Cost intensity ({_curr}/m²·a)"}),
                        use_container_width=True,
                        hide_index=True,
                    )

    if not uploaded_file:
        st.write("Please upload the project Excel file to see benchmark results.")
