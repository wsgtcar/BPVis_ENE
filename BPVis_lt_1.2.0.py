import io
from pathlib import Path
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
import re
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
    page_title="WSGT_BPVis_ENE 2.2.16",
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
    "Equipment", "HotWater", "Pumps", "Other", "On-site_Generation"
]
ENERGY_SOURCE_ORDER = ["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling", "Biomass"]

# --- NEW: naming convention (legacy PV -> On-site Generation)
ONSITE_GENERATION_ENDUSE = "On-site_Generation"  # internal (no suffix)
LEGACY_PV_ENDUSE = "PV_Generation"  # legacy internal name from older templates
ONSITE_GENERATION_LABEL = "On-site Generation"  # UI label

# --- UI display name mapping (do NOT change internal keys used for calculations / saving)
UI_NAME_MAP = {
    ONSITE_GENERATION_ENDUSE: ONSITE_GENERATION_LABEL,
    LEGACY_PV_ENDUSE: ONSITE_GENERATION_LABEL,  # legacy display alias
}

# --- NEW: track which End Use(s) represent on-site generation (so NET logic stays correct even if the user renames it)
_ONSITE_ENDUSES_KEY = "_onsite_generation_enduses"


def get_onsite_generation_enduses(enduses=None):
    """Return a list of End Use names that should be treated as on-site generation credits.

    - Primary: user/session defined list in _ONSITE_ENDUSES_KEY
    - Fallbacks (if list not present / not in provided enduses): canonical name, then token-based heuristic.
    """
    try:
        lst = st.session_state.get(_ONSITE_ENDUSES_KEY)
    except Exception:
        lst = None

    if not isinstance(lst, list) or len(lst) == 0:
        lst = [ONSITE_GENERATION_ENDUSE]

    # normalize legacy PV token in stored list
    norm = []
    for x in lst:
        xs = str(x)
        if xs == LEGACY_PV_ENDUSE:
            xs = ONSITE_GENERATION_ENDUSE
        norm.append(xs)

    # If enduses is provided, filter to those present; if none present, try heuristic
    if enduses is not None:
        try:
            end_list = [str(e) for e in list(enduses)]
        except Exception:
            end_list = [str(e) for e in enduses]

        present = [x for x in norm if x in end_list]
        if present:
            return present

        # Canonical name present?
        if ONSITE_GENERATION_ENDUSE in end_list:
            return [ONSITE_GENERATION_ENDUSE]

        # Token-based heuristic (supports renamed columns like PV_Roof, On-site_Gen, etc.)
        try:
            pat = re.compile(r"(pv|on\s*-?site|onsite)", re.IGNORECASE)
            guess = [e for e in end_list if pat.search(str(e))]
            if guess:
                return guess
        except Exception:
            pass

        return []

    return norm


def ui_name(name: str) -> str:
    """Return user-facing label for internal End Use / Load names.

    Rules:
    - Keep internal keys untouched for calculations (this function is for UI only).
    - Apply explicit aliases (e.g., PV_Generation -> On-site Generation).
    - Replace underscores used as word separators with spaces for display.
    """
    s = str(name)
    out = UI_NAME_MAP.get(s, s)
    # Convert internal word separators to UI-friendly spaces
    try:
        out = out.replace("_", " ")
        out = re.sub(r"\s{2,}", " ", out).strip()
    except Exception:
        pass
    return out


def _apply_ui_names_plotly(fig):
    """Mutate Plotly figure so category labels show UI-friendly names (e.g., On-site_Generation -> On-site Generation)."""
    try:
        # Traces (legend + categorical axis arrays)
        for tr in getattr(fig, "data", []) or []:
            try:
                if hasattr(tr, "name") and isinstance(tr.name, str):
                    tr.name = ui_name(tr.name)
            except Exception:
                pass
            try:
                if hasattr(tr, "legendgroup") and isinstance(tr.legendgroup, str):
                    tr.legendgroup = ui_name(tr.legendgroup)
            except Exception:
                pass
            # Map categorical x/y values when they are strings
            try:
                if hasattr(tr, "x") and tr.x is not None:
                    tr.x = [ui_name(v) if isinstance(v, str) else v for v in list(tr.x)]
            except Exception:
                pass
            try:
                if hasattr(tr, "y") and tr.y is not None:
                    tr.y = [ui_name(v) if isinstance(v, str) else v for v in list(tr.y)]
            except Exception:
                pass

            # Pie/Sunburst/Treemap-style categorical labels
            try:
                if hasattr(tr, "labels") and tr.labels is not None:
                    tr.labels = [ui_name(v) if isinstance(v, str) else v for v in list(tr.labels)]
            except Exception:
                pass
            try:
                if hasattr(tr, "ids") and tr.ids is not None:
                    tr.ids = [ui_name(v) if isinstance(v, str) else v for v in list(tr.ids)]
            except Exception:
                pass

        # Layout axis category arrays (if present)
        try:
            for k in list(getattr(fig, "layout", {}).keys()):
                if not (str(k).startswith("xaxis") or str(k).startswith("yaxis")):
                    continue
                ax = getattr(fig.layout, k, None)
                if ax is None:
                    continue
                try:
                    if getattr(ax, "categoryarray", None) is not None:
                        ax.categoryarray = [ui_name(v) if isinstance(v, str) else v for v in list(ax.categoryarray)]
                except Exception:
                    pass
        except Exception:
            pass

    except Exception:
        return fig
    return fig


_ST_PLOTLY_CHART = st.plotly_chart


def st_plotly_chart(*args, **kwargs):
    """Wrapper around st.plotly_chart that applies UI label mapping before rendering."""
    try:
        if args and args[0] is not None:
            fig0 = _apply_ui_names_plotly(args[0])
            args = (fig0,) + tuple(args[1:])
        elif "figure_or_data" in kwargs and kwargs["figure_or_data"] is not None:
            kwargs["figure_or_data"] = _apply_ui_names_plotly(kwargs["figure_or_data"])
    except Exception:
        pass
    return _ST_PLOTLY_CHART(*args, **kwargs)



def _canon_enduse_name(name: str) -> str:
    """Canonicalize legacy PV naming to On-site Generation."""
    n = str(name or "").strip()
    if n.lower() in {LEGACY_PV_ENDUSE.lower(), "pv", "pv_generation", "pv generation"}:
        return ONSITE_GENERATION_ENDUSE
    return n


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
    "On-site_Generation": "#a9c724",
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

# --- NEW: keep immutable defaults for the Color Settings sidebar
DEFAULT_COLOR_MAP = dict(color_map)
DEFAULT_COLOR_MAP_SOURCES = dict(color_map_sources)

# --- NEW: default palette for Loads (used in Loads Analysis and Color Settings)
DEFAULT_COLOR_MAP_LOADS = {k: v for k, v in DEFAULT_COLOR_MAP.items() if k not in set(DEFAULT_COLOR_MAP_SOURCES.keys())}

# --- Scenario color helper (scenario names are dynamic, so defaults are generated by order/name)
def default_scenario_color_map(scenario_names=None):
    out = {}
    try:
        for i, name in enumerate(list(scenario_names or [])):
            out[str(name)] = SCENARIO_COLOR_PALETTE[i % len(SCENARIO_COLOR_PALETTE)]
    except Exception:
        pass
    return out

# --- NEW: ensure a default project name exists before rendering the title
if "project_name" not in st.session_state:
    st.session_state["project_name"] = "Building Performance Dashboard"

# =========================
# Sidebar — template download & file upload
# =========================
st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## BPVis ENE")
st.sidebar.write("Version 2.2.16")

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
    # Canonicalize legacy PV naming
    df_.columns = ["Month" if c == "Month" else _canon_enduse_name(c) for c in df_.columns]
    return df_


def loads_balace_sheet(file_bytes: bytes) -> pd.DataFrame:
    """Load 'Loads_Balance' sheet and strip '_load' suffix from columns."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df_loads = pd.read_excel(xls, sheet_name="Loads_Balance")
    df_loads.columns = [c.removesuffix("_load") for c in df_loads.columns]
    # Canonicalize legacy PV naming
    df_loads.columns = [_canon_enduse_name(c) for c in df_loads.columns]
    return df_loads



# =========================
# NEW — Raw data state (Energy_Balance + Loads_Balance)
# =========================
RAW_SHEET_ENERGY = "Energy_Balance"
RAW_SHEET_LOADS = "Loads_Balance"
_RAW_TOKEN_KEY = "_raw_dfs_workbook_token"
_RAW_ENERGY_KEY = "raw_energy_balance_df"
_RAW_LOADS_KEY = "raw_loads_balance_df"
_RAW_ENERGY_DRAFT_KEY = "raw_energy_balance_df_draft"
_RAW_LOADS_DRAFT_KEY = "raw_loads_balance_df_draft"
_RAW_COMMIT_VERSION_KEY = "_raw_data_commit_version"
_RAW_ENERGY_SCENARIO_OVERRIDES_KEY = "raw_energy_balance_scenario_overrides"
_RAW_ENERGY_SCENARIO_DRAFTS_KEY = "raw_energy_balance_scenario_drafts"
_RAW_ENERGY_SCENARIO_DIRTY_KEY = "raw_energy_balance_scenario_dirty"
RAW_SCENARIO_ENERGY_SHEET = "Energy_Balance_Scenarios"


def _workbook_token(file_bytes: bytes, filename: str = "") -> str:
    try:
        return f"{filename}|{hashlib.md5(file_bytes).hexdigest()}"
    except Exception:
        return f"{filename}|{hash(file_bytes)}"


def sanitize_energy_balance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Energy_Balance is clean: Month as str, other cols numeric (kWh). Canonicalizes legacy PV naming."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    if "Month" not in out.columns:
        out.insert(0, "Month", "")
    out["Month"] = out["Month"].astype(str)
    for c in out.columns:
        if c == "Month":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # Canonicalize legacy PV naming (e.g., PV_Generation -> On-site_Generation)
    rename_map = {}
    for c in out.columns:
        if c == "Month":
            continue
        canon = _canon_enduse_name(c)
        if canon != c:
            rename_map[c] = canon
    if rename_map:
        out = out.rename(columns=rename_map)

    return out


def sanitize_loads_balance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Loads_Balance is clean: weekday as str, other cols numeric (kW). Canonicalizes legacy PV naming."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    if "weekday" in out.columns:
        out["weekday"] = out["weekday"].astype(str)
    for c in out.columns:
        if c == "weekday":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # Canonicalize legacy PV naming (e.g., PV_Generation -> On-site_Generation)
    if LEGACY_PV_ENDUSE in out.columns and ONSITE_GENERATION_ENDUSE not in out.columns:
        out = out.rename(columns={LEGACY_PV_ENDUSE: ONSITE_GENERATION_ENDUSE})

    return out


def _active_scenario_name() -> Optional[str]:
    """Return the currently selected scenario name, if available."""
    try:
        name = st.session_state.get("active_scenario")
        return str(name) if name is not None and str(name).strip() else None
    except Exception:
        return None


def _scenario_energy_overrides() -> Dict[str, pd.DataFrame]:
    """Return the scenario-specific Energy_Balance override dict from session state."""
    overrides = st.session_state.get(_RAW_ENERGY_SCENARIO_OVERRIDES_KEY)
    if not isinstance(overrides, dict):
        overrides = {}
        st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = overrides
    return overrides


def _scenario_energy_drafts() -> Dict[str, pd.DataFrame]:
    """Return scenario-specific Raw Data draft buffers from session state."""
    drafts = st.session_state.get(_RAW_ENERGY_SCENARIO_DRAFTS_KEY)
    if not isinstance(drafts, dict):
        drafts = {}
        st.session_state[_RAW_ENERGY_SCENARIO_DRAFTS_KEY] = drafts
    return drafts


def _scenario_energy_dirty_flags() -> Dict[str, bool]:
    """Return scenario-specific dirty flags for Energy_Balance drafts.

    A dirty draft is a user-edited scenario Energy_Balance that has been captured
    in session state but has not necessarily been committed with Update Data yet.
    Dirty drafts are preserved across scenario switching and included in project export.
    """
    flags = st.session_state.get(_RAW_ENERGY_SCENARIO_DIRTY_KEY)
    if not isinstance(flags, dict):
        flags = {}
        st.session_state[_RAW_ENERGY_SCENARIO_DIRTY_KEY] = flags
    return flags


def _energy_df_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Robust equality check for sanitized Energy_Balance dataframes."""
    try:
        aa = sanitize_energy_balance_df(a).copy()
        bb = sanitize_energy_balance_df(b).copy()
        if set(aa.columns) != set(bb.columns):
            return False
        # Preserve the column order from aa and append any missing columns defensively.
        cols = list(aa.columns)
        bb = bb[cols]
        aa = aa.reset_index(drop=True)
        bb = bb.reset_index(drop=True)
        return aa.equals(bb)
    except Exception:
        return False


def _mark_scenario_energy_draft_dirty(scenario_name: str, is_dirty: bool = True) -> None:
    if scenario_name is None or not str(scenario_name).strip():
        return
    flags = _scenario_energy_dirty_flags()
    if is_dirty:
        flags[str(scenario_name)] = True
    else:
        flags.pop(str(scenario_name), None)
    st.session_state[_RAW_ENERGY_SCENARIO_DIRTY_KEY] = flags


def promote_scenario_energy_drafts_to_overrides(scenario_name: Optional[str] = None, only_dirty: bool = True) -> None:
    """Legacy helper for old captured scenario Energy_Balance drafts.

    The Raw Data editor now commits scenario-specific Energy_Balance overrides only
    when the user clicks **Update Data**. This helper is intentionally not used by
    normal scenario switching, because unsubmitted form edits must not be promoted.
    """
    drafts = _scenario_energy_drafts()
    flags = _scenario_energy_dirty_flags()
    overrides = _scenario_energy_overrides()

    names = [str(scenario_name)] if scenario_name is not None and str(scenario_name).strip() else list(drafts.keys())
    changed = False
    for name in names:
        if name not in drafts or not isinstance(drafts.get(name), pd.DataFrame):
            continue
        if only_dirty and not bool(flags.get(name, False)):
            continue
        overrides[name] = sanitize_energy_balance_df(drafts[name]).copy(deep=True)
        flags.pop(name, None)
        changed = True

    if changed:
        st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = overrides
        st.session_state[_RAW_ENERGY_SCENARIO_DIRTY_KEY] = flags


def effective_scenario_energy_overrides_for_export() -> Dict[str, pd.DataFrame]:
    """Return committed scenario-specific Energy_Balance overrides for workbook export.

    Raw-data table edits are intentionally committed only when the user clicks
    **Update Data**. This keeps the editor from triggering full app reruns on
    every cell edit and prevents unsubmitted browser-side edits from being
    mixed into project export.
    """
    out: Dict[str, pd.DataFrame] = {}
    overrides = _scenario_energy_overrides()
    for name, df in overrides.items():
        if isinstance(df, pd.DataFrame):
            out[str(name)] = sanitize_energy_balance_df(df).copy(deep=True)
    return out

def get_global_energy_balance_df(file_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """Return the global/base Energy_Balance dataframe, ignoring scenario overrides."""
    tok = _workbook_token(file_bytes, filename)
    if st.session_state.get(_RAW_TOKEN_KEY) != tok or _RAW_ENERGY_KEY not in st.session_state:
        try:
            df_ = energy_balance_sheet(file_bytes)
        except Exception:
            df_ = pd.DataFrame()
        st.session_state[_RAW_ENERGY_KEY] = sanitize_energy_balance_df(df_)
    return st.session_state.get(_RAW_ENERGY_KEY, pd.DataFrame())


def get_scenario_energy_balance_override(scenario_name: Optional[str]) -> Optional[pd.DataFrame]:
    """Return a scenario-specific Energy_Balance override if one exists."""
    if scenario_name is None or not str(scenario_name).strip():
        return None
    overrides = _scenario_energy_overrides()
    df = overrides.get(str(scenario_name))
    if isinstance(df, pd.DataFrame):
        return sanitize_energy_balance_df(df)
    return None


def set_scenario_energy_balance_override(scenario_name: str, df: pd.DataFrame) -> None:
    """Commit a scenario-specific Energy_Balance override."""
    if scenario_name is None or not str(scenario_name).strip():
        return
    overrides = _scenario_energy_overrides()
    overrides[str(scenario_name)] = sanitize_energy_balance_df(df)
    st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = overrides


def delete_scenario_energy_balance_override(scenario_name: str) -> None:
    """Remove the active scenario's Energy_Balance override so it falls back to global data."""
    if scenario_name is None or not str(scenario_name).strip():
        return
    overrides = _scenario_energy_overrides()
    drafts = _scenario_energy_drafts()
    dirty = _scenario_energy_dirty_flags()
    overrides.pop(str(scenario_name), None)
    drafts.pop(str(scenario_name), None)
    dirty.pop(str(scenario_name), None)
    st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = overrides
    st.session_state[_RAW_ENERGY_SCENARIO_DRAFTS_KEY] = drafts
    st.session_state[_RAW_ENERGY_SCENARIO_DIRTY_KEY] = dirty


def get_energy_balance_df(
        file_bytes: bytes,
        filename: str = "",
        scenario_name: Optional[str] = None,
        use_scenario_override: bool = True,
) -> pd.DataFrame:
    """Return Energy_Balance data for calculations.

    If a scenario-specific override exists for the requested/current active scenario,
    it is used. Otherwise the global/base Energy_Balance sheet is returned.
    """
    base = get_global_energy_balance_df(file_bytes, filename)
    if not use_scenario_override:
        return base

    sc_name = str(scenario_name) if scenario_name is not None and str(scenario_name).strip() else _active_scenario_name()
    override = get_scenario_energy_balance_override(sc_name)
    if override is not None:
        return override
    return base


def get_loads_balance_df(file_bytes: bytes, filename: str = "") -> pd.DataFrame:
    """Return the (possibly edited) Loads_Balance dataframe (columns without _load)."""
    tok = _workbook_token(file_bytes, filename)
    # keep a single token for both raw dfs
    if st.session_state.get(_RAW_TOKEN_KEY) != tok or _RAW_LOADS_KEY not in st.session_state:
        try:
            df_ = loads_balace_sheet(file_bytes)
        except Exception:
            df_ = pd.DataFrame()
        st.session_state[_RAW_LOADS_KEY] = sanitize_loads_balance_df(df_)
        # ensure the token is set when we successfully (re)seed raw data
        st.session_state[_RAW_TOKEN_KEY] = tok
    return st.session_state.get(_RAW_LOADS_KEY, pd.DataFrame())


def _energy_balance_to_excel_df(df_no_suffix: pd.DataFrame) -> pd.DataFrame:
    """Add _kWh suffix back (except Month) when saving to Excel."""
    df = sanitize_energy_balance_df(df_no_suffix)
    out = df.copy()
    new_cols = []
    for c in out.columns:
        if c == "Month":
            new_cols.append("Month")
        else:
            new_cols.append(f"{c}_kWh" if not str(c).endswith("_kWh") else str(c))
    out.columns = new_cols
    return out


def parse_scenario_energy_overrides_df(df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Parse the optional Energy_Balance_Scenarios sheet into scenario-specific wide dataframes.

    Preferred schema: Scenario | Month | End_Use | kWh. A wide fallback schema with
    Scenario + Month + one column per end use is also supported.
    """
    out: Dict[str, pd.DataFrame] = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "Scenario" not in df.columns:
        return out

    work = df.copy()

    # Long format: Scenario | Month | End_Use | kWh
    if {"Scenario", "Month", "End_Use", "kWh"}.issubset(work.columns):
        work = work.dropna(subset=["Scenario", "Month", "End_Use"])
        if work.empty:
            return out
        work["Scenario"] = work["Scenario"].astype(str)
        work["Month"] = work["Month"].astype(str)
        work["End_Use"] = work["End_Use"].apply(lambda x: _canon_enduse_name(str(x)))
        work["kWh"] = pd.to_numeric(work["kWh"], errors="coerce").fillna(0.0)
        for sc_name, group in work.groupby("Scenario"):
            if not str(sc_name).strip():
                continue
            wide = group.pivot_table(index="Month", columns="End_Use", values="kWh", aggfunc="sum").reset_index()
            # Preserve standard month order where possible.
            try:
                wide["Month"] = pd.Categorical(wide["Month"].astype(str), categories=MONTH_ORDER, ordered=True)
                wide = wide.sort_values("Month", kind="stable")
                wide["Month"] = wide["Month"].astype(str)
            except Exception:
                pass
            out[str(sc_name)] = sanitize_energy_balance_df(wide)
        return out

    # Wide fallback: Scenario | Month | <End Use 1> | <End Use 2> ...
    if "Month" in work.columns:
        for sc_name, group in work.groupby(work["Scenario"].astype(str)):
            if not str(sc_name).strip():
                continue
            wide = group.drop(columns=["Scenario"], errors="ignore").copy()
            out[str(sc_name)] = sanitize_energy_balance_df(wide)
    return out


def build_scenario_energy_overrides_df(overrides: Optional[Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Build a human-readable long sheet for scenario-specific Energy_Balance overrides."""
    rows = []
    if not isinstance(overrides, dict):
        return pd.DataFrame(columns=["Scenario", "Month", "End_Use", "kWh"])

    for sc_name, df in overrides.items():
        if not str(sc_name).strip() or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        clean = sanitize_energy_balance_df(df)
        if "Month" not in clean.columns:
            continue
        long_df = clean.melt(id_vars="Month", var_name="End_Use", value_name="kWh")
        for _, r in long_df.iterrows():
            rows.append({
                "Scenario": str(sc_name),
                "Month": str(r.get("Month", "")),
                "End_Use": _canon_enduse_name(str(r.get("End_Use", ""))),
                "kWh": _to_float_lcc(r.get("kWh"), 0.0) if "_to_float_lcc" in globals() else pd.to_numeric(r.get("kWh"), errors="coerce"),
            })
    return pd.DataFrame(rows, columns=["Scenario", "Month", "End_Use", "kWh"])


def _loads_balance_to_excel_df(df_no_suffix: pd.DataFrame) -> pd.DataFrame:
    """Add _load suffix back for load columns when saving to Excel."""
    df = sanitize_loads_balance_df(df_no_suffix)
    out = df.copy()
    meta_cols = {"hoy", "doy", "day", "month", "weekday", "hour", "Grid_Injection"}
    new_cols = []
    for c in out.columns:
        c_str = str(c)
        if c_str in meta_cols:
            new_cols.append(c_str)
        else:
            new_cols.append(f"{c_str}_load" if not c_str.endswith("_load") else c_str)
    out.columns = new_cols
    return out

# =========================
# NEW — Configuration I/O helpers (Save/Load Project settings)
# =========================
SHEET_PROJECT = "Project_Data"
SHEET_FACTORS = "Emission_Factors"
SHEET_TARIFFS = "Energy_Tariffs"
SHEET_MAPPING = "EndUse_to_Source"
SHEET_EFFICIENCY = "Efficiency_Factors"
SHEET_SCENARIOS = "Scenarios"
SHEET_COLORS = "Color_Settings"
SHEET_LCC_GLOBAL = "LCC_Global"
SHEET_LCC_INVESTMENTS = "LCC_Investments"
SHEET_RAW_ENERGY_SCENARIOS = RAW_SCENARIO_ENERGY_SHEET
SHEET_MODEL_INPUTS_QA = "Model_Inputs_QA"


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
        "colors": sheets.get(SHEET_COLORS),
        "lcc_global": sheets.get(SHEET_LCC_GLOBAL),
        "lcc_investments": sheets.get(SHEET_LCC_INVESTMENTS),
        "scenario_energy": sheets.get(SHEET_RAW_ENERGY_SCENARIOS),
        "model_inputs": sheets.get(SHEET_MODEL_INPUTS_QA),
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
            out[_canon_enduse_name(eu)] = es
    return out


def parse_efficiency_df(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    out = {}
    if df is not None and {"End_Use", "Efficiency_Factor"}.issubset(df.columns):
        for _, row in df.iterrows():
            eu = str(row["End_Use"])
            try:
                out[_canon_enduse_name(eu)] = float(row["Efficiency_Factor"])
            except Exception:
                pass
    return out



def parse_color_settings_df(df: Optional[pd.DataFrame]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Parse Color_Settings sheet into (End_Use colors, Energy_Source colors, Load colors, Scenario colors)."""
    end_use_map: Dict[str, str] = {}
    source_map: Dict[str, str] = {}
    load_map: Dict[str, str] = {}
    scenario_map: Dict[str, str] = {}

    if df is None or df.empty:
        return end_use_map, source_map, load_map, scenario_map

    def _clean_color(col):
        col = str(col or "").strip()
        if not col:
            return ""
        if not col.startswith("#"):
            col = f"#{col}"
        return col

    # Preferred schema: Type | Name | Color
    if {"Type", "Name", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            typ = str(row.get("Type", "")).strip()
            name = str(row.get("Name", "")).strip()
            col = _clean_color(row.get("Color", ""))
            if not name or not col:
                continue
            typ_l = typ.lower()

            if typ_l in ["end_use", "end use", "enduse", "end-use"]:
                end_use_map[_canon_enduse_name(name)] = col
            elif typ_l in ["energy_source", "energy source", "energysource", "energy-source", "source"]:
                source_map[name] = col
            elif typ_l in ["load", "loads"]:
                load_map[_canon_enduse_name(name)] = col
            elif typ_l in ["scenario", "scenarios"]:
                scenario_map[name] = col

        return end_use_map, source_map, load_map, scenario_map

    # Fallback schema(s)
    if {"End_Use", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("End_Use", "")).strip()
            col = _clean_color(row.get("Color", ""))
            if name and col:
                end_use_map[_canon_enduse_name(name)] = col

    if {"Energy_Source", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("Energy_Source", "")).strip()
            col = _clean_color(row.get("Color", ""))
            if name and col:
                source_map[name] = col

    if {"Load", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("Load", "")).strip()
            col = _clean_color(row.get("Color", ""))
            if name and col:
                load_map[_canon_enduse_name(name)] = col

    if {"Scenario", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("Scenario", "")).strip()
            col = _clean_color(row.get("Color", ""))
            if name and col:
                scenario_map[name] = col

    return end_use_map, source_map, load_map, scenario_map


def build_color_settings_df(
        color_map_end_use: Dict[str, str],
        color_map_sources_in: Dict[str, str],
        color_map_loads_in: Optional[Dict[str, str]] = None,
        color_map_scenarios_in: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Build Color_Settings sheet from the current color maps."""
    rows = []
    for k, v in (color_map_end_use or {}).items():
        rows.append({"Type": "End_Use", "Name": str(k), "Color": str(v)})
    for k, v in (color_map_sources_in or {}).items():
        rows.append({"Type": "Energy_Source", "Name": str(k), "Color": str(v)})
    for k, v in (color_map_loads_in or {}).items():
        rows.append({"Type": "Load", "Name": str(k), "Color": str(v)})
    for k, v in (color_map_scenarios_in or {}).items():
        rows.append({"Type": "Scenario", "Name": str(k), "Color": str(v)})
    return pd.DataFrame(rows)



# =========================
# NEW — Scenario Manager helpers
# =========================

def _canon_scenario_payload(payload: dict) -> dict:
    """Canonicalize scenario payload keys so legacy PV naming still works."""
    if not isinstance(payload, dict):
        return {}
    for section in ("mapping", "efficiency"):
        d = payload.get(section)
        if isinstance(d, dict):
            payload[section] = {_canon_enduse_name(k): v for k, v in d.items()}

    # Canonicalize any measure parameter labels that reference the legacy end-use name.
    measures = payload.get("crrem_measures")
    if isinstance(measures, list):
        for rec in measures:
            if isinstance(rec, dict):
                p = rec.get("Parameter")
                if isinstance(p, str):
                    p_s = p.strip()
                    # Legacy PV annual production label -> current on-site generation label
                    if p_s == "PV_Generation → PV Annual Production (kWh/a)":
                        rec["Parameter"] = "On-site_Generation → Annual Production (kWh/a)"
                        continue
                    # Generic legacy end-use name replacement (keep other text)
                    if LEGACY_PV_ENDUSE in p_s and ONSITE_GENERATION_ENDUSE not in p_s:
                        p_s = p_s.replace(LEGACY_PV_ENDUSE, ONSITE_GENERATION_ENDUSE)
                    # If the legacy label text remains, update it too
                    if "PV Annual Production" in p_s:
                        p_s = p_s.replace("PV Annual Production", "On-site Generation Annual Production")
                    rec["Parameter"] = p_s
    # Canonicalize LCC records and selected operational end uses.
    lcc = payload.get("lcc")
    if isinstance(lcc, dict):
        selected = lcc.get("selected_operational_end_uses")
        if isinstance(selected, list):
            lcc["selected_operational_end_uses"] = [_canon_enduse_name(str(x)) for x in selected if str(x).strip()]
        investments = lcc.get("investments")
        if isinstance(investments, list):
            for rec in investments:
                if isinstance(rec, dict):
                    if "Assigned End Uses" in rec:
                        vals = re.split(r"[,;|/\n]+", str(rec.get("Assigned End Uses", "")))
                        rec["Assigned End Uses"] = ", ".join([_canon_enduse_name(str(x).strip()) for x in vals if str(x).strip()])
                    elif "Assigned End Use" in rec:
                        rec["Assigned End Uses"] = _canon_enduse_name(str(rec.get("Assigned End Use", "")))
                        rec.pop("Assigned End Use", None)
        payload["lcc"] = lcc

    lcc_global = payload.get("lcc_global")
    if isinstance(lcc_global, dict):
        selected = lcc_global.get("selected_operational_end_uses")
        if isinstance(selected, list):
            lcc_global["selected_operational_end_uses"] = [_canon_enduse_name(str(x)) for x in selected if str(x).strip()]
        payload["lcc_global"] = lcc_global

    return payload


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
        payload = _canon_scenario_payload(payload)
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



# =========================
# LCC — scenario-specific helpers
# =========================
LCC_INVESTMENT_COLUMNS = [
    "Measure Name",
    "Assigned End Uses",
    "Investment Year",
    "Investment Cost",
    "Annual Maintenance Cost (%)",
    "Life Length (years)",
]

LCC_COST_TYPE_COLORS = {
    "Energy": CRREM_COLOR_BASELINE,
    "Investment": CRREM_COLOR_LIMIT,
    "Maintenance": "#d3b402",
    "Replacement": "#833fd1",
}

# Global LCC assumptions are stored once in session state and then duplicated into
# each scenario payload only for workbook persistence/backwards compatibility.
LCC_GLOBAL_STATE_KEY = "lcc_global_payload"
LCC_GLOBAL_DRAFT_PREFIX = "lcc_draft_"



def _safe_state_key(text: str) -> str:
    """Return a stable Streamlit-safe key fragment."""
    try:
        return re.sub(r"[^0-9A-Za-z_]+", "_", str(text)).strip("_")[:80]
    except Exception:
        return "key"


def _to_float_lcc(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        s = str(x).strip().replace("%", "").replace(" ", "").replace(",", ".")
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _to_int_lcc(x, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return int(default)
        s = str(x).strip().replace(",", ".")
        if s == "":
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)


def _lcc_default_selected_enduses(end_uses: list) -> list:
    try:
        onsite = set(get_onsite_generation_enduses(end_uses)) | {ONSITE_GENERATION_ENDUSE, LEGACY_PV_ENDUSE}
        out = [str(u) for u in end_uses if str(u) not in onsite]
        return out if out else [str(u) for u in end_uses]
    except Exception:
        return [str(u) for u in end_uses]


def _default_lcc_global_payload(end_uses: list) -> dict:
    """Default LCC assumptions that apply equally to all scenarios."""
    return {
        "analysis_period": 30,
        "interest_rate_pct": 4.0,
        "capex_inflation_pct": 2.0,
        "energy_inflation_pct": {src: 2.0 for src in ENERGY_SOURCE_ORDER},
        "selected_operational_end_uses": _lcc_default_selected_enduses(end_uses),
        "payback_reference_scenario": "",
    }


def _default_lcc_payload(end_uses: list) -> dict:
    """Default scenario-specific LCC configuration. Only investment measures are scenario-specific."""
    return {
        "investments": [],
    }


def _normalize_lcc_global_payload(lcc_global_payload, end_uses: list) -> dict:
    """Merge saved global LCC assumptions with defaults and sanitize references to current end uses.

    Backwards compatible: this also accepts the old scenario-specific `lcc` payload shape and extracts
    the global settings from it.
    """
    defaults = _default_lcc_global_payload(end_uses)
    if not isinstance(lcc_global_payload, dict):
        return defaults

    out = deepcopy(defaults)
    out["analysis_period"] = max(1, _to_int_lcc(lcc_global_payload.get("analysis_period"), defaults["analysis_period"]))
    out["interest_rate_pct"] = _to_float_lcc(lcc_global_payload.get("interest_rate_pct"), defaults["interest_rate_pct"])
    out["capex_inflation_pct"] = _to_float_lcc(lcc_global_payload.get("capex_inflation_pct"), defaults["capex_inflation_pct"])

    energy_inf = lcc_global_payload.get("energy_inflation_pct", {})
    if not isinstance(energy_inf, dict):
        energy_inf = {}
    out["energy_inflation_pct"] = {
        src: _to_float_lcc(energy_inf.get(src), defaults["energy_inflation_pct"].get(src, 2.0))
        for src in ENERGY_SOURCE_ORDER
    }

    selected = lcc_global_payload.get("selected_operational_end_uses", defaults["selected_operational_end_uses"])
    if not isinstance(selected, list):
        selected = defaults["selected_operational_end_uses"]
    enduse_set = {str(u) for u in end_uses}
    selected = [_canon_enduse_name(str(u)) for u in selected if str(u).strip()]
    selected = [u for u in selected if u in enduse_set]
    out["selected_operational_end_uses"] = selected if selected else defaults["selected_operational_end_uses"]

    out["payback_reference_scenario"] = str(lcc_global_payload.get("payback_reference_scenario", "") or "")
    return out


def _normalize_lcc_payload(lcc_payload, end_uses: list) -> dict:
    """Merge saved scenario-specific LCC payload with defaults.

    Global assumptions are intentionally ignored here and handled by `_normalize_lcc_global_payload`,
    so switching scenarios cannot overwrite global LCC parameters.
    """
    defaults = _default_lcc_payload(end_uses)
    if not isinstance(lcc_payload, dict):
        return defaults

    out = deepcopy(defaults)
    out["investments"] = _lcc_investments_df_to_records(
        _lcc_investments_records_to_df(lcc_payload.get("investments", []), end_uses=end_uses)
    )
    return out


def _lcc_parse_assigned_enduses(value, end_uses: Optional[list] = None) -> list:
    """Parse one or multiple assigned end uses from a text/list cell.

    The data editor uses a text field because Streamlit does not reliably provide a per-cell multiselect
    column across versions. Accepted separators: comma, semicolon, pipe, slash, or newline.
    """
    valid = [str(u) for u in (end_uses or [])]
    valid_set = set(valid)

    if isinstance(value, list):
        raw_items = value
    else:
        raw = "" if value is None else str(value)
        # Support legacy single end-use strings and user-entered multi-use strings.
        raw_items = re.split(r"[,;|/\n]+", raw)

    out = []
    for item in raw_items:
        item_s = _canon_enduse_name(str(item).strip())
        if not item_s:
            continue
        # Exact match first, then case-insensitive match to make manual typing more forgiving.
        if valid_set and item_s not in valid_set:
            match = next((v for v in valid if v.lower() == item_s.lower()), None)
            if not match:
                match = next((v for v in valid if ui_name(v).lower() == item_s.replace("_", " ").lower()), None)
            if match:
                item_s = match
            else:
                continue
        if item_s not in out:
            out.append(item_s)

    if not out and valid:
        defaults = _lcc_default_selected_enduses(valid)
        out = [defaults[0]] if defaults else [valid[0]]
    return out


def _lcc_format_assigned_enduses(value, end_uses: Optional[list] = None) -> str:
    """Return assigned end uses as a stable comma-separated string for the data editor."""
    parsed = _lcc_parse_assigned_enduses(value, end_uses=end_uses)
    return ", ".join(parsed)


def _lcc_investments_records_to_df(records, end_uses: Optional[list] = None) -> pd.DataFrame:
    """Convert saved LCC investment records to a stable dataframe."""
    try:
        if records is None:
            df = pd.DataFrame(columns=LCC_INVESTMENT_COLUMNS)
        elif isinstance(records, pd.DataFrame):
            df = records.copy()
        else:
            df = pd.DataFrame(list(records))
    except Exception:
        df = pd.DataFrame(columns=LCC_INVESTMENT_COLUMNS)

    # Backwards compatibility for v1.4.0 payloads with singular Assigned End Use.
    if "Assigned End Uses" not in df.columns and "Assigned End Use" in df.columns:
        df["Assigned End Uses"] = df["Assigned End Use"]

    for c in LCC_INVESTMENT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[LCC_INVESTMENT_COLUMNS].copy()

    default_enduses = _lcc_default_selected_enduses(end_uses or []) if end_uses else []
    default_assigned = default_enduses[0] if default_enduses else ((end_uses or [""])[0] if end_uses else "")

    df["Measure Name"] = df["Measure Name"].fillna("").astype(str)
    df["Assigned End Uses"] = df["Assigned End Uses"].apply(
        lambda x: _lcc_format_assigned_enduses(x if x is not None and str(x).strip() else default_assigned, end_uses=end_uses)
    )
    df["Investment Year"] = df["Investment Year"].apply(lambda x: _to_int_lcc(x, 0))
    df["Investment Cost"] = df["Investment Cost"].apply(lambda x: _to_float_lcc(x, 0.0))
    df["Annual Maintenance Cost (%)"] = df["Annual Maintenance Cost (%)"].apply(lambda x: _to_float_lcc(x, 0.0))
    df["Life Length (years)"] = df["Life Length (years)"].apply(lambda x: _to_int_lcc(x, 0))

    # Drop fully empty rows, but keep rows with a name even if costs are temporarily zero while editing.
    keep = (
        df["Measure Name"].astype(str).str.strip().ne("") |
        df["Investment Cost"].astype(float).ne(0.0) |
        df["Annual Maintenance Cost (%)"].astype(float).ne(0.0)
    )
    return df.loc[keep].reset_index(drop=True)


def _lcc_investments_df_to_records(df) -> list:
    """Convert the LCC investments dataframe to JSON-serializable records."""
    df = _lcc_investments_records_to_df(df)
    records = []
    try:
        for _, r in df.iterrows():
            name = str(r.get("Measure Name", "")).strip()
            assigned = _lcc_format_assigned_enduses(r.get("Assigned End Uses", ""))
            inv_cost = _to_float_lcc(r.get("Investment Cost"), 0.0)
            maint_pct = _to_float_lcc(r.get("Annual Maintenance Cost (%)"), 0.0)
            if not name and inv_cost == 0.0 and maint_pct == 0.0:
                continue
            records.append({
                "Measure Name": name,
                "Assigned End Uses": assigned,
                "Investment Year": _to_int_lcc(r.get("Investment Year"), 0),
                "Investment Cost": inv_cost,
                "Annual Maintenance Cost (%)": maint_pct,
                "Life Length (years)": max(0, _to_int_lcc(r.get("Life Length (years)"), 0)),
            })
    except Exception:
        return []
    return records


def _lcc_global_draft_key(name: str) -> str:
    """Return the session-state key used for uncommitted/global LCC form values."""
    return f"{LCC_GLOBAL_DRAFT_PREFIX}{name}"


def _lcc_energy_inflation_draft_key(src: str) -> str:
    return _lcc_global_draft_key(f"energy_inflation_pct_{_safe_state_key(src)}")


def _seed_lcc_global_draft_from_payload(lcc_global_payload: dict, end_uses: list, force: bool = False) -> None:
    """Seed the global LCC form draft keys from a committed payload.

    The draft keys are widget-bound inside the LCC form. This function is called before
    those widgets are rendered. When force=False, existing draft values are preserved so
    scenario switching cannot reset unsubmitted LCC edits.
    """
    valid = [str(u) for u in end_uses]
    lcc_global = _normalize_lcc_global_payload(lcc_global_payload, valid)

    key = _lcc_global_draft_key("analysis_period")
    if force or key not in st.session_state:
        st.session_state[key] = int(lcc_global.get("analysis_period", 30))

    for short_key, payload_key, default_val in [
        ("interest_rate_pct", "interest_rate_pct", 4.0),
        ("capex_inflation_pct", "capex_inflation_pct", 2.0),
    ]:
        key = _lcc_global_draft_key(short_key)
        if force or key not in st.session_state:
            val = float(lcc_global.get(payload_key, default_val))
            st.session_state[key] = val
            st.session_state[f"{key}_txt"] = f"{val:.4f}"
        elif f"{key}_txt" not in st.session_state:
            st.session_state[f"{key}_txt"] = f"{_to_float_lcc(st.session_state.get(key), default_val):.4f}"

    energy_inf = lcc_global.get("energy_inflation_pct", {}) or {}
    for src in ENERGY_SOURCE_ORDER:
        key = _lcc_energy_inflation_draft_key(src)
        if force or key not in st.session_state:
            val = float(energy_inf.get(src, 2.0))
            st.session_state[key] = val
            st.session_state[f"{key}_txt"] = f"{val:.4f}"
        elif f"{key}_txt" not in st.session_state:
            st.session_state[f"{key}_txt"] = f"{_to_float_lcc(st.session_state.get(key), 2.0):.4f}"

    selected_key = _lcc_global_draft_key("selected_operational_end_uses")
    if force or selected_key not in st.session_state:
        selected = lcc_global.get("selected_operational_end_uses", _lcc_default_selected_enduses(valid))
        selected = [_canon_enduse_name(str(u)) for u in selected if _canon_enduse_name(str(u)) in set(valid)]
        st.session_state[selected_key] = selected or _lcc_default_selected_enduses(valid)
    else:
        # Keep existing draft filter, but remove end uses that no longer exist in the uploaded data.
        selected = st.session_state.get(selected_key, [])
        if not isinstance(selected, list):
            selected = []
        selected = [_canon_enduse_name(str(u)) for u in selected if _canon_enduse_name(str(u)) in set(valid)]
        if not selected:
            selected = _lcc_default_selected_enduses(valid)
        st.session_state[selected_key] = selected

    ref_key = _lcc_global_draft_key("payback_reference_scenario")
    if force or ref_key not in st.session_state:
        st.session_state[ref_key] = str(lcc_global.get("payback_reference_scenario", "") or "")


def _capture_lcc_global_from_draft_widgets(end_uses: list) -> dict:
    """Capture the LCC global form draft values. Used only when the user submits the LCC form."""
    defaults = _default_lcc_global_payload(end_uses)
    selected = st.session_state.get(_lcc_global_draft_key("selected_operational_end_uses"), defaults["selected_operational_end_uses"])
    if not isinstance(selected, list):
        selected = defaults["selected_operational_end_uses"]

    payload = {
        "analysis_period": max(1, _to_int_lcc(st.session_state.get(_lcc_global_draft_key("analysis_period"), defaults["analysis_period"]), defaults["analysis_period"])),
        "interest_rate_pct": _to_float_lcc(st.session_state.get(_lcc_global_draft_key("interest_rate_pct"), defaults["interest_rate_pct"]), defaults["interest_rate_pct"]),
        "capex_inflation_pct": _to_float_lcc(st.session_state.get(_lcc_global_draft_key("capex_inflation_pct"), defaults["capex_inflation_pct"]), defaults["capex_inflation_pct"]),
        "energy_inflation_pct": {
            src: _to_float_lcc(
                st.session_state.get(_lcc_energy_inflation_draft_key(src), defaults["energy_inflation_pct"].get(src, 2.0)),
                defaults["energy_inflation_pct"].get(src, 2.0),
            )
            for src in ENERGY_SOURCE_ORDER
        },
        "selected_operational_end_uses": [_canon_enduse_name(str(u)) for u in selected if str(u).strip()],
        "payback_reference_scenario": str(st.session_state.get(_lcc_global_draft_key("payback_reference_scenario"), "") or ""),
    }
    return _normalize_lcc_global_payload(payload, end_uses)


def _capture_lcc_global_from_widgets(end_uses: list) -> dict:
    """Backward-compatible alias for older internal calls. Captures the LCC form draft."""
    return _capture_lcc_global_from_draft_widgets(end_uses)


def _get_lcc_global_state_payload(end_uses: list) -> dict:
    """Return the committed global LCC payload used for every scenario calculation.

    This function deliberately does not read the live/draft form widget keys. Draft values
    become active only when the user clicks the LCC update button.
    """
    saved = st.session_state.get(LCC_GLOBAL_STATE_KEY)

    if not isinstance(saved, dict):
        scenarios = st.session_state.get("scenarios", {})
        active_name = st.session_state.get("active_scenario")
        if isinstance(scenarios, dict):
            active_payload = scenarios.get(active_name, {}) if active_name else {}
            if isinstance(active_payload, dict) and isinstance(active_payload.get("lcc_global"), dict):
                saved = active_payload.get("lcc_global")
            if not isinstance(saved, dict):
                for payload_i in scenarios.values():
                    if isinstance(payload_i, dict) and isinstance(payload_i.get("lcc_global"), dict):
                        saved = payload_i.get("lcc_global")
                        break
            if not isinstance(saved, dict):
                for payload_i in scenarios.values():
                    if isinstance(payload_i, dict) and isinstance(payload_i.get("lcc"), dict):
                        lcc_i = payload_i.get("lcc")
                        if any(k in lcc_i for k in ["analysis_period", "selected_operational_end_uses", "energy_inflation_pct"]):
                            saved = lcc_i
                            break

    payload = _normalize_lcc_global_payload(saved, end_uses) if isinstance(saved, dict) else _default_lcc_global_payload(end_uses)
    st.session_state[LCC_GLOBAL_STATE_KEY] = deepcopy(payload)
    return payload


def _capture_lcc_from_widgets(end_uses: list) -> dict:
    """Capture current scenario-specific LCC investment assumptions."""
    return {
        "investments": _lcc_investments_df_to_records(
            st.session_state.get("lcc_investments_df", pd.DataFrame(columns=LCC_INVESTMENT_COLUMNS))
        ),
    }


def _load_lcc_global_into_widgets(lcc_global_payload: dict, end_uses: list) -> None:
    """Initialize committed global LCC state and seed the draft form keys."""
    lcc_global = _normalize_lcc_global_payload(lcc_global_payload, end_uses)
    st.session_state[LCC_GLOBAL_STATE_KEY] = deepcopy(lcc_global)
    st.session_state["_lcc_global_initialized"] = True
    _seed_lcc_global_draft_from_payload(lcc_global, end_uses, force=True)


def _ensure_lcc_global_state(end_uses: list, scenarios: Optional[dict] = None, active_payload: Optional[dict] = None) -> None:
    """Initialize committed global LCC state once and preserve draft keys across scenario switches."""
    if st.session_state.get("_lcc_global_initialized") and isinstance(st.session_state.get(LCC_GLOBAL_STATE_KEY), dict):
        committed = _get_lcc_global_state_payload(end_uses)
        _seed_lcc_global_draft_from_payload(committed, end_uses, force=False)
        return

    source = None
    if isinstance(active_payload, dict) and isinstance(active_payload.get("lcc_global"), dict):
        source = active_payload.get("lcc_global")
    elif isinstance(active_payload, dict) and isinstance(active_payload.get("lcc"), dict):
        source = active_payload.get("lcc")

    if source is None and isinstance(scenarios, dict):
        for payload in scenarios.values():
            if isinstance(payload, dict) and isinstance(payload.get("lcc_global"), dict):
                source = payload.get("lcc_global")
                break
        if source is None:
            for payload in scenarios.values():
                if isinstance(payload, dict) and isinstance(payload.get("lcc"), dict):
                    source = payload.get("lcc")
                    break

    _load_lcc_global_into_widgets(source or _default_lcc_global_payload(end_uses), end_uses)


def _load_lcc_into_widgets(payload: dict, end_uses: list) -> None:
    """Seed scenario-specific LCC investment state from a scenario payload."""
    lcc = _normalize_lcc_payload((payload or {}).get("lcc", {}), end_uses)
    inv_df = _lcc_investments_records_to_df(lcc.get("investments", []), end_uses=end_uses)
    st.session_state["lcc_investments_df"] = inv_df
    st.session_state["lcc_investments_draft_df"] = inv_df.copy(deep=True)


def _apply_lcc_global_to_all_scenarios(end_uses: list) -> None:
    """Persist the committed global LCC assumptions into every scenario payload for workbook save/load."""
    try:
        scenarios = st.session_state.get("scenarios", {})
        if not isinstance(scenarios, dict):
            return
        lcc_global = _get_lcc_global_state_payload(end_uses)
        for name, payload in list(scenarios.items()):
            if not isinstance(payload, dict):
                payload = {}
            payload["lcc_global"] = deepcopy(lcc_global)
            payload["lcc"] = _normalize_lcc_payload(payload.get("lcc", {}), end_uses)
            scenarios[name] = payload
        st.session_state["scenarios"] = scenarios
    except Exception:
        pass


def _sync_lcc_global_widget_state(end_uses: list) -> None:
    """Keep global LCC draft keys valid without committing them.

    This is used before export and scenario captures. The committed global payload remains
    the single source of truth until the user submits the LCC form.
    """
    committed = _get_lcc_global_state_payload(end_uses)
    _seed_lcc_global_draft_from_payload(committed, end_uses, force=False)


def parse_lcc_global_df(df: Optional[pd.DataFrame], end_uses: list) -> Optional[dict]:
    """Parse the human-readable LCC_Global sheet into the global LCC payload."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if not {"Key", "Value"}.issubset(df.columns):
        return None

    kv = dict(zip(df["Key"].astype(str), df["Value"]))
    energy_inf = {}
    for src in ENERGY_SOURCE_ORDER:
        for key in [f"Energy_Inflation_{src}", f"Energy Inflation {src}", f"{src} Inflation"]:
            if key in kv:
                energy_inf[src] = _to_float_lcc(kv.get(key), 2.0)
                break

    selected_raw = kv.get("Selected_Operational_End_Uses", kv.get("Selected Operational End Uses", ""))
    if isinstance(selected_raw, str):
        selected = _lcc_parse_assigned_enduses(selected_raw, end_uses=end_uses)
    elif isinstance(selected_raw, list):
        selected = _lcc_parse_assigned_enduses(selected_raw, end_uses=end_uses)
    else:
        selected = _lcc_default_selected_enduses(end_uses)

    payload = {
        "analysis_period": _to_int_lcc(kv.get("Analysis_Period", kv.get("Analysis Period")), 30),
        "interest_rate_pct": _to_float_lcc(kv.get("Interest_Rate_Pct", kv.get("Interest Rate (%)")), 4.0),
        "capex_inflation_pct": _to_float_lcc(kv.get("CAPEX_Inflation_Pct", kv.get("CAPEX Inflation (%)")), 2.0),
        "energy_inflation_pct": energy_inf,
        "selected_operational_end_uses": selected,
        "payback_reference_scenario": str(kv.get("Payback_Reference_Scenario", kv.get("Payback Reference Scenario", "")) or ""),
    }
    return _normalize_lcc_global_payload(payload, end_uses)


def build_lcc_global_df(lcc_global: dict, end_uses: list) -> pd.DataFrame:
    """Build the human-readable LCC_Global sheet from the current global LCC payload."""
    payload = _normalize_lcc_global_payload(lcc_global, end_uses)
    rows = [
        {"Key": "Analysis_Period", "Value": int(payload.get("analysis_period", 30))},
        {"Key": "Interest_Rate_Pct", "Value": float(payload.get("interest_rate_pct", 4.0))},
        {"Key": "CAPEX_Inflation_Pct", "Value": float(payload.get("capex_inflation_pct", 2.0))},
        {"Key": "Selected_Operational_End_Uses", "Value": ", ".join(payload.get("selected_operational_end_uses", []))},
        {"Key": "Payback_Reference_Scenario", "Value": str(payload.get("payback_reference_scenario", "") or "")},
    ]
    energy_inf = payload.get("energy_inflation_pct", {}) or {}
    for src in ENERGY_SOURCE_ORDER:
        rows.append({"Key": f"Energy_Inflation_{src}", "Value": float(energy_inf.get(src, 2.0))})
    return pd.DataFrame(rows)


def parse_lcc_investments_sheet(df: Optional[pd.DataFrame], end_uses: list) -> Dict[str, list]:
    """Parse the human-readable LCC_Investments sheet into scenario -> records."""
    out: Dict[str, list] = {}
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "Scenario" not in df.columns:
        return out
    work = df.copy()
    if "Assigned End Uses" not in work.columns and "Assigned End Use" in work.columns:
        work["Assigned End Uses"] = work["Assigned End Use"]
    for sc_name, group in work.groupby(work["Scenario"].astype(str)):
        if not str(sc_name).strip():
            continue
        inv_df = _lcc_investments_records_to_df(group.drop(columns=["Scenario"], errors="ignore"), end_uses=end_uses)
        out[str(sc_name)] = _lcc_investments_df_to_records(inv_df)
    return out


def build_lcc_investments_sheet(scenarios: Dict[str, dict], end_uses: list) -> pd.DataFrame:
    """Build the human-readable LCC_Investments sheet from scenario payloads."""
    rows = []
    if not isinstance(scenarios, dict):
        return pd.DataFrame(columns=["Scenario"] + LCC_INVESTMENT_COLUMNS)
    for sc_name, payload in scenarios.items():
        lcc = _normalize_lcc_payload((payload or {}).get("lcc", {}), end_uses)
        inv_df = _lcc_investments_records_to_df(lcc.get("investments", []), end_uses=end_uses)
        for _, r in inv_df.iterrows():
            row = {"Scenario": str(sc_name)}
            for col in LCC_INVESTMENT_COLUMNS:
                row[col] = r.get(col, "")
            rows.append(row)
    return pd.DataFrame(rows, columns=["Scenario"] + LCC_INVESTMENT_COLUMNS)


def merge_lcc_sheets_into_scenarios(
        scenarios: Dict[str, dict],
        lcc_global_df: Optional[pd.DataFrame],
        lcc_investments_df: Optional[pd.DataFrame],
        end_uses: list,
) -> Dict[str, dict]:
    """Merge dedicated LCC sheets into scenario payloads after loading a workbook.

    Dedicated sheets override the JSON payload when present, while the JSON payload remains
    supported for backwards compatibility.
    """
    if not isinstance(scenarios, dict):
        scenarios = {}

    global_payload = parse_lcc_global_df(lcc_global_df, end_uses)
    investments_by_scenario = parse_lcc_investments_sheet(lcc_investments_df, end_uses)

    if investments_by_scenario:
        for sc_name in investments_by_scenario.keys():
            if sc_name not in scenarios:
                scenarios[sc_name] = default_scenario_payload(end_uses, None)

    for sc_name, payload in list(scenarios.items()):
        if not isinstance(payload, dict):
            payload = default_scenario_payload(end_uses, None)
        payload["lcc"] = _normalize_lcc_payload(payload.get("lcc", {}), end_uses)
        if sc_name in investments_by_scenario:
            payload["lcc"] = {"investments": investments_by_scenario.get(sc_name, [])}
        if global_payload is not None:
            payload["lcc_global"] = deepcopy(global_payload)
        else:
            payload["lcc_global"] = _normalize_lcc_global_payload(payload.get("lcc_global", payload.get("lcc", {})), end_uses)
        scenarios[sc_name] = payload

    return scenarios


def _lcc_energy_rows_for_payload(df_energy: pd.DataFrame, payload: dict, selected_end_uses: list) -> pd.DataFrame:
    """Return row-level annual energy cost basis for a scenario payload before escalation."""
    if df_energy is None or df_energy.empty or "Month" not in df_energy.columns:
        return pd.DataFrame(columns=["End_Use", "Energy_Source", "kWh", "Tariff", "Annual Cost"])

    df = df_energy.melt(id_vars="Month", var_name="End_Use", value_name="kWh").copy()
    df["End_Use"] = df["End_Use"].apply(lambda x: _canon_enduse_name(str(x)))
    selected = [_canon_enduse_name(str(u)) for u in (selected_end_uses or []) if str(u).strip()]
    if selected:
        df = df[df["End_Use"].isin(set(selected))].copy()
    if df.empty:
        return pd.DataFrame(columns=["End_Use", "Energy_Source", "kWh", "Tariff", "Annual Cost"])

    payload = payload or {}
    eff = payload.get("efficiency", {}) or {}
    mapping = payload.get("mapping", {}) or {}
    tariffs = payload.get("tariffs", {}) or {}
    pv_cfg = payload.get("pv", {}) or {}
    pv_scale = _to_float_lcc(pv_cfg.get("scale", 1.0), 1.0)

    df["Efficiency_Factor"] = df["End_Use"].map(lambda u: _to_float_lcc(eff.get(u, 1.0), 1.0)).replace(0.0, 1.0)
    df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce").fillna(0.0) / df["Efficiency_Factor"]

    onsite_enduses = set(get_onsite_generation_enduses(df["End_Use"].unique()))
    pv_mask = df["End_Use"].isin(onsite_enduses)
    if pv_mask.any():
        df.loc[pv_mask, "kWh"] = -df.loc[pv_mask, "kWh"].abs() * pv_scale
    df.loc[~pv_mask, "kWh"] = df.loc[~pv_mask, "kWh"].clip(lower=0.0)

    df["Energy_Source"] = df["End_Use"].map(lambda u: str(mapping.get(u, "Electricity")))
    df.loc[~df["Energy_Source"].isin(ENERGY_SOURCE_ORDER), "Energy_Source"] = "Electricity"
    if pv_mask.any():
        df.loc[pv_mask, "Energy_Source"] = "Electricity"

    df["Tariff"] = df["Energy_Source"].map(lambda s: _to_float_lcc(tariffs.get(s, 0.0), 0.0)).fillna(0.0)
    df["Annual Cost"] = df["kWh"] * df["Tariff"]
    return df[["End_Use", "Energy_Source", "kWh", "Tariff", "Annual Cost"]]


def compute_lcc_cashflow_table(
        df_energy: pd.DataFrame,
        payload: dict,
        end_uses: list,
        project_year: int,
        lcc_global: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute annual nominal and discounted LCC cash-flow rows for one scenario.

    Scenario payload controls scenario-specific energy assumptions and investment measures.
    lcc_global controls analysis period, discount rate, inflation and operational filter and is shared by all scenarios.
    """
    payload = payload or {}
    lcc = _normalize_lcc_payload(payload.get("lcc", {}), end_uses)
    global_payload = lcc_global if isinstance(lcc_global, dict) else payload.get("lcc_global", payload.get("lcc", {}))
    lcc_assumptions = _normalize_lcc_global_payload(global_payload, end_uses)

    analysis_period = max(1, _to_int_lcc(lcc_assumptions.get("analysis_period", 30), 30))
    start_year = int(project_year)
    years = list(range(start_year, start_year + analysis_period))
    discount_rate = _to_float_lcc(lcc_assumptions.get("interest_rate_pct", 0.0), 0.0) / 100.0
    capex_inflation = _to_float_lcc(lcc_assumptions.get("capex_inflation_pct", 0.0), 0.0) / 100.0
    energy_inf = lcc_assumptions.get("energy_inflation_pct", {}) or {}
    selected_end_uses = lcc_assumptions.get("selected_operational_end_uses", _lcc_default_selected_enduses(end_uses))

    rows = []

    # Operational energy cost, escalated independently per energy source.
    energy_rows = _lcc_energy_rows_for_payload(df_energy, payload, selected_end_uses)
    if not energy_rows.empty:
        grouped_energy = energy_rows.groupby(["End_Use", "Energy_Source"], as_index=False).agg(
            kWh=("kWh", "sum"),
            Annual_Base_Cost=("Annual Cost", "sum"),
        )
        for y in years:
            offset = int(y - start_year)
            for _, r in grouped_energy.iterrows():
                src = str(r["Energy_Source"])
                rate = _to_float_lcc(energy_inf.get(src, 0.0), 0.0) / 100.0
                nominal = float(r["Annual_Base_Cost"]) * ((1.0 + rate) ** offset)
                discounted = nominal / ((1.0 + discount_rate) ** offset) if (1.0 + discount_rate) != 0 else nominal
                rows.append({
                    "Year": int(y),
                    "Year Offset": offset,
                    "Cost Type": "Energy",
                    "End_Use": str(r["End_Use"]),
                    "Energy_Source": src,
                    "Measure Name": "Operational energy",
                    "Nominal Cost": nominal,
                    "Discounted Cost": discounted,
                })

    # Investments, annual maintenance and replacement.
    # Measures can be assigned to several end uses; CAPEX/O&M/replacement costs are allocated equally.
    inv_df = _lcc_investments_records_to_df(lcc.get("investments", []), end_uses=end_uses)
    for _, r in inv_df.iterrows():
        measure = str(r.get("Measure Name", "")).strip() or "Unnamed measure"
        assigned_list = _lcc_parse_assigned_enduses(r.get("Assigned End Uses", ""), end_uses=end_uses)
        if not assigned_list:
            assigned_list = _lcc_default_selected_enduses(end_uses)[:1]
        allocation = 1.0 / max(1, len(assigned_list))

        inv_year = _to_int_lcc(r.get("Investment Year"), start_year)
        inv_cost = max(0.0, _to_float_lcc(r.get("Investment Cost"), 0.0))
        maint_pct = max(0.0, _to_float_lcc(r.get("Annual Maintenance Cost (%)"), 0.0)) / 100.0
        life = max(0, _to_int_lcc(r.get("Life Length (years)"), 0))

        def _append_allocated_cost(year: int, cost_type: str, nominal_total: float) -> None:
            offset_local = int(year - start_year)
            discounted_total = nominal_total / ((1.0 + discount_rate) ** offset_local) if (1.0 + discount_rate) != 0 else nominal_total
            for assigned in assigned_list:
                rows.append({
                    "Year": int(year),
                    "Year Offset": offset_local,
                    "Cost Type": cost_type,
                    "End_Use": assigned,
                    "Energy_Source": "",
                    "Measure Name": measure,
                    "Nominal Cost": float(nominal_total) * allocation,
                    "Discounted Cost": float(discounted_total) * allocation,
                })

        # Initial investment in selected investment year, escalated by CAPEX inflation from project start.
        if start_year <= inv_year <= years[-1] and inv_cost > 0.0:
            offset = int(inv_year - start_year)
            nominal = inv_cost * ((1.0 + capex_inflation) ** offset)
            _append_allocated_cost(inv_year, "Investment", nominal)

        # Annual maintenance as percentage of investment cost, escalated with CAPEX/O&M inflation.
        if inv_cost > 0.0 and maint_pct > 0.0:
            for y in years:
                if y < inv_year:
                    continue
                offset = int(y - start_year)
                nominal = inv_cost * maint_pct * ((1.0 + capex_inflation) ** offset)
                _append_allocated_cost(y, "Maintenance", nominal)

        # Replacement cost equals initial investment cost corrected by CAPEX inflation until replacement year.
        if inv_cost > 0.0 and life > 0:
            repl_year = inv_year + life
            while repl_year <= years[-1]:
                if repl_year >= start_year:
                    offset = int(repl_year - start_year)
                    nominal = inv_cost * ((1.0 + capex_inflation) ** offset)
                    _append_allocated_cost(repl_year, "Replacement", nominal)
                repl_year += life

    if not rows:
        return pd.DataFrame(columns=[
            "Year", "Year Offset", "Cost Type", "End_Use", "Energy_Source", "Measure Name",
            "Nominal Cost", "Discounted Cost",
        ])

    out = pd.DataFrame(rows)
    out["Nominal Cost"] = pd.to_numeric(out["Nominal Cost"], errors="coerce").fillna(0.0)
    out["Discounted Cost"] = pd.to_numeric(out["Discounted Cost"], errors="coerce").fillna(0.0)
    return out


def discounted_payback_period(active_cf: pd.DataFrame, reference_cf: pd.DataFrame, project_year: int) -> Optional[float]:
    """Return discounted payback period in years for active scenario vs reference scenario."""
    if active_cf is None or active_cf.empty or reference_cf is None or reference_cf.empty:
        return None
    years = sorted(set(active_cf["Year"].astype(int).tolist()) | set(reference_cf["Year"].astype(int).tolist()))
    if not years:
        return None
    active = active_cf.groupby("Year")["Discounted Cost"].sum().reindex(years).fillna(0.0)
    ref = reference_cf.groupby("Year")["Discounted Cost"].sum().reindex(years).fillna(0.0)
    incremental = ref - active  # positive means active scenario saves money against reference
    cumulative = incremental.cumsum()

    if cumulative.iloc[0] >= 0:
        return 0.0
    prev_cum = float(cumulative.iloc[0])
    prev_offset = int(years[0] - int(project_year))
    for idx in range(1, len(years)):
        curr_cum = float(cumulative.iloc[idx])
        curr_offset = int(years[idx] - int(project_year))
        if curr_cum >= 0:
            annual_gain = curr_cum - prev_cum
            if annual_gain <= 0:
                return float(curr_offset)
            frac = abs(prev_cum) / annual_gain
            return float(prev_offset + frac * (curr_offset - prev_offset))
        prev_cum = curr_cum
        prev_offset = curr_offset
    return None


def _format_payback(pb: Optional[float]) -> str:
    if pb is None or (isinstance(pb, float) and np.isnan(pb)):
        return "Not reached"
    return f"{float(pb):,.1f} years"


# =========================
# Report generation helpers (PDF)
# =========================
REPORT_VERSION = "2.2.16"


def _report_sanitize_filename(text: str) -> str:
    try:
        out = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")
        return out or "BPVis_Report"
    except Exception:
        return "BPVis_Report"


def _report_active_payload(end_uses: list) -> Tuple[str, dict]:
    """Return active scenario name and a fresh active scenario payload for the report."""
    active = str(st.session_state.get("active_scenario", "Base") or "Base")
    scenarios = st.session_state.get("scenarios", {})
    if not isinstance(scenarios, dict):
        scenarios = {}
    payload = deepcopy(scenarios.get(active, {})) if isinstance(scenarios.get(active, {}), dict) else {}
    try:
        # Keep report aligned with the latest committed sidebar/widget values.
        payload = capture_scenario_from_widgets(end_uses)
        scenarios[active] = payload
        st.session_state["scenarios"] = scenarios
    except Exception:
        pass
    return active, payload


def _report_prepare_energy_rows(df_energy: pd.DataFrame, payload: dict, apply_efficiency: bool = False) -> pd.DataFrame:
    """Prepare monthly long-format energy rows for report calculations."""
    if df_energy is None or df_energy.empty or "Month" not in df_energy.columns:
        return pd.DataFrame(columns=["Month", "End_Use", "kWh", "Energy_Source"])
    payload = payload or {}
    df = df_energy.melt(id_vars="Month", var_name="End_Use", value_name="kWh").copy()
    df["End_Use"] = df["End_Use"].apply(lambda x: _canon_enduse_name(str(x)))
    df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce").fillna(0.0)
    if apply_efficiency:
        eff = payload.get("efficiency", {}) or {}
        df["Efficiency_Factor"] = df["End_Use"].map(lambda u: _to_float_lcc(eff.get(u, 1.0), 1.0)).replace(0.0, 1.0)
        df["kWh"] = df["kWh"] / df["Efficiency_Factor"]
    mapping = payload.get("mapping", {}) or {}
    df["Energy_Source"] = df["End_Use"].map(lambda u: str(mapping.get(u, "Electricity")))
    df.loc[~df["Energy_Source"].isin(ENERGY_SOURCE_ORDER), "Energy_Source"] = "Electricity"
    return df[["Month", "End_Use", "kWh", "Energy_Source"]]


def _report_factor_maps(payload: dict) -> Tuple[dict, dict]:
    payload = payload or {}
    factors = payload.get("factors", {}) or {}
    tariffs = payload.get("tariffs", {}) or {}
    factor_map = {src: _to_float_lcc(factors.get(src, 0.0), 0.0) for src in ENERGY_SOURCE_ORDER}
    tariff_map = {src: _to_float_lcc(tariffs.get(src, 0.0), 0.0) for src in ENERGY_SOURCE_ORDER}
    return factor_map, tariff_map


# Consistent report-only chart typography. These settings affect the generated PDF only,
# not the native Streamlit/Plotly charts shown in the app.
REPORT_CHART_TITLE_SIZE = 10
REPORT_AXIS_LABEL_SIZE = 7
REPORT_TICK_LABEL_SIZE = 6
REPORT_LEGEND_FONT_SIZE = 6
REPORT_LEGEND_NCOL = 4
REPORT_LINE_WIDTH = 1.15
REPORT_MARKER_SIZE = 2.2


def _report_unique_order(items) -> list:
    """Return unique string items while preserving order and removing blanks."""
    out = []
    seen = set()
    for item in list(items or []):
        s = str(item)
        if not s or s.lower() == "nan" or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def _report_colors_for(labels, color_dict=None, fallback="#777777"):
    out = []
    cmap = color_dict or {}
    label_list = list(labels or [])
    for i, lab in enumerate(label_list):
        col = cmap.get(str(lab), cmap.get(_canon_enduse_name(str(lab)), None))
        if not col:
            col = SCENARIO_COLOR_PALETTE[i % len(SCENARIO_COLOR_PALETTE)] if labels is not None else fallback
        out.append(col)
    return out


def _report_apply_axis_style(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=REPORT_TICK_LABEL_SIZE, pad=2)
    try:
        ax.yaxis.get_offset_text().set_fontsize(REPORT_TICK_LABEL_SIZE)
        ax.xaxis.get_offset_text().set_fontsize(REPORT_TICK_LABEL_SIZE)
    except Exception:
        pass


def _report_reduce_xticks(ax, labels=None, max_ticks: int = 14, rotation: float = 45):
    """Prevent dense year/month labels from overlapping in the PDF report."""
    try:
        import numpy as _np
        if labels is not None:
            labels = [str(x) for x in labels]
            n = len(labels)
            if n > 0:
                step = max(1, int(_np.ceil(n / float(max_ticks))))
                ticks = list(range(0, n, step))
                if (n - 1) not in ticks:
                    ticks.append(n - 1)
                ax.set_xticks(ticks)
                ax.set_xticklabels([labels[i] for i in ticks], rotation=rotation, ha="right" if rotation else "center")
        else:
            ticks = list(ax.get_xticks())
            n = len(ticks)
            if n > max_ticks:
                step = max(1, int(_np.ceil(n / float(max_ticks))))
                keep = ticks[::step]
                if ticks[-1] not in keep:
                    keep = list(keep) + [ticks[-1]]
                ax.set_xticks(keep)
        ax.tick_params(axis="x", labelsize=REPORT_TICK_LABEL_SIZE, pad=2)
    except Exception:
        pass


def _report_apply_legend(ax, ncol: Optional[int] = None, bottom_anchor: float = -0.22):
    """Consistent compact legend styling for all report diagrams."""
    try:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        dedup_handles = []
        dedup_labels = []
        seen = set()
        for h, lab in zip(handles, labels):
            if lab in seen:
                continue
            seen.add(lab)
            dedup_handles.append(h)
            dedup_labels.append(lab)
        n = len(dedup_labels)
        cols = ncol or min(REPORT_LEGEND_NCOL, max(1, n))
        ax.legend(
            dedup_handles,
            dedup_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, bottom_anchor),
            ncol=cols,
            fontsize=REPORT_LEGEND_FONT_SIZE,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.9,
            labelspacing=0.35,
            borderaxespad=0.0,
        )
    except Exception:
        pass


def _report_fig_to_png_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
    return buf


def _report_stacked_monthly_chart(df: pd.DataFrame, value_col: str, category_col: str, title: str, y_label: str, color_dict: dict) -> io.BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
        return _report_fig_to_png_bytes(fig)
    work = df.copy()
    # The same helper is used for calendar-month charts and annual LCC charts.
    # Use canonical month ordering only when the x values are month names; otherwise keep numeric/text order.
    raw_x = work["Month"].astype(str)
    if set(raw_x).intersection(set(MONTH_ORDER)):
        work["_report_x"] = pd.Categorical(raw_x, categories=MONTH_ORDER, ordered=True)
        x_order = [m for m in MONTH_ORDER if m in set(raw_x)]
    else:
        work["_report_x"] = raw_x
        try:
            x_order = [str(v) for v in sorted(pd.to_numeric(raw_x, errors="coerce").dropna().unique().tolist())]
        except Exception:
            x_order = []
        if not x_order:
            x_order = list(raw_x.dropna().unique())
    cats = _report_unique_order([c for c in END_USE_ORDER + ENERGY_SOURCE_ORDER + sorted(work[category_col].dropna().astype(str).unique().tolist()) if c in set(work[category_col].astype(str))])
    pivot = work.groupby(["_report_x", category_col], observed=False)[value_col].sum().unstack(fill_value=0.0).reindex(x_order).fillna(0.0)
    pos_bottom = np.zeros(len(pivot.index))
    neg_bottom = np.zeros(len(pivot.index))
    for cat, col in zip(cats, _report_colors_for(cats, color_dict)):
        if cat not in pivot.columns:
            continue
        vals = pivot[cat].astype(float).values
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        if np.any(pos):
            ax.bar(pivot.index.astype(str), pos, bottom=pos_bottom, label=ui_name(cat), color=col, width=0.72)
            pos_bottom += pos
        if np.any(neg):
            ax.bar(pivot.index.astype(str), neg, bottom=neg_bottom, label=ui_name(cat), color=col, width=0.72)
            neg_bottom += neg
    totals = pivot.sum(axis=1).astype(float).values
    ax.plot(pivot.index.astype(str), totals, color="black", linestyle="--", linewidth=REPORT_LINE_WIDTH, marker="o", markersize=REPORT_MARKER_SIZE, label="Net total")
    ax.set_title(title, fontsize=REPORT_CHART_TITLE_SIZE, weight="bold")
    ax.set_ylabel(y_label, fontsize=REPORT_AXIS_LABEL_SIZE)
    if set(raw_x).intersection(set(MONTH_ORDER)):
        _report_reduce_xticks(ax, pivot.index.astype(str).tolist(), max_ticks=12, rotation=45)
    else:
        _report_reduce_xticks(ax, pivot.index.astype(str).tolist(), max_ticks=10, rotation=45)
    _report_apply_axis_style(ax)
    _report_apply_legend(ax, bottom_anchor=-0.24)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    return _report_fig_to_png_bytes(fig)


def _report_annual_stacked_chart(df: pd.DataFrame, value_col: str, category_col: str, title: str, y_label: str, color_dict: dict) -> io.BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.75))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
        return _report_fig_to_png_bytes(fig)
    totals = df.groupby(category_col, as_index=True)[value_col].sum()
    cats = _report_unique_order([c for c in END_USE_ORDER + ENERGY_SOURCE_ORDER + sorted(totals.index.astype(str).tolist()) if c in set(totals.index.astype(str))])
    pos_bottom = 0.0
    neg_bottom = 0.0
    for cat, col in zip(cats, _report_colors_for(cats, color_dict)):
        val = float(totals.get(cat, 0.0))
        if val >= 0:
            ax.bar(["Total"], [val], bottom=[pos_bottom], label=ui_name(cat), color=col, width=0.48)
            pos_bottom += val
        else:
            ax.bar(["Total"], [val], bottom=[neg_bottom], label=ui_name(cat), color=col, width=0.48)
            neg_bottom += val
    net = float(totals.sum())
    ax.axhline(net, color="black", linestyle="--", linewidth=1.3)
    ax.text(0, net, f" {net:,.0f}", fontsize=REPORT_TICK_LABEL_SIZE, va="bottom" if net >= 0 else "top")
    ax.set_title(title, fontsize=REPORT_CHART_TITLE_SIZE, weight="bold")
    ax.set_ylabel(y_label, fontsize=REPORT_AXIS_LABEL_SIZE)
    _report_apply_axis_style(ax)
    _report_apply_legend(ax, bottom_anchor=-0.20)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    return _report_fig_to_png_bytes(fig)


def _report_pie_chart(series: pd.Series, title: str, center_text: str, color_dict: dict) -> io.BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.75))
    if series is None or len(series) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
        return _report_fig_to_png_bytes(fig)
    s = pd.Series(series).copy()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if s.empty or float(s.sum()) == 0.0:
        ax.text(0.5, 0.5, "No positive values available", ha="center", va="center")
        ax.set_axis_off()
        return _report_fig_to_png_bytes(fig)
    labels = [ui_name(str(x)) for x in s.index.astype(str)]
    colors = _report_colors_for(s.index.astype(str).tolist(), color_dict)
    wedges, texts, autotexts = ax.pie(
        s.values,
        labels=None,
        autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white"),
        textprops=dict(fontsize=REPORT_TICK_LABEL_SIZE),
    )
    ax.text(0, 0, center_text, ha="center", va="center", fontsize=9, weight="bold")
    ax.set_title(title, fontsize=REPORT_CHART_TITLE_SIZE, weight="bold")
    ax.legend(
        wedges, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(REPORT_LEGEND_NCOL, max(1, len(labels))), fontsize=REPORT_LEGEND_FONT_SIZE,
        frameon=False, handlelength=1.3, columnspacing=0.9, labelspacing=0.35
    )
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    return _report_fig_to_png_bytes(fig)


def _report_line_chart(series_dict: Dict[str, pd.Series], title: str, y_label: str, color_map_in: Optional[dict] = None, dashed: Optional[set] = None) -> io.BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    dashed = dashed or set()
    if not series_dict:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
        return _report_fig_to_png_bytes(fig)
    for i, (name, ser) in enumerate(series_dict.items()):
        if ser is None or len(ser) == 0:
            continue
        ser = pd.Series(ser).dropna()
        if ser.empty:
            continue
        col = (color_map_in or {}).get(name, SCENARIO_COLOR_PALETTE[i % len(SCENARIO_COLOR_PALETTE)])
        ax.plot(ser.index.astype(int), ser.values.astype(float), label=name, color=col, linewidth=REPORT_LINE_WIDTH, linestyle="--" if name in dashed else "-", marker="o", markersize=REPORT_MARKER_SIZE)
    ax.set_title(title, fontsize=REPORT_CHART_TITLE_SIZE, weight="bold")
    ax.set_xlabel("Year", fontsize=REPORT_AXIS_LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=REPORT_AXIS_LABEL_SIZE)
    _report_apply_axis_style(ax)
    _report_reduce_xticks(ax, max_ticks=10, rotation=0)
    _report_apply_legend(ax, bottom_anchor=-0.20)
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    return _report_fig_to_png_bytes(fig)


def _report_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, y_label: str, color: Optional[str] = None) -> io.BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
        return _report_fig_to_png_bytes(fig)
    ax.bar(df[x].astype(str), pd.to_numeric(df[y], errors="coerce").fillna(0.0), color=color or CRREM_COLOR_BASELINE)
    ax.set_title(title, fontsize=REPORT_CHART_TITLE_SIZE, weight="bold")
    ax.set_ylabel(y_label, fontsize=REPORT_AXIS_LABEL_SIZE)
    _report_reduce_xticks(ax, df[x].astype(str).tolist(), max_ticks=12, rotation=45)
    _report_apply_axis_style(ax)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _report_fig_to_png_bytes(fig)


def _report_heatmap(df_loads: pd.DataFrame, load_col: str, title: str) -> io.BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    try:
        work = df_loads.copy()
        work["doy"] = pd.to_numeric(work["doy"], errors="coerce")
        work["hour"] = pd.to_numeric(work["hour"], errors="coerce")
        work[load_col] = pd.to_numeric(work[load_col], errors="coerce")
        piv = work.pivot_table(index="hour", columns="doy", values=load_col, aggfunc="mean").sort_index()
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="inferno")
        ax.set_xlabel("Day of year", fontsize=REPORT_AXIS_LABEL_SIZE)
        ax.set_ylabel("Hour", fontsize=REPORT_AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=REPORT_CHART_TITLE_SIZE, weight="bold")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(load_col, fontsize=REPORT_AXIS_LABEL_SIZE)
        cbar.ax.tick_params(labelsize=REPORT_TICK_LABEL_SIZE)
        _report_reduce_xticks(ax, max_ticks=8, rotation=0)
        ax.tick_params(axis="both", labelsize=REPORT_TICK_LABEL_SIZE, pad=2)
    except Exception:
        ax.text(0.5, 0.5, "No heatmap data available", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _report_fig_to_png_bytes(fig)


def _report_load_duration(df_loads: pd.DataFrame, load_col: str, title: str) -> io.BytesIO:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    vals = pd.to_numeric(df_loads.get(load_col, pd.Series(dtype=float)), errors="coerce").dropna().sort_values(ascending=False).reset_index(drop=True)
    if vals.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        pct = (np.arange(1, len(vals) + 1) / len(vals)) * 100.0
        ax.plot(pct, vals.values, color=CRREM_COLOR_BASELINE, linewidth=REPORT_LINE_WIDTH)
        ax.fill_between(pct, vals.values, alpha=0.18, color=CRREM_COLOR_BASELINE)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage of hours (%)", fontsize=REPORT_AXIS_LABEL_SIZE)
        ax.set_ylabel(f"{ui_name(load_col)} (kW)", fontsize=REPORT_AXIS_LABEL_SIZE)
        ax.set_title(title, fontsize=REPORT_CHART_TITLE_SIZE, weight="bold")
        _report_apply_axis_style(ax)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _report_fig_to_png_bytes(fig)


def _report_table_flowable(rows, col_widths=None, font_size=7):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    if not rows:
        rows = [["No data", ""]]
    table = Table(rows, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b8c0cc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _report_format_number(x, decimals=1, prefix="", suffix=""):
    try:
        return f"{prefix}{float(x):,.{decimals}f}{suffix}"
    except Exception:
        return f"{prefix}{x}{suffix}"


def _report_add_chart(story, styles, title: str, png_buf: io.BytesIO, caption: str = ""):
    """Add a chart image to the PDF without distorting its aspect ratio.

    The previous implementation forced every image into a fixed width/height box,
    which stretched wide or tall diagrams. This version reads the generated PNG
    size and scales it proportionally to fit within an A4-safe content box.
    """
    from reportlab.platypus import Paragraph, Spacer, Image, KeepTogether
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader

    max_w = 17.0 * cm
    max_h = 10.2 * cm

    try:
        data = png_buf.getvalue()
        reader = ImageReader(io.BytesIO(data))
        iw, ih = reader.getSize()
        if iw and ih:
            scale = min(max_w / float(iw), max_h / float(ih))
            img_w = float(iw) * scale
            img_h = float(ih) * scale
        else:
            img_w, img_h = max_w, 9.2 * cm
        img = Image(io.BytesIO(data), width=img_w, height=img_h)
    except Exception:
        # Fallback still preserves proportions through ReportLab's proportional mode.
        try:
            png_buf.seek(0)
        except Exception:
            pass
        img = Image(png_buf, width=max_w, height=max_h, kind="proportional")

    block = [Paragraph(title, styles["Heading3"]), img]
    if caption:
        block.append(Paragraph(caption, styles["CaptionSmall"]))
    block.append(Spacer(1, 0.18 * cm))
    story.append(KeepTogether(block))


def _report_crrem_limits_for_context(crrem: dict, target_id: str, crrem_use: str, mixed_records, years: list) -> Tuple[pd.Series, pd.Series]:
    """Return CRREM carbon and EUI limits in kgCO2e/m2.a and kWh/m2.a for report years."""
    if not crrem or not years:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        pt_df = crrem["property_types"].copy()
        pc = crrem["pathways_carbon"].copy()
        pe = crrem["pathways_eui"].copy()
        pc_t = pc.loc[pc["target"].astype(str) == target_id]
        pe_t = pe.loc[pe["target"].astype(str) == target_id]
        carbon_pivot = pc_t.pivot_table(index="year", columns="property_type_code", values="kgco2e_per_m2_yr")
        eui_pivot = pe_t.pivot_table(index="year", columns="property_type_code", values="kwh_per_m2_yr")
        years_avail = [int(y) for y in years if int(y) in carbon_pivot.index and int(y) in eui_pivot.index]
        if not years_avail:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        if str(crrem_use) != "Mixed Use":
            code_row = pt_df.loc[pt_df["app_use"].astype(str) == str(crrem_use)]
            if code_row.empty:
                return pd.Series(dtype=float), pd.Series(dtype=float)
            p_code = str(code_row.iloc[0]["crrem_code"])
            return carbon_pivot[p_code].reindex(years_avail).astype(float), eui_pivot[p_code].reindex(years_avail).astype(float)
        mixed_df = _mixed_use_records_to_df(mixed_records)
        if mixed_df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        total_share = float(mixed_df["Area Share %"].sum()) or 100.0
        use_to_code = dict(zip(pt_df["app_use"].astype(str), pt_df["crrem_code"].astype(str)))
        carbon_limit = pd.Series(0.0, index=years_avail)
        eui_limit = pd.Series(0.0, index=years_avail)
        for _, row in mixed_df.iterrows():
            u = str(row.get("Use Type", ""))
            w = float(row.get("Area Share %", 0.0) or 0.0) / total_share
            c = use_to_code.get(u)
            if c and c in carbon_pivot.columns:
                carbon_limit = carbon_limit + w * carbon_pivot[c].reindex(years_avail).astype(float)
                eui_limit = eui_limit + w * eui_pivot[c].reindex(years_avail).astype(float)
        return carbon_limit, eui_limit
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float)


def _report_eui_series_for_payload(df_energy: pd.DataFrame, payload: dict, years: list, project_area: float) -> pd.Series:
    """Return constant annual gross EUI series excluding on-site generation, with efficiency factors applied."""
    try:
        rows = _report_prepare_energy_rows(df_energy, payload, apply_efficiency=True)
        onsite = set(get_onsite_generation_enduses(rows["End_Use"].unique())) | {ONSITE_GENERATION_ENDUSE, LEGACY_PV_ENDUSE}
        consumption = rows.loc[~rows["End_Use"].isin(onsite), "kWh"].clip(lower=0.0).sum()
        val = float(consumption) / float(project_area) if project_area else 0.0
        return pd.Series({int(y): val for y in years}, dtype=float)
    except Exception:
        return pd.Series({int(y): 0.0 for y in years}, dtype=float)


def generate_bpvis_pdf_report(file_bytes: bytes, filename: str = "") -> bytes:
    """Generate an A4 PDF report for the active scenario, excluding the Raw Data tab."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, KeepTogether
    except Exception as exc:
        raise RuntimeError("Report generation requires matplotlib and reportlab to be installed.") from exc

    # Data and context
    base_df = get_energy_balance_df(file_bytes, filename)
    end_uses = [str(c) for c in base_df.columns if str(c) != "Month"]
    active_name, payload = _report_active_payload(end_uses)
    df_energy = get_energy_balance_df(file_bytes, filename, scenario_name=active_name)
    df_loads = get_loads_balance_df(file_bytes, filename)

    project_name_r = str(st.session_state.get("project_name", "Building Performance Dashboard") or "Building Performance Dashboard")
    project_area_r = float(st.session_state.get("project_area", 0.0) or 0.0)
    project_year_r = int(st.session_state.get("project_year", 2025) or 2025)
    project_country_r = str(st.session_state.get("project_country", "Germany") or "Germany")
    building_use_r = str(st.session_state.get("building_use", "Office") or "Office")
    currency_r = str(st.session_state.get("currency_symbol", "€") or "€")
    colors_eu = st.session_state.get("color_map_enduse", DEFAULT_COLOR_MAP)
    colors_src = st.session_state.get("color_map_sources", DEFAULT_COLOR_MAP_SOURCES)
    colors_loads = st.session_state.get("color_map_loads", DEFAULT_COLOR_MAP_LOADS)
    colors_scenarios = st.session_state.get("color_map_scenarios", default_scenario_color_map(list(st.session_state.get("scenarios", {}).keys())))
    scenario_color = colors_scenarios.get(active_name, CRREM_COLOR_BASELINE)
    lcc_global = _get_lcc_global_state_payload(end_uses)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title=f"BPVis ENE Report - {project_name_r}",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=10))
    styles.add(ParagraphStyle(name="CaptionSmall", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9, textColor=colors.HexColor("#4b5563"), spaceAfter=4))
    styles["Heading2"].fontSize = 13
    styles["Heading3"].fontSize = 10
    styles["BodyText"].fontSize = 8.5
    styles["BodyText"].leading = 10.5

    story = []

    def _page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(A4[0] - 1.5 * cm, 0.75 * cm, f"BPVis ENE {REPORT_VERSION} | Page {doc_obj.page}")
        canvas.restoreState()

    def add_section(title: str):
        if story:
            story.append(PageBreak())
        story.append(Paragraph(title, styles["SectionTitle"]))

    def add_kpi_table(title: str, kpis: list):
        story.append(Paragraph(title, styles["Heading3"]))
        rows = [["KPI", "Value"]] + [[str(k), str(v)] for k, v in kpis]
        story.append(_report_table_flowable(rows, col_widths=[8.0 * cm, 8.0 * cm], font_size=7.2))
        story.append(Spacer(1, 0.25 * cm))

    def add_input_table(title: str, inputs: list):
        story.append(Paragraph(title, styles["Heading3"]))
        rows = [["Input", "Value"]] + [[str(k), str(v)] for k, v in inputs]
        story.append(_report_table_flowable(rows, col_widths=[8.0 * cm, 8.0 * cm], font_size=7.0))
        story.append(Spacer(1, 0.25 * cm))

    # Cover page
    logo_candidates = [Path("WS_Logo.jpg"), Path("WS_Logo.png"), Path("Pamo_Icon_Black.png")]
    logo_path = next((p for p in logo_candidates if p.exists()), None)
    if logo_path is not None:
        try:
            story.append(Image(str(logo_path), width=12.0 * cm, height=3.0 * cm, kind="proportional"))
            story.append(Spacer(1, 0.7 * cm))
        except Exception:
            pass
    story.append(Paragraph("BPVis ENE - Automated Project Report", styles["ReportTitle"]))
    story.append(Paragraph(f"Active scenario: <b>{active_name}</b>", styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))
    project_rows = [
        ["Project Data", "Value"],
        ["Project name", project_name_r],
        ["Building use", building_use_r],
        ["Country", project_country_r],
        ["Project year", str(project_year_r)],
        ["Project area", f"{project_area_r:,.1f} m²"],
        ["Currency", currency_r],
        ["Report version", REPORT_VERSION],
    ]
    story.append(_report_table_flowable(project_rows, col_widths=[6.0 * cm, 10.0 * cm], font_size=8.5))
    story.append(Spacer(1, 0.45 * cm))
    story.append(Paragraph("This report presents the current active scenario only. It excludes the Raw Data tab and is formatted for A4 output. Each section starts on a new page.", styles["BodyText"]))

    # Energy balance without factors
    add_section("1. Energy Balance (without Factors)")
    rows_raw = _report_prepare_energy_rows(df_energy, payload, apply_efficiency=False)
    totals_eu = rows_raw.groupby("End_Use", as_index=True)["kWh"].sum()
    totals_src = rows_raw.groupby("Energy_Source", as_index=True)["kWh"].sum()
    eui_gross = totals_eu[totals_eu > 0].sum() / project_area_r if project_area_r else 0.0
    net_eui = totals_eu.sum() / project_area_r if project_area_r else 0.0
    add_kpi_table("Energy KPIs", [
        ("Total annual consumption", f"{totals_eu[totals_eu > 0].sum():,.0f} kWh/a"),
        ("Net annual energy", f"{totals_eu.sum():,.0f} kWh/a"),
        ("Gross EUI", f"{eui_gross:,.1f} kWh/m²·a"),
        ("Net EUI", f"{net_eui:,.1f} kWh/m²·a"),
        ("Monthly average net energy", f"{rows_raw.groupby('Month')['kWh'].sum().mean():,.0f} kWh/month"),
    ])
    _report_add_chart(story, styles, "Monthly energy by end use", _report_stacked_monthly_chart(rows_raw, "kWh", "End_Use", "Monthly Energy by End Use", "kWh/month", colors_eu), "Monthly values are raw Energy_Balance values by end use. The dashed line is the monthly net total.")
    _report_add_chart(story, styles, "Annual energy by end use", _report_annual_stacked_chart(rows_raw, "kWh", "End_Use", "Annual Energy by End Use", "kWh/a", colors_eu), "Annual values are the sum of all monthly Energy_Balance values. Negative on-site generation is shown below zero.")
    pie_eu = (totals_eu[totals_eu > 0] / project_area_r) if project_area_r else totals_eu[totals_eu > 0]
    _report_add_chart(story, styles, "Energy use intensity by end use", _report_pie_chart(pie_eu, "EUI Share by End Use", f"{eui_gross:,.1f}\nkWh/m²·a", colors_eu), "The donut uses positive consumption only and divides annual kWh by project area.")
    _report_add_chart(story, styles, "Monthly energy by energy source", _report_stacked_monthly_chart(rows_raw.groupby(["Month", "Energy_Source"], as_index=False)["kWh"].sum(), "kWh", "Energy_Source", "Monthly Energy by Energy Source", "kWh/month", colors_src), "Monthly values are grouped by the active scenario end-use to energy-source mapping.")
    _report_add_chart(story, styles, "Annual energy by energy source", _report_annual_stacked_chart(rows_raw, "kWh", "Energy_Source", "Annual Energy by Energy Source", "kWh/a", colors_src), "Annual source totals are calculated using the active scenario energy-source mapping.")
    pie_src = (totals_src[totals_src > 0] / project_area_r) if project_area_r else totals_src[totals_src > 0]
    _report_add_chart(story, styles, "Energy use intensity by energy source", _report_pie_chart(pie_src, "EUI Share by Energy Source", f"{pie_src.sum():,.1f}\nkWh/m²·a", colors_src), "Source intensity is annual kWh per source divided by project area.")
    add_input_table("Relevant inputs", [
        ("Scenario", active_name),
        ("Project area", f"{project_area_r:,.1f} m²"),
        ("Energy-source mapping", ", ".join([f"{ui_name(k)}={v}" for k, v in (payload.get("mapping", {}) or {}).items()])),
    ])

    # Energy balance with factors
    add_section("2. Energy Balance (with Factors)")
    rows_eff = _report_prepare_energy_rows(df_energy, payload, apply_efficiency=True)
    totals_eff_eu = rows_eff.groupby("End_Use", as_index=True)["kWh"].sum()
    totals_eff_src = rows_eff.groupby("Energy_Source", as_index=True)["kWh"].sum()
    eui_eff_gross = totals_eff_eu[totals_eff_eu > 0].sum() / project_area_r if project_area_r else 0.0
    net_eui_eff = totals_eff_eu.sum() / project_area_r if project_area_r else 0.0
    add_kpi_table("Factored energy KPIs", [
        ("Total annual consumption", f"{totals_eff_eu[totals_eff_eu > 0].sum():,.0f} kWh/a"),
        ("Net annual energy", f"{totals_eff_eu.sum():,.0f} kWh/a"),
        ("Gross EUI", f"{eui_eff_gross:,.1f} kWh/m²·a"),
        ("Net EUI", f"{net_eui_eff:,.1f} kWh/m²·a"),
        ("Monthly average net energy", f"{rows_eff.groupby('Month')['kWh'].sum().mean():,.0f} kWh/month"),
    ])
    _report_add_chart(story, styles, "Monthly factored energy by end use", _report_stacked_monthly_chart(rows_eff, "kWh", "End_Use", "Monthly Energy by End Use - with Factors", "kWh/month", colors_eu), "Each end-use kWh is divided by its efficiency factor before aggregation.")
    _report_add_chart(story, styles, "Annual factored energy by end use", _report_annual_stacked_chart(rows_eff, "kWh", "End_Use", "Annual Energy by End Use - with Factors", "kWh/a", colors_eu), "Annual values are factored kWh after applying active scenario efficiency factors.")
    pie_eff_eu = (totals_eff_eu[totals_eff_eu > 0] / project_area_r) if project_area_r else totals_eff_eu[totals_eff_eu > 0]
    _report_add_chart(story, styles, "Factored EUI by end use", _report_pie_chart(pie_eff_eu, "Factored EUI Share by End Use", f"{eui_eff_gross:,.1f}\nkWh/m²·a", colors_eu), "The donut is based on positive factored consumption divided by project area.")
    _report_add_chart(story, styles, "Monthly factored energy by source", _report_stacked_monthly_chart(rows_eff.groupby(["Month", "Energy_Source"], as_index=False)["kWh"].sum(), "kWh", "Energy_Source", "Monthly Energy by Source - with Factors", "kWh/month", colors_src), "Factored monthly kWh are grouped by energy source.")
    _report_add_chart(story, styles, "Annual factored energy by source", _report_annual_stacked_chart(rows_eff, "kWh", "Energy_Source", "Annual Energy by Source - with Factors", "kWh/a", colors_src), "Source totals are after efficiency factors and source mapping.")
    pie_eff_src = (totals_eff_src[totals_eff_src > 0] / project_area_r) if project_area_r else totals_eff_src[totals_eff_src > 0]
    _report_add_chart(story, styles, "Factored EUI by energy source", _report_pie_chart(pie_eff_src, "Factored EUI Share by Energy Source", f"{pie_eff_src.sum():,.1f}\nkWh/m²·a", colors_src), "Source EUI is annual factored kWh divided by project area.")
    add_input_table("Relevant inputs", [(f"Efficiency - {ui_name(k)}", f"{float(v):,.4f}") for k, v in (payload.get("efficiency", {}) or {}).items()])

    # CO2 emissions
    add_section("3. CO₂ Emissions (with Factors)")
    factor_map, tariff_map = _report_factor_maps(payload)
    rows_co2 = rows_eff.copy()
    rows_co2["CO2_factor_kg_per_kWh"] = rows_co2["Energy_Source"].map(factor_map).fillna(0.0)
    rows_co2["kgCO2"] = rows_co2["kWh"] * rows_co2["CO2_factor_kg_per_kWh"]
    totals_co2_eu = rows_co2.groupby("End_Use", as_index=True)["kgCO2"].sum()
    totals_co2_src = rows_co2.groupby("Energy_Source", as_index=True)["kgCO2"].sum()
    gross_co2_int = totals_co2_eu[totals_co2_eu > 0].sum() / project_area_r if project_area_r else 0.0
    net_co2_int = totals_co2_eu.sum() / project_area_r if project_area_r else 0.0
    add_kpi_table("CO₂ KPIs", [
        ("Total annual CO₂", f"{totals_co2_eu.sum():,.0f} kgCO₂/a"),
        ("Monthly average CO₂", f"{rows_co2.groupby('Month')['kgCO2'].sum().mean():,.0f} kgCO₂/month"),
        ("Net CO₂ intensity", f"{net_co2_int:,.1f} kgCO₂/m²·a"),
        ("Gross CO₂ intensity", f"{gross_co2_int:,.1f} kgCO₂/m²·a"),
    ])
    _report_add_chart(story, styles, "Monthly CO₂ by end use", _report_stacked_monthly_chart(rows_co2, "kgCO2", "End_Use", "Monthly CO₂ by End Use", "kgCO₂/month", colors_eu), "CO₂ is calculated as factored kWh multiplied by the mapped energy-source emission factor.")
    _report_add_chart(story, styles, "Annual CO₂ by end use", _report_annual_stacked_chart(rows_co2, "kgCO2", "End_Use", "Annual CO₂ by End Use", "kgCO₂/a", colors_eu), "Annual CO₂ is the sum of all monthly emissions by end use.")
    pie_co2_eu = (totals_co2_eu[totals_co2_eu > 0] / project_area_r) if project_area_r else totals_co2_eu[totals_co2_eu > 0]
    _report_add_chart(story, styles, "CO₂ intensity by end use", _report_pie_chart(pie_co2_eu, "CO₂ Intensity Share by End Use", f"{gross_co2_int:,.1f}\nkg/m²·a", colors_eu), "The donut uses positive annual kgCO₂ divided by project area.")
    _report_add_chart(story, styles, "Monthly CO₂ by energy source", _report_stacked_monthly_chart(rows_co2.groupby(["Month", "Energy_Source"], as_index=False)["kgCO2"].sum(), "kgCO2", "Energy_Source", "Monthly CO₂ by Energy Source", "kgCO₂/month", colors_src), "Monthly CO₂ is grouped by energy source.")
    _report_add_chart(story, styles, "Annual CO₂ by energy source", _report_annual_stacked_chart(rows_co2, "kgCO2", "Energy_Source", "Annual CO₂ by Energy Source", "kgCO₂/a", colors_src), "Annual source emissions are factored kWh times source-specific emission factors.")
    pie_co2_src = (totals_co2_src[totals_co2_src > 0] / project_area_r) if project_area_r else totals_co2_src[totals_co2_src > 0]
    _report_add_chart(story, styles, "CO₂ intensity by energy source", _report_pie_chart(pie_co2_src, "CO₂ Intensity Share by Energy Source", f"{pie_co2_src.sum():,.1f}\nkg/m²·a", colors_src), "Source CO₂ intensity is annual kgCO₂ per source divided by project area.")
    add_input_table("Relevant inputs", [(f"Emission factor - {src}", f"{factor_map.get(src, 0.0):,.5f} kgCO₂/kWh") for src in ENERGY_SOURCE_ORDER])

    # Energy cost
    add_section("4. Energy Cost (with Factors)")
    rows_cost = rows_eff.copy()
    rows_cost["Tariff"] = rows_cost["Energy_Source"].map(tariff_map).fillna(0.0)
    rows_cost["cost"] = rows_cost["kWh"] * rows_cost["Tariff"]
    totals_cost_eu = rows_cost.groupby("End_Use", as_index=True)["cost"].sum()
    totals_cost_src = rows_cost.groupby("Energy_Source", as_index=True)["cost"].sum()
    gross_cost_int = totals_cost_eu[totals_cost_eu > 0].sum() / project_area_r if project_area_r else 0.0
    net_cost_int = totals_cost_eu.sum() / project_area_r if project_area_r else 0.0
    add_kpi_table("Cost KPIs", [
        ("Total annual cost", f"{currency_r} {totals_cost_eu.sum():,.0f}/a"),
        ("Monthly average cost", f"{currency_r} {rows_cost.groupby('Month')['cost'].sum().mean():,.0f}/month"),
        ("Net cost intensity", f"{currency_r} {net_cost_int:,.2f}/m²·a"),
        ("Gross cost intensity", f"{currency_r} {gross_cost_int:,.2f}/m²·a"),
    ])
    _report_add_chart(story, styles, "Monthly cost by end use", _report_stacked_monthly_chart(rows_cost, "cost", "End_Use", "Monthly Cost by End Use", f"{currency_r}/month", colors_eu), "Cost is factored kWh multiplied by the active scenario tariff of the mapped energy source.")
    _report_add_chart(story, styles, "Annual cost by end use", _report_annual_stacked_chart(rows_cost, "cost", "End_Use", "Annual Cost by End Use", f"{currency_r}/a", colors_eu), "Annual costs are the sum of monthly costs. Negative on-site generation appears as avoided cost if present.")
    pie_cost_eu = (totals_cost_eu[totals_cost_eu > 0] / project_area_r) if project_area_r else totals_cost_eu[totals_cost_eu > 0]
    _report_add_chart(story, styles, "Cost intensity by end use", _report_pie_chart(pie_cost_eu, "Cost Intensity Share by End Use", f"{currency_r} {gross_cost_int:,.2f}\n/m²·a", colors_eu), "The donut uses positive annual costs divided by project area.")
    _report_add_chart(story, styles, "Monthly cost by energy source", _report_stacked_monthly_chart(rows_cost.groupby(["Month", "Energy_Source"], as_index=False)["cost"].sum(), "cost", "Energy_Source", "Monthly Cost by Energy Source", f"{currency_r}/month", colors_src), "Monthly energy costs are grouped by source.")
    _report_add_chart(story, styles, "Annual cost by energy source", _report_annual_stacked_chart(rows_cost, "cost", "Energy_Source", "Annual Cost by Energy Source", f"{currency_r}/a", colors_src), "Annual source costs are factored kWh times source-specific tariffs.")
    pie_cost_src = (totals_cost_src[totals_cost_src > 0] / project_area_r) if project_area_r else totals_cost_src[totals_cost_src > 0]
    _report_add_chart(story, styles, "Cost intensity by energy source", _report_pie_chart(pie_cost_src, "Cost Intensity Share by Energy Source", f"{currency_r} {pie_cost_src.sum():,.2f}\n/m²·a", colors_src), "Source cost intensity is annual cost per source divided by project area.")
    add_input_table("Relevant inputs", [(f"Tariff - {src}", f"{currency_r} {tariff_map.get(src, 0.0):,.5f}/kWh") for src in ENERGY_SOURCE_ORDER])

    # Loads
    add_section("5. Loads Analysis")
    load_cols = [c for c in df_loads.columns if c not in ["hoy", "doy", "day", "month", "weekday", "hour"]]
    load_cols = [c for c in load_cols if pd.to_numeric(df_loads.get(c, pd.Series(dtype=float)), errors="coerce").notna().any()]

    if load_cols:
        story.append(Paragraph(
            "This section includes all load profiles available in the project's Loads_Balance data, not only the load selected in the interactive Streamlit view.",
            styles["BodyText"]
        ))

        for idx_load, selected_load in enumerate(load_cols, start=1):
            if idx_load > 1:
                story.append(Spacer(1, 0.35 * cm))
            story.append(Paragraph(f"Load profile {idx_load}: {ui_name(selected_load)}", styles["Heading3"]))

            s_load = pd.to_numeric(df_loads[selected_load], errors="coerce").dropna()
            specific = (s_load / project_area_r) * 1000.0 if project_area_r else s_load * 0.0
            add_kpi_table(f"Load KPIs - {ui_name(selected_load)}", [
                ("Total load", f"{s_load.sum():,.0f} kWh"),
                ("Maximum load", f"{s_load.max():,.1f} kW"),
                ("Minimum load", f"{s_load.min():,.1f} kW"),
                ("Maximum specific load", f"{specific.max():,.1f} W/m²"),
                ("95th percentile specific load", f"{np.percentile(specific.dropna(), 95):,.1f} W/m²" if not specific.dropna().empty else "n/a"),
                ("80th percentile specific load", f"{np.percentile(specific.dropna(), 80):,.1f} W/m²" if not specific.dropna().empty else "n/a"),
            ])
            if "month" in df_loads.columns:
                monthly_load = df_loads.assign(_load=pd.to_numeric(df_loads[selected_load], errors="coerce")).groupby("month", as_index=False)["_load"].sum()
                _report_add_chart(story, styles, "Monthly load sum", _report_bar_chart(monthly_load, "month", "_load", f"Monthly Load Sum - {ui_name(selected_load)}", "kWh", colors_loads.get(selected_load, CRREM_COLOR_BASELINE)), "Monthly load sum is calculated by summing the hourly load profile by month.")
            _report_add_chart(story, styles, "Hourly load heatmap", _report_heatmap(df_loads, selected_load, f"Hourly Load Heatmap - {ui_name(selected_load)}"), "The heatmap shows hourly load intensity by day of year and hour.")
            _report_add_chart(story, styles, "Load duration curve", _report_load_duration(df_loads, selected_load, f"Load Duration Curve - {ui_name(selected_load)}"), "The load duration curve sorts hourly load values descending and plots load against percentage of annual hours.")
            try:
                peaks = df_loads.loc[:, [c for c in ["month", "day", "weekday", "hour", selected_load] if c in df_loads.columns]].copy()
                peaks[selected_load] = pd.to_numeric(peaks[selected_load], errors="coerce")
                peaks = peaks.sort_values(selected_load, ascending=False).head(5)
                rows_peak = [["month", "day", "weekday", "hour", selected_load]] + peaks.fillna("").astype(str).values.tolist()
                story.append(Paragraph(f"Top 5 peak loads - {ui_name(selected_load)}", styles["Heading3"]))
                story.append(_report_table_flowable(rows_peak, font_size=6.8))
            except Exception:
                pass
    else:
        story.append(Paragraph("No Loads_Balance data available.", styles["BodyText"]))
    add_input_table("Relevant inputs", [("Load profiles included", ", ".join([ui_name(c) for c in load_cols]) if load_cols else "n/a"), ("Number of load profiles", str(len(load_cols))), ("Project area", f"{project_area_r:,.1f} m²")])

    # Benchmark
    add_section("6. Benchmark")
    try:
        benchmark_df = load_benchmark_data(building_use_r)
        eui_net = net_eui_eff
        co2_net = net_co2_int
        cost_net = net_cost_int
        bench_kpis = [("Net EUI", f"{eui_net:,.1f} kWh/m²·a"), ("Net CO₂ intensity", f"{co2_net:,.1f} kgCO₂/m²·a"), ("Net energy cost", f"{currency_r} {cost_net:,.2f}/m²·a")]
        add_kpi_table("Benchmark KPIs", bench_kpis)
        bench_df = pd.DataFrame({"KPI": ["EUI", "CO₂", "Cost"], "Value": [eui_net, co2_net, cost_net]})
        _report_add_chart(story, styles, "Benchmark KPI overview", _report_bar_chart(bench_df, "KPI", "Value", "Project KPI Values", "Intensity / cost", CRREM_COLOR_BASELINE), "Benchmark categories depend on the selected building use and the benchmark template available in the app.")
        if benchmark_df is not None and not benchmark_df.empty:
            story.append(Paragraph("Benchmark data source loaded successfully for the selected building use.", styles["BodyText"]))
        else:
            story.append(Paragraph("No benchmark threshold sheet was found for the selected building use. The report therefore shows project KPI values only.", styles["BodyText"]))
    except Exception:
        add_kpi_table("Benchmark KPIs", [("Net EUI", f"{net_eui_eff:,.1f} kWh/m²·a"), ("Net CO₂ intensity", f"{net_co2_int:,.1f} kgCO₂/m²·a"), ("Net energy cost", f"{currency_r} {net_cost_int:,.2f}/m²·a")])
    add_input_table("Relevant inputs", [("Building use", building_use_r), ("Country", project_country_r)])

    # CRREM
    add_section("7. CRREM-Analysis")
    crrem = load_crrem_dataset(project_country_r)
    if crrem is None:
        story.append(Paragraph("CRREM dataset was not found. CRREM diagrams could not be generated.", styles["BodyText"]))
    else:
        target_label = str(st.session_state.get("crrem_target_select", "1.5°C") or "1.5°C")
        target_id = "1.5C" if target_label.startswith("1.5") else "2C"
        crrem_use = str(payload.get("crrem_use_type", st.session_state.get("crrem_use_type", "Office")) or "Office")
        mixed_records = payload.get("crrem_mixed_use", st.session_state.get("crrem_mixed_use_df", []))
        analysis_period = max(1, _to_int_lcc(lcc_global.get("analysis_period", 30), 30))
        ef_grid = crrem.get("ef_grid")
        max_year = int(min(int(project_year_r + analysis_period - 1), int(ef_grid.index.max()))) if ef_grid is not None and len(ef_grid) else int(project_year_r + analysis_period - 1)
        years_crrem = list(range(int(project_year_r), max_year + 1))
        carbon_limit, eui_limit = _report_crrem_limits_for_context(crrem, target_id, crrem_use, mixed_records, years_crrem)
        if not carbon_limit.empty:
            years_crrem = list(carbon_limit.index.astype(int))
        annual_em_t = compute_crrem_like_scenario_emissions_series(df_energy, payload, crrem, project_year_r, years_crrem).reindex(years_crrem).fillna(0.0)
        carbon_asset = (annual_em_t * 1000.0 / project_area_r) if project_area_r else annual_em_t * 0.0
        eui_asset = _report_eui_series_for_payload(df_energy, payload, years_crrem, project_area_r)
        stranding_c = find_stranding_year(carbon_asset, carbon_limit) if not carbon_limit.empty else None
        stranding_e = find_stranding_year(eui_asset, eui_limit) if not eui_limit.empty else None
        add_kpi_table("CRREM KPIs", [
            ("CRREM target", target_label),
            ("CRREM use type", crrem_use),
            ("Stranding year - Carbon", "Not stranded" if stranding_c is None else str(stranding_c)),
            ("Stranding year - EUI", "Not stranded" if stranding_e is None else str(stranding_e)),
        ])
        _report_add_chart(story, styles, "Carbon intensity vs CRREM pathway", _report_line_chart({"Project": carbon_asset, "CRREM limit": carbon_limit}, "Carbon Intensity vs CRREM", "kgCO₂e/m²·a", {"Project": scenario_color, "CRREM limit": CRREM_COLOR_LIMIT}, dashed={"CRREM limit"}), "Project annual emissions use CRREM decarbonization logic. The CRREM limit is the selected pathway for the project country and use type.")
        _report_add_chart(story, styles, "EUI vs CRREM pathway", _report_line_chart({"Project": eui_asset, "CRREM limit": eui_limit}, "EUI vs CRREM", "kWh/m²·a", {"Project": scenario_color, "CRREM limit": CRREM_COLOR_LIMIT}, dashed={"CRREM limit"}), "Project EUI is the annual factored consumption intensity. The CRREM limit is taken from the pathway dataset.")
        total_em_t = annual_em_t
        total_em_limit_t = carbon_limit * project_area_r / 1000.0 if not carbon_limit.empty else pd.Series(dtype=float)
        _report_add_chart(story, styles, "Total annual emissions", _report_line_chart({"Project": total_em_t, "CRREM limit": total_em_limit_t}, "Total Annual Emissions", "tCO₂e/a", {"Project": scenario_color, "CRREM limit": CRREM_COLOR_LIMIT}, dashed={"CRREM limit"}), "Annual emissions are converted from intensity to total emissions using project area.")
        _report_add_chart(story, styles, "Cumulative emissions", _report_line_chart({"Project cumulative": total_em_t.cumsum(), "CRREM cumulative limit": total_em_limit_t.cumsum()}, "Cumulative Emissions", "tCO₂e", {"Project cumulative": scenario_color, "CRREM cumulative limit": CRREM_COLOR_LIMIT}, dashed={"CRREM cumulative limit"}), "Cumulative emissions are the running sum of annual decarbonized emissions.")
    add_input_table("Relevant inputs", [("Country", project_country_r), ("Project year", project_year_r), ("Emission factors", ", ".join([f"{k}: {v:.4f}" for k, v in factor_map.items()])), ("CRREM target", str(st.session_state.get("crrem_target_select", "1.5°C")))])

    # LCC Analysis
    add_section("8. LCC-Analysis")
    cf = compute_lcc_cashflow_table(df_energy, payload, end_uses, project_year_r, lcc_global=lcc_global)
    if cf.empty:
        story.append(Paragraph("No LCC cash-flow data available. Add LCC assumptions and investment measures in the LCC-Analysis tab.", styles["BodyText"]))
    else:
        by_year = cf.groupby("Year", as_index=True).agg({"Nominal Cost": "sum", "Discounted Cost": "sum"})
        by_year["Cumulative Nominal Cost"] = by_year["Nominal Cost"].cumsum()
        by_year["Cumulative Discounted Cost"] = by_year["Discounted Cost"].cumsum()
        by_type = cf.groupby("Cost Type", as_index=True).agg({"Nominal Cost": "sum", "Discounted Cost": "sum"})
        by_end = cf.groupby("End_Use", as_index=True)["Nominal Cost"].sum()
        total_nom = float(by_year["Nominal Cost"].sum())
        total_disc = float(by_year["Discounted Cost"].sum())
        ref_name = str(lcc_global.get("payback_reference_scenario", "") or "")
        pb = None
        if ref_name and ref_name in st.session_state.get("scenarios", {}) and ref_name != active_name:
            try:
                ref_payload = st.session_state["scenarios"].get(ref_name, {}) or {}
                ref_df = get_energy_balance_df(file_bytes, filename, scenario_name=ref_name)
                ref_cf = compute_lcc_cashflow_table(ref_df, ref_payload, end_uses, project_year_r, lcc_global=lcc_global)
                pb = discounted_payback_period(cf, ref_cf, project_year_r)
            except Exception:
                pb = None
        add_kpi_table("LCC KPIs", [
            ("Total nominal LCC", f"{currency_r} {total_nom:,.0f}"),
            ("Total discounted LCC", f"{currency_r} {total_disc:,.0f}"),
            ("Nominal LCC intensity", f"{currency_r} {total_nom / project_area_r:,.2f}/m²" if project_area_r else "n/a"),
            ("Discounted LCC intensity", f"{currency_r} {total_disc / project_area_r:,.2f}/m²" if project_area_r else "n/a"),
            ("Discounted payback period", _format_payback(pb)),
        ])
        type_df = by_type.reset_index().rename(columns={"Nominal Cost": "Cost"})
        _report_add_chart(story, styles, "Annual LCC balance", _report_stacked_monthly_chart(cf.rename(columns={"Year": "Month"}), "Nominal Cost", "Cost Type", "Annual LCC Balance", f"{currency_r}/a", LCC_COST_TYPE_COLORS), "Annual LCC balance stacks energy, investment, maintenance and replacement costs by year.")
        _report_add_chart(story, styles, "Cumulative LCC", _report_line_chart({"Nominal": by_year["Cumulative Nominal Cost"], "Discounted": by_year["Cumulative Discounted Cost"]}, "Cumulative LCC", currency_r, {"Nominal": scenario_color, "Discounted": scenario_color}, dashed={"Discounted"}), "Nominal cost is undiscounted cash flow. Discounted cost is future cash flow converted to present value using the interest rate.")
        _report_add_chart(story, styles, "LCC by cost type", _report_pie_chart(by_type["Nominal Cost"], "Nominal LCC by Cost Type", f"{currency_r} {total_nom:,.0f}", LCC_COST_TYPE_COLORS), "Cost-type shares use total nominal cost over the analysis period.")
        _report_add_chart(story, styles, "LCC by assigned end use", _report_pie_chart(by_end / project_area_r if project_area_r else by_end, "Nominal LCC per m² by End Use", f"{currency_r} {total_nom / project_area_r:,.0f}\n/m²" if project_area_r else f"{currency_r} {total_nom:,.0f}", colors_eu), "Costs assigned to multiple end uses are allocated equally across the assigned end uses.")
    add_input_table("Relevant inputs", [
        ("Analysis period", f"{int(lcc_global.get('analysis_period', 30))} years"),
        ("Interest / discount rate", f"{float(lcc_global.get('interest_rate_pct', 0.0)):,.2f} %"),
        ("CAPEX/O&M inflation", f"{float(lcc_global.get('capex_inflation_pct', 0.0)):,.2f} %"),
        ("Operational end-use filter", ", ".join([ui_name(u) for u in lcc_global.get("selected_operational_end_uses", [])])),
        ("Energy inflation", ", ".join([f"{src}: {float((lcc_global.get('energy_inflation_pct', {}) or {}).get(src, 0.0)):,.2f}%" for src in ENERGY_SOURCE_ORDER])),
    ])

    # Scenarios section - active scenario only per report rule
    add_section("9. Scenarios")
    story.append(Paragraph("The report is generated for the active scenario only. Multi-scenario comparison diagrams remain available interactively in the app.", styles["BodyText"]))
    if not cf.empty:
        by_year = cf.groupby("Year", as_index=True).agg({"Nominal Cost": "sum", "Discounted Cost": "sum"})
        energy_by_year = cf.loc[cf["Cost Type"] == "Energy"].groupby("Year")["Nominal Cost"].sum().reindex(by_year.index).fillna(0.0)
        _report_add_chart(story, styles, "Annual energy cost", _report_line_chart({active_name: energy_by_year}, "Annual Energy Cost", f"{currency_r}/a", {active_name: scenario_color}), "Annual energy cost is calculated from active-scenario factored kWh, tariffs and energy inflation.")
        _report_add_chart(story, styles, "Cumulative LCC", _report_line_chart({"Nominal": by_year["Nominal Cost"].cumsum(), "Discounted": by_year["Discounted Cost"].cumsum()}, "Cumulative LCC - Active Scenario", currency_r, {"Nominal": scenario_color, "Discounted": scenario_color}, dashed={"Discounted"}), "Solid line is nominal cumulative cost; dashed line is discounted cumulative cost.")
    if crrem is not None:
        try:
            analysis_period = max(1, _to_int_lcc(lcc_global.get("analysis_period", 30), 30))
            years_sc = list(range(project_year_r, project_year_r + analysis_period))
            annual_em_sc = compute_crrem_like_scenario_emissions_series(df_energy, payload, crrem, project_year_r, years_sc).reindex(years_sc).fillna(0.0)
            target_id = "1.5C" if str(st.session_state.get("crrem_target_select", "1.5°C")).startswith("1.5") else "2C"
            carbon_limit_sc, _ = _report_crrem_limits_for_context(crrem, target_id, str(payload.get("crrem_use_type", "Office")), payload.get("crrem_mixed_use", []), years_sc)
            crrem_total = carbon_limit_sc * project_area_r / 1000.0 if not carbon_limit_sc.empty else pd.Series(dtype=float)
            _report_add_chart(story, styles, "Annual emissions", _report_line_chart({active_name: annual_em_sc, "CRREM-Baseline": crrem_total}, "Annual Emissions - Active Scenario", "tCO₂e/a", {active_name: scenario_color, "CRREM-Baseline": CRREM_COLOR_LIMIT}, dashed={"CRREM-Baseline"}), "Annual emissions use the same decarbonization pathway logic as the CRREM tab.")
            _report_add_chart(story, styles, "Cumulative emissions", _report_line_chart({active_name: annual_em_sc.cumsum(), "CRREM-Baseline": crrem_total.cumsum()}, "Cumulative Emissions - Active Scenario", "tCO₂e", {active_name: scenario_color, "CRREM-Baseline": CRREM_COLOR_LIMIT}, dashed={"CRREM-Baseline"}), "Cumulative emissions are the running sum of annual decarbonized emissions.")
        except Exception:
            pass
    add_input_table("Relevant inputs", [("Active scenario", active_name), ("Scenario color", scenario_color), ("Scenario-specific raw Energy_Balance override", "Yes" if get_scenario_energy_balance_override(active_name) is not None else "No")])

    # Model Inputs QA
    add_section("10. Model Inputs QA")
    try:
        mi_df = model_inputs_df_for_scenario(st.session_state.get("model_inputs_qa_df"), active_name)
        mi_qa = evaluate_model_inputs_qa_df(mi_df)
        mi_summary = model_inputs_qa_summary(mi_df)
        add_kpi_table("Model Inputs QA KPIs", [
            ("Input completeness", f"{mi_summary['completeness']} %"),
            ("Required inputs", f"{mi_summary['required']}"),
            ("Missing required", f"{mi_summary['missing']}"),
            ("Assumption-tagged inputs", f"{mi_summary['assumptions']}"),
            ("QA review flags", f"{mi_summary['review']}"),
        ])
        story.append(Paragraph("The report includes global model setup inputs and scenario-specific model inputs for the active scenario only. Assumption-tagged inputs should be reviewed and replaced with documented project references where possible.", styles["BodyText"]))
        story.append(Spacer(1, 0.25 * cm))
        for cat in MODEL_INPUT_CATEGORIES:
            sub_cat = mi_qa.loc[mi_qa["Category"].astype(str) == cat].copy()
            if sub_cat.empty:
                continue
            story.append(Paragraph(cat, styles["Heading3"]))
            for (scope_i, item_type_i, item_name_i), sub in sub_cat.groupby(["Scope", "Item Type", "Item Name"], dropna=False):
                story.append(Paragraph(f"{scope_i} — {item_type_i}: {item_name_i}", styles["BodyText"]))
                rows_mi = [["Parameter", "Value", "Unit", "Source", "QA", "Justification"]]
                for _, r in sub.iterrows():
                    rows_mi.append([
                        str(r.get("Parameter", "")),
                        str(r.get("Value", "")),
                        str(r.get("Unit", "")),
                        str(r.get("Source Type", "")),
                        str(r.get("QA Status", "")),
                        str(r.get("Range Justification", "")),
                    ])
                story.append(_report_table_flowable(rows_mi, col_widths=[3.5 * cm, 2.5 * cm, 1.5 * cm, 2.6 * cm, 2.2 * cm, 3.2 * cm], font_size=5.5))
                story.append(Spacer(1, 0.15 * cm))
    except Exception:
        story.append(Paragraph("Model Inputs QA data could not be loaded for this report.", styles["BodyText"]))
    add_input_table("Relevant inputs", [("Register source", "Model_Inputs_QA workbook sheet / Model Inputs QA tab"), ("Global inputs", "General Model Setup"), ("Scenario-specific inputs", "Room types, envelope components and systems"), ("Assumption tag", "Source Type = Assumption")])


    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    buf.seek(0)
    return buf.getvalue()


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
        "lcc": _default_lcc_payload(end_uses),
        "lcc_global": _default_lcc_global_payload(end_uses),
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
        "lcc": _capture_lcc_from_widgets(end_uses),
        "lcc_global": _get_lcc_global_state_payload(end_uses),
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

    _set_num("co2_Emissions_Electricity", float(f.get("Electricity", 0.300)), "{:.5f}")
    _set_num("co2_Emissions_Green_Electricity", float(f.get("Green Electricity", 0.000)), "{:.5f}")
    _set_num("co2_emissions_dh", float(f.get("District Heating", 0.260)), "{:.5f}")
    _set_num("co2_emissions_dc", float(f.get("District Cooling", 0.280)), "{:.5f}")
    _set_num("co2_emissions_gas", float(f.get("Gas", 0.180)), "{:.5f}")
    _set_num("co2_emissions_biomass", float(f.get("Biomass", 0.000)), "{:.5f}")

    _set_num("cost_electricity", float(t.get("Electricity", 0.35)), "{:.5f}")
    _set_num("cost_green_electricity", float(t.get("Green Electricity", 0.40)), "{:.5f}")
    _set_num("cost_dh", float(t.get("District Heating", 0.16)), "{:.5f}")
    _set_num("cost_dc", float(t.get("District Cooling", 0.16)), "{:.5f}")
    _set_num("cost_gas", float(t.get("Gas", 0.12)), "{:.5f}")
    _set_num("cost_biomass", float(t.get("Biomass", 0.10)), "{:.5f}")

    for use in end_uses:
        st.session_state[f"source_{use}"] = str(m.get(use, "Electricity"))
        _set_num(f"eff_{use}", float(e.get(use, 1.0)), "{:.5f}")

    _set_num("pv_scale", float(pv.get("scale", 1.0)), "{:.5f}")
    st.session_state["pv_sc_enabled"] = bool(pv.get("enabled", False))

    # CRREM measures (scenario-specific)
    st.session_state["crrem_measures_df"] = _measures_records_to_df(payload.get("crrem_measures", []))
    # Keep the measures editor in sync after loading a project/scenario (so the table shows saved measures)
    st.session_state["crrem_measures_draft_df"] = st.session_state["crrem_measures_df"].copy(deep=True)

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

    # LCC investments are scenario-specific. Global LCC assumptions are initialized once in the LCC tab.
    _load_lcc_into_widgets(payload, end_uses)


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
# Model Inputs QA helpers
# =========================
MODEL_INPUT_GLOBAL_SCENARIO = "GLOBAL"

MODEL_INPUT_CATEGORIES = [
    "General Model Setup",
    "Room Types",
    "Thermal Envelope",
    "AHU / Ventilation",
    "Heating",
    "Cooling",
    "Domestic Hot Water",
    "Pumps / Auxiliaries",
    "Controls",
    "Other / Custom Inputs",
]

MODEL_INPUT_ENVELOPE_COMPONENT_TYPES = [
    "External Walls",
    "Roofs",
    "Floors / Slabs",
    "Glazings",
    "External Doors",
    "Shading Devices",
    "Thermal Bridges",
    "Infiltration / Airtightness",
    "Other Envelope Component",
]

MODEL_INPUT_SYSTEM_TYPES = [
    "AHU / Ventilation",
    "Heating",
    "Cooling",
    "Domestic Hot Water",
    "Pumps / Auxiliaries",
    "Controls",
    "Other System",
]

MODEL_INPUT_SOURCE_TYPES = [
    "Design Document",
    "Specification",
    "Simulation Model",
    "Calculation",
    "Standard / Reference",
    "Measurement",
    "Assumption",
    "Other",
]

MODEL_INPUT_DHW_DEMAND_UNITS = [
    "L/person·day",
    "L/person·h",
    "L/m²·day",
    "L/m²·h",
    "m³/year",
    "m³/day",
    "L/day",
    "kWh/year",
    "Other",
]


MODEL_INPUT_COMMON_UNITS = [
    "-", "m²", "m³", "L", "m", "cm", "mm", "°", "°C", "K",
    "W", "kW", "MW", "W/m²", "W/person", "W/K", "W/mK", "W/m²K",
    "kWh", "kWh/a", "MWh/a", "%", "1/h", "m³/h", "m³/s", "L/s",
    "L/s.person", "m³/h.person", "L/s.m²", "m³/h.m²", "W/(L/s)",
    "kW/(m³/s)", "W/(m³/h)", "m²/person", "person/m²", "person/room",
    "min", "h", "h/day", "h/week", "full-load hours/a", "months", "years",
    "schedule name", "rule", "yes/no", "Other / Custom",
]


def _norm_model_input_text(s: str) -> str:
    """Normalize labels for robust parameter/unit lookup."""
    try:
        s = str(s or "").strip().lower()
        s = s.replace("²", "2").replace("³", "3")
        s = s.replace("·", ".")
        s = re.sub(r"\s+", " ", s)
        return s
    except Exception:
        return str(s or "").strip().lower()


def _unique_preserve_order(values) -> list:
    out = []
    seen = set()
    for v in values or []:
        vs = str(v or "").strip()
        if vs == "":
            continue
        if vs not in seen:
            out.append(vs)
            seen.add(vs)
    return out


def _split_model_input_multi_value(value: str) -> list:
    """Parse comma/semicolon/pipe separated values from saved text fields."""
    if value is None:
        return []
    parts = re.split(r"[,;|]", str(value))
    return [p.strip() for p in parts if p and p.strip()]


def _model_input_unit_options(category: str, item_type: str, parameter: str, current_unit: str = "") -> list:
    """Return metric unit dropdown options appropriate for each standard Model Inputs QA parameter.

    The options are intentionally practical, not exhaustive. They cover common metric units
    used in building-energy simulation model documentation. Custom parameters and unusual
    project-specific inputs can still use 'Other / Custom'.
    """
    cat = _norm_model_input_text(category)
    it = _norm_model_input_text(item_type)
    par = _norm_model_input_text(parameter)
    cur = str(current_unit or "").strip()

    # Exact/specific parameters first.
    exact = {
        "simulation timestep": ["min", "h"],
        "annual simulation period": ["months", "years", "days"],
        "modelled floor area": ["m²"],
        "area": ["m²"],
        "area / quantity": ["m²", "m", "piece", "unit"],
        "main performance value": ["-", "W/m²K", "W/m²", "kW", "kWh/a", "%"],
        "occupancy density": ["m²/person", "person/m²", "person/room"],
        "lighting power density": ["W/m²", "W/person", "W/room"],
        "equipment power density": ["W/m²", "W/person", "W/room"],
        "people sensible gain": ["W/person"],
        "people latent gain": ["W/person"],
        "heating delivery": ["-"],
        "cooling delivery": ["-"],
        "heating setpoint": ["°C"],
        "cooling setpoint": ["°C"],
        "night setback / setup": ["°C", "K", "rule"],
        "occupancy schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "lighting schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "daylight-controlled dimming": ["yes/no"],
        "equipment schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "outdoor air per person": ["L/s.person", "m³/h.person"],
        "outdoor air per area": ["L/s.m²", "m³/h.m²"],
        "demand controlled ventilation permitted": ["yes/no"],
        "demand controlled ventilation": ["yes/no"],
        "natural ventilation permitted": ["yes/no"],
        "demand controlled ventilation rule": ["rule", "-"],
        "natural ventilation rule": ["rule", "-"],
        "u-value": ["W/m²K"],
        "thermal bridge allowance": ["%", "W/K", "W/mK", "W/m²K"],
        "thermal mass": ["-", "kJ/m²K", "Wh/m²K", "kJ/m³K", "MJ/m³K"],
        "solar absorptance": ["-", "%"],
        "orientation / exposure": ["-", "°"],
        "orientation": ["-", "°"],
        "shgc / g-value": ["-", "%"],
        "visible transmittance": ["-", "%"],
        "frame fraction": ["%", "-"],
        "associated shading device": ["-"],
        "shading device / control": ["-", "rule"],
        "shading type": ["-"],
        "geometry / projection": ["m", "cm", "mm", "°", "rule"],
        "control rule": ["rule", "-"],
        "free cooling rule": ["rule", "-"],
        "free cooling / economizer rule": ["rule", "-"],
        "reduction factor": ["-", "%"],
        "infiltration rate": ["1/h", "L/s.m²", "m³/h.m²"],
        "infiltration schedule / rule": ["rule", "schedule name", "-"],
        "airtightness test value": ["n50", "q50", "m³/(h·m²)"],
        "served room types": ["room types"],
        "served room types / zones": ["room types"],
        "supply airflow": ["m³/h", "L/s", "m³/s"],
        "outdoor airflow": ["m³/h", "L/s", "m³/s"],
        "exhaust airflow": ["m³/h", "L/s", "m³/s"],
        "heat recovery type": ["-"],
        "heat recovery efficiency": ["%", "-"],
        "specific fan power": ["W/(L/s)", "kW/(m³/s)", "W/(m³/h)"],
        "supply air temperature": ["°C"],
        "ahu operation schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "humidification / dehumidification": ["yes/no", "rule", "-"],
        "system type": ["-"],
        "system description": ["-"],
        "energy source": ["-"],
        "overall efficiency / seasonal cop": ["-", "%", "COP", "SCOP", "EER", "SEER"],
        "generator efficiency / cop": ["-", "%", "COP", "SCOP"],
        "efficiency / control assumption": ["-", "%", "COP", "SCOP", "EER", "SEER", "rule"],
        "supply temperature": ["°C"],
        "return temperature": ["°C"],
        "design heating load": ["kW", "W", "MW"],
        "design cooling load": ["kW", "W", "MW"],
        "heating operation schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "cooling operation schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "distribution / storage losses": ["%", "kWh/a", "MWh/a", "W", "kW"],
        "hot water demand": MODEL_INPUT_DHW_DEMAND_UNITS,
        "dhw demand calculation method": ["-", "rule"],
        "storage volume": ["L", "m³"],
        "storage losses": ["kWh/a", "MWh/a", "W", "kW", "%"],
        "circulation losses": ["kWh/a", "MWh/a", "%", "W", "kW"],
        "dhw operation schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "operation schedule": ["schedule name", "h/day", "h/week", "full-load hours/a", "-"],
        "green roof / cool roof assumption": ["yes/no", "rule", "-"],
        "ground temperature / boundary condition": ["°C", "rule", "-"],
        "modelling rule": ["rule", "-"],
    }
    if par in exact:
        opts = list(exact[par])
    elif "schedule" in par:
        opts = ["schedule name", "h/day", "h/week", "full-load hours/a", "-"]
    elif "temperature" in par or "setpoint" in par:
        opts = ["°C", "K", "rule"]
    elif "airflow" in par or "flow" in par:
        opts = ["m³/h", "L/s", "m³/s"]
    elif "efficiency" in par or "cop" in par or "eer" in par:
        opts = ["-", "%", "COP", "SCOP", "EER", "SEER"]
    elif "loss" in par:
        opts = ["%", "kWh/a", "MWh/a", "W", "kW"]
    elif "area" in par:
        opts = ["m²"]
    elif "rule" in par:
        opts = ["rule", "-"]
    elif "permitted" in par or par.startswith("is "):
        opts = ["yes/no"]
    else:
        opts = ["-", "Other / Custom"]

    # Keep legacy/stored units available so older projects do not silently lose information.
    opts = _unique_preserve_order(([cur] if cur and cur not in opts else []) + opts + ["Other / Custom"])
    return opts


MODEL_INPUT_OBJECT_SCOPE_OPTIONS = ["Global", "Active scenario only"]

MODEL_INPUT_QA_COLUMNS = [
    "Scenario",
    "Scope",
    "Category",
    "Item Type",
    "Item Name",
    "Parameter",
    "Value",
    "Unit",
    "Required",
    "Source Type",
    "Source Document / Reference",
    "Reference / Target",
    "Min Check",
    "Max Check",
    "Usual Min",
    "Usual Max",
    "Range Justification",
    "Notes",
]

# Report/backwards-compatible alias. The previous version used a flat list of categories;
# the new implementation still exports one flat sheet, but the UI renders it as objects.
MODEL_INPUT_DISPLAY_CATEGORIES = MODEL_INPUT_CATEGORIES


def _mi_row(
        scenario: str,
        scope: str,
        category: str,
        item_type: str,
        item_name: str,
        parameter: str,
        value: str = "",
        unit: str = "-",
        required: bool = False,
        source_type: str = "Assumption",
        source_ref: str = "",
        reference: str = "",
        min_check=None,
        max_check=None,
        usual_min=None,
        usual_max=None,
        range_justification: str = "",
        notes: str = "",
) -> dict:
    return {
        "Scenario": str(scenario or MODEL_INPUT_GLOBAL_SCENARIO),
        "Scope": str(scope or "Scenario"),
        "Category": str(category or "Other / Custom Inputs"),
        "Item Type": str(item_type or "Custom"),
        "Item Name": str(item_name or "General"),
        "Parameter": str(parameter or ""),
        "Value": "" if value is None else str(value),
        "Unit": "" if unit is None else str(unit),
        "Required": bool(required),
        "Source Type": str(source_type or "Assumption"),
        "Source Document / Reference": "" if source_ref is None else str(source_ref),
        "Reference / Target": "" if reference is None else str(reference),
        "Min Check": min_check,
        "Max Check": max_check,
        "Usual Min": usual_min,
        "Usual Max": usual_max,
        "Range Justification": "" if range_justification is None else str(range_justification),
        "Notes": "" if notes is None else str(notes),
    }


def default_model_inputs_global_rows() -> list:
    """Global model setup inputs. These are intentionally project-wide, not scenario-specific."""
    return [
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Simulation tool", unit="-", required=True, source_type="Simulation Model", reference="e.g. IESVE / EnergyPlus / DesignBuilder"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Simulation tool version", unit="-", required=True, source_type="Simulation Model", reference="Version used for the final simulation results"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Weather file", unit="-", required=True, source_type="Simulation Model", reference="Weather station/file used in annual simulation"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Climate zone", unit="-", required=False, source_type="Standard / Reference", source_ref="ASHRAE / LEED / local code documentation", reference="Required when comparing against a formal baseline"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Simulation timestep", unit="min", required=False, source_type="Simulation Model", reference="Typically 60 min or sub-hourly", min_check=1, max_check=60),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Annual simulation period", unit="months", required=True, source_type="Simulation Model", reference="Should normally cover 12 months", min_check=12, max_check=12),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Modelled floor area", unit="m²", required=True, source_type="Design Document", source_ref="Area schedule / simulation model geometry", reference="Should match project area or document difference", min_check=1),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Area basis", unit="-", required=False, source_type="Design Document", reference="GFA / NFA / conditioned area / treated floor area"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Geometry / zoning source", unit="-", required=True, source_type="Design Document", source_ref="Architectural drawings / BIM / simulation model", reference="Trace model version and drawing issue"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Simulation purpose / rating basis", unit="-", required=False, source_type="Standard / Reference", reference="Design support / LEED / ASHRAE 90.1 PRM / local code / internal benchmark"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Excluded areas / systems", unit="-", required=False, source_type="Design Document", source_ref="Energy model report / compliance narrative", reference="Document any excluded systems or loads"),
        _mi_row(MODEL_INPUT_GLOBAL_SCENARIO, "Global", "General Model Setup", "General", "Simulation Model", "Modeller / reviewer", unit="-", required=False, source_type="Other", reference="Responsible person and QA reviewer"),
    ]


def _model_input_scope_values(scope: str, scenario: str) -> Tuple[str, str]:
    """Return (scenario_value, scope_value) for a global/scenario Model Inputs QA object."""
    scope_clean = str(scope or "Scenario").strip()
    if scope_clean.lower().startswith("global"):
        return MODEL_INPUT_GLOBAL_SCENARIO, "Global"
    return str(scenario or "Base"), "Scenario"


def room_type_template(item_name: str, scenario: str, scope: str = "Global") -> list:
    sc, scope_value = _model_input_scope_values(scope, scenario)
    return [
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Area", unit="m²", required=True, source_type="Design Document", source_ref="Area schedule / room book", min_check=0),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Occupancy density", unit="m²/person", required=False, source_type="Design Document", source_ref="Room data sheet / LEED calculator / design brief", min_check=1, max_check=100),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Lighting power density", unit="W/m²", required=True, source_type="Design Document", source_ref="Lighting concept / ASHRAE baseline / design brief", min_check=0, max_check=50),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Equipment power density", unit="W/m²", required=True, source_type="Design Document", source_ref="Equipment schedule / design brief", min_check=0, max_check=150),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "People sensible gain", unit="W/person", required=False, source_type="Assumption", reference="Document activity/metabolic assumption", min_check=0, max_check=200),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "People latent gain", unit="W/person", required=False, source_type="Assumption", reference="Document activity/metabolic assumption", min_check=0, max_check=200),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Heating delivery", value="", unit="-", required=False, source_type="Design Document", source_ref="HVAC concept / room data sheet", reference="Radiant Ceiling / Fan Coil / Floor Heating / Radiator / Air System / custom"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Cooling delivery", value="", unit="-", required=False, source_type="Design Document", source_ref="HVAC concept / room data sheet", reference="Radiant Ceiling / Fan Coil / Floor Cooling / Fan Coil / Air System / custom"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Heating setpoint", unit="°C", required=True, source_type="Design Document", source_ref="Owner requirements / room data sheet", min_check=10, max_check=26),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Cooling setpoint", unit="°C", required=True, source_type="Design Document", source_ref="Owner requirements / room data sheet", min_check=18, max_check=35),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Night setback / setup", unit="°C or rule", required=False, source_type="Assumption", reference="Document unoccupied temperature control logic"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Occupancy schedule", unit="-", required=True, source_type="Design Document", source_ref="Operation concept / design brief", reference="Weekday/weekend and holiday operation"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Lighting schedule", unit="-", required=True, source_type="Assumption", source_ref="Operation concept / lighting controls", reference="Equivalent full-load hours or schedule name"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Daylight-Controlled dimming", value="No", unit="yes/no", required=False, source_type="Design Document", source_ref="Lighting control concept / room data sheet", reference="Tick yes if daylight sensors or daylight-linked automatic dimming are modelled"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Equipment schedule", unit="-", required=True, source_type="Assumption", source_ref="Operation concept / equipment load assumptions", reference="Equivalent full-load hours or schedule name"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Outdoor air per person", unit="L/s.person", required=False, source_type="Standard / Reference", source_ref="ASHRAE 62.1 / EN 16798 / project brief", min_check=0, max_check=50),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Outdoor air per area", unit="L/s.m²", required=False, source_type="Standard / Reference", source_ref="ASHRAE 62.1 / EN 16798 / project brief", min_check=0, max_check=10),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Demand controlled ventilation permitted", unit="yes/no", required=False, source_type="Design Document", source_ref="Controls narrative"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Demand controlled ventilation rule", unit="-", required=False, source_type="Design Document", source_ref="Controls narrative", reference="CO₂ setpoint, minimum outdoor air, occupancy sensor logic"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Natural ventilation permitted", unit="yes/no", required=False, source_type="Design Document", source_ref="Ventilation / façade / controls concept"),
        _mi_row(sc, scope_value, "Room Types", "Room Type", item_name, "Natural ventilation rule", unit="-", required=False, source_type="Design Document", source_ref="Ventilation / controls concept", reference="Opening schedule, temperature limits, CO₂ limits, wind/rain lockout"),
    ]


def envelope_component_template(component_type: str, item_name: str, scenario: str, scope: str = "Scenario") -> list:
    sc, scope_value = _model_input_scope_values(scope, scenario)
    ct = str(component_type)
    if ct == "External Walls":
        params = [
            ("Area", "m²", True, "Design Document", "Envelope schedule / model geometry", "", 0, None),
            ("U-value", "W/m²K", True, "Design Document", "U-value calculation / envelope specification", "Compare with code/baseline", 0.05, 2.5),
            ("Thermal Mass", "-", False, "Design Document", "Construction build-up / material layer schedule", "Very low / low / medium / high / very high, or enter thermal capacity as custom value", None, None),
            ("Thermal bridge allowance", "% or W/K", False, "Calculation", "Envelope calculation", "Document method", 0, 50),
            ("Solar absorptance", "-", False, "Assumption", "Material specification", "", 0, 1),
            ("Orientation / exposure", "-", False, "Design Document", "Model geometry", "", None, None),
        ]
    elif ct == "Roofs":
        params = [
            ("Area", "m²", True, "Design Document", "Envelope schedule / model geometry", "", 0, None),
            ("U-value", "W/m²K", True, "Design Document", "U-value calculation / envelope specification", "Compare with code/baseline", 0.04, 1.5),
            ("Thermal Mass", "-", False, "Design Document", "Construction build-up / material layer schedule", "Very low / low / medium / high / very high, or enter thermal capacity as custom value", None, None),
            ("Solar absorptance", "-", False, "Assumption", "Roof finish specification", "", 0, 1),
            ("Green roof / cool roof assumption", "yes/no/rule", False, "Design Document", "Roof concept", "", None, None),
        ]
    elif ct == "Floors / Slabs":
        params = [
            ("Area", "m²", True, "Design Document", "Envelope schedule / model geometry", "", 0, None),
            ("U-value", "W/m²K", True, "Design Document", "U-value calculation / envelope specification", "Compare with code/baseline", 0.04, 2.0),
            ("Thermal Mass", "-", False, "Design Document", "Construction build-up / material layer schedule", "Very low / low / medium / high / very high, or enter thermal capacity as custom value", None, None),
            ("Ground temperature / boundary condition", "-", False, "Simulation Model", "Model input", "", None, None),
        ]
    elif ct == "Glazings":
        params = [
            ("Area", "m²", True, "Design Document", "Façade schedule / model geometry", "", 0, None),
            ("U-value", "W/m²K", True, "Design Document", "Glazing specification", "Compare with code/baseline", 0.5, 6.0),
            ("SHGC / g-value", "-", True, "Design Document", "Glazing specification", "", 0.05, 0.9),
            ("Visible transmittance", "-", False, "Design Document", "Glazing specification", "", 0.05, 0.9),
            ("Frame fraction", "%", False, "Design Document", "Façade schedule", "", 0, 80),
            ("Orientation", "-", False, "Design Document", "Model geometry", "", None, None),
            ("Associated shading device", "-", False, "Design Document", "Shading concept / façade schedule", "Select an existing Shading Device object where applicable", None, None),
            ("Shading device / control", "-", False, "Design Document", "Shading concept / controls", "Internal/external, fixed/dynamic, schedule/rule", None, None),
        ]
    elif ct == "Shading Devices":
        params = [
            ("Shading type", "-", True, "Design Document", "Façade / shading concept", "Fixed, movable, electrochromic, internal blind, etc.", None, None),
            ("Geometry / projection", "m or rule", False, "Design Document", "Façade drawings", "Overhang/fin depth, spacing, angle", None, None),
            ("Control rule", "-", False, "Design Document", "Controls narrative", "Solar radiation, glare, schedule, temperature, manual", None, None),
            ("Reduction factor", "-", False, "Assumption", "Simulation model", "Document if simplified", 0, 1),
        ]
    elif ct == "Infiltration / Airtightness":
        params = [
            ("Infiltration rate", "1/h or L/s.m²", True, "Assumption", "Airtightness concept / model input", "Flag assumption unless measured/specified", 0, 5),
            ("Infiltration schedule / rule", "-", False, "Assumption", "Simulation model", "Wind/stack/schedule logic", None, None),
            ("Airtightness test value", "n50 or q50", False, "Measurement", "Blower-door test / specification", "", 0, 20),
        ]
    else:
        params = [
            ("Area / quantity", "m² or unit", False, "Design Document", "Project documentation", "", 0, None),
            ("Main performance value", "project unit", False, "Assumption", "Project documentation", "Document unit and method", None, None),
            ("Modelling rule", "-", False, "Simulation Model", "Energy model input", "", None, None),
        ]
    return [_mi_row(sc, scope_value, "Thermal Envelope", ct, item_name, p, unit=u, required=req, source_type=src, source_ref=ref, reference=tgt, min_check=mn, max_check=mx) for p, u, req, src, ref, tgt, mn, mx in params]


def system_template(system_type: str, item_name: str, scenario: str, scope: str = "Scenario") -> list:
    sc, scope_value = _model_input_scope_values(scope, scenario)
    stype = str(system_type)
    if stype == "AHU / Ventilation":
        params = [
            ("Served Room Types", "room types", True, "Design Document", "MEP concept / zone list", "Select one or more existing Room Type objects served by this AHU", None, None),
            ("Supply airflow", "m³/h", True, "Design Document", "Air balance / AHU schedule", "", 0, None),
            ("Outdoor airflow", "m³/h", True, "Design Document", "Air balance / ventilation calculation", "", 0, None),
            ("Exhaust airflow", "m³/h", False, "Design Document", "Air balance", "", 0, None),
            ("Heat recovery type", "-", False, "Design Document", "AHU specification", "Plate, rotary, run-around, none", None, None),
            ("Heat recovery efficiency", "%", False, "Design Document", "AHU specification", "Sensible/total effectiveness used in model", 0, 95),
            ("Specific fan power", "W/(L/s)", False, "Design Document", "Fan/AHU schedule", "Check against selected standard/project target", 0, 8),
            ("Supply air temperature", "°C", False, "Design Document", "Controls sequence / AHU schedule", "", 10, 35),
            ("AHU operation schedule", "-", True, "Design Document", "Controls narrative / operation concept", "", None, None),
            ("Demand controlled ventilation", "yes/no", False, "Design Document", "Controls narrative", "", None, None),
            ("Demand controlled ventilation rule", "-", False, "Design Document", "Controls narrative", "CO₂ setpoint, occupancy sensor, min OA", None, None),
            ("Free cooling / economizer rule", "-", False, "Design Document", "Controls narrative", "Outdoor air temperature/enthalpy limits", None, None),
            ("Humidification / dehumidification", "yes/no/rule", False, "Design Document", "MEP concept / controls", "", None, None),
        ]
    elif stype == "Heating":
        params = [
            ("System type", "-", True, "Design Document", "MEP concept", "Boiler, heat pump, district heating, etc.", None, None),
            ("Energy source", "-", True, "Design Document", "MEP concept / energy source mapping", "Must be consistent with Energy_Balance mapping", None, None),
            ("Overall efficiency / seasonal COP", "-", True, "Design Document", "Equipment datasheet / model input", "Boiler efficiency or heat-pump SCOP", 0, 10),
            ("Supply temperature", "°C", False, "Design Document", "Heating concept", "", 20, 90),
            ("Return temperature", "°C", False, "Design Document", "Heating concept", "", 15, 80),
            ("Design heating load", "kW", False, "Simulation Model", "Loads analysis / sizing report", "", 0, None),
            ("Heating operation schedule", "-", False, "Design Document", "Controls narrative", "", None, None),
            ("Control rule", "-", False, "Design Document", "Controls narrative", "Weather compensation, setback, room control", None, None),
            ("Distribution / storage losses", "% or kWh/a", False, "Assumption", "MEP calculation / model input", "Document method", 0, 50),
        ]
    elif stype == "Cooling":
        params = [
            ("System type", "-", True, "Design Document", "MEP concept", "Chiller, heat pump, district cooling, VRF, etc.", None, None),
            ("Energy source", "-", True, "Design Document", "MEP concept / energy source mapping", "Must be consistent with Energy_Balance mapping", None, None),
            ("Overall efficiency / seasonal COP", "-", True, "Design Document", "Equipment datasheet / model input", "Chiller COP, SEER, EER, DC assumption", 0, 15),
            ("Supply temperature", "°C", False, "Design Document", "Cooling concept", "", 4, 25),
            ("Return temperature", "°C", False, "Design Document", "Cooling concept", "", 6, 30),
            ("Design cooling load", "kW", False, "Simulation Model", "Loads analysis / sizing report", "", 0, None),
            ("Cooling operation schedule", "-", False, "Design Document", "Controls narrative", "", None, None),
            ("Free cooling rule", "-", False, "Design Document", "Controls narrative", "", None, None),
            ("Distribution / storage losses", "% or kWh/a", False, "Assumption", "MEP calculation / model input", "Document method", 0, 50),
        ]
    elif stype == "Domestic Hot Water":
        params = [
            ("System type", "-", True, "Design Document", "DHW concept", "Boiler, heat pump, district heating, electric, solar thermal", None, None),
            ("Energy source", "-", True, "Design Document", "DHW concept / energy source mapping", "", None, None),
            ("Generator efficiency / COP", "-", True, "Design Document", "Equipment datasheet / model input", "", 0, 10),
            ("Hot Water Demand", "L/person·day", True, "Calculation", "DHW calculation / design brief / plumbing fixture schedule", "Document basis: persons, area, fixtures, schedule or annual volume", 0, None),
            ("DHW demand calculation method", "-", False, "Calculation", "DHW calculation / design brief", "Persons, fixtures, litres/day, schedule", None, None),
            ("Storage volume", "L", False, "Design Document", "DHW specification", "", 0, None),
            ("Storage losses", "kWh/a or W", False, "Assumption", "DHW calculation / model input", "", 0, None),
            ("Circulation losses", "kWh/a or %", False, "Assumption", "DHW calculation / model input", "", 0, None),
            ("DHW operation schedule", "-", False, "Design Document", "Operation concept", "", None, None),
        ]
    else:
        params = [
            ("System description", "-", True, "Design Document", "MEP concept / simulation model", "", None, None),
            ("Energy source", "-", False, "Design Document", "MEP concept / energy source mapping", "", None, None),
            ("Efficiency / control assumption", "-", False, "Assumption", "Model input / specification", "", None, None),
            ("Operation schedule", "-", False, "Design Document", "Controls / operation concept", "", None, None),
        ]
    return [_mi_row(sc, scope_value, stype if stype in MODEL_INPUT_CATEGORIES else "Other / Custom Inputs", stype, item_name, p, unit=u, required=req, source_type=src, source_ref=ref, reference=tgt, min_check=mn, max_check=mx) for p, u, req, src, ref, tgt, mn, mx in params]


def _model_input_usual_range(category: str, item_type: str, parameter: str, unit: str = "") -> Tuple[float, float]:
    """Return informative usual-value bounds for standard Model Inputs QA parameters.

    These ranges are pragmatic QA plausibility ranges for common building energy simulation inputs.
    They are not code-compliance limits and should be overridden by project standards where needed.
    Custom/user-defined parameters intentionally return NaN/NaN.
    """
    cat = str(category or "").strip().lower()
    it = str(item_type or "").strip().lower()
    p = str(parameter or "").strip().lower()
    u = str(unit or "").strip().lower()

    # Unit-sensitive DHW demand ranges.
    if p == "hot water demand":
        if "person" in u and "day" in u:
            return 5.0, 80.0
        if "person" in u and "h" in u:
            return 0.1, 10.0
        if "m²" in u and "day" in u or "m2" in u and "day" in u:
            return 0.01, 10.0
        if "m²" in u and "h" in u or "m2" in u and "h" in u:
            return 0.001, 2.0
        if "m³/year" in u or "m3/year" in u:
            return 0.1, 100000.0
        if "m³/day" in u or "m3/day" in u:
            return 0.01, 500.0
        if "l/day" in u:
            return 1.0, 500000.0
        if "kwh/year" in u:
            return 1.0, 1000000.0
        return 0.0, np.nan

    # General setup.
    if cat == "general model setup":
        general_ranges = {
            "simulation timestep": (5.0, 60.0),
            "annual simulation period": (12.0, 12.0),
            "modelled floor area": (10.0, 10000000.0),
        }
        return general_ranges.get(p, (np.nan, np.nan))

    # Room types.
    if it == "room type" or cat == "room types":
        room_ranges = {
            "area": (1.0, 100000.0),
            "occupancy density": (3.0, 50.0),
            "lighting power density": (2.0, 25.0),
            "equipment power density": (1.0, 80.0),
            "people sensible gain": (40.0, 120.0),
            "people latent gain": (20.0, 100.0),
            "heating setpoint": (18.0, 23.0),
            "cooling setpoint": (22.0, 28.0),
            "outdoor air per person": (3.0, 20.0),
            "outdoor air per area": (0.05, 5.0),
        }
        return room_ranges.get(p, (np.nan, np.nan))

    # Envelope components.
    if cat == "thermal envelope":
        if it == "external walls":
            ranges = {
                "area": (1.0, 500000.0),
                "u-value": (0.15, 0.25),
                "thermal bridge allowance": (0.0, 20.0),
                "solar absorptance": (0.20, 0.90),
            }
            return ranges.get(p, (np.nan, np.nan))
        if it == "roofs":
            ranges = {
                "area": (1.0, 500000.0),
                "u-value": (0.08, 0.25),
                "solar absorptance": (0.20, 0.90),
            }
            return ranges.get(p, (np.nan, np.nan))
        if it == "floors / slabs":
            ranges = {
                "area": (1.0, 500000.0),
                "u-value": (0.10, 0.35),
            }
            return ranges.get(p, (np.nan, np.nan))
        if it == "glazings":
            ranges = {
                "area": (1.0, 300000.0),
                "u-value": (0.70, 1.80),
                "shgc / g-value": (0.20, 0.60),
                "visible transmittance": (0.35, 0.75),
                "frame fraction": (10.0, 45.0),
            }
            return ranges.get(p, (np.nan, np.nan))
        if it == "external doors":
            ranges = {
                "area": (0.5, 10000.0),
                "u-value": (0.80, 2.50),
            }
            return ranges.get(p, (np.nan, np.nan))
        if it == "shading devices":
            ranges = {
                "reduction factor": (0.10, 0.90),
            }
            return ranges.get(p, (np.nan, np.nan))
        if it == "thermal bridges":
            ranges = {
                "thermal bridge allowance": (0.0, 20.0),
                "main performance value": (0.0, 1.0),
            }
            return ranges.get(p, (np.nan, np.nan))
        if it == "infiltration / airtightness":
            ranges = {
                "infiltration rate": (0.05, 1.00),
                "airtightness test value": (0.30, 5.00),
            }
            return ranges.get(p, (np.nan, np.nan))

    # Systems.
    if it == "ahu / ventilation":
        ranges = {
            "supply airflow": (50.0, 250000.0),
            "outdoor airflow": (10.0, 250000.0),
            "exhaust airflow": (0.0, 250000.0),
            "heat recovery efficiency": (50.0, 85.0),
            "specific fan power": (0.50, 3.00),
            "supply air temperature": (14.0, 22.0),
        }
        return ranges.get(p, (np.nan, np.nan))
    if it == "heating":
        ranges = {
            "overall efficiency / seasonal cop": (0.70, 5.50),
            "supply temperature": (28.0, 65.0),
            "return temperature": (22.0, 55.0),
            "design heating load": (0.1, 10000.0),
            "distribution / storage losses": (0.0, 25.0),
        }
        return ranges.get(p, (np.nan, np.nan))
    if it == "cooling":
        ranges = {
            "overall efficiency / seasonal cop": (2.0, 8.0),
            "supply temperature": (6.0, 18.0),
            "return temperature": (10.0, 24.0),
            "design cooling load": (0.1, 10000.0),
            "distribution / storage losses": (0.0, 25.0),
        }
        return ranges.get(p, (np.nan, np.nan))
    if it == "domestic hot water":
        ranges = {
            "generator efficiency / cop": (0.70, 5.00),
            "storage volume": (5.0, 50000.0),
            "storage losses": (0.0, 100000.0),
            "circulation losses": (0.0, 100000.0),
        }
        return ranges.get(p, (np.nan, np.nan))

    return np.nan, np.nan


def _apply_model_input_usual_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Populate usual-value bounds for standard parameters while preserving custom rows."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "Usual Min" not in out.columns:
        out["Usual Min"] = np.nan
    if "Usual Max" not in out.columns:
        out["Usual Max"] = np.nan
    for idx, row in out.iterrows():
        mn, mx = _model_input_usual_range(
            row.get("Category", ""), row.get("Item Type", ""), row.get("Parameter", ""), row.get("Unit", "")
        )
        if pd.notna(mn):
            out.at[idx, "Usual Min"] = float(mn)
        if pd.notna(mx):
            out.at[idx, "Usual Max"] = float(mx)
    return out


def _is_model_input_out_of_usual_range(value, usual_min, usual_max) -> bool:
    num = _extract_first_number(value)
    if num is None:
        return False
    try:
        if pd.notna(usual_min) and float(num) < float(usual_min):
            return True
    except Exception:
        pass
    try:
        if pd.notna(usual_max) and float(num) > float(usual_max):
            return True
    except Exception:
        pass
    return False


def _format_usual_range(usual_min, usual_max, unit: str = "") -> str:
    unit_s = str(unit or "").strip()
    try:
        has_min = pd.notna(usual_min)
    except Exception:
        has_min = False
    try:
        has_max = pd.notna(usual_max)
    except Exception:
        has_max = False
    if has_min and has_max:
        core = f"{float(usual_min):g} – {float(usual_max):g}"
    elif has_min:
        core = f">= {float(usual_min):g}"
    elif has_max:
        core = f"<= {float(usual_max):g}"
    else:
        return ""
    return f"{core} {unit_s}".strip()


def default_model_inputs_qa_df() -> pd.DataFrame:
    """Return the default global register. Scenario-specific objects are added by the user."""
    return sanitize_model_inputs_qa_df(pd.DataFrame(default_model_inputs_global_rows()))


def sanitize_model_inputs_qa_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Clean Model Inputs QA data and provide backwards compatibility with the v2.2.0 flat table."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        out = pd.DataFrame(default_model_inputs_global_rows())
    else:
        out = df.copy()

    # Backwards compatibility from the first flat implementation.
    cat_series = out["Category"].astype(str) if "Category" in out.columns else pd.Series([""] * len(out), index=out.index)
    if "Scenario" not in out.columns:
        out["Scenario"] = np.where(cat_series.eq("General Model Setup"), MODEL_INPUT_GLOBAL_SCENARIO, "Base")
    if "Scope" not in out.columns:
        out["Scope"] = np.where(out["Scenario"].astype(str).eq(MODEL_INPUT_GLOBAL_SCENARIO), "Global", "Scenario")
    if "Item Type" not in out.columns:
        out["Item Type"] = cat_series.replace("", "Other / Custom Inputs")
    if "Item Name" not in out.columns:
        out["Item Name"] = np.where(cat_series.eq("General Model Setup"), "Simulation Model", cat_series.replace("", "Custom"))

    for col in MODEL_INPUT_QA_COLUMNS:
        if col not in out.columns:
            if col == "Required":
                out[col] = False
            elif col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
                out[col] = np.nan
            elif col == "Scenario":
                out[col] = MODEL_INPUT_GLOBAL_SCENARIO
            elif col == "Scope":
                out[col] = "Scenario"
            else:
                out[col] = ""

    out = out[MODEL_INPUT_QA_COLUMNS].copy()
    for col in ["Scenario", "Scope", "Category", "Item Type", "Item Name", "Parameter", "Value", "Unit", "Source Type", "Source Document / Reference", "Reference / Target", "Range Justification", "Notes"]:
        out[col] = out[col].fillna("").astype(str)

    out["Scenario"] = out["Scenario"].replace("", MODEL_INPUT_GLOBAL_SCENARIO)
    out["Scope"] = out["Scope"].replace("", "Scenario")
    out.loc[out["Scenario"].astype(str).eq(MODEL_INPUT_GLOBAL_SCENARIO), "Scope"] = "Global"
    out.loc[out["Scope"].astype(str).str.lower().eq("global"), "Scenario"] = MODEL_INPUT_GLOBAL_SCENARIO
    out["Category"] = out["Category"].replace("", "Other / Custom Inputs")
    # Do not use Series as the replacement value in Series.replace(); pandas raises
    # ValueError for scalar-to-Series replacement. Fill row-wise with masks instead.
    _item_type_empty = out["Item Type"].astype(str).str.strip().eq("")
    out.loc[_item_type_empty, "Item Type"] = out.loc[_item_type_empty, "Category"].astype(str)
    out["Item Name"] = out["Item Name"].replace("", "General")
    out["Source Type"] = out["Source Type"].replace("", "Assumption")
    # Backwards compatibility: v2.2.9 used a free-text AHU parameter name.
    out.loc[out["Parameter"].astype(str).eq("Served room types / zones"), "Parameter"] = "Served Room Types"
    out.loc[out["Parameter"].astype(str).eq("Served Room Types") & out["Unit"].astype(str).str.strip().eq(""), "Unit"] = "room types"

    def _to_bool(x):
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in {"1", "true", "yes", "y", "x", "required"}

    out["Required"] = out["Required"].apply(_to_bool)
    for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop fully empty parameter rows.
    try:
        out = out.loc[out["Parameter"].astype(str).str.strip() != ""].reset_index(drop=True)
    except Exception:
        pass

    # Ensure global setup rows exist once.
    existing_global_params = set(out.loc[out["Scenario"].astype(str).eq(MODEL_INPUT_GLOBAL_SCENARIO), "Parameter"].astype(str))
    add_rows = [r for r in default_model_inputs_global_rows() if str(r.get("Parameter")) not in existing_global_params]
    if add_rows:
        out = pd.concat([out, pd.DataFrame(add_rows)], ignore_index=True)
        out = out[MODEL_INPUT_QA_COLUMNS].copy()
        for col in ["Min Check", "Max Check"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Backwards compatibility for v2.2.4: ensure existing Room Type objects also receive
    # the standard heating/cooling delivery parameters introduced after earlier projects
    # may already have created room types.
    try:
        room_items = out.loc[
            out["Category"].astype(str).eq("Room Types")
            & out["Item Type"].astype(str).eq("Room Type"),
            ["Scenario", "Scope", "Item Name"]
        ].drop_duplicates()
        room_delivery_defaults = {
            "Heating delivery": {
                "source_ref": "HVAC concept / room data sheet",
                "reference": "Radiant Ceiling / Fan Coil / Floor Heating / Radiator / Air System / custom",
            },
            "Cooling delivery": {
                "source_ref": "HVAC concept / room data sheet",
                "reference": "Radiant Ceiling / Fan Coil / Floor Cooling / Fan Coil / Air System / custom",
            },
        }
        delivery_rows = []
        for _, item in room_items.iterrows():
            sc = str(item.get("Scenario", "Base"))
            scope = str(item.get("Scope", "Scenario")) or "Scenario"
            nm = str(item.get("Item Name", "Room Type"))
            existing_params = set(out.loc[
                out["Scenario"].astype(str).eq(sc)
                & out["Category"].astype(str).eq("Room Types")
                & out["Item Type"].astype(str).eq("Room Type")
                & out["Item Name"].astype(str).eq(nm),
                "Parameter"
            ].astype(str))
            for param, meta in room_delivery_defaults.items():
                if param not in existing_params:
                    delivery_rows.append(_mi_row(
                        sc,
                        scope if scope in ["Global", "Scenario"] else "Scenario",
                        "Room Types",
                        "Room Type",
                        nm,
                        param,
                        value="",
                        unit="-",
                        required=False,
                        source_type="Design Document",
                        source_ref=meta["source_ref"],
                        reference=meta["reference"],
                    ))
        if delivery_rows:
            out = pd.concat([out, pd.DataFrame(delivery_rows)], ignore_index=True)
            out = out[MODEL_INPUT_QA_COLUMNS].copy()
            for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    except Exception:
        pass

    # Backwards compatibility for v2.2.16: ensure existing Room Type objects
    # include the standard daylight-controlled dimming parameter.
    try:
        room_items = out.loc[
            out["Category"].astype(str).eq("Room Types")
            & out["Item Type"].astype(str).eq("Room Type"),
            ["Scenario", "Scope", "Item Name"]
        ].drop_duplicates()
        daylight_rows = []
        for _, item in room_items.iterrows():
            sc = str(item.get("Scenario", "Base"))
            scope = str(item.get("Scope", "Scenario")) or "Scenario"
            nm = str(item.get("Item Name", "Room Type"))
            existing_params = set(out.loc[
                out["Scenario"].astype(str).eq(sc)
                & out["Category"].astype(str).eq("Room Types")
                & out["Item Type"].astype(str).eq("Room Type")
                & out["Item Name"].astype(str).eq(nm),
                "Parameter"
            ].astype(str))
            if "Daylight-Controlled dimming" not in existing_params:
                daylight_rows.append(_mi_row(
                    sc,
                    scope if scope in ["Global", "Scenario"] else "Scenario",
                    "Room Types",
                    "Room Type",
                    nm,
                    "Daylight-Controlled dimming",
                    value="No",
                    unit="yes/no",
                    required=False,
                    source_type="Design Document",
                    source_ref="Lighting control concept / room data sheet",
                    reference="Tick yes if daylight sensors or daylight-linked automatic dimming are modelled",
                ))
        if daylight_rows:
            out = pd.concat([out, pd.DataFrame(daylight_rows)], ignore_index=True)
            out = out[MODEL_INPUT_QA_COLUMNS].copy()
            for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    except Exception:
        pass

    # Backwards compatibility for v2.2.16: ensure existing opaque envelope
    # construction objects include the standard Thermal Mass parameter.
    try:
        thermal_mass_types = ["External Walls", "Roofs", "Floors / Slabs"]
        env_items = out.loc[
            out["Category"].astype(str).eq("Thermal Envelope")
            & out["Item Type"].astype(str).isin(thermal_mass_types),
            ["Scenario", "Scope", "Item Type", "Item Name"]
        ].drop_duplicates()
        tm_rows = []
        for _, item in env_items.iterrows():
            sc = str(item.get("Scenario", "Base"))
            scope = str(item.get("Scope", "Scenario")) or "Scenario"
            it = str(item.get("Item Type", "External Walls"))
            nm = str(item.get("Item Name", it))
            existing_params = set(out.loc[
                out["Scenario"].astype(str).eq(sc)
                & out["Category"].astype(str).eq("Thermal Envelope")
                & out["Item Type"].astype(str).eq(it)
                & out["Item Name"].astype(str).eq(nm),
                "Parameter"
            ].astype(str))
            if "Thermal Mass" not in existing_params:
                tm_rows.append(_mi_row(
                    sc,
                    scope if scope in ["Global", "Scenario"] else "Scenario",
                    "Thermal Envelope",
                    it,
                    nm,
                    "Thermal Mass",
                    value="",
                    unit="-",
                    required=False,
                    source_type="Design Document",
                    source_ref="Construction build-up / material layer schedule",
                    reference="Very low / low / medium / high / very high, or enter thermal capacity as custom value",
                ))
        if tm_rows:
            out = pd.concat([out, pd.DataFrame(tm_rows)], ignore_index=True)
            out = out[MODEL_INPUT_QA_COLUMNS].copy()
            for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    except Exception:
        pass

    # Backwards compatibility for v2.2.5: ensure existing Domestic Hot Water
    # system objects include a standard Hot Water Demand parameter.
    try:
        dhw_items = out.loc[
            out["Category"].astype(str).eq("Domestic Hot Water")
            & out["Item Type"].astype(str).eq("Domestic Hot Water"),
            ["Scenario", "Scope", "Item Name"]
        ].drop_duplicates()
        dhw_rows = []
        for _, item in dhw_items.iterrows():
            sc = str(item.get("Scenario", "Base"))
            scope = str(item.get("Scope", "Scenario")) or "Scenario"
            nm = str(item.get("Item Name", "DHW System"))
            existing_params = set(out.loc[
                out["Scenario"].astype(str).eq(sc)
                & out["Category"].astype(str).eq("Domestic Hot Water")
                & out["Item Type"].astype(str).eq("Domestic Hot Water")
                & out["Item Name"].astype(str).eq(nm),
                "Parameter"
            ].astype(str))
            if "Hot Water Demand" not in existing_params:
                dhw_rows.append(_mi_row(
                    sc,
                    scope if scope in ["Global", "Scenario"] else "Scenario",
                    "Domestic Hot Water",
                    "Domestic Hot Water",
                    nm,
                    "Hot Water Demand",
                    value="",
                    unit="L/person·day",
                    required=True,
                    source_type="Calculation",
                    source_ref="DHW calculation / design brief / plumbing fixture schedule",
                    reference="Document basis: persons, area, fixtures, schedule or annual volume",
                    min_check=0,
                    max_check=np.nan,
                ))
        if dhw_rows:
            out = pd.concat([out, pd.DataFrame(dhw_rows)], ignore_index=True)
            out = out[MODEL_INPUT_QA_COLUMNS].copy()
            for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    except Exception:
        pass


    # Backwards compatibility for v2.2.6: ensure existing Glazing objects include
    # a standard link to an existing Shading Device object.
    try:
        glazing_items = out.loc[
            out["Category"].astype(str).eq("Thermal Envelope")
            & out["Item Type"].astype(str).eq("Glazings"),
            ["Scenario", "Scope", "Item Name"]
        ].drop_duplicates()
        glazing_rows = []
        for _, item in glazing_items.iterrows():
            sc = str(item.get("Scenario", "Base"))
            scope = str(item.get("Scope", "Scenario")) or "Scenario"
            nm = str(item.get("Item Name", "Glazing"))
            existing_params = set(out.loc[
                out["Scenario"].astype(str).eq(sc)
                & out["Category"].astype(str).eq("Thermal Envelope")
                & out["Item Type"].astype(str).eq("Glazings")
                & out["Item Name"].astype(str).eq(nm),
                "Parameter"
            ].astype(str))
            if "Associated shading device" not in existing_params:
                glazing_rows.append(_mi_row(
                    sc,
                    scope if scope in ["Global", "Scenario"] else "Scenario",
                    "Thermal Envelope",
                    "Glazings",
                    nm,
                    "Associated shading device",
                    value="",
                    unit="-",
                    required=False,
                    source_type="Design Document",
                    source_ref="Shading concept / façade schedule",
                    reference="Select an existing Shading Device object where applicable",
                ))
        if glazing_rows:
            out = pd.concat([out, pd.DataFrame(glazing_rows)], ignore_index=True)
            out = out[MODEL_INPUT_QA_COLUMNS].copy()
            for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    except Exception:
        pass

    # Ensure every structured object has a standard ID parameter for traceability.
    # This is stored like any other parameter, so it survives project export/reload.
    try:
        object_keys = out[["Scenario", "Scope", "Category", "Item Type", "Item Name"]].drop_duplicates()
        id_rows = []
        for _, item in object_keys.iterrows():
            sc = str(item.get("Scenario", MODEL_INPUT_GLOBAL_SCENARIO))
            scope = str(item.get("Scope", "Global" if sc == MODEL_INPUT_GLOBAL_SCENARIO else "Scenario"))
            cat = str(item.get("Category", "Other / Custom Inputs"))
            it = str(item.get("Item Type", cat))
            nm = str(item.get("Item Name", "General"))
            has_id = out.loc[
                out["Scenario"].astype(str).eq(sc)
                & out["Scope"].astype(str).eq(scope)
                & out["Category"].astype(str).eq(cat)
                & out["Item Type"].astype(str).eq(it)
                & out["Item Name"].astype(str).eq(nm)
                & out["Parameter"].astype(str).eq("ID")
            ].shape[0] > 0
            if not has_id:
                id_rows.append(_mi_row(
                    sc,
                    scope if scope in ["Global", "Scenario"] else ("Global" if sc == MODEL_INPUT_GLOBAL_SCENARIO else "Scenario"),
                    cat,
                    it,
                    nm,
                    "ID",
                    value=_model_input_object_id(cat, it, nm, sc),
                    unit="-",
                    required=True,
                    source_type="Other",
                    source_ref="Model Inputs QA",
                    reference="Unique object identifier for traceability",
                    notes="Automatically added object ID",
                ))
        if id_rows:
            out = pd.concat([out, pd.DataFrame(id_rows)], ignore_index=True)
            out = out[MODEL_INPUT_QA_COLUMNS].copy()
            for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
    except Exception:
        pass

    out = _apply_model_input_usual_ranges(out)
    for col in ["Min Check", "Max Check", "Usual Min", "Usual Max"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.reset_index(drop=True)


def parse_model_inputs_qa_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return sanitize_model_inputs_qa_df(df)


def build_model_inputs_qa_df() -> pd.DataFrame:
    return sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))


def _extract_first_number(value) -> Optional[float]:
    try:
        s = str(value).replace(",", ".")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        return float(m.group(0))
    except Exception:
        return None


def model_inputs_df_for_scenario(df: Optional[pd.DataFrame], scenario_name: Optional[str]) -> pd.DataFrame:
    """Return global rows plus rows belonging to the selected scenario."""
    out = sanitize_model_inputs_qa_df(df)
    sc = str(scenario_name or "Base")
    mask = out["Scenario"].astype(str).eq(MODEL_INPUT_GLOBAL_SCENARIO) | out["Scenario"].astype(str).eq(sc)
    return out.loc[mask].reset_index(drop=True)



def _format_model_input_value_for_comparison(row) -> str:
    """Return a compact display value for scenario-comparison tables."""
    try:
        val = str(row.get("Value", "")).strip()
        unit = str(row.get("Unit", "")).strip()
        src = str(row.get("Source Type", "")).strip()
        if val == "":
            base = "Missing"
        elif unit and unit not in ["-", "nan", "None"]:
            base = f"{val} {unit}"
        else:
            base = val
        if src == "Assumption":
            base = f"⚠ {base} (Assumption)"
        return base
    except Exception:
        return "Missing"


def model_inputs_scenario_differences(df: Optional[pd.DataFrame], scenario_names: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return a wide table and a chart table with inputs that differ across scenarios.

    Comparison is based on the effective input set per scenario: global rows plus
    rows belonging to each scenario. Rows are matched by Category, Item Type,
    Item Name and Parameter. Missing objects/parameters in a scenario are shown
    as "Not defined" and are treated as differences.
    """
    try:
        scenario_names = [str(x) for x in scenario_names if str(x).strip()]
    except Exception:
        scenario_names = []
    if len(scenario_names) <= 1:
        return pd.DataFrame(), pd.DataFrame()

    base_df = sanitize_model_inputs_qa_df(df)
    key_cols = ["Category", "Item Type", "Item Name", "Parameter"]
    scenario_maps = {}
    all_keys = set()

    for sc in scenario_names:
        eff = model_inputs_df_for_scenario(base_df, sc).copy()
        if eff.empty:
            scenario_maps[sc] = {}
            continue

        # If the same parameter exists as both global and scenario-specific,
        # use the scenario-specific row as the effective value.
        eff["_priority"] = np.where(eff["Scenario"].astype(str).eq(MODEL_INPUT_GLOBAL_SCENARIO), 0, 1)
        eff = eff.sort_values(by=key_cols + ["_priority"], kind="stable")
        eff = eff.drop_duplicates(subset=key_cols, keep="last")

        smap = {}
        for _, r in eff.iterrows():
            key = tuple(str(r.get(c, "")) for c in key_cols)
            all_keys.add(key)
            smap[key] = {
                "display": _format_model_input_value_for_comparison(r),
                "raw": str(r.get("Value", "")).strip(),
                "unit": str(r.get("Unit", "")).strip(),
                "scope": str(r.get("Scope", "")),
                "source": str(r.get("Source Type", "")),
            }
        scenario_maps[sc] = smap

    rows = []
    chart_rows = []
    for key in sorted(all_keys, key=lambda k: (k[0], k[1], k[2], k[3])):
        values = []
        raw_values = []
        for sc in scenario_names:
            item = scenario_maps.get(sc, {}).get(key)
            if item is None:
                values.append("Not defined")
                raw_values.append("__NOT_DEFINED__")
            else:
                values.append(item.get("display", "Missing"))
                raw_values.append(f"{item.get('raw','')}|{item.get('unit','')}|{item.get('source','')}")
        # Show only inputs with at least two different effective values/sources.
        if len(set(raw_values)) <= 1:
            continue
        category, item_type, item_name, parameter = key
        row = {
            "Category": category,
            "Object Type": item_type,
            "Object Name": item_name,
            "Parameter": parameter,
        }
        for sc, val in zip(scenario_names, values):
            row[sc] = val
        rows.append(row)

        # For the chart, count scenarios that deviate from the most common value.
        try:
            vc = pd.Series(raw_values).value_counts(dropna=False)
            common = str(vc.index[0]) if not vc.empty else raw_values[0]
        except Exception:
            common = raw_values[0]
        for sc, raw in zip(scenario_names, raw_values):
            if str(raw) != str(common):
                chart_rows.append({"Scenario": sc, "Category": category, "Differing inputs": 1})

    diff_df = pd.DataFrame(rows)
    chart_df = pd.DataFrame(chart_rows)
    if not chart_df.empty:
        chart_df = chart_df.groupby(["Scenario", "Category"], as_index=False)["Differing inputs"].sum()
    return diff_df, chart_df

def evaluate_model_inputs_qa_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add QA status columns for completeness, assumption tagging, sanity ranges and usual-value bounds."""
    out = sanitize_model_inputs_qa_df(df)
    statuses = []
    messages = []
    numeric_values = []
    for _, row in out.iterrows():
        val_raw = str(row.get("Value", "")).strip()
        src_type = str(row.get("Source Type", "")).strip()
        required = bool(row.get("Required", False))
        min_v = row.get("Min Check", np.nan)
        max_v = row.get("Max Check", np.nan)
        usual_min = row.get("Usual Min", np.nan)
        usual_max = row.get("Usual Max", np.nan)
        justification = str(row.get("Range Justification", "")).strip()
        num = _extract_first_number(val_raw)
        numeric_values.append(num)
        msg_parts = []
        status = "OK"
        if required and val_raw == "":
            status = "Missing"
            msg_parts.append("required value missing")
        if src_type == "Assumption":
            if status == "OK":
                status = "Assumption"
            msg_parts.append("review assumption/source")
        if num is not None:
            try:
                if pd.notna(min_v) and num < float(min_v):
                    status = "Review" if status not in ["Missing"] else status
                    msg_parts.append(f"below sanity minimum ({float(min_v):g})")
            except Exception:
                pass
            try:
                if pd.notna(max_v) and num > float(max_v):
                    status = "Review" if status not in ["Missing"] else status
                    msg_parts.append(f"above sanity maximum ({float(max_v):g})")
            except Exception:
                pass
            if status != "Missing" and _is_model_input_out_of_usual_range(val_raw, usual_min, usual_max):
                status = "Out of usual range"
                msg_parts.append(f"outside usual range ({_format_usual_range(usual_min, usual_max, row.get('Unit', ''))})")
                if not justification:
                    msg_parts.append("justification missing")
        statuses.append(status)
        messages.append("; ".join([m for m in msg_parts if m]) if msg_parts else "")
    out["Numeric Value"] = numeric_values
    out["QA Status"] = statuses
    out["QA Message"] = messages
    return out

def model_inputs_qa_summary(df: Optional[pd.DataFrame]) -> Dict[str, int]:
    qa = evaluate_model_inputs_qa_df(df)
    total = int(len(qa))
    required = int(qa["Required"].sum()) if not qa.empty else 0
    missing = int((qa["QA Status"] == "Missing").sum()) if not qa.empty else 0
    assumptions = int((qa["Source Type"].astype(str) == "Assumption").sum()) if not qa.empty else 0
    review = int(qa["QA Status"].isin(["Review", "Out of usual range"]).sum()) if not qa.empty else 0
    complete_required = max(0, required - missing)
    completeness = int(round((complete_required / required) * 100)) if required > 0 else 100
    return {
        "total": total,
        "required": required,
        "missing": missing,
        "assumptions": assumptions,
        "review": review,
        "completeness": completeness,
    }


def _style_model_inputs_qa(row):
    """Pandas Styler helper: assumption rows yellow; missing/review rows red/orange."""
    status = str(row.get("QA Status", ""))
    source = str(row.get("Source Type", ""))
    if status == "Missing":
        color = "background-color: #f8d7da"
    elif status == "Out of usual range":
        color = "background-color: #f5b7b1"
    elif status == "Review":
        color = "background-color: #ffe5b4"
    elif source == "Assumption" or status == "Assumption":
        color = "background-color: #fff3cd"
    else:
        color = ""
    return [color for _ in row]


def _safe_model_input_key(*parts) -> str:
    raw = "_".join([str(x) for x in parts])
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    return safe[:120]


def _model_input_object_id(category: str, item_type: str, item_name: str, scenario: str = "") -> str:
    """Create a stable, readable default ID for Model Inputs QA objects."""
    try:
        prefix_map = {
            "General Model Setup": "GEN",
            "Room Types": "ROOM",
            "Thermal Envelope": "ENV",
            "AHU / Ventilation": "AHU",
            "Heating": "HEAT",
            "Cooling": "COOL",
            "Domestic Hot Water": "DHW",
            "Pumps / Auxiliaries": "PUMP",
            "Controls": "CTRL",
            "Other / Custom Inputs": "CUSTOM",
        }
        prefix = prefix_map.get(str(category), str(item_type or category or "OBJ"))
        base = f"{prefix}_{item_type}_{item_name}"
        safe = re.sub(r"[^0-9A-Za-z]+", "_", str(base)).strip("_").upper()
        safe = re.sub(r"_+", "_", safe)
        return safe[:80] or "MODEL_INPUT_OBJECT"
    except Exception:
        return "MODEL_INPUT_OBJECT"


def _model_input_ensure_unique_name(existing_rows: list, item_key: tuple, requested_name: str = "", allow_current: bool = False) -> str:
    """Return a unique Item Name within the same Scenario/Category/Item Type namespace."""
    scenario_i, category_i, item_type_i, item_name_i = [str(x) for x in item_key]
    existing_names = {
        str(r.get("Item Name", ""))
        for r in existing_rows
        if str(r.get("Scenario", "")) == scenario_i
        and str(r.get("Category", "")) == category_i
        and str(r.get("Item Type", "")) == item_type_i
        and (allow_current is False or str(r.get("Item Name", "")) != item_name_i)
    }
    requested = str(requested_name or "").strip()
    base = requested if requested else item_name_i
    if base not in existing_names:
        return base
    n = 2
    while f"{base} {n}" in existing_names:
        n += 1
    return f"{base} {n}"


def add_model_input_rows(rows: list) -> None:
    df = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    st.session_state["model_inputs_qa_df"] = sanitize_model_inputs_qa_df(df)


def add_model_room_type(item_name: str, scenario_name: str, scope: str = "Global") -> None:
    nm = str(item_name or "Room Type").strip() or "Room Type"
    add_model_input_rows(room_type_template(nm, str(scenario_name or "Base"), scope=scope))


def add_model_envelope_component(component_type: str, item_name: str, scenario_name: str, scope: str = "Scenario") -> None:
    nm = str(item_name or component_type or "Component").strip() or "Component"
    add_model_input_rows(envelope_component_template(str(component_type or "Other Envelope Component"), nm, str(scenario_name or "Base"), scope=scope))


def add_model_system(system_type: str, item_name: str, scenario_name: str, scope: str = "Scenario") -> None:
    nm = str(item_name or system_type or "System").strip() or "System"
    add_model_input_rows(system_template(str(system_type or "Other System"), nm, str(scenario_name or "Base"), scope=scope))


def add_model_custom_parameter(scenario_name: str, scope: str, category: str, item_type: str, item_name: str, parameter_name: str) -> None:
    sc = MODEL_INPUT_GLOBAL_SCENARIO if str(scope).lower() == "global" else str(scenario_name or "Base")
    add_model_input_rows([_mi_row(sc, "Global" if sc == MODEL_INPUT_GLOBAL_SCENARIO else "Scenario", category, item_type, item_name, str(parameter_name or "Custom parameter"), unit="-", required=False, source_type="Assumption")])


def duplicate_model_inputs_for_scenario(source_scenario: str, target_scenario: str) -> None:
    df = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))
    src = str(source_scenario)
    tgt = str(target_scenario)
    subset = df.loc[df["Scenario"].astype(str).eq(src)].copy()
    if not subset.empty:
        subset["Scenario"] = tgt
        df = pd.concat([df, subset], ignore_index=True)
        st.session_state["model_inputs_qa_df"] = sanitize_model_inputs_qa_df(df)


def rename_model_inputs_scenario(old_name: str, new_name: str) -> None:
    df = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))
    df.loc[df["Scenario"].astype(str).eq(str(old_name)), "Scenario"] = str(new_name)
    st.session_state["model_inputs_qa_df"] = sanitize_model_inputs_qa_df(df)


def delete_model_inputs_scenario(scenario_name: str) -> None:
    df = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))
    df = df.loc[~df["Scenario"].astype(str).eq(str(scenario_name))].reset_index(drop=True)
    st.session_state["model_inputs_qa_df"] = sanitize_model_inputs_qa_df(df)

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
                          scenarios_df: Optional[pd.DataFrame] = None,
                          colors_df: Optional[pd.DataFrame] = None,
                          lcc_global_df: Optional[pd.DataFrame] = None,
                          lcc_investments_df: Optional[pd.DataFrame] = None,
                          model_inputs_qa_df: Optional[pd.DataFrame] = None,
                          energy_balance_df: Optional[pd.DataFrame] = None,
                          scenario_energy_overrides_df: Optional[pd.DataFrame] = None,
                          loads_balance_df: Optional[pd.DataFrame] = None) -> bytes:
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
    if colors_df is not None:
        sheets[SHEET_COLORS] = colors_df
    if lcc_global_df is not None:
        sheets[SHEET_LCC_GLOBAL] = lcc_global_df
    if lcc_investments_df is not None:
        sheets[SHEET_LCC_INVESTMENTS] = lcc_investments_df
    if model_inputs_qa_df is not None:
        sheets[SHEET_MODEL_INPUTS_QA] = sanitize_model_inputs_qa_df(model_inputs_qa_df)


    # overwrite raw data sheets if provided
    if energy_balance_df is not None:
        sheets[RAW_SHEET_ENERGY] = _energy_balance_to_excel_df(energy_balance_df)
    if scenario_energy_overrides_df is not None:
        sheets[SHEET_RAW_ENERGY_SCENARIOS] = scenario_energy_overrides_df
    if loads_balance_df is not None:
        sheets[RAW_SHEET_LOADS] = _loads_balance_to_excel_df(loads_balance_df)

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



def compute_crrem_like_scenario_emissions_series(
        df_energy: pd.DataFrame,
        payload: dict,
        crrem: Optional[dict],
        project_year: int,
        years: list,
) -> pd.Series:
    """Return annual total emissions in tCO₂e/a using the same decarbonization logic as CRREM.

    This mirrors the CRREM tab logic:
    - apply scenario-specific efficiency factors;
    - apply scenario-specific source mapping;
    - apply scenario-specific on-site generation scale only when enabled;
    - clamp net electricity to zero, so exported on-site generation is not credited;
    - calculate base-year emissions from scenario emission factors;
    - multiply the full annual emissions balance by the CRREM grid-electricity decarbonization multiplier.
    """
    years_i = [int(y) for y in (years or [])]
    if not years_i:
        return pd.Series(dtype=float)

    payload = payload or {}
    try:
        df_base = sanitize_energy_balance_df(df_energy)
    except Exception:
        df_base = df_energy.copy() if isinstance(df_energy, pd.DataFrame) else pd.DataFrame()

    if df_base is None or df_base.empty or "Month" not in df_base.columns:
        return pd.Series({int(y): 0.0 for y in years_i}, dtype=float)

    df_m = df_base.melt(id_vars="Month", var_name="End_Use", value_name="kWh").copy()
    df_m["End_Use"] = df_m["End_Use"].apply(lambda x: _canon_enduse_name(str(x)))
    df_m["kWh"] = pd.to_numeric(df_m["kWh"], errors="coerce").fillna(0.0)

    eff = payload.get("efficiency", {}) or {}
    mapping = payload.get("mapping", {}) or {}
    factors = payload.get("factors", {}) or {}
    pv_cfg = payload.get("pv", {}) or {}

    df_m["Efficiency_Factor"] = df_m["End_Use"].map(lambda u: _to_float_lcc(eff.get(u, 1.0), 1.0)).fillna(1.0)
    df_m["Efficiency_Factor"] = df_m["Efficiency_Factor"].replace(0.0, 1.0)
    df_m["kWh_adj"] = df_m["kWh"] / df_m["Efficiency_Factor"]

    onsite_enduses = set(get_onsite_generation_enduses(df_m["End_Use"].unique()))
    pv_mask = df_m["End_Use"].isin(onsite_enduses)
    if pv_mask.any():
        pv_apply_scale = bool(pv_cfg.get("enabled", False))
        pv_scale = _to_float_lcc(pv_cfg.get("scale", 1.0), 1.0)
        scale = pv_scale if pv_apply_scale else 1.0
        df_m.loc[pv_mask, "kWh_adj"] = -df_m.loc[pv_mask, "kWh_adj"].abs() * float(scale)

    df_m.loc[~pv_mask, "kWh_adj"] = df_m.loc[~pv_mask, "kWh_adj"].clip(lower=0.0)
    df_m["Energy_Source"] = df_m["End_Use"].map(lambda u: str(mapping.get(u, "Electricity"))).fillna("Electricity")
    df_m.loc[~df_m["Energy_Source"].isin(ENERGY_SOURCE_ORDER), "Energy_Source"] = "Electricity"
    if pv_mask.any():
        df_m.loc[pv_mask, "Energy_Source"] = "Electricity"

    annual_kwh_by_source = df_m.groupby("Energy_Source", as_index=True)["kWh_adj"].sum()
    if "Electricity" in annual_kwh_by_source.index:
        annual_kwh_by_source.loc["Electricity"] = max(float(annual_kwh_by_source.loc["Electricity"]), 0.0)

    base_factors = {
        "Electricity": _to_float_lcc(factors.get("Electricity", 0.0), 0.0),
        "Green Electricity": 0.0,
        "Gas": _to_float_lcc(factors.get("Gas", 0.0), 0.0),
        "District Heating": _to_float_lcc(factors.get("District Heating", 0.0), 0.0),
        "District Cooling": _to_float_lcc(factors.get("District Cooling", 0.0), 0.0),
        "Biomass": _to_float_lcc(factors.get("Biomass", 0.0), 0.0),
    }

    emissions_base_kg = 0.0
    for src, kwh in annual_kwh_by_source.items():
        if str(src) == "Green Electricity":
            continue
        emissions_base_kg += float(kwh) * float(base_factors.get(str(src), 0.0))

    # Fallback: if CRREM data is unavailable, keep emissions flat instead of failing the chart.
    try:
        ef_grid = (crrem or {}).get("ef_grid")
        if ef_grid is not None and isinstance(ef_grid, pd.Series) and not ef_grid.empty:
            multipliers = compute_decarb_multiplier(ef_grid, int(project_year), years_i)
        else:
            multipliers = pd.Series({int(y): 1.0 for y in years_i}, dtype=float)
    except Exception:
        multipliers = pd.Series({int(y): 1.0 for y in years_i}, dtype=float)

    return pd.Series({int(y): (float(emissions_base_kg) * float(multipliers.loc[int(y)])) / 1000.0 for y in years_i}, dtype=float)

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
    saved_colors_enduse, saved_colors_sources, saved_colors_loads, saved_colors_scenarios = parse_color_settings_df(cfg_saved.get("colors"))
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
        "colors_enduse": saved_colors_enduse,
        "colors_sources": saved_colors_sources,
        "colors_loads": saved_colors_loads,
        "colors_scenarios": saved_colors_scenarios,
        "scenarios_df": cfg_saved.get("scenarios"),
        "lcc_global_df": cfg_saved.get("lcc_global"),
        "lcc_investments_df": cfg_saved.get("lcc_investments"),
        "scenario_energy_df": cfg_saved.get("scenario_energy"),
        "model_inputs_df": cfg_saved.get("model_inputs"),
        "file_bytes": file_bytes,
    }

    # --- Seed Project Data from file on each new upload (token-based)
    #     This keeps Project Data global (not scenario-dependent) and ensures it reloads correctly from the workbook.
    wb_token = f"{uploaded_file.name}|{hashlib.md5(file_bytes).hexdigest()}"

    # --- Seed Raw Data (Energy_Balance / Loads_Balance) from file on each new upload (token-based)
    #     Raw Data is global (not scenario-dependent) and can be edited in-app in the "Raw Data" tab.
    if st.session_state.get(_RAW_TOKEN_KEY) != wb_token:
        try:
            st.session_state[_RAW_ENERGY_KEY] = sanitize_energy_balance_df(energy_balance_sheet(file_bytes))
        except Exception:
            st.session_state[_RAW_ENERGY_KEY] = pd.DataFrame()
        try:
            st.session_state[_RAW_LOADS_KEY] = sanitize_loads_balance_df(loads_balace_sheet(file_bytes))
        except Exception:
            st.session_state[_RAW_LOADS_KEY] = pd.DataFrame()

        # Scenario-specific Energy_Balance overrides are optional and stored in a dedicated long sheet.
        try:
            st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = parse_scenario_energy_overrides_df(
                cfg_saved.get("scenario_energy")
            )
        except Exception:
            st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = {}
        st.session_state[_RAW_ENERGY_SCENARIO_DRAFTS_KEY] = {}
        st.session_state[_RAW_ENERGY_SCENARIO_DIRTY_KEY] = {}

        # Model Inputs QA is a global project input register. It is committed via the tab form and saved with the workbook.
        try:
            st.session_state["model_inputs_qa_df"] = parse_model_inputs_qa_df(cfg_saved.get("model_inputs"))
        except Exception:
            st.session_state["model_inputs_qa_df"] = default_model_inputs_qa_df()

        # Draft copies are edited in the UI; committed copies drive calculations.
        st.session_state[_RAW_ENERGY_DRAFT_KEY] = st.session_state[_RAW_ENERGY_KEY].copy(deep=True)
        st.session_state[_RAW_LOADS_DRAFT_KEY] = st.session_state[_RAW_LOADS_KEY].copy(deep=True)

        st.session_state[_RAW_TOKEN_KEY] = wb_token

    # Ensure draft buffers exist (e.g., when restoring session state)
    if _RAW_COMMIT_VERSION_KEY not in st.session_state:
        st.session_state[_RAW_COMMIT_VERSION_KEY] = 0
    if _RAW_ENERGY_DRAFT_KEY not in st.session_state and _RAW_ENERGY_KEY in st.session_state:
        st.session_state[_RAW_ENERGY_DRAFT_KEY] = st.session_state[_RAW_ENERGY_KEY].copy(deep=True)
    if _RAW_LOADS_DRAFT_KEY not in st.session_state and _RAW_LOADS_KEY in st.session_state:
        st.session_state[_RAW_LOADS_DRAFT_KEY] = st.session_state[_RAW_LOADS_KEY].copy(deep=True)
    if _RAW_ENERGY_SCENARIO_DIRTY_KEY not in st.session_state or not isinstance(st.session_state.get(_RAW_ENERGY_SCENARIO_DIRTY_KEY), dict):
        st.session_state[_RAW_ENERGY_SCENARIO_DIRTY_KEY] = {}
    if "model_inputs_qa_df" not in st.session_state or not isinstance(st.session_state.get("model_inputs_qa_df"), pd.DataFrame):
        try:
            st.session_state["model_inputs_qa_df"] = parse_model_inputs_qa_df(cfg_saved.get("model_inputs"))
        except Exception:
            st.session_state["model_inputs_qa_df"] = default_model_inputs_qa_df()

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

        # --- Seed Color Settings from file (or defaults)
        try:
            enduse_base = dict(DEFAULT_COLOR_MAP)
            enduse_base.update(preloaded.get("colors_enduse") or {})
            source_base = dict(DEFAULT_COLOR_MAP_SOURCES)
            source_base.update(preloaded.get("colors_sources") or {})
            loads_base = dict(DEFAULT_COLOR_MAP_LOADS)
            loads_base.update(preloaded.get("colors_loads") or {})
            scenarios_base = default_scenario_color_map([])
            scenarios_base.update(preloaded.get("colors_scenarios") or {})
            st.session_state["color_map_enduse"] = enduse_base
            st.session_state["color_map_sources"] = source_base
            st.session_state["color_map_loads"] = loads_base
            st.session_state["color_map_scenarios"] = scenarios_base
        except Exception:
            st.session_state["color_map_enduse"] = dict(DEFAULT_COLOR_MAP)
            st.session_state["color_map_sources"] = dict(DEFAULT_COLOR_MAP_SOURCES)
            st.session_state["color_map_loads"] = dict(DEFAULT_COLOR_MAP_LOADS)
            st.session_state["color_map_scenarios"] = default_scenario_color_map([])

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
tab1, tab1_factors, tab2, tab3, tab4, tab5, tab6, tab_lcc, tab7, tab_model_qa, tab8 = st.tabs(
    ["Energy Balance (without Factors)", "Energy Balance (with Factors)", "CO2 Emissions (with Factors)", "Energy Cost (with Factors)", "Loads Analysis", "Benchmark",
     "CRREM-Analysis", "LCC-Analysis", "Scenarios", "Model Inputs QA", "Raw Data"])

# =========================
# Tab 1 — Energy Balance (Energy Balance Tab)
# =========================
with tab1:
    if uploaded_file:
        # ---- Load data
        df = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name)

        # ---- Wide->Long transform for plotting and grouping
        df_melted = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

        # ---- Scenario Manager initialization (backwards compatible)
        end_uses = df_melted["End_Use"].unique().tolist()
        _just_initialized_scenarios = False

        if "scenarios" not in st.session_state:
            scenarios_from_file, active_from_file = parse_scenarios_sheet(
                preloaded.get("scenarios_df") if preloaded else None
            )
            if scenarios_from_file:
                scenarios_from_file = merge_lcc_sheets_into_scenarios(
                    scenarios_from_file,
                    preloaded.get("lcc_global_df") if preloaded else None,
                    preloaded.get("lcc_investments_df") if preloaded else None,
                    end_uses,
                )
                st.session_state["scenarios"] = scenarios_from_file
                st.session_state["active_scenario"] = active_from_file or list(scenarios_from_file.keys())[0]
            else:
                base_payload = default_scenario_payload(end_uses, preloaded)
                scenarios_from_file = merge_lcc_sheets_into_scenarios(
                    {"Base": base_payload},
                    preloaded.get("lcc_global_df") if preloaded else None,
                    preloaded.get("lcc_investments_df") if preloaded else None,
                    end_uses,
                )
                st.session_state["scenarios"] = scenarios_from_file
                st.session_state["active_scenario"] = "Base" if "Base" in scenarios_from_file else list(scenarios_from_file.keys())[0]
            st.session_state["_prev_active_scenario"] = st.session_state["active_scenario"]
            load_scenario_into_widgets(st.session_state["scenarios"][st.session_state["active_scenario"]], end_uses)
            _just_initialized_scenarios = True

        # Resolve Energy_Balance again after scenario initialization so first render also uses
        # an active scenario-specific raw-data override when one was loaded from the workbook.
        df = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name, scenario_name=st.session_state.get("active_scenario"))
        df_melted = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")
        end_uses = df_melted["End_Use"].unique().tolist()
        if _just_initialized_scenarios and st.session_state.get("active_scenario") in st.session_state.get("scenarios", {}):
            load_scenario_into_widgets(st.session_state["scenarios"][st.session_state["active_scenario"]], end_uses)

        # ---- Sidebar: scenario manager UI
        with st.sidebar.expander("Scenario Manager", expanded=True):
            st.caption("Manage and select active project's scenario")
            scenarios = st.session_state.get("scenarios", {})
            scenario_names = list(scenarios.keys()) if scenarios else ["Base"]

            # ------------------------------------------------------------------
            # Robust active-scenario handling
            # ------------------------------------------------------------------
            # IMPORTANT:
            # The selectbox uses its own widget key ("active_scenario_selector").
            # The canonical active scenario is stored in "active_scenario".
            # This avoids StreamlitAPIException errors caused by assigning to a
            # widget-bound key after the widget has been instantiated in the same run
            # (which happened when creating, renaming or deleting scenarios).
            current_active = str(st.session_state.get("active_scenario", scenario_names[0]))
            if current_active not in scenario_names:
                current_active = scenario_names[0]
                st.session_state["active_scenario"] = current_active

            # Keep the selector valid BEFORE the widget is rendered.
            # Do NOT force it back to current_active when the user has just selected
            # another valid scenario: on the following rerun, Streamlit already stores
            # the user's new dropdown value in active_scenario_selector, while
            # active_scenario still contains the previous canonical value. If we
            # overwrite the selector here, manual switching becomes impossible.
            #
            # For button actions (New / Duplicate / Rename / Delete), _activate_scenario
            # sets _active_scenario_selector_sync_to. That flag is consumed here on
            # the next run, safely before the selectbox widget is instantiated.
            selector_sync_to = st.session_state.pop("_active_scenario_selector_sync_to", None)
            if selector_sync_to in scenario_names:
                st.session_state["active_scenario_selector"] = selector_sync_to
            elif st.session_state.get("active_scenario_selector") not in scenario_names:
                st.session_state["active_scenario_selector"] = current_active

            selector_value = st.session_state.get("active_scenario_selector", current_active)
            active_idx = scenario_names.index(selector_value) if selector_value in scenario_names else (scenario_names.index(current_active) if current_active in scenario_names else 0)
            active_selected = st.selectbox(
                "Active Scenario",
                scenario_names,
                index=active_idx,
                key="active_scenario_selector",
            )

            def _end_uses_for_scenario(_scenario_name: str, _fallback_end_uses: list) -> list:
                """Return End Uses from the scenario-specific Energy_Balance if available."""
                try:
                    _df = get_energy_balance_df(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        scenario_name=_scenario_name,
                    )
                    if isinstance(_df, pd.DataFrame) and "Month" in _df.columns:
                        return _df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")["End_Use"].unique().tolist()
                except Exception:
                    pass
                return list(_fallback_end_uses)

            def _save_current_scenario_payload(_scenario_name: str) -> None:
                """Persist current sidebar/widget values to the canonical active scenario payload."""
                try:
                    if _scenario_name in scenarios:
                        _eu = _end_uses_for_scenario(_scenario_name, end_uses)
                        scenarios[_scenario_name] = capture_scenario_from_widgets(_eu)
                        st.session_state["scenarios"] = scenarios
                except Exception:
                    pass

            def _activate_scenario(_scenario_name: str) -> None:
                """Set canonical active scenario and load its widget state.

                Do not assign to active_scenario_selector here. The selector will be
                synchronized at the beginning of the next run before widget rendering.
                """
                try:
                    _scenario_name = str(_scenario_name)
                    if _scenario_name in scenarios:
                        _eu = _end_uses_for_scenario(_scenario_name, end_uses)
                        load_scenario_into_widgets(scenarios[_scenario_name], _eu)
                        st.session_state["active_scenario"] = _scenario_name
                        st.session_state["_prev_active_scenario"] = _scenario_name
                        # Do not write to the selectbox key in this run because the
                        # widget may already exist. Instead, request a safe pre-widget
                        # sync for the next rerun.
                        st.session_state["_active_scenario_selector_sync_to"] = _scenario_name
                except Exception:
                    _scenario_name = str(_scenario_name)
                    st.session_state["active_scenario"] = _scenario_name
                    st.session_state["_prev_active_scenario"] = _scenario_name
                    st.session_state["_active_scenario_selector_sync_to"] = _scenario_name

            # Manual scenario switch from the selectbox.
            if active_selected != current_active:
                _save_current_scenario_payload(current_active)
                _activate_scenario(active_selected)
                st.rerun()

            # Scenario actions (stacked vertically for clarity)
            if st.button("New", use_container_width=True, key="scenario_btn_new"):
                _save_current_scenario_payload(current_active)
                base_name = "Scenario"
                n = 1
                new_name = f"{base_name} {n}"
                while new_name in scenarios:
                    n += 1
                    new_name = f"{base_name} {n}"
                scenarios[new_name] = default_scenario_payload(end_uses, preloaded)
                # New scenarios intentionally start from the global Energy_Balance unless the user creates an override.
                try:
                    cmap_sc = st.session_state.get("color_map_scenarios", {})
                    cmap_sc[new_name] = SCENARIO_COLOR_PALETTE[(len(scenarios) - 1) % len(SCENARIO_COLOR_PALETTE)]
                    st.session_state["color_map_scenarios"] = cmap_sc
                except Exception:
                    pass
                st.session_state["scenarios"] = scenarios
                _activate_scenario(new_name)
                st.rerun()

            if st.button("Duplicate", use_container_width=True, key="scenario_btn_duplicate"):
                _save_current_scenario_payload(current_active)
                base_name = f"{current_active} Copy"
                new_name = base_name
                i = 2
                while new_name in scenarios:
                    new_name = f"{base_name} {i}"
                    i += 1
                scenarios[new_name] = deepcopy(scenarios[current_active])
                try:
                    overrides = _scenario_energy_overrides()
                    if current_active in overrides and isinstance(overrides.get(current_active), pd.DataFrame):
                        overrides[new_name] = overrides[current_active].copy(deep=True)
                        st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = overrides
                except Exception:
                    pass
                try:
                    duplicate_model_inputs_for_scenario(current_active, new_name)
                except Exception:
                    pass
                try:
                    cmap_sc = st.session_state.get("color_map_scenarios", {})
                    cmap_sc[new_name] = cmap_sc.get(current_active, SCENARIO_COLOR_PALETTE[(len(scenarios) - 1) % len(SCENARIO_COLOR_PALETTE)])
                    st.session_state["color_map_scenarios"] = cmap_sc
                except Exception:
                    pass
                st.session_state["scenarios"] = scenarios
                _activate_scenario(new_name)
                st.rerun()

            rename_to = st.text_input("Rename to", value="", key="scenario_rename_to")
            if st.button("Rename", use_container_width=True, key="scenario_btn_rename"):
                rename_to_clean = str(rename_to).strip()
                if rename_to_clean and rename_to_clean not in scenarios and current_active in scenarios:
                    _save_current_scenario_payload(current_active)
                    scenarios[rename_to_clean] = scenarios.pop(current_active)
                    try:
                        overrides = _scenario_energy_overrides()
                        drafts = _scenario_energy_drafts()
                        if current_active in overrides:
                            overrides[rename_to_clean] = overrides.pop(current_active)
                        if current_active in drafts:
                            drafts[rename_to_clean] = drafts.pop(current_active)
                        dirty = _scenario_energy_dirty_flags()
                        if current_active in dirty:
                            dirty[rename_to_clean] = dirty.pop(current_active)
                        st.session_state[_RAW_ENERGY_SCENARIO_OVERRIDES_KEY] = overrides
                        st.session_state[_RAW_ENERGY_SCENARIO_DRAFTS_KEY] = drafts
                        st.session_state[_RAW_ENERGY_SCENARIO_DIRTY_KEY] = dirty
                    except Exception:
                        pass
                    try:
                        rename_model_inputs_scenario(current_active, rename_to_clean)
                    except Exception:
                        pass
                    try:
                        cmap_sc = st.session_state.get("color_map_scenarios", {})
                        if current_active in cmap_sc:
                            cmap_sc[rename_to_clean] = cmap_sc.pop(current_active)
                        st.session_state["color_map_scenarios"] = cmap_sc
                    except Exception:
                        pass
                    st.session_state["scenarios"] = scenarios
                    _activate_scenario(rename_to_clean)
                    st.rerun()
                elif rename_to_clean in scenarios:
                    st.warning("A scenario with this name already exists.")

            if st.button("Delete", use_container_width=True, key="scenario_btn_delete"):
                if len(scenarios) > 1 and current_active in scenarios:
                    # Choose the next available scenario in the current ordering.
                    old_names = list(scenarios.keys())
                    old_idx = old_names.index(current_active) if current_active in old_names else 0

                    try:
                        delete_scenario_energy_balance_override(current_active)
                    except Exception:
                        pass
                    try:
                        delete_model_inputs_scenario(current_active)
                    except Exception:
                        pass

                    scenarios.pop(current_active, None)
                    try:
                        cmap_sc = st.session_state.get("color_map_scenarios", {})
                        cmap_sc.pop(current_active, None)
                        st.session_state["color_map_scenarios"] = cmap_sc
                    except Exception:
                        pass
                    remaining_names = list(scenarios.keys())
                    new_active = remaining_names[min(old_idx, len(remaining_names) - 1)]

                    st.session_state["scenarios"] = scenarios
                    _activate_scenario(new_active)
                    st.rerun()
            st.caption("Scenarios store CO₂ factors, tariffs, source mapping, efficiency factors, On-site generation settings, CRREM measures and scenario-specific LCC investment measures. Optional scenario-specific Energy_Balance overrides are managed in Raw Data. Global LCC parameters are shared across scenarios.")

        # ---- Sidebar: project info (prefill from saved if available)
        with st.sidebar.expander("Project Data"):
            st.caption("Enter Project's Basic Informations")

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
            st.caption("Assign Emission Factors per source")
            def_f = preloaded["factors"] if preloaded else {}
            co2_Emissions_Electricity = numeric_input("CO2 Factor Electricity", float(def_f.get("Electricity", 0.300)),
                                                      key="co2_Emissions_Electricity", min_value=0.0, max_value=1.0,
                                                      fmt="{:.5f}")
            co2_Emissions_Green_Electricity = numeric_input("CO2 Factor Green Electricity",
                                                            float(def_f.get("Green Electricity", 0.000)),
                                                            key="co2_Emissions_Green_Electricity", min_value=0.0,
                                                            max_value=1.0, fmt="{:.5f}")
            co2_emissions_dh = numeric_input("CO2 Factor District Heating", float(def_f.get("District Heating", 0.260)),
                                             key="co2_emissions_dh", min_value=0.0, max_value=1.0, fmt="{:.5f}")
            co2_emissions_dc = numeric_input("CO2 Factor District Cooling", float(def_f.get("District Cooling", 0.280)),
                                             key="co2_emissions_dc", min_value=0.0, max_value=1.0, fmt="{:.5f}")
            co2_emissions_gas = numeric_input("CO2 Factor Gas", float(def_f.get("Gas", 0.180)), key="co2_emissions_gas",
                                              min_value=0.0, max_value=1.0, fmt="{:.5f}")
            co2_emissions_biomass = numeric_input("CO2 Factor Biomass", float(def_f.get("Biomass", 0.000)),
                                                  key="co2_emissions_biomass",
                                                  min_value=0.0, max_value=5.0, fmt="{:.5f}")

        # --- Energy Cost (€/kWh) ---
        with st.sidebar.expander("Energy Tariffs"):
            st.caption("Assign energy cost per source (per kWh)")
            default_currency = preloaded["currency"] if (
                    preloaded and preloaded["currency"] in ["€", "$", "£"]) else "€"
            currency_symbol = st.selectbox("Currency", ["€", "$", "£"], index=["€", "$", "£"].index(default_currency),
                                           key="currency_symbol")

            def_t = preloaded["tariffs"] if preloaded else {}
            cost_electricity = numeric_input(f"Cost Electricity ({currency_symbol}/kWh)",
                                             float(def_t.get("Electricity", 0.3500)), key="cost_electricity",
                                             min_value=0.0, max_value=100.0, fmt="{:.5f}")
            cost_green_electricity = numeric_input(f"Cost Green Electricity ({currency_symbol}/kWh)",
                                                   float(def_t.get("Green Electricity", 0.4000)),
                                                   key="cost_green_electricity", min_value=0.0, max_value=100.0,
                                                   fmt="{:.5f}")
            cost_dh = numeric_input(f"Cost District Heating ({currency_symbol}/kWh)",
                                    float(def_t.get("District Heating", 0.1600)), key="cost_dh", min_value=0.0,
                                    max_value=100.0, fmt="{:.5f}")
            cost_dc = numeric_input(f"Cost District Cooling ({currency_symbol}/kWh)",
                                    float(def_t.get("District Cooling", 0.1600)), key="cost_dc", min_value=0.0,
                                    max_value=100.0, fmt="{:.5f}")
            cost_gas = numeric_input(f"Cost Gas ({currency_symbol}/kWh)", float(def_t.get("Gas", 0.1200)), key="cost_gas",
                                     min_value=0.0, max_value=100.0, fmt="{:.5f}")
            cost_biomass = numeric_input(f"Cost Biomass ({currency_symbol}/kWh)", float(def_t.get("Biomass", 0.1000)),
                                         key="cost_biomass",
                                         min_value=0.0, max_value=100.0, fmt="{:.5f}")

        # ---- Sidebar: efficiency factors per End_Use (used in 'Energy Balance with Factors' tab)
        with st.sidebar.expander("Efficiency Factors"):
            st.caption("Assign efficiency factors per End Use (dimensionless; kWh is divided by factor)")
            def_eff = preloaded["efficiency"] if (preloaded and preloaded.get("efficiency")) else {}
            for use in df_melted["End_Use"].unique().tolist():
                numeric_input(
                    f"Efficiency Factor {use}",
                    float(def_eff.get(use, 1.0)),
                    key=f"eff_{use}",
                    min_value=0.0001,
                    max_value=1000.0,
                    fmt="{:.5f}",
                )

        # ---- Sidebar: map End_Use -> Energy_Source (user-controlled)
        with st.sidebar.expander("Assign Energy Sources"):
            st.caption("Assign Energy Sources per End Use")
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

        # ---- Sidebar: Color Settings (global; used across all charts)
        with st.sidebar.expander("Color Settings", expanded=False):
            # Ensure the dicts exist (seeded from workbook if available)
            if "color_map_enduse" not in st.session_state or not isinstance(st.session_state.get("color_map_enduse"), dict):
                st.session_state["color_map_enduse"] = dict(DEFAULT_COLOR_MAP)
            if "color_map_sources" not in st.session_state or not isinstance(st.session_state.get("color_map_sources"), dict):
                st.session_state["color_map_sources"] = dict(DEFAULT_COLOR_MAP_SOURCES)
            if "color_map_loads" not in st.session_state or not isinstance(st.session_state.get("color_map_loads"), dict):
                st.session_state["color_map_loads"] = dict(DEFAULT_COLOR_MAP_LOADS)
            if "color_map_scenarios" not in st.session_state or not isinstance(st.session_state.get("color_map_scenarios"), dict):
                st.session_state["color_map_scenarios"] = default_scenario_color_map(list(st.session_state.get("scenarios", {}).keys()))

            def _rand_hex(_name: str) -> str:
                try:
                    return "#" + hashlib.md5(str(_name).encode("utf-8")).hexdigest()[:6]
                except Exception:
                    return "#777777"

            def _k_safe(s: str) -> str:
                s = str(s)
                return "".join([(c if c.isalnum() else "_") for c in s])[:60]

            # Ensure colors exist for all End Uses found in the workbook
            for _eu in end_uses:
                if _eu not in st.session_state["color_map_enduse"]:
                    st.session_state["color_map_enduse"][_eu] = _rand_hex(f"enduse::{_eu}")

            # Ensure colors exist for all Energy Sources in use (or selectable)
            try:
                _sources_in_use = sorted(set(list(mapping_dict.values())) | set(ENERGY_SOURCE_ORDER))
            except Exception:
                _sources_in_use = list(ENERGY_SOURCE_ORDER)

            for _src in _sources_in_use:
                if _src not in st.session_state["color_map_sources"]:
                    st.session_state["color_map_sources"][_src] = _rand_hex(f"source::{_src}")

            # Detect loads from Loads_Balance (if present)
            try:
                _df_loads_sidebar = get_loads_balance_df(uploaded_file.getvalue(), uploaded_file.name)
                _load_cols = [c for c in _df_loads_sidebar.columns if c not in ["hoy", "doy", "day", "month", "weekday", "hour"]]
            except Exception:
                _load_cols = []

            for _ld in _load_cols:
                if _ld not in st.session_state["color_map_loads"]:
                    # Prefer same palette as End Uses when names overlap; otherwise deterministic color
                    st.session_state["color_map_loads"][_ld] = st.session_state["color_map_enduse"].get(_ld, _rand_hex(f"load::{_ld}"))

            # Ensure scenario colors exist for all current scenarios.
            try:
                _scenario_names_for_colors = list(st.session_state.get("scenarios", {}).keys())
            except Exception:
                _scenario_names_for_colors = []
            for _i_sc_col, _sc_col in enumerate(_scenario_names_for_colors):
                if _sc_col not in st.session_state["color_map_scenarios"]:
                    st.session_state["color_map_scenarios"][_sc_col] = SCENARIO_COLOR_PALETTE[_i_sc_col % len(SCENARIO_COLOR_PALETTE)]
            # Remove colors for scenarios that no longer exist.
            try:
                st.session_state["color_map_scenarios"] = {
                    str(k): v for k, v in st.session_state["color_map_scenarios"].items()
                    if str(k) in set(map(str, _scenario_names_for_colors))
                }
            except Exception:
                pass

            # Key prefix stable per workbook (prevents stale color-picker state after uploading a new file)
            _tok = st.session_state.get("_loaded_workbook_token", "default")
            try:
                _tok_short = hashlib.md5(str(_tok).encode("utf-8")).hexdigest()[:8]
            except Exception:
                _tok_short = "default"

            # Reset button — restores original palettes defined in the app code
            if st.button("Reset Colors", use_container_width=True, key=f"reset_colors_{_tok_short}"):
                st.session_state["color_map_enduse"] = dict(DEFAULT_COLOR_MAP)
                st.session_state["color_map_sources"] = dict(DEFAULT_COLOR_MAP_SOURCES)
                st.session_state["color_map_loads"] = dict(DEFAULT_COLOR_MAP_LOADS)
                st.session_state["color_map_scenarios"] = default_scenario_color_map(list(st.session_state.get("scenarios", {}).keys()))

                # Clear color-picker widget state so new defaults are reflected immediately
                for _k in list(st.session_state.keys()):
                    if str(_k).startswith(("cp_eu_", "cp_src_", "cp_ld_", "cp_sc_")):
                        try:
                            del st.session_state[_k]
                        except Exception:
                            pass
                st.rerun()

            st.caption("Customize colors for End Uses, Energy Sources, and Loads. These settings are saved with the project.")

            st.markdown("**End Uses**")
            for _eu in end_uses:
                _key = f"cp_eu_{_tok_short}_{_k_safe(_eu)}"
                _val = st.session_state["color_map_enduse"].get(_eu, _rand_hex(f"enduse::{_eu}"))
                _new = st.color_picker(ui_name(str(_eu)), value=_val, key=_key)
                st.session_state["color_map_enduse"][_eu] = _new

            st.markdown("---")
            st.markdown("**Energy Sources**")
            for _src in _sources_in_use:
                _key = f"cp_src_{_tok_short}_{_k_safe(_src)}"
                _val = st.session_state["color_map_sources"].get(_src, _rand_hex(f"source::{_src}"))
                _new = st.color_picker(str(_src), value=_val, key=_key)
                st.session_state["color_map_sources"][_src] = _new

            st.markdown("---")
            st.markdown("**Loads**")
            if _load_cols:
                for _ld in _load_cols:
                    _key = f"cp_ld_{_tok_short}_{_k_safe(_ld)}"
                    _val = st.session_state["color_map_loads"].get(_ld, _rand_hex(f"load::{_ld}"))
                    _new = st.color_picker(ui_name(str(_ld)), value=_val, key=_key)
                    st.session_state["color_map_loads"][_ld] = _new
            else:
                st.caption("No Loads_Balance sheet found (or no load columns detected).")

            st.markdown("---")
            st.markdown("**Scenarios**")
            if _scenario_names_for_colors:
                for _sc in _scenario_names_for_colors:
                    _key = f"cp_sc_{_tok_short}_{_k_safe(_sc)}"
                    _val = st.session_state["color_map_scenarios"].get(_sc, _rand_hex(f"scenario::{_sc}"))
                    _new = st.color_picker(str(_sc), value=_val, key=_key)
                    st.session_state["color_map_scenarios"][_sc] = _new
            else:
                st.caption("No scenarios found yet.")


        # ---- Apply current color settings to plotting maps
        color_map = st.session_state.get("color_map_enduse", color_map)
        color_map_sources = st.session_state.get("color_map_sources", color_map_sources)
        color_map_loads = st.session_state.get("color_map_loads", DEFAULT_COLOR_MAP_LOADS)
        color_map_scenarios = st.session_state.get("color_map_scenarios", default_scenario_color_map(list(st.session_state.get("scenarios", {}).keys())))

        # ---- Persist current widget values back into the active scenario (for switching/comparison/save)
        if "scenarios" in st.session_state and st.session_state.get("active_scenario") in st.session_state["scenarios"]:
            st.session_state["scenarios"][st.session_state["active_scenario"]] = capture_scenario_from_widgets(end_uses)
            if st.session_state.get("_lcc_global_initialized"):
                _apply_lcc_global_to_all_scenarios(end_uses)

        # ---- Save Project button (exports current inputs into the workbook)
        with st.sidebar:
            if st.button("Save Project", use_container_width=True):
                # Ensure the active scenario payload is up-to-date (including CRREM measures)
                try:
                    if "scenarios" in st.session_state and st.session_state.get("active_scenario") in st.session_state[
                        "scenarios"]:
                        st.session_state["scenarios"][
                            st.session_state["active_scenario"]] = capture_scenario_from_widgets(end_uses)
                        _apply_lcc_global_to_all_scenarios(end_uses)
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
                lcc_global_df = None
                lcc_investments_df = None
                if "scenarios" in st.session_state:
                    # Keep global LCC parameters and scenario-specific LCC investments synchronized before export.
                    try:
                        if st.session_state.get("_lcc_global_initialized"):
                            _sync_lcc_global_widget_state(end_uses)
                        _apply_lcc_global_to_all_scenarios(end_uses)
                    except Exception:
                        pass

                    scenarios_df = build_scenarios_sheet(
                        st.session_state.get("scenarios", {}),
                        st.session_state.get("active_scenario")
                    )
                    try:
                        lcc_global_df = build_lcc_global_df(_get_lcc_global_state_payload(end_uses), end_uses)
                        lcc_investments_df = build_lcc_investments_sheet(st.session_state.get("scenarios", {}), end_uses)
                    except Exception:
                        lcc_global_df = None
                        lcc_investments_df = None

                colors_df = build_color_settings_df(
                    st.session_state.get("color_map_enduse", DEFAULT_COLOR_MAP),
                    st.session_state.get("color_map_sources", DEFAULT_COLOR_MAP_SOURCES),
                    st.session_state.get("color_map_loads", DEFAULT_COLOR_MAP_LOADS),
                    st.session_state.get("color_map_scenarios", default_scenario_color_map(list(st.session_state.get("scenarios", {}).keys()))),
                )

                updated_bytes = write_config_to_excel(
                    preloaded["file_bytes"],
                    project_df,
                    factors_df,
                    tariffs_df,
                    mapping_df,
                    efficiency_df,
                    scenarios_df=scenarios_df,
                    colors_df=colors_df,
                    lcc_global_df=lcc_global_df,
                    lcc_investments_df=lcc_investments_df,
                    model_inputs_qa_df=build_model_inputs_qa_df(),
                    energy_balance_df=st.session_state.get(_RAW_ENERGY_KEY),
                    scenario_energy_overrides_df=build_scenario_energy_overrides_df(
                        effective_scenario_energy_overrides_for_export()
                    ),
                    loads_balance_df=st.session_state.get(_RAW_LOADS_KEY),
                )

                st.success("Project settings saved to workbook.")
                st.download_button(
                    label="Download Updated Workbook",
                    data=updated_bytes,
                    file_name=uploaded_file.name.replace(".xlsx", "_with_project.xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


            # ---- Generate Report button (active scenario only)
            if st.button("Generate Report", use_container_width=True, key="btn_generate_report"):
                try:
                    with st.spinner("Generating A4 PDF report for the active scenario..."):
                        if "scenarios" in st.session_state and st.session_state.get("active_scenario") in st.session_state["scenarios"]:
                            st.session_state["scenarios"][st.session_state["active_scenario"]] = capture_scenario_from_widgets(end_uses)
                            if st.session_state.get("_lcc_global_initialized"):
                                _apply_lcc_global_to_all_scenarios(end_uses)
                        report_pdf = generate_bpvis_pdf_report(uploaded_file.getvalue(), uploaded_file.name)
                        st.session_state["_generated_report_pdf"] = report_pdf
                        st.session_state["_generated_report_name"] = f"{_report_sanitize_filename(st.session_state.get('project_name', 'BPVis_Project'))}_{_report_sanitize_filename(st.session_state.get('active_scenario', 'Scenario'))}_Report_v2_2_8.pdf"
                    st.success("Report generated successfully.")
                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")

            if st.session_state.get("_generated_report_pdf"):
                st.download_button(
                    label="Download Report (PDF)",
                    data=st.session_state["_generated_report_pdf"],
                    file_name=st.session_state.get("_generated_report_name", "BPVis_Report_v2_2_8.pdf"),
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_generated_report_pdf",
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
        st.metric("Active Scenario", active_selected)

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
            (totals["End_Use"] != "On-site_Generation")]

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
                text=f"{net_eui:,.1f}<br>kWh/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=50, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart_per_source.update_traces(textinfo="value+percent", textfont_size=18,
                                                        textfont_color="white")

        # ---- On-site Generation coverage (share of on-site generation vs consumption-only EUI)
        totals_indexed = totals.set_index("End_Use")
        pv_value = totals_indexed.loc["On-site_Generation", "kWh_per_m2"] if "On-site_Generation" in totals_indexed.index else 0.0
        pv_coverage = abs((pv_value / eui) * 100) if eui != 0 else 0.0

        # ---- Layout: charts and KPIs (kept identical)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy")
            st_plotly_chart(monthly_chart, use_container_width=True)
        with col2:
            st.subheader("Annual Energy")
            st_plotly_chart(annual_chart, use_container_width=True)

        # KPI calculations (kept identical logic)
        monthly_avr = (totals["kWh"].sum()) / 12
        net_total = totals["kWh"].sum()
        total_energy = totals.loc[totals["kWh"] > 0, "kWh"].sum()
        pv_total = abs(df_melted.groupby("End_Use")["kWh"].sum().get("On-site_Generation", 0.0))

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st_plotly_chart(energy_intensity_chart, use_container_width=True)
        with col2:
            st.subheader("Energy KPI's")
            st.metric(label="Monthly Average Energy Consumption", value=f"{monthly_avr:,.0f} kWh")
            st.metric(label="Total Annual Energy Consumption", value=f"{total_energy:,.0f} kWh")
            st.metric(label="Net Annual Energy Consumption", value=f"{net_total:,.0f} kWh")
            st.metric(label="EUI", value=f"{eui:,.1f} kWh/m2.a")
            st.metric(label="Net EUI", value=f"{net_eui:,.1f} kWh/m2.a")
            st.metric(label="On-site Generation Production", value=f"{pv_total:,.1f} kWh")
            st.metric(label="On-site Generation Coverage", value=f"{pv_coverage:,.1f} %")

        st.markdown("---")
        st.write("## Energy Balance (per Energy Source)")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy Demand")
            st_plotly_chart(monthly_chart_source, use_container_width=True)
        with col2:
            st.subheader("Annual Energy Demand")
            st_plotly_chart(annual_chart_per_source, use_container_width=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st_plotly_chart(energy_intensity_chart_per_source, use_container_width=True)
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
# Tab 5b — Model Inputs QA
# =========================
with tab_model_qa:
    if uploaded_file:
        st.write("## Model Inputs QA")
        st.metric("Active Scenario", active_selected)

        st.caption(
            "This tab documents the inputs and assumptions used in the energy simulation. "
            "General model setup inputs are global. Room types are global by default; room types, envelope components, systems and custom parameters can also be created as active-scenario-specific objects when required."
        )
        st.info(
            "Inputs tagged as **Assumption** are highlighted in the QA review table and should be replaced by documented project references where possible. "
            "The sanity ranges are informative quality-control checks, not automatic code-compliance limits."
        )

        if "model_inputs_qa_df" not in st.session_state or not isinstance(st.session_state.get("model_inputs_qa_df"), pd.DataFrame):
            st.session_state["model_inputs_qa_df"] = default_model_inputs_qa_df()
        st.session_state["model_inputs_qa_df"] = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))

        if st.session_state.get("_model_inputs_qa_flash") == "updated":
            st.success("Model Inputs QA updated.")
            del st.session_state["_model_inputs_qa_flash"]

        all_model_inputs_df = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))
        current_model_inputs_df = model_inputs_df_for_scenario(all_model_inputs_df, active_selected)
        qa_eval = evaluate_model_inputs_qa_df(current_model_inputs_df)
        summary = model_inputs_qa_summary(current_model_inputs_df)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Input completeness", f"{summary['completeness']} %")
        with m2:
            st.metric("Required inputs", f"{summary['required']}")
        with m3:
            st.metric("Missing required", f"{summary['missing']}")
        with m4:
            st.metric("Assumption-tagged", f"{summary['assumptions']}")
        with m5:
            st.metric("QA review flags", f"{summary['review']}")

        if summary["missing"] > 0:
            st.warning("Some required inputs for the active scenario are still missing.")
        if summary["assumptions"] > 0:
            st.warning("Assumption-tagged inputs should be reviewed before issuing final simulation results.")

        st.write("### Add model input objects")
        st.caption(
            "Use the controls below to add room types, envelope constructions/components, HVAC/DHW systems and custom parameters. "
            "Each new object can be stored as a global object or as an active-scenario-specific object. Room types are global by default. The controls are inside a form, so typing or changing dropdowns does not rerun the page."
        )

        add_room_submit = False
        add_env_submit = False
        add_system_submit = False

        with st.form("model_inputs_qa_add_object_form", clear_on_submit=False):
            add_col1, add_col2, add_col3 = st.columns(3)
            with add_col1:
                with st.expander("Add room type", expanded=False):
                    new_room_type_name = st.text_input("Room type name", value="Office", key="mi_new_room_type_name")
                    new_room_scope_label = st.selectbox(
                        "Scope",
                        MODEL_INPUT_OBJECT_SCOPE_OPTIONS,
                        index=0,
                        key="mi_new_room_scope",
                        help="Room types are global by default, but can be made scenario-specific when the room definition changes between scenarios.",
                    )
                    add_room_submit = st.form_submit_button("Add Room Type", use_container_width=True)
            with add_col2:
                with st.expander("Add envelope component", expanded=False):
                    new_env_type = st.selectbox("Construction/component type", MODEL_INPUT_ENVELOPE_COMPONENT_TYPES, key="mi_new_envelope_type")
                    new_env_name = st.text_input("Component name", value="New component", key="mi_new_envelope_name")
                    new_env_scope_label = st.selectbox("Scope", MODEL_INPUT_OBJECT_SCOPE_OPTIONS, index=1, key="mi_new_envelope_scope")
                    add_env_submit = st.form_submit_button("Add Envelope Component", use_container_width=True)
            with add_col3:
                with st.expander("Add system", expanded=False):
                    new_system_type = st.selectbox("System type", MODEL_INPUT_SYSTEM_TYPES, key="mi_new_system_type")
                    new_system_name = st.text_input("System name", value="New system", key="mi_new_system_name")
                    new_system_scope_label = st.selectbox("Scope", MODEL_INPUT_OBJECT_SCOPE_OPTIONS, index=1, key="mi_new_system_scope")
                    add_system_submit = st.form_submit_button("Add System", use_container_width=True)

        if add_room_submit:
            add_model_room_type(new_room_type_name, active_selected, scope="Global" if new_room_scope_label == "Global" else "Scenario")
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()
        if add_env_submit:
            add_model_envelope_component(new_env_type, new_env_name, active_selected, scope="Global" if new_env_scope_label == "Global" else "Scenario")
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()
        if add_system_submit:
            add_model_system(new_system_type, new_system_name, active_selected, scope="Global" if new_system_scope_label == "Global" else "Scenario")
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()

        st.markdown("---")
        st.write("### Structured input editor")
        st.caption(
            "Edits are collected in a form. The app updates calculations/export data only after **Update Model Inputs QA** is clicked. "
            "Mark a parameter or complete item for removal and click update to delete it."
        )

        # Work with the full dataframe so non-active scenario rows are preserved.
        mi_full = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df")).reset_index().rename(columns={"index": "_orig_index"})
        mask_active = mi_full["Scenario"].astype(str).eq(MODEL_INPUT_GLOBAL_SCENARIO) | mi_full["Scenario"].astype(str).eq(str(active_selected))
        mi_edit = mi_full.loc[mask_active].copy()
        mi_keep_other = mi_full.loc[~mask_active].drop(columns=["_orig_index"]).copy()

        _editor_key_token = hashlib.md5(str(st.session_state.get(_RAW_TOKEN_KEY, "model_inputs_structured")).encode("utf-8")).hexdigest()[:8]
        edited_rows = []
        deleted_orig_indices = set()
        deleted_item_keys = set()
        duplicate_item_requests = []
        rename_item_requests = []
        remove_item_requests = []
        custom_param_add_requests = []

        def _value_widget(row, key_prefix):
            param = str(row.get("Parameter", ""))
            unit = str(row.get("Unit", ""))
            value = str(row.get("Value", ""))
            p_low = param.lower()
            u_low = unit.lower()
            if p_low in ["served room types", "served room types / zones"]:
                opts = [str(x).strip() for x in room_type_options if str(x).strip()]
                current_values = _split_model_input_multi_value(value)
                for _v in current_values:
                    if _v and _v not in opts:
                        opts.append(_v)
                if not opts:
                    st.caption("No Room Type objects are defined yet. Add Room Types in General Model Setup first.")
                return ", ".join(st.multiselect(
                    "Value",
                    options=opts,
                    default=[v for v in current_values if v in opts],
                    key=f"{key_prefix}_value_rooms",
                    help="Select all Room Type objects served by this AHU.",
                ))
            if p_low == "daylight-controlled dimming":
                checked = str(value).strip().lower() in {"yes", "true", "1", "x"}
                return "Yes" if st.checkbox(
                    "Value",
                    value=checked,
                    key=f"{key_prefix}_value_daylight_dimming",
                    help="Tick if automatic daylight-linked dimming is included in the simulation for this room type.",
                ) else "No"
            if p_low == "thermal mass":
                mass_options = [
                    "",
                    "Very Low",
                    "Low",
                    "Medium",
                    "High",
                    "Very High",
                    "Other / Custom thermal capacity",
                ]
                current_value = value.strip()
                current_l = current_value.lower()
                option_lookup = {opt.lower(): opt for opt in mass_options}
                selected_default = option_lookup.get(current_l, "Other / Custom thermal capacity" if current_value else "")
                selected = st.selectbox(
                    "Value",
                    mass_options,
                    index=mass_options.index(selected_default),
                    key=f"{key_prefix}_value_thermal_mass_select",
                    help="Choose a qualitative thermal-mass class or enter a numeric thermal-capacity value below.",
                )
                custom_default = "" if current_l in option_lookup else current_value
                custom_value = st.text_input(
                    "Custom thermal capacity value",
                    value=custom_default,
                    key=f"{key_prefix}_value_thermal_mass_custom",
                    help="Optional numeric value, e.g. 165. Use the Unit field to select kJ/m²K, Wh/m²K, kJ/m³K, etc.",
                )
                custom_value = str(custom_value or "").strip()
                return custom_value if custom_value else selected
            if p_low in ["heating delivery", "cooling delivery"]:
                delivery_options = [
                    "",
                    "Radiant Ceiling",
                    "Fan Coil",
                    "Floor Heating",
                    "Floor Cooling",
                    "Radiator",
                    "Chilled Beam",
                    "Air Handling Unit / Air System",
                    "VAV / CAV Air System",
                    "VRF Indoor Unit",
                    "Other / Custom",
                ]
                current_value = value.strip()
                selected_default = current_value if current_value in delivery_options else ("Other / Custom" if current_value else "")
                selected = st.selectbox(
                    "Value",
                    delivery_options,
                    index=delivery_options.index(selected_default),
                    key=f"{key_prefix}_value_delivery_select",
                    help="Choose a typical delivery system or type a custom value below. The custom value is used when filled.",
                )
                custom_default = "" if current_value in delivery_options else current_value
                custom_value = st.text_input(
                    "Custom delivery type",
                    value=custom_default,
                    key=f"{key_prefix}_value_delivery_custom",
                    help="Optional. Use this when the delivery type is not covered by the dropdown.",
                )
                custom_value = str(custom_value or "").strip()
                return custom_value if custom_value else selected
            if p_low == "associated shading device":
                opts = ["", "None"] + [x for x in shading_device_options if str(x).strip()]
                opts = list(dict.fromkeys(opts))
                if value and value not in opts:
                    opts = [value] + opts
                val = value if value in opts else ""
                return st.selectbox("Value", opts, index=opts.index(val), key=f"{key_prefix}_value")
            if "energy source" in p_low:
                opts = [""] + ENERGY_SOURCE_ORDER + ["Other"]
                val = value if value in opts else ""
                return st.selectbox("Value", opts, index=opts.index(val), key=f"{key_prefix}_value")
            if u_low.strip() == "yes/no" or p_low.endswith("permitted") or p_low in ["demand controlled ventilation", "natural ventilation permitted"]:
                opts = ["", "Yes", "No", "Not applicable"]
                val = value if value in opts else ""
                return st.selectbox("Value", opts, index=opts.index(val), key=f"{key_prefix}_value")
            return st.text_input("Value", value=value, key=f"{key_prefix}_value")

        custom_rows_to_add = []

        try:
            room_type_options = sorted(set(
                mi_edit.loc[
                    mi_edit["Category"].astype(str).eq("Room Types")
                    & mi_edit["Item Type"].astype(str).eq("Room Type"),
                    "Item Name"
                ].dropna().astype(str).tolist()
            ))
        except Exception:
            room_type_options = []

        try:
            shading_device_options = sorted(set(
                mi_edit.loc[
                    mi_edit["Category"].astype(str).eq("Thermal Envelope")
                    & mi_edit["Item Type"].astype(str).eq("Shading Devices"),
                    "Item Name"
                ].dropna().astype(str).tolist()
            ))
        except Exception:
            shading_device_options = []

        def _render_model_input_item(item, source_df):
            scenario_i = str(item.get("Scenario", MODEL_INPUT_GLOBAL_SCENARIO))
            scope_i = str(item.get("Scope", "Scenario"))
            category_i = str(item.get("Category", "Other / Custom Inputs"))
            item_type_i = str(item.get("Item Type", category_i))
            item_name_i = str(item.get("Item Name", "General"))
            item_key_tuple = (scenario_i, category_i, item_type_i, item_name_i)
            scope_label_short = "Global" if scope_i == "Global" or scenario_i == MODEL_INPUT_GLOBAL_SCENARIO else "Scenario-Specific"
            _object_label = f"{item_type_i} | {item_name_i} | {scope_label_short}"

            rows_item = source_df.loc[
                source_df["Scenario"].astype(str).eq(scenario_i)
                & source_df["Scope"].astype(str).eq(scope_i)
                & source_df["Category"].astype(str).eq(category_i)
                & source_df["Item Type"].astype(str).eq(item_type_i)
                & source_df["Item Name"].astype(str).eq(item_name_i)
            ].copy()

            with st.expander(_object_label, expanded=False):
                st.markdown(f"### **{item_name_i}**")

                # Keep the ID/action controls together so object management is easy to find.
                id_rows_item = rows_item.loc[rows_item["Parameter"].astype(str).eq("ID")].copy()
                rows_item = rows_item.loc[~rows_item["Parameter"].astype(str).eq("ID")].copy()

                with st.expander("Object ID", expanded=False):
                    item_delete_key = _safe_model_input_key("mi", _editor_key_token, "delete_item", scenario_i, category_i, item_type_i, item_name_i)
                    item_duplicate_key = _safe_model_input_key("mi", _editor_key_token, "duplicate_item", scenario_i, category_i, item_type_i, item_name_i)
                    item_rename_key = _safe_model_input_key("mi", _editor_key_token, "rename_item", scenario_i, category_i, item_type_i, item_name_i)
                    item_duplicate_name_key = _safe_model_input_key("mi", _editor_key_token, "duplicate_item_name", scenario_i, category_i, item_type_i, item_name_i)
                    item_rename_name_key = _safe_model_input_key("mi", _editor_key_token, "rename_item_name", scenario_i, category_i, item_type_i, item_name_i)

                    id_record_to_save = None
                    if not id_rows_item.empty:
                        id_row = id_rows_item.iloc[0]
                        id_orig_idx = int(id_row.get("_orig_index", -1))
                        id_key_prefix = _safe_model_input_key("mi", _editor_key_token, id_orig_idx, scenario_i, category_i, item_type_i, item_name_i, "ID")
                        st.caption("Object identifier used for traceability in saved projects and reports.")
                        id_c1, id_c2, id_c3 = st.columns([1.6, 1.2, 2.4])
                        with id_c1:
                            object_id_value = st.text_input("ID", value=str(id_row.get("Value", "")), key=f"{id_key_prefix}_value")
                        with id_c2:
                            current_id_source = str(id_row.get("Source Type", "Other"))
                            if current_id_source not in MODEL_INPUT_SOURCE_TYPES:
                                current_id_source = "Other"
                            object_id_source = st.selectbox("Source Type", MODEL_INPUT_SOURCE_TYPES, index=MODEL_INPUT_SOURCE_TYPES.index(current_id_source), key=f"{id_key_prefix}_source")
                        with id_c3:
                            object_id_ref = st.text_input("Source document / reference", value=str(id_row.get("Source Document / Reference", "Model Inputs QA")), key=f"{id_key_prefix}_source_ref")
                        object_id_notes = st.text_input("ID notes", value=str(id_row.get("Notes", "")), key=f"{id_key_prefix}_notes")
                        id_record_to_save = _mi_row(
                            scenario_i, scope_i, category_i, item_type_i, item_name_i,
                            "ID", value=object_id_value, unit="-", required=True,
                            source_type=object_id_source, source_ref=object_id_ref,
                            reference="Unique object identifier for traceability",
                            min_check=np.nan, max_check=np.nan, usual_min=np.nan, usual_max=np.nan,
                            range_justification=str(id_row.get("Range Justification", "")), notes=object_id_notes,
                        )
                    else:
                        st.caption("No ID row was found for this object. It will be added automatically after update.")

                    duplicate_item = False
                    rename_item = False
                    remove_item = False
                    duplicate_item_name = f"{item_name_i} Copy"
                    rename_item_name = item_name_i

                    if category_i != "General Model Setup":
                        duplicate_item_name = st.text_input(
                            "Name for duplicate",
                            value=f"{item_name_i} Copy",
                            key=item_duplicate_name_key,
                            help="Edit this name before clicking Duplicate. If the name already exists, the app adds a numeric suffix automatically.",
                        )
                        duplicate_item = st.form_submit_button(
                            "Duplicate this complete object",
                            key=item_duplicate_key,
                            use_container_width=True,
                            help="Creates a copy of this object using the current form values. The copied object receives the name above and keeps the same Global / Scenario-Specific scope.",
                        )

                        rename_item_name = st.text_input(
                            "New object name",
                            value=item_name_i,
                            key=item_rename_name_key,
                            help="Edit this name and click Rename. If the name already exists, the app adds a numeric suffix automatically.",
                        )
                        rename_item = st.form_submit_button(
                            "Rename this complete object",
                            key=item_rename_key,
                            use_container_width=True,
                            help="Renames this object and keeps all parameter values, references, assumptions and QA justifications.",
                        )

                        remove_item = st.form_submit_button(
                            "Remove this complete object",
                            key=item_delete_key,
                            use_container_width=True,
                            help="Removes this complete object after the form is submitted.",
                        )
                    else:
                        st.caption("General setup is a single global object and is not duplicated, renamed or removed.")

                    if remove_item:
                        deleted_item_keys.add(item_key_tuple)
                        remove_item_requests.append(item_key_tuple)
                    if duplicate_item:
                        duplicate_item_requests.append((item_key_tuple, str(duplicate_item_name or "").strip()))
                    if rename_item:
                        rename_item_requests.append((item_key_tuple, str(rename_item_name or "").strip()))
                    if id_record_to_save is not None and not remove_item:
                        edited_rows.append(id_record_to_save)

                with st.expander("Add custom parameter", expanded=False):
                    custom_key_prefix = _safe_model_input_key("mi", _editor_key_token, "custom", scenario_i, category_i, item_type_i, item_name_i)
                    custom_param_name = st.text_input("Custom parameter name", value="", key=f"{custom_key_prefix}_name")
                    custom_param_unit_choice = st.selectbox(
                        "Unit",
                        MODEL_INPUT_COMMON_UNITS,
                        index=MODEL_INPUT_COMMON_UNITS.index("-") if "-" in MODEL_INPUT_COMMON_UNITS else 0,
                        key=f"{custom_key_prefix}_unit_select",
                    )
                    if custom_param_unit_choice == "Other / Custom":
                        custom_param_unit = st.text_input("Custom unit", value="", key=f"{custom_key_prefix}_unit_custom").strip() or "Other / Custom"
                    else:
                        custom_param_unit = custom_param_unit_choice
                    custom_param_required = st.checkbox("Required", value=False, key=f"{custom_key_prefix}_required")
                    custom_param_source = st.selectbox(
                        "Source Type",
                        MODEL_INPUT_SOURCE_TYPES,
                        index=MODEL_INPUT_SOURCE_TYPES.index("Assumption"),
                        key=f"{custom_key_prefix}_source",
                    )
                    custom_param_ref = st.text_input("Source document / reference", value="", key=f"{custom_key_prefix}_source_ref")
                    custom_param_submit = st.form_submit_button(
                        "Add custom parameter to this object",
                        key=f"{custom_key_prefix}_submit",
                        use_container_width=True,
                        help="Adds this parameter immediately to the current object. Current form edits are committed at the same time so no changes are lost.",
                    )
                    if custom_param_submit:
                        if str(custom_param_name).strip():
                            custom_param_add_requests.append(_mi_row(
                                scenario_i, scope_i, category_i, item_type_i, item_name_i,
                                str(custom_param_name).strip(),
                                value="", unit=custom_param_unit, required=custom_param_required,
                                source_type=custom_param_source, source_ref=custom_param_ref,
                                reference="Custom user-defined parameter", min_check=np.nan, max_check=np.nan,
                                notes="User-defined parameter",
                            ))
                        else:
                            st.warning("Enter a custom parameter name before adding it.")

                for _, row in rows_item.iterrows():
                    orig_idx = int(row.get("_orig_index", -1))
                    parameter = str(row.get("Parameter", ""))
                    key_prefix = _safe_model_input_key("mi", _editor_key_token, orig_idx, scenario_i, category_i, item_type_i, item_name_i, parameter)
                    if str(row.get("Source Type", "")) == "Assumption":
                        st.markdown("<div style='background-color:#fff3cd;padding:4px 8px;border-radius:4px;margin-top:6px;'><b>Assumption-tagged input</b></div>", unsafe_allow_html=True)
                    c1, c2, c3, c4, c5 = st.columns([2.2, 2.0, 0.9, 0.8, 0.8])
                    with c1:
                        st.markdown(f"**{parameter}**")
                    with c2:
                        value = _value_widget(row, key_prefix)
                    with c3:
                        _unit_current = str(row.get("Unit", ""))
                        _unit_options = _model_input_unit_options(category_i, item_type_i, parameter, _unit_current)
                        _unit_default = _unit_current if _unit_current in _unit_options else (_unit_options[0] if _unit_options else "-")
                        _unit_selected = st.selectbox(
                            "Unit",
                            _unit_options,
                            index=_unit_options.index(_unit_default) if _unit_default in _unit_options else 0,
                            key=f"{key_prefix}_unit_select",
                            help="Metric unit options are filtered according to the component/system type and parameter.",
                        )
                        if _unit_selected == "Other / Custom":
                            unit = st.text_input(
                                "Custom unit",
                                value="" if _unit_current in _unit_options else _unit_current,
                                key=f"{key_prefix}_unit_custom",
                            ).strip() or "Other / Custom"
                        else:
                            unit = _unit_selected
                    with c4:
                        required = st.checkbox("Required", value=bool(row.get("Required", False)), key=f"{key_prefix}_required")
                    with c5:
                        remove_param = st.checkbox("Remove", value=False, key=f"{key_prefix}_remove")

                    c6, c7, c8, c9, c10, c11 = st.columns([1.3, 2.1, 2.0, 0.9, 0.9, 2.2])
                    with c6:
                        current_source = str(row.get("Source Type", "Assumption"))
                        if current_source not in MODEL_INPUT_SOURCE_TYPES:
                            current_source = "Other"
                        source_type = st.selectbox("Source Type", MODEL_INPUT_SOURCE_TYPES, index=MODEL_INPUT_SOURCE_TYPES.index(current_source), key=f"{key_prefix}_source")
                    with c7:
                        source_ref = st.text_input("Source document / reference", value=str(row.get("Source Document / Reference", "")), key=f"{key_prefix}_source_ref")
                    with c8:
                        reference = st.text_input("Reference / target", value=str(row.get("Reference / Target", "")), key=f"{key_prefix}_reference")
                    with c9:
                        min_default = "" if pd.isna(row.get("Min Check", np.nan)) else str(row.get("Min Check"))
                        min_check_txt = st.text_input("Min", value=min_default, key=f"{key_prefix}_min")
                    with c10:
                        max_default = "" if pd.isna(row.get("Max Check", np.nan)) else str(row.get("Max Check"))
                        max_check_txt = st.text_input("Max", value=max_default, key=f"{key_prefix}_max")
                    with c11:
                        notes = st.text_input("Notes", value=str(row.get("Notes", "")), key=f"{key_prefix}_notes")

                    try:
                        min_check = float(str(min_check_txt).replace(",", ".")) if str(min_check_txt).strip() != "" else np.nan
                    except Exception:
                        min_check = np.nan
                    try:
                        max_check = float(str(max_check_txt).replace(",", ".")) if str(max_check_txt).strip() != "" else np.nan
                    except Exception:
                        max_check = np.nan

                    usual_min, usual_max = _model_input_usual_range(category_i, item_type_i, parameter, unit)
                    if pd.isna(usual_min):
                        usual_min = row.get("Usual Min", np.nan)
                    if pd.isna(usual_max):
                        usual_max = row.get("Usual Max", np.nan)
                    usual_range_label = _format_usual_range(usual_min, usual_max, unit)
                    if usual_range_label:
                        st.caption(f"Usual range for QA: {usual_range_label}")
                    range_justification = str(row.get("Range Justification", ""))
                    if _is_model_input_out_of_usual_range(value, usual_min, usual_max):
                        st.markdown(
                            "<div style='background-color:#f5b7b1;color:#6b0000;padding:6px 10px;border-radius:4px;margin-top:6px;'>"
                            "🚩 <b>Value out of usual range.</b> Please review and document why this value is applicable for the project."
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        range_justification = st.text_input(
                            "Value out of usual range. Please justify",
                            value=str(row.get("Range Justification", "")),
                            key=f"{key_prefix}_range_justification",
                        )

                    if remove_param:
                        deleted_orig_indices.add(orig_idx)
                    else:
                        edited_rows.append(_mi_row(
                            scenario_i, scope_i, category_i, item_type_i, item_name_i,
                            parameter, value=value, unit=unit, required=required,
                            source_type=source_type, source_ref=source_ref,
                            reference=reference, min_check=min_check, max_check=max_check,
                            usual_min=usual_min, usual_max=usual_max, range_justification=range_justification,
                            notes=notes,
                        ))
                    st.markdown("---")

        with st.form("model_inputs_qa_structured_form", clear_on_submit=False):
            for cat in MODEL_INPUT_CATEGORIES:
                if cat == "Room Types":
                    continue
                if cat == "General Model Setup":
                    cat_df = mi_edit.loc[mi_edit["Category"].astype(str).isin(["General Model Setup", "Room Types"])].copy()
                else:
                    cat_df = mi_edit.loc[mi_edit["Category"].astype(str) == cat].copy()
                if cat_df.empty:
                    continue
                with st.expander(cat, expanded=False):
                    if cat == "General Model Setup":
                        st.caption("This section contains global simulation setup inputs and room-type definitions. Room types are global by default, but can also be created for the active scenario only.")
                    item_df_keys = cat_df[["Scenario", "Scope", "Category", "Item Type", "Item Name"]].drop_duplicates().reset_index(drop=True)
                    if cat == "General Model Setup":
                        general_item_keys = item_df_keys.loc[item_df_keys["Category"].astype(str).eq("General Model Setup")]
                        room_item_keys = item_df_keys.loc[item_df_keys["Category"].astype(str).eq("Room Types")]
                        for _, item in general_item_keys.iterrows():
                            _render_model_input_item(item, cat_df)
                        if not room_item_keys.empty:
                            with st.expander("Room Types", expanded=False):
                                for _, item in room_item_keys.iterrows():
                                    _render_model_input_item(item, cat_df)
                    elif cat == "Thermal Envelope":
                        st.caption("Envelope components are grouped by construction/component type. Open a component-type group first, then open the specific construction/object to edit its parameters.")
                        try:
                            _item_types_present = item_df_keys["Item Type"].dropna().astype(str).tolist()
                        except Exception:
                            _item_types_present = []
                        _ordered_env_types = [t for t in MODEL_INPUT_ENVELOPE_COMPONENT_TYPES if t in set(_item_types_present)]
                        _extra_env_types = sorted([t for t in set(_item_types_present) if t not in set(_ordered_env_types)])
                        for _env_type in _ordered_env_types + _extra_env_types:
                            _env_type_keys = item_df_keys.loc[item_df_keys["Item Type"].astype(str).eq(str(_env_type))].copy()
                            if _env_type_keys.empty:
                                continue
                            with st.expander(str(_env_type), expanded=False):
                                for _, item in _env_type_keys.iterrows():
                                    _render_model_input_item(item, cat_df)
                    else:
                        for _, item in item_df_keys.iterrows():
                            _render_model_input_item(item, cat_df)

            c_upd, c_reset = st.columns([1, 1])
            with c_upd:
                update_model_inputs = st.form_submit_button("Update Model Inputs QA", use_container_width=True)
            with c_reset:
                reset_global_inputs = st.form_submit_button("Reset global setup inputs", use_container_width=True)

        if reset_global_inputs:
            df_current = sanitize_model_inputs_qa_df(st.session_state.get("model_inputs_qa_df"))
            df_current = df_current.loc[~df_current["Scenario"].astype(str).eq(MODEL_INPUT_GLOBAL_SCENARIO)].copy()
            df_current = pd.concat([pd.DataFrame(default_model_inputs_global_rows()), df_current], ignore_index=True)
            st.session_state["model_inputs_qa_df"] = sanitize_model_inputs_qa_df(df_current)
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()

        def _model_input_filtered_rows_current_form():
            """Return edited/custom rows from the current form, excluding removed objects and duplicate parameters."""
            rows_filtered_local = []
            seen_param_keys_local = set()
            for rec in list(edited_rows) + list(custom_rows_to_add):
                item_key = (str(rec.get("Scenario")), str(rec.get("Category")), str(rec.get("Item Type")), str(rec.get("Item Name")))
                param_key = item_key + (str(rec.get("Parameter")),)
                if item_key in deleted_item_keys:
                    continue
                if param_key in seen_param_keys_local:
                    continue
                seen_param_keys_local.add(param_key)
                rows_filtered_local.append(rec)
            return rows_filtered_local

        def _model_input_next_copy_name(existing_rows: list, item_key: tuple, requested_name: str = "") -> str:
            """Create a stable unique object name for a duplicated Model Inputs QA object."""
            scenario_i, category_i, item_type_i, item_name_i = [str(x) for x in item_key]
            requested = str(requested_name or "").strip() or f"{item_name_i} Copy"
            return _model_input_ensure_unique_name(existing_rows, item_key, requested, allow_current=False)

        def _model_input_next_rename_name(existing_rows: list, item_key: tuple, requested_name: str = "") -> str:
            """Create a stable unique object name when renaming an existing Model Inputs QA object."""
            scenario_i, category_i, item_type_i, item_name_i = [str(x) for x in item_key]
            requested = str(requested_name or "").strip() or item_name_i
            return _model_input_ensure_unique_name(existing_rows, item_key, requested, allow_current=True)

        if remove_item_requests:
            # A remove button is a form submit action. Commit the current form values first,
            # excluding the selected object, so unrelated unsaved edits are not lost.
            rows_filtered = _model_input_filtered_rows_current_form()
            combined = pd.concat([mi_keep_other, pd.DataFrame(rows_filtered)], ignore_index=True)
            combined = sanitize_model_inputs_qa_df(combined)
            st.session_state["model_inputs_qa_df"] = combined
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()

        if duplicate_item_requests:
            # A duplicate button is a form submit action. Commit the current form values first,
            # then add a copy of the requested object so unsaved edits are not lost.
            rows_filtered = _model_input_filtered_rows_current_form()
            duplicate_rows = []
            for dup_key, requested_name in duplicate_item_requests:
                dup_key = tuple(str(x) for x in dup_key)
                if dup_key in deleted_item_keys:
                    continue
                new_item_name = _model_input_next_copy_name(list(rows_filtered) + list(duplicate_rows), dup_key, requested_name)
                for rec in rows_filtered:
                    rec_key = (str(rec.get("Scenario")), str(rec.get("Category")), str(rec.get("Item Type")), str(rec.get("Item Name")))
                    if rec_key != dup_key:
                        continue
                    new_rec = dict(rec)
                    new_rec["Item Name"] = new_item_name
                    if str(new_rec.get("Parameter", "")) == "ID":
                        new_rec["Value"] = _model_input_object_id(str(new_rec.get("Category", "")), str(new_rec.get("Item Type", "")), new_item_name, str(new_rec.get("Scenario", "")))
                    # Keep traceability without changing the original source document/reference.
                    prev_notes = str(new_rec.get("Notes", "")).strip()
                    copy_note = f"Duplicated from {dup_key[3]}"
                    new_rec["Notes"] = copy_note if not prev_notes else f"{prev_notes}; {copy_note}"
                    duplicate_rows.append(new_rec)
            combined = pd.concat([mi_keep_other, pd.DataFrame(rows_filtered + duplicate_rows)], ignore_index=True)
            combined = sanitize_model_inputs_qa_df(combined)
            st.session_state["model_inputs_qa_df"] = combined
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()

        if rename_item_requests:
            # A rename button is a form submit action. Commit the current form values first,
            # then rename all rows belonging to the selected object.
            rows_filtered = _model_input_filtered_rows_current_form()
            rename_map = {}
            for rename_key, requested_name in rename_item_requests:
                rename_key = tuple(str(x) for x in rename_key)
                if rename_key in deleted_item_keys:
                    continue
                new_item_name = _model_input_next_rename_name(rows_filtered, rename_key, requested_name)
                rename_map[rename_key] = new_item_name

            renamed_rows = []
            for rec in rows_filtered:
                rec_key = (str(rec.get("Scenario")), str(rec.get("Category")), str(rec.get("Item Type")), str(rec.get("Item Name")))
                if rec_key in rename_map:
                    new_rec = dict(rec)
                    new_rec["Item Name"] = rename_map[rec_key]
                    renamed_rows.append(new_rec)
                else:
                    renamed_rows.append(rec)
            combined = pd.concat([mi_keep_other, pd.DataFrame(renamed_rows)], ignore_index=True)
            combined = sanitize_model_inputs_qa_df(combined)
            st.session_state["model_inputs_qa_df"] = combined
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()

        if custom_param_add_requests:
            # A custom-parameter button is a form submit action. Commit the current form
            # values first, then append the requested parameter(s), so the user does not
            # need to scroll to the bottom Update button and unsaved edits are preserved.
            rows_filtered = _model_input_filtered_rows_current_form()
            seen_param_keys = set()
            rows_with_custom = []
            for rec in rows_filtered:
                param_key = (
                    str(rec.get("Scenario")), str(rec.get("Category")), str(rec.get("Item Type")),
                    str(rec.get("Item Name")), str(rec.get("Parameter"))
                )
                seen_param_keys.add(param_key)
                rows_with_custom.append(rec)
            for rec in custom_param_add_requests:
                param_key = (
                    str(rec.get("Scenario")), str(rec.get("Category")), str(rec.get("Item Type")),
                    str(rec.get("Item Name")), str(rec.get("Parameter"))
                )
                if param_key not in seen_param_keys:
                    seen_param_keys.add(param_key)
                    rows_with_custom.append(rec)
            combined = pd.concat([mi_keep_other, pd.DataFrame(rows_with_custom)], ignore_index=True)
            combined = sanitize_model_inputs_qa_df(combined)
            st.session_state["model_inputs_qa_df"] = combined
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()

        if update_model_inputs:
            # Remove full objects selected for deletion and append object-level custom parameters.
            rows_filtered = _model_input_filtered_rows_current_form()
            combined = pd.concat([mi_keep_other, pd.DataFrame(rows_filtered)], ignore_index=True)
            combined = sanitize_model_inputs_qa_df(combined)
            st.session_state["model_inputs_qa_df"] = combined
            st.session_state["_model_inputs_qa_flash"] = "updated"
            st.rerun()

        st.markdown("---")
        with st.expander("Scenario input differences", expanded=False):
            st.caption(
                "This overview compares the effective Model Inputs QA values across all project scenarios. "
                "Global values are included in every scenario; scenario-specific values override matching global values. "
                "Only parameters with different values, units, source tags, or missing definitions are shown."
            )
            _sc_names_mi = list(st.session_state.get("scenarios", {}).keys())
            if len(_sc_names_mi) <= 1:
                st.info("Only one scenario is available. Add more scenarios to compare Model Inputs QA assumptions.")
            else:
                _diff_df_mi, _diff_chart_mi = model_inputs_scenario_differences(
                    st.session_state.get("model_inputs_qa_df"),
                    _sc_names_mi,
                )
                if _diff_df_mi.empty:
                    st.success("No differing Model Inputs QA values were found between scenarios.")
                else:
                    st.write("### Differing parameters between scenarios")
                    st.dataframe(_diff_df_mi, use_container_width=True, height=420)
                    st.caption(
                        "Cells marked with ⚠ are assumption-tagged values. 'Not defined' means the object or parameter exists in at least one scenario, "
                        "but not in that scenario's effective input set."
                    )
                    if not _diff_chart_mi.empty:
                        st.write("### Difference count by scenario and category")
                        _scenario_color_map_mi = st.session_state.get(
                            "color_map_scenarios",
                            default_scenario_color_map(_sc_names_mi),
                        )
                        fig_mi_diff = px.bar(
                            _diff_chart_mi,
                            x="Category",
                            y="Differing inputs",
                            color="Scenario",
                            barmode="group",
                            color_discrete_map=_scenario_color_map_mi,
                            text_auto=".0f",
                            height=420,
                            title="Model Inputs QA differences by category",
                        )
                        fig_mi_diff.update_layout(
                            xaxis_title="Model input category",
                            yaxis_title="Number of differing inputs",
                            legend_title_text="Scenario",
                            margin=dict(l=40, r=20, t=55, b=95),
                        )
                        fig_mi_diff.update_xaxes(tickangle=-30)
                        fig_mi_diff.update_traces(textfont_size=12)
                        st_plotly_chart(fig_mi_diff, use_container_width=True, key="mi_scenario_difference_chart")

        with st.expander("QA interpretation", expanded=False):
            st.markdown(
                """
**Global** rows apply to all scenarios. **Scenario** rows apply only to the active scenario shown above.  
**Missing** means a required input has no value.  
**Assumption** means the value is explicitly tagged as an assumption and should be reviewed.  
**Review** means a numeric value is outside the configured sanity range.  
**Out of usual range** means a numeric value is outside the defined usual QA range and should be justified.  
**OK** means the field is filled and has no current sanity-check warning.

The QA is intended as traceability and model-quality control. It does not replace formal ASHRAE 90.1, LEED, local-code, or certification documentation.
                """
            )
            st.write("### QA review table")
            qa_eval = evaluate_model_inputs_qa_df(model_inputs_df_for_scenario(st.session_state.get("model_inputs_qa_df"), active_selected))
            qa_cols = [
                "Scope", "Category", "Item Type", "Item Name", "Parameter", "Value", "Unit", "Required", "Source Type",
                "Source Document / Reference", "Reference / Target", "Usual Min", "Usual Max", "Range Justification", "QA Status", "QA Message", "Notes"
            ]
            try:
                st.dataframe(
                    qa_eval[qa_cols].style.apply(_style_model_inputs_qa, axis=1),
                    use_container_width=True,
                    height=560,
                )
            except Exception:
                st.dataframe(qa_eval[qa_cols], use_container_width=True, height=560)

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")


# =========================
# Tab 6 — CRREM-Analysis
# =========================
with tab6:
    if uploaded_file:
        st.write(f"## CRREM-Analysis ({st.session_state.get('project_country', 'Germany')})")
        st.metric("Active Scenario", active_selected)

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
                df_crrem = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name)
                df_crrem_m = df_crrem.melt(id_vars="Month", var_name="End_Use", value_name="kWh")

                # Apply efficiency factors (scenario-specific)
                eff_map_crrem = {use: st.session_state.get(f"eff_{use}", 1.0) for use in df_crrem_m["End_Use"].unique()}
                df_crrem_m["Efficiency_Factor"] = df_crrem_m["End_Use"].map(eff_map_crrem).fillna(1.0)
                df_crrem_m["kWh_adj"] = df_crrem_m["kWh"] / df_crrem_m["Efficiency_Factor"]

                # Apply PV scaling (scenario-specific). For CRREM carbon, On-site Generation always offsets Electricity (EF=0).
                pv_apply_scale = bool(st.session_state.get("pv_sc_enabled", False))
                pv_scale = float(st.session_state.get("pv_scale", 1.0))
                pv_mask = df_crrem_m["End_Use"].astype(str) == "On-site_Generation"
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
                # On-site Generation always offsets Electricity
                df_crrem_m.loc[pv_mask, "Energy_Source"] = "Electricity"

                # Annual kWh per source (net, PV included as negative electricity)
                annual_kwh_by_source = df_crrem_m.groupby("Energy_Source", as_index=True)["kWh_adj"].sum()

                # Clamp net electricity to >= 0 (On-site Generation offsets electricity up to demand; no export credit)
                if "Electricity" in annual_kwh_by_source.index:
                    annual_kwh_by_source.loc["Electricity"] = max(float(annual_kwh_by_source.loc["Electricity"]), 0.0)

                # CRREM EUI is consumption-only (exclude On-site_Generation)
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
                                              legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center",
                                                          x=0.5),
                                              margin=dict(l=40, r=20, t=50, b=85))
                        fig_tot.update_yaxes(rangemode="tozero")
                        st_plotly_chart(fig_tot, use_container_width=True, key=f"crrem_tot_emis_{project_label}")

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
                                              legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center",
                                                          x=0.5),
                                              margin=dict(l=40, r=20, t=50, b=85))
                        fig_cum.update_yaxes(rangemode="tozero")
                        st_plotly_chart(fig_cum, use_container_width=True, key=f"crrem_cum_emis_{project_label}")

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
                                                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center",
                                                            x=0.5),
                                                margin=dict(l=40, r=20, t=50, b=85))
                        fig_e_tot.update_yaxes(rangemode="tozero")
                        st_plotly_chart(fig_e_tot, use_container_width=True, key=f"crrem_tot_energy_{project_label}")

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
                                                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center",
                                                            x=0.5),
                                                margin=dict(l=40, r=20, t=50, b=85))
                        fig_e_cum.update_yaxes(rangemode="tozero")
                        st_plotly_chart(fig_e_cum, use_container_width=True, key=f"crrem_cum_energy_{project_label}")

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
                        st_plotly_chart(fig_exc_c, use_container_width=True,
                                        key=f"crrem_cum_exceed_carbon_{project_label}")

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
                        st_plotly_chart(fig_exc_e, use_container_width=True,
                                        key=f"crrem_cum_exceed_energy_{project_label}")

                    # Optional: headroom (limit - project)
                    show_headroom = st.checkbox(
                        "Show headroom (limit − project) charts",
                        value=True,
                        key=f"crrem_show_headroom_{project_label}",
                        help="Positive values indicate compliance; negative values indicate exceedance.",
                    )
                    if show_headroom:
                        h1, h2 = st.columns(2)
                        with h1:
                            headroom_c = (carbon_limit_s - carbon_project_s).astype(float)
                            bar_colors = [CRREM_COLOR_MEASURES if v >= 0 else CRREM_COLOR_LIMIT for v in
                                          headroom_c.values]
                            fig_hc = go.Figure(
                                go.Bar(x=years_list, y=headroom_c.values, marker_color=bar_colors, name="Headroom"))
                            fig_hc.update_layout(height=420, yaxis_title="kgCO₂e/m²·a", title="Carbon headroom",
                                                 margin=dict(l=40, r=20, t=45, b=45))
                            st_plotly_chart(fig_hc, use_container_width=True,
                                            key=f"crrem_headroom_carbon_{project_label}")
                        with h2:
                            headroom_e = (eui_limit_s - eui_project_s).astype(float)
                            bar_colors = [CRREM_COLOR_MEASURES if v >= 0 else CRREM_COLOR_LIMIT for v in
                                          headroom_e.values]
                            fig_he = go.Figure(
                                go.Bar(x=years_list, y=headroom_e.values, marker_color=bar_colors, name="Headroom"))
                            fig_he.update_layout(height=420, yaxis_title="kWh/m²·a", title="EUI headroom",
                                                 margin=dict(l=40, r=20, t=45, b=45))
                            st_plotly_chart(fig_he, use_container_width=True,
                                            key=f"crrem_headroom_energy_{project_label}")


                st.write("## Prognose without measures")
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
                    st_plotly_chart(fig, use_container_width=True)

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
                    st_plotly_chart(fig2, use_container_width=True)

                with st.expander("Additional CRREM diagrams — Baseline", expanded=False):
                    st.caption(
                        "Totals and cumulative charts use your Project Area (m²) and the same CRREM pathway years as the intensity plots.")
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
                    end_uses_no_pv = [u for u in end_uses_all if u != "On-site_Generation"]

                    param_specs = {}
                    param_options = []


                    def _add_param(label: str, spec: dict):
                        param_options.append(label)
                        param_specs[label] = spec


                    # Emission Factors (numeric)
                    _add_param("Emission Factors → Electricity",
                               {"kind": "ef", "source": "Electricity", "dtype": "float"})
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
                    _add_param("On-site_Generation → Annual Production (kWh/a)", {"kind": "pv", "dtype": "float"})
                    # Backwards compatible label (legacy PV naming)
                    try:
                        param_specs["PV_Generation → PV Annual Production (kWh/a)"] = param_specs["On-site_Generation → Annual Production (kWh/a)"]
                    except Exception:
                        pass

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
                        st.caption(
                            "Edits in this table are **applied only when you click `Update Measures`** (same logic as in `Raw Data`). "
                            "This avoids recalculating the CRREM measures charts on every cell edit."
                        )

                        # Flash message (shown after rerun)
                        if st.session_state.get("_crrem_measures_flash") == "updated":
                            st.success("Measures updated and applied to CRREM calculations.")
                            del st.session_state["_crrem_measures_flash"]

                        # Ensure committed + draft measures exist
                        if "crrem_measures_df" not in st.session_state or not isinstance(
                                st.session_state.get("crrem_measures_df"), pd.DataFrame):
                            st.session_state["crrem_measures_df"] = pd.DataFrame(
                                columns=["Parameter", "Year", "New Value"])

                        if "crrem_measures_draft_df" not in st.session_state or not isinstance(
                                st.session_state.get("crrem_measures_draft_df"), pd.DataFrame):
                            st.session_state["crrem_measures_draft_df"] = st.session_state["crrem_measures_df"].copy(deep=True)

                        # Add measure row to DRAFT (does not affect calculations until Update Measures)
                        if st.button("Add measure", key="crrem_add_measure_btn", use_container_width=False):
                            df_tmp = st.session_state["crrem_measures_draft_df"].copy()
                            default_param = param_options[0] if param_options else ""
                            df_tmp = pd.concat(
                                [
                                    df_tmp,
                                    pd.DataFrame(
                                        [{"Parameter": default_param, "Year": int(project_year_val), "New Value": ""}]),
                                ],
                                ignore_index=True,
                            )
                            st.session_state["crrem_measures_draft_df"] = df_tmp

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

                        with st.form("crrem_measures_form", clear_on_submit=False):
                            edited_measures = st.data_editor(
                                st.session_state["crrem_measures_draft_df"].copy(deep=True),
                                **editor_kwargs
                            )

                            # Persist edits into DRAFT on every rerun (do not apply to calculations yet).
                            st.session_state["crrem_measures_draft_df"] = edited_measures

                            apply_measures = st.form_submit_button("Update Measures", use_container_width=False)

                        # Deleting measures works on DRAFT (apply afterwards to affect calculations)
                        draft_df = st.session_state.get("crrem_measures_draft_df", pd.DataFrame()).copy()

                        if not draft_df.empty:

                            def _fmt_measure_idx(i):
                                try:
                                    p = str(draft_df.loc[i, "Parameter"]) if pd.notna(
                                        draft_df.loc[i, "Parameter"]) else ""
                                except Exception:
                                    p = ""
                                try:
                                    yv = draft_df.loc[i, "Year"]
                                    y = str(int(float(yv))) if pd.notna(yv) and str(yv).strip() != "" else ""
                                except Exception:
                                    y = ""
                                return f"{i + 1}: {p} @ {y}"


                            def _crrem_delete_selected_measures():
                                sel = st.session_state.get("crrem_measures_delete_idx", [])
                                if sel:
                                    df = st.session_state.get("crrem_measures_draft_df", pd.DataFrame()).copy()
                                    try:
                                        df = df.drop(sel).reset_index(drop=True)
                                    except Exception:
                                        df = df.iloc[[j for j in range(len(df)) if j not in set(sel)]].reset_index(
                                            drop=True)
                                    st.session_state["crrem_measures_draft_df"] = df
                                # Clear selection (safe to mutate in callback)
                                st.session_state["crrem_measures_delete_idx"] = []


                            st.multiselect(
                                "Select measure rows to delete",
                                options=list(draft_df.index),
                                format_func=_fmt_measure_idx,
                                key="crrem_measures_delete_idx",
                            )
                            st.button(
                                "Delete selected measures",
                                key="crrem_delete_measures_btn",
                                on_click=_crrem_delete_selected_measures,
                                use_container_width=False,
                            )

                        if apply_measures:
                            committed = edited_measures.copy(deep=True)
                            st.session_state["crrem_measures_df"] = committed
                            st.session_state["crrem_measures_draft_df"] = committed.copy(deep=True)

                            # Persist measures into the active scenario payload (saved in Scenarios sheet)
                            try:
                                _sc = st.session_state.get("scenarios", {})
                                _act = st.session_state.get("active_scenario")
                                if _act in _sc:
                                    _sc[_act]["crrem_measures"] = _measures_df_to_records(committed)
                                    st.session_state["scenarios"] = _sc
                            except Exception:
                                pass

                            st.session_state["_crrem_measures_flash"] = "updated"
                            st.rerun()

                        st.caption(
                            "Note: tariff measures are stored but do not affect the CRREM Carbon/EUI charts yet.")

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
                        src_base["On-site_Generation"] = "Electricity"

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
                                if ((int(ym) <= int(year)) if inclusive else (int(ym) < int(year))) and int(ym) >= int(
                                        y0):
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

                            # PV annual production override (kWh/a). If set via measures, overrides baseline PV (and sidebar On-site Generation scale).
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

                                if eu == "On-site_Generation":
                                    # On-site Generation always offsets electricity; enforce negative generation
                                    if pv_annual_override_y is not None:
                                        # Absolute annual On-site Generation production (kWh/a) provided by measures
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
                                    df_meas_tl["Category"] = df_meas_tl["Parameter"].astype(str).str.split("→").str[
                                        0].str.strip()
                                    df_meas_tl["Parameter"] = df_meas_tl["Parameter"].astype(str).str.strip()
                                    df_meas_tl = df_meas_tl.sort_values(by="Year", ascending=True)

                                    fig_tl = px.scatter(
                                        df_meas_tl,
                                        x="Year",
                                        y="Parameter",
                                        color="Category",
                                        hover_data={"New Value": True, "Year": True, "Category": True,
                                                    "Parameter": True},
                                    )
                                    fig_tl.update_layout(height=420, xaxis_title="Year", yaxis_title="",
                                                         legend_title="",
                                                         margin=dict(l=20, r=20, t=50, b=30))

                                    fig_tl.update_traces(marker=dict(size=20, symbol="square"))
                                    st_plotly_chart(fig_tl, use_container_width=True)
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
                                figm.add_vline(x=stranding_carbon_meas, line_width=3, line_dash="dash",
                                               line_color="black")
                            st_plotly_chart(figm, use_container_width=True, key="crrem_carbon_measures_chart")

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
                            st_plotly_chart(fige, use_container_width=True, key="crrem_eui_measures_chart")

                        with st.expander("Additional CRREM diagrams — With measures", expanded=False):
                            st.caption(
                                "Totals and cumulative charts are computed from the with-measures trajectories shown above.")
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
                        "Notes: Green Electricity and On-site Generation offset are treated with EF=0. On-site Generation offsets Electricity consumption (no export credit).")

    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")


# =========================
# Tab 6b — LCC-Analysis
# =========================
with tab_lcc:
    if uploaded_file:
        st.write("## LCC-Analysis")
        st.metric("Active Scenario", active_selected)

        # Current project/scenario context
        df_lcc_energy = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name).copy()
        end_uses_lcc = [str(c) for c in df_lcc_energy.columns if str(c) != "Month"]
        project_year_lcc = int(st.session_state.get("project_year", 2025))
        project_area_lcc = float(st.session_state.get("project_area", 0.0) or 0.0)
        currency_lcc = st.session_state.get("currency_symbol", "€")

        scenarios_lcc = st.session_state.get("scenarios", {}) or {}
        active_payload_lcc = scenarios_lcc.get(active_selected, capture_scenario_from_widgets(end_uses_lcc))
        active_payload_lcc["lcc"] = _normalize_lcc_payload(active_payload_lcc.get("lcc", {}), end_uses_lcc)

        valid_enduses_lcc = [str(u) for u in end_uses_lcc]
        _ensure_lcc_global_state(valid_enduses_lcc, scenarios=scenarios_lcc, active_payload=active_payload_lcc)

        if "lcc_investments_df" not in st.session_state or not isinstance(st.session_state.get("lcc_investments_df"), pd.DataFrame):
            _load_lcc_into_widgets(active_payload_lcc, end_uses_lcc)

        # Global LCC parameters use a draft/commit pattern. They are not pushed into calculations
        # or scenarios until the user clicks the LCC update button.
        lcc_global_committed = _get_lcc_global_state_payload(valid_enduses_lcc)
        _seed_lcc_global_draft_from_payload(lcc_global_committed, valid_enduses_lcc, force=False)
        scenarios_lcc = st.session_state.get("scenarios", {}) or {}
        active_payload_lcc = scenarios_lcc.get(active_selected, active_payload_lcc)

        with st.expander("LCC-Analysis", expanded=True):
            st.caption(
                "Global LCC parameters apply to all scenarios. Investment measures remain scenario-specific. "
                "Energy costs use the active scenario tariffs, efficiency factors, energy-source assignment and selected operational end uses. "
                "Submit once to avoid recalculation on every cell edit."
            )

            # Scenario reference options for discounted payback.
            scenario_names_lcc = list(scenarios_lcc.keys()) if scenarios_lcc else [active_selected]
            ref_options_lcc = [""] + [s for s in scenario_names_lcc if s != active_selected]
            ref_key_lcc = _lcc_global_draft_key("payback_reference_scenario")
            ref_default_lcc = st.session_state.get(ref_key_lcc, lcc_global_committed.get("payback_reference_scenario", ""))
            if ref_default_lcc not in ref_options_lcc:
                if "Base" in ref_options_lcc and "Base" != active_selected:
                    ref_default_lcc = "Base"
                elif len(ref_options_lcc) > 1:
                    ref_default_lcc = ref_options_lcc[1]
                else:
                    ref_default_lcc = ""
                # Safe here: this happens before the selectbox is instantiated in this run.
                st.session_state[ref_key_lcc] = ref_default_lcc

            # Draft editor state mirrors the committed scenario-specific LCC investment table.
            if "lcc_investments_draft_df" not in st.session_state or not isinstance(st.session_state.get("lcc_investments_draft_df"), pd.DataFrame):
                st.session_state["lcc_investments_draft_df"] = _lcc_investments_records_to_df(
                    active_payload_lcc.get("lcc", {}).get("investments", []),
                    end_uses=valid_enduses_lcc,
                )

            if st.session_state.get("_lcc_flash") == "updated":
                st.success("LCC inputs updated. Global LCC parameters were applied to all scenarios; investment measures were applied to the active scenario.")
                del st.session_state["_lcc_flash"]

            st.write("### LCC inputs")
            st.caption(
                "Global parameters are shared by all scenarios. Investment measures remain scenario-specific. "
                "Values in this expander are drafts until `Update LCC Inputs` is clicked, so charts do not recalculate on every edit."
            )

            with st.form("lcc_analysis_form", clear_on_submit=False):
                st.write("### Global LCC parameters")
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.number_input(
                        "Analysis Period (years)",
                        min_value=1,
                        max_value=100,
                        step=1,
                        format="%d",
                        key=_lcc_global_draft_key("analysis_period"),
                    )
                with p2:
                    numeric_input(
                        "Interest Rate / Discount Rate (%)",
                        float(st.session_state.get(_lcc_global_draft_key("interest_rate_pct"), 4.0)),
                        key=_lcc_global_draft_key("interest_rate_pct"),
                        min_value=-100.0,
                        max_value=100.0,
                        fmt="{:.4f}",
                    )
                with p3:
                    numeric_input(
                        "CAPEX / O&M Inflation Rate (%)",
                        float(st.session_state.get(_lcc_global_draft_key("capex_inflation_pct"), 2.0)),
                        key=_lcc_global_draft_key("capex_inflation_pct"),
                        min_value=-100.0,
                        max_value=100.0,
                        fmt="{:.4f}",
                    )

                st.write("### Energy inflation rate by source")
                inf_cols = st.columns(3)
                for i, src in enumerate(ENERGY_SOURCE_ORDER):
                    with inf_cols[i % 3]:
                        src_key = _lcc_energy_inflation_draft_key(src)
                        numeric_input(
                            f"{src} Inflation (%)",
                            float(st.session_state.get(src_key, 2.0)),
                            key=src_key,
                            min_value=-100.0,
                            max_value=100.0,
                            fmt="{:.4f}",
                        )

                st.write("### Operational cost filter")
                st.multiselect(
                    "Operational End Uses included in LCC energy cost",
                    options=valid_enduses_lcc,
                    default=st.session_state.get(_lcc_global_draft_key("selected_operational_end_uses"), _lcc_default_selected_enduses(valid_enduses_lcc)),
                    format_func=ui_name,
                    key=_lcc_global_draft_key("selected_operational_end_uses"),
                    help="Only the selected End Uses are included in the operational energy-cost part of the LCC analysis.",
                )

                st.selectbox(
                    "Discounted Payback Reference Scenario",
                    options=ref_options_lcc,
                    index=ref_options_lcc.index(st.session_state.get(ref_key_lcc, "")) if st.session_state.get(ref_key_lcc, "") in ref_options_lcc else 0,
                    format_func=lambda x: "None" if str(x) == "" else str(x),
                    key=ref_key_lcc,
                    help="Discounted payback is calculated against this reference scenario using discounted incremental cash flows.",
                )

                st.write("### Investment, maintenance and replacement assumptions")
                st.caption(
                    "Replacement cost is not entered separately. It is calculated as the initial investment cost escalated by the CAPEX/O&M inflation rate up to each replacement year. "
                    "For measures that affect several uses, enter multiple Assigned End Uses separated by commas (example: Heating, Cooling). "
                    "The measure cost is allocated equally across the assigned End Uses."
                )

                draft_analysis_period_for_editor = max(
                    1,
                    _to_int_lcc(
                        st.session_state.get(_lcc_global_draft_key("analysis_period"), lcc_global_committed.get("analysis_period", 30)),
                        int(lcc_global_committed.get("analysis_period", 30)),
                    ),
                )

                editor_kwargs_lcc = {
                    "num_rows": "dynamic",
                    "use_container_width": True,
                    "key": "lcc_investments_editor",
                }
                if hasattr(st, "column_config"):
                    editor_kwargs_lcc["column_config"] = {
                        "Measure Name": st.column_config.TextColumn("Measure Name", required=False),
                        "Assigned End Uses": st.column_config.TextColumn(
                            "Assigned End Uses",
                            help="Enter one or more End Uses separated by commas, e.g. Heating, Cooling.",
                            required=True,
                        ),
                        "Investment Year": st.column_config.NumberColumn(
                            "Investment Year",
                            min_value=int(project_year_lcc),
                            max_value=int(project_year_lcc + draft_analysis_period_for_editor - 1),
                            step=1,
                            format="%d",
                            required=True,
                        ),
                        "Investment Cost": st.column_config.NumberColumn(
                            f"Investment Cost ({currency_lcc})",
                            min_value=0.0,
                            step=1000.0,
                            format="%.2f",
                        ),
                        "Annual Maintenance Cost (%)": st.column_config.NumberColumn(
                            "Annual Maintenance Cost (% of investment)",
                            min_value=0.0,
                            max_value=100.0,
                            step=0.1,
                            format="%.2f",
                        ),
                        "Life Length (years)": st.column_config.NumberColumn(
                            "Life Length (years)",
                            min_value=0,
                            max_value=200,
                            step=1,
                            format="%d",
                        ),
                    }

                edited_lcc_investments = st.data_editor(
                    _lcc_investments_records_to_df(
                        st.session_state.get("lcc_investments_draft_df", pd.DataFrame(columns=LCC_INVESTMENT_COLUMNS)),
                        end_uses=valid_enduses_lcc,
                    ),
                    **editor_kwargs_lcc,
                )

                apply_lcc_inputs = st.form_submit_button("Update LCC Inputs", use_container_width=False)

            if apply_lcc_inputs:
                committed_lcc_df = _lcc_investments_records_to_df(edited_lcc_investments, end_uses=valid_enduses_lcc)
                st.session_state["lcc_investments_df"] = committed_lcc_df
                st.session_state["lcc_investments_draft_df"] = committed_lcc_df.copy(deep=True)

                # Commit global LCC draft values only on submit. Until this point, charts and
                # scenario payloads continue using the previous committed global assumptions.
                committed_lcc_global = _capture_lcc_global_from_draft_widgets(valid_enduses_lcc)
                st.session_state[LCC_GLOBAL_STATE_KEY] = deepcopy(committed_lcc_global)
                st.session_state["_lcc_global_initialized"] = True

                try:
                    if "scenarios" in st.session_state and st.session_state.get("active_scenario") in st.session_state["scenarios"]:
                        _act_lcc = st.session_state.get("active_scenario")
                        st.session_state["scenarios"][_act_lcc]["lcc"] = _capture_lcc_from_widgets(valid_enduses_lcc)
                        st.session_state["scenarios"][_act_lcc]["lcc_global"] = deepcopy(committed_lcc_global)
                        _apply_lcc_global_to_all_scenarios(valid_enduses_lcc)
                except Exception:
                    pass

                st.session_state["_lcc_flash"] = "updated"
                st.rerun()

        # Use the latest committed values for calculations. Global LCC assumptions are shared by all scenarios.
        lcc_global_active = _get_lcc_global_state_payload(valid_enduses_lcc)
        _apply_lcc_global_to_all_scenarios(valid_enduses_lcc)
        scenarios_lcc = st.session_state.get("scenarios", {}) or {}
        if "scenarios" in st.session_state and active_selected in st.session_state["scenarios"]:
            st.session_state["scenarios"][active_selected]["lcc"] = _capture_lcc_from_widgets(valid_enduses_lcc)
            st.session_state["scenarios"][active_selected]["lcc_global"] = deepcopy(lcc_global_active)
            # Re-apply after updating the active scenario so the operational filter and all other
            # global LCC assumptions remain identical in every scenario payload.
            _apply_lcc_global_to_all_scenarios(valid_enduses_lcc)
            scenarios_lcc = st.session_state.get("scenarios", {}) or {}
            active_payload_lcc = scenarios_lcc.get(active_selected, st.session_state["scenarios"][active_selected])
        else:
            active_payload_lcc["lcc"] = _capture_lcc_from_widgets(valid_enduses_lcc)
            active_payload_lcc["lcc_global"] = deepcopy(lcc_global_active)

        active_lcc_cashflow = compute_lcc_cashflow_table(
            df_lcc_energy,
            active_payload_lcc,
            valid_enduses_lcc,
            project_year_lcc,
            lcc_global=lcc_global_active,
        )

        if active_lcc_cashflow.empty:
            st.info("No LCC cash flows available. Add investment data and/or select operational End Uses in the LCC input expander.")
        else:
            total_nominal_lcc = float(active_lcc_cashflow["Nominal Cost"].sum())
            total_discounted_lcc = float(active_lcc_cashflow["Discounted Cost"].sum())
            cost_per_m2_nominal = total_nominal_lcc / project_area_lcc if project_area_lcc > 0 else np.nan
            cost_per_m2_discounted = total_discounted_lcc / project_area_lcc if project_area_lcc > 0 else np.nan

            by_type_lcc = active_lcc_cashflow.groupby("Cost Type", as_index=False).agg(
                Nominal_Cost=("Nominal Cost", "sum"),
                Discounted_Cost=("Discounted Cost", "sum"),
            )
            type_totals = dict(zip(by_type_lcc["Cost Type"], by_type_lcc["Nominal_Cost"]))

            # Discounted payback vs reference scenario.
            ref_scenario_lcc = str(lcc_global_active.get("payback_reference_scenario", "") or "")
            ref_lcc_cashflow = pd.DataFrame()
            payback_value = None
            if ref_scenario_lcc and ref_scenario_lcc in scenarios_lcc and ref_scenario_lcc != active_selected:
                ref_payload_lcc = deepcopy(scenarios_lcc.get(ref_scenario_lcc, {}))
                ref_df_lcc_energy = get_energy_balance_df(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    scenario_name=ref_scenario_lcc,
                ).copy()
                ref_valid_enduses_lcc = [str(c) for c in ref_df_lcc_energy.columns if str(c) != "Month"]
                ref_lcc_global_active = _normalize_lcc_global_payload(lcc_global_active, ref_valid_enduses_lcc)
                ref_payload_lcc["lcc"] = _normalize_lcc_payload(ref_payload_lcc.get("lcc", {}), ref_valid_enduses_lcc)
                ref_payload_lcc["lcc_global"] = deepcopy(ref_lcc_global_active)
                ref_lcc_cashflow = compute_lcc_cashflow_table(
                    ref_df_lcc_energy,
                    ref_payload_lcc,
                    ref_valid_enduses_lcc,
                    project_year_lcc,
                    lcc_global=ref_lcc_global_active,
                )
                payback_value = discounted_payback_period(active_lcc_cashflow, ref_lcc_cashflow, project_year_lcc)

            st.write("## LCC Balance")

            annual_by_type = active_lcc_cashflow.groupby(["Year", "Cost Type"], as_index=False).agg(
                Nominal_Cost=("Nominal Cost", "sum"),
                Discounted_Cost=("Discounted Cost", "sum"),
            )
            annual_totals = active_lcc_cashflow.groupby("Year", as_index=False).agg(
                Nominal_Cost=("Nominal Cost", "sum"),
                Discounted_Cost=("Discounted Cost", "sum"),
            )
            annual_totals["Cumulative Nominal Cost"] = annual_totals["Nominal_Cost"].cumsum()
            annual_totals["Cumulative Discounted Cost"] = annual_totals["Discounted_Cost"].cumsum()

            ref_annual_totals = pd.DataFrame()
            if ref_lcc_cashflow is not None and not ref_lcc_cashflow.empty:
                ref_annual_totals = ref_lcc_cashflow.groupby("Year", as_index=False).agg(
                    Nominal_Cost=("Nominal Cost", "sum"),
                    Discounted_Cost=("Discounted Cost", "sum"),
                )
                ref_annual_totals["Cumulative Nominal Cost"] = ref_annual_totals["Nominal_Cost"].cumsum()
                ref_annual_totals["Cumulative Discounted Cost"] = ref_annual_totals["Discounted_Cost"].cumsum()

            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader("Annual LCC Balance")
                fig_lcc_annual = px.bar(
                    annual_by_type,
                    x="Year",
                    y="Nominal_Cost",
                    color="Cost Type",
                    barmode="relative",
                    color_discrete_map=LCC_COST_TYPE_COLORS,
                    height=620,
                    text_auto=".0f",
                    labels={"Nominal_Cost": f"Nominal Cost ({currency_lcc})"},
                )
                line_lcc_total = px.line(
                    annual_totals,
                    x="Year",
                    y="Nominal_Cost",
                    markers=True,
                    labels={"Nominal_Cost": "Total annual cost"},
                )
                for tr in line_lcc_total.data:
                    tr.name = "Total annual cost"
                    tr.line.width = 5
                    tr.line.color = "black"
                    tr.line.dash = "dash"
                    tr.marker.size = 10
                    fig_lcc_annual.add_trace(tr)
                fig_lcc_annual.update_traces(textfont_size=12, textfont_color="white")
                fig_lcc_annual.update_layout(
                    yaxis_title=f"Nominal Cost ({currency_lcc}/a)",
                    xaxis_title="Year",
                    legend_title_text="Cost Type",
                    margin=dict(l=40, r=20, t=50, b=80),
                    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                )
                st_plotly_chart(fig_lcc_annual, use_container_width=True, key="lcc_annual_balance")

            with c2:
                st.subheader("LCC KPI's")
                st.metric("Total Nominal Cost", f"{currency_lcc} {total_nominal_lcc:,.0f}")
                st.metric("Total Discounted Cost", f"{currency_lcc} {total_discounted_lcc:,.0f}")
                if project_area_lcc > 0:
                    st.metric("Nominal Cost per m²", f"{currency_lcc} {cost_per_m2_nominal:,.2f}/m²")
                    st.metric("Discounted Cost per m²", f"{currency_lcc} {cost_per_m2_discounted:,.2f}/m²")
                else:
                    st.metric("Nominal Cost per m²", "n/a")
                    st.metric("Discounted Cost per m²", "n/a")
                st.metric("Discounted Payback Period", _format_payback(payback_value))
                st.metric("Energy Cost", f"{currency_lcc} {type_totals.get('Energy', 0.0):,.0f}")
                st.metric("Investment Cost", f"{currency_lcc} {type_totals.get('Investment', 0.0):,.0f}")
                st.metric("Maintenance Cost", f"{currency_lcc} {type_totals.get('Maintenance', 0.0):,.0f}")
                st.metric("Replacement Cost", f"{currency_lcc} {type_totals.get('Replacement', 0.0):,.0f}")

            c3, c4 = st.columns([3, 1])
            with c3:
                st.subheader("Cumulative LCC")
                fig_lcc_cum = go.Figure()
                fig_lcc_cum.add_trace(go.Scatter(
                    x=annual_totals["Year"],
                    y=annual_totals["Cumulative Nominal Cost"],
                    mode="lines+markers",
                    name="Cumulative nominal cost",
                    line=dict(color=CRREM_COLOR_BASELINE),
                    marker=dict(color=CRREM_COLOR_BASELINE),
                ))
                fig_lcc_cum.add_trace(go.Scatter(
                    x=annual_totals["Year"],
                    y=annual_totals["Cumulative Discounted Cost"],
                    mode="lines+markers",
                    name="Cumulative discounted cost",
                    line=dict(color=CRREM_COLOR_MEASURES),
                    marker=dict(color=CRREM_COLOR_MEASURES),
                ))
                if not ref_annual_totals.empty:
                    fig_lcc_cum.add_trace(go.Scatter(
                        x=ref_annual_totals["Year"],
                        y=ref_annual_totals["Cumulative Nominal Cost"],
                        mode="lines+markers",
                        name=f"{ref_scenario_lcc} cumulative nominal",
                        line=dict(color="#9ca3af", dash="dash"),
                        marker=dict(color="#9ca3af"),
                    ))
                    fig_lcc_cum.add_trace(go.Scatter(
                        x=ref_annual_totals["Year"],
                        y=ref_annual_totals["Cumulative Discounted Cost"],
                        mode="lines+markers",
                        name=f"{ref_scenario_lcc} cumulative discounted",
                        line=dict(color="#6b7280", dash="dot"),
                        marker=dict(color="#6b7280"),
                    ))
                fig_lcc_cum.update_layout(
                    height=620,
                    yaxis_title=f"Cost ({currency_lcc})",
                    xaxis_title="Year",
                    legend_title="",
                    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                    margin=dict(l=40, r=20, t=50, b=80),
                )
                fig_lcc_cum.update_yaxes(rangemode="tozero")
                st_plotly_chart(fig_lcc_cum, use_container_width=True, key="lcc_cumulative")

            with c4:
                st.subheader("LCC Cost Type Share")
                pie_type = by_type_lcc.copy()
                pie_type = pie_type[pie_type["Nominal_Cost"] > 0]
                if not pie_type.empty:
                    fig_type_pie = px.pie(
                        pie_type,
                        names="Cost Type",
                        values="Nominal_Cost",
                        color="Cost Type",
                        color_discrete_map=LCC_COST_TYPE_COLORS,
                        hole=0.5,
                        height=620,
                    )
                    fig_type_pie.update_traces(textinfo="value+percent", textfont_size=16, textfont_color="white")
                    fig_type_pie.update_layout(
                        annotations=[dict(
                            text=f"{currency_lcc} {total_nominal_lcc / project_area_lcc:,.2f}<br>per m²" if project_area_lcc > 0 else f"{currency_lcc} {total_nominal_lcc:,.0f}",
                            x=0.5,
                            y=0.5,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=32, color="black"),
                        )],
                        showlegend=True,
                    )
                    st_plotly_chart(fig_type_pie, use_container_width=True, key="lcc_cost_type_pie")
                else:
                    st.info("No positive LCC costs available for the cost-type pie chart.")

            st.markdown("---")
            st.write("## LCC Breakdown")

            b1, b2 = st.columns(2)
            with b1:
                st.subheader("Total Cost per m² by End Use")
                enduse_breakdown = active_lcc_cashflow.groupby("End_Use", as_index=False).agg(
                    Nominal_Cost=("Nominal Cost", "sum"),
                    Discounted_Cost=("Discounted Cost", "sum"),
                )
                if project_area_lcc > 0:
                    enduse_breakdown["Nominal Cost per m²"] = enduse_breakdown["Nominal_Cost"] / project_area_lcc
                else:
                    enduse_breakdown["Nominal Cost per m²"] = np.nan
                pie_enduse = enduse_breakdown[enduse_breakdown["Nominal Cost per m²"] > 0].copy()
                if not pie_enduse.empty:
                    fig_enduse_pie = px.pie(
                        pie_enduse,
                        names="End_Use",
                        values="Nominal Cost per m²",
                        color="End_Use",
                        color_discrete_map=color_map,
                        hole=0.5,
                        height=800,
                        category_orders={"End_Use": END_USE_ORDER},
                    )
                    fig_enduse_pie.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")
                    fig_enduse_pie.update_layout(showlegend=True)
                    st_plotly_chart(fig_enduse_pie, use_container_width=True, key="lcc_enduse_pie")
                else:
                    st.info("No positive End Use costs available for the pie chart.")

            with b2:
                st.subheader("Operational Energy Cost per m² by Energy Source")
                source_breakdown = active_lcc_cashflow.loc[active_lcc_cashflow["Cost Type"] == "Energy"].groupby("Energy_Source", as_index=False).agg(
                    Nominal_Cost=("Nominal Cost", "sum"),
                    Discounted_Cost=("Discounted Cost", "sum"),
                )
                if project_area_lcc > 0 and not source_breakdown.empty:
                    source_breakdown["Nominal Cost per m²"] = source_breakdown["Nominal_Cost"] / project_area_lcc
                else:
                    source_breakdown["Nominal Cost per m²"] = np.nan
                pie_source = source_breakdown[source_breakdown["Nominal Cost per m²"] > 0].copy()
                if not pie_source.empty:
                    fig_source_pie = px.pie(
                        pie_source,
                        names="Energy_Source",
                        values="Nominal Cost per m²",
                        color="Energy_Source",
                        color_discrete_map=color_map_sources,
                        hole=0.5,
                        height=800,
                        category_orders={"Energy_Source": ENERGY_SOURCE_ORDER},
                    )
                    fig_source_pie.update_traces(textinfo="value+percent", textfont_size=18, textfont_color="white")
                    fig_source_pie.update_layout(showlegend=True)
                    st_plotly_chart(fig_source_pie, use_container_width=True, key="lcc_source_pie")
                else:
                    st.info("No positive operational energy costs available for the energy-source pie chart.")

            with st.expander("LCC cash-flow table", expanded=False):
                cashflow_display = active_lcc_cashflow.copy()
                cashflow_display["End_Use"] = cashflow_display["End_Use"].apply(ui_name)
                st.dataframe(cashflow_display, use_container_width=True)

            st.caption(
                "Discounted Payback Period is calculated against the selected reference scenario from discounted incremental cash flows: "
                "reference scenario cost minus active scenario cost. If no reference scenario is selected, payback is shown as not reached."
            )

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
            # Base data is resolved per scenario so scenario-specific Energy_Balance overrides
            # are reflected in the comparison charts.

            rows = []
            energy_rows = []  # per-source, per-scenario (factored) end energy intensity
            cost_rows = []  # per-source, per-scenario (factored) cost intensity
            emissions_rows = []  # per-source, per-scenario (factored) emissions intensity

            # per-end-use, per-scenario (factored) intensities (used for secondary charts)
            energy_use_rows = []
            cost_use_rows = []
            emissions_use_rows = []

            for name, payload in scenarios.items():
                payload = payload or {}
                eff = (payload.get("efficiency") or {})
                mapping = (payload.get("mapping") or {})
                factors = (payload.get("factors") or {})
                tariffs = (payload.get("tariffs") or {})

                df_s_raw = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name, scenario_name=str(name))
                df_base = df_s_raw.melt(id_vars="Month", var_name="End_Use", value_name="kWh")
                df_s = df_base.copy()
                df_s["Efficiency_Factor"] = df_s["End_Use"].map(lambda u: float(eff.get(u, 1.0))).fillna(1.0)

                # Apply efficiency factors (kWh is divided by factor)
                df_s["kWh_factored"] = df_s["kWh"] / df_s["Efficiency_Factor"]

                # Apply per-scenario On-site Generation scale (on-site generation end use(s))
                # In Scenarios tab net KPIs, PV is always considered as an offset.
                # To model a "no on-site generation" scenario, set On-site Generation scale to 0.0.
                pv_cfg = (payload.get("pv") or {}) if isinstance(payload, dict) else {}
                pv_scale = float(pv_cfg.get("scale", 1.0))
                onsite_enduses = get_onsite_generation_enduses(df_s["End_Use"].unique())
                onsite_set = set(onsite_enduses)
                pv_mask = df_s["End_Use"].isin(onsite_set)
                if pv_mask.any():
                    df_s.loc[pv_mask, "kWh_factored"] = df_s.loc[pv_mask, "kWh_factored"] * pv_scale

                # Enforce sign convention for net calculations:
                # - On-site_Generation is always treated as a negative credit (generation)
                # - All other end uses are treated as consumption only (clip negatives to 0)
                pv_mask = df_s["End_Use"].isin(onsite_set)
                df_s["kWh_signed"] = df_s["kWh_factored"]
                df_s.loc[pv_mask, "kWh_signed"] = -df_s.loc[pv_mask, "kWh_factored"].abs()
                df_s.loc[~pv_mask, "kWh_signed"] = df_s.loc[~pv_mask, "kWh_factored"].clip(lower=0.0)

                df_s["Energy_Source"] = df_s["End_Use"].map(lambda u: str(mapping.get(u, "Electricity")))

                # On-site generation end uses always offset Electricity
                if pv_mask.any():
                    df_s.loc[pv_mask, "Energy_Source"] = "Electricity"

                # Annual energy (net includes PV as a negative contribution)
                totals_use = df_s.groupby("End_Use", as_index=False)["kWh_signed"].sum()
                net_kwh = float(totals_use["kWh_signed"].sum())
                gross_kwh = float(
                    totals_use.loc[
                        (~totals_use["End_Use"].isin(onsite_set)) & (totals_use["kWh_signed"] > 0),
                        "kWh_signed"
                    ].sum()
                )
                pv_kwh = float(abs(totals_use.loc[totals_use["End_Use"].isin(onsite_set), "kWh_signed"].sum()))

                # Net CO2 and net cost (including PV credit as signed kWh)
                df_net = df_s.copy()
                df_net["co2_factor"] = df_net["Energy_Source"].map(lambda s: float(factors.get(s, 0.0))).fillna(0.0)
                df_net["tariff"] = df_net["Energy_Source"].map(lambda s: float(tariffs.get(s, 0.0))).fillna(0.0)

                co2_kg = float((df_net["kWh_signed"] * df_net["co2_factor"]).sum())
                cost_val = float((df_net["kWh_signed"] * df_net["tariff"]).sum())
                # Gross CO2 and gross cost (excluding On-site_Generation)
                df_gross = df_s.loc[~pv_mask].copy()
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


                    # Per-end-use breakdown (net, including PV) for secondary charts (intensities)
                    grp_eu = df_src.groupby("End_Use", as_index=False).agg(
                        kWh=("kWh_signed", "sum"),
                        cost=("cost", "sum"),
                        co2_kg=("co2_kg", "sum"),
                    )
                    for _, r in grp_eu.iterrows():
                        eu = r["End_Use"]
                        energy_use_rows.append({
                            "Scenario": str(name),
                            "End_Use": eu,
                            "End Energy (kWh/m²·a)": float(r["kWh"]) / _area,
                        })
                        cost_use_rows.append({
                            "Scenario": str(name),
                            "End_Use": eu,
                            f"Cost ({_curr}/m²·a)": float(r["cost"]) / _area,
                        })
                        emissions_use_rows.append({
                            "Scenario": str(name),
                            "End_Use": eu,
                            "Emissions (kgCO₂e/m²·a)": float(r["co2_kg"]) / _area,
                        })

                rows.append({
                    "Scenario": str(name),
                    "Net Energy (kWh/a)": net_kwh,
                    "Gross Consumption (kWh/a)": gross_kwh,
                    "On-site Generation (kWh/a)": pv_kwh,
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
                "On-site Generation (kWh/a)",
                "Net CO2 (t/a)",
                f"Net Cost ({_curr}/a)",
                "Net EUI (kWh/m²·a)",
                "Gross EUI (kWh/m²·a)",
            ]].copy()
            _saved_scenario_colors = st.session_state.get("color_map_scenarios", {})
            if not isinstance(_saved_scenario_colors, dict):
                _saved_scenario_colors = {}
            scenario_color_map = {}
            for i, s in enumerate(scenario_order):
                scenario_color_map[str(s)] = str(_saved_scenario_colors.get(str(s), SCENARIO_COLOR_PALETTE[i % len(SCENARIO_COLOR_PALETTE)]))

            with st.expander("Life Cycle Comparission", expanded=True):
                st.caption(
                    "Life-cycle comparison uses the committed global LCC assumptions from the LCC-Analysis tab. "
                    "Scenario-specific Energy_Balance overrides and LCC investment measures are included."
                )

                # Build one cumulative LCC trajectory per scenario. Nominal costs are shown as solid lines,
                # discounted costs as dashed lines, using the same scenario color for both.
                all_enduses_lcc_cmp = []
                try:
                    _base_lcc_df_cmp = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name)
                    all_enduses_lcc_cmp.extend([c for c in _base_lcc_df_cmp.columns if c != "Month"])
                except Exception:
                    pass
                try:
                    for _sc_name in scenario_order:
                        _df_tmp_lcc = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name, scenario_name=str(_sc_name))
                        all_enduses_lcc_cmp.extend([c for c in _df_tmp_lcc.columns if c != "Month"])
                except Exception:
                    pass
                all_enduses_lcc_cmp = list(dict.fromkeys([_canon_enduse_name(str(u)) for u in all_enduses_lcc_cmp if str(u).strip()]))

                if not all_enduses_lcc_cmp:
                    st.info("No Energy_Balance data available for life-cycle comparison.")
                else:
                    project_year_lcc_cmp = int(st.session_state.get("project_year", 2025))
                    lcc_global_cmp = _get_lcc_global_state_payload(all_enduses_lcc_cmp)
                    analysis_period_lcc_cmp = max(1, _to_int_lcc(lcc_global_cmp.get("analysis_period", 30), 30))
                    lcc_years_cmp = list(range(project_year_lcc_cmp, project_year_lcc_cmp + analysis_period_lcc_cmp))

                    fig_lcc_cmp = go.Figure()
                    fig_energy_cost_annual_cmp = go.Figure()
                    lcc_summary_rows = []
                    energy_cost_annual_rows = []

                    for _idx_sc, _sc_name in enumerate(scenario_order):
                        _payload_sc = scenarios.get(_sc_name, {}) or {}
                        _color_sc = scenario_color_map.get(_sc_name, SCENARIO_COLOR_PALETTE[_idx_sc % len(SCENARIO_COLOR_PALETTE)])
                        try:
                            _df_energy_sc = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name, scenario_name=str(_sc_name))
                            _end_uses_sc = [_canon_enduse_name(str(c)) for c in _df_energy_sc.columns if c != "Month"]
                            _cf_sc = compute_lcc_cashflow_table(
                                _df_energy_sc,
                                _payload_sc,
                                _end_uses_sc,
                                project_year_lcc_cmp,
                                lcc_global=lcc_global_cmp,
                            )
                        except Exception:
                            _cf_sc = pd.DataFrame()

                        if _cf_sc is None or _cf_sc.empty:
                            continue

                        _annual_lcc_sc = (
                            _cf_sc.groupby("Year", as_index=True)[["Nominal Cost", "Discounted Cost"]]
                            .sum()
                            .reindex(lcc_years_cmp)
                            .fillna(0.0)
                        )
                        _cum_nominal_sc = _annual_lcc_sc["Nominal Cost"].cumsum()
                        _cum_discounted_sc = _annual_lcc_sc["Discounted Cost"].cumsum()

                        # Annual operational energy cost (nominal) for life-cycle comparison.
                        try:
                            _energy_cf_sc = _cf_sc.loc[_cf_sc["Cost Type"].astype(str) == "Energy"].copy()
                            _annual_energy_cost_sc = (
                                _energy_cf_sc.groupby("Year", as_index=True)["Nominal Cost"]
                                .sum()
                                .reindex(lcc_years_cmp)
                                .fillna(0.0)
                            )
                        except Exception:
                            _annual_energy_cost_sc = pd.Series({int(_y): 0.0 for _y in lcc_years_cmp}, dtype=float)

                        fig_energy_cost_annual_cmp.add_trace(go.Scatter(
                            x=lcc_years_cmp,
                            y=_annual_energy_cost_sc.values,
                            mode="lines+markers",
                            name=str(_sc_name),
                            line=dict(color=_color_sc, width=2.5),
                            marker=dict(color=_color_sc, size=5),
                        ))
                        for _y_ec, _v_ec in _annual_energy_cost_sc.items():
                            energy_cost_annual_rows.append({
                                "Scenario": _sc_name,
                                "Year": int(_y_ec),
                                f"Annual Energy Cost ({_curr}/a)": float(_v_ec),
                            })

                        fig_lcc_cmp.add_trace(go.Scatter(
                            x=lcc_years_cmp,
                            y=_cum_nominal_sc.values,
                            mode="lines+markers",
                            name=f"{_sc_name} — Nominal",
                            line=dict(color=_color_sc, width=2.5, dash="solid"),
                            marker=dict(color=_color_sc, size=5),
                        ))
                        fig_lcc_cmp.add_trace(go.Scatter(
                            x=lcc_years_cmp,
                            y=_cum_discounted_sc.values,
                            mode="lines+markers",
                            name=f"{_sc_name} — Discounted",
                            line=dict(color=_color_sc, width=2.5, dash="dash"),
                            marker=dict(color=_color_sc, size=5),
                        ))
                        lcc_summary_rows.append({
                            "Scenario": _sc_name,
                            f"Cumulative Nominal LCC ({_curr})": float(_cum_nominal_sc.iloc[-1]) if len(_cum_nominal_sc) else 0.0,
                            f"Cumulative Discounted LCC ({_curr})": float(_cum_discounted_sc.iloc[-1]) if len(_cum_discounted_sc) else 0.0,
                        })

                    # CRREM baseline/reference line for the annual and cumulative emissions comparison.
                    # Scenario lines use the same decarbonization multiplier used in the CRREM tab.
                    crrem_baseline_x_cmp = []
                    crrem_baseline_annual_y_cmp = []
                    crrem_baseline_cumulative_y_cmp = []
                    _crrem_cmp = None
                    try:
                        _crrem_cmp = load_crrem_dataset(st.session_state.get("project_country", "Germany"))
                        if _crrem_cmp is not None and _area and float(_area) > 0:
                            _target_label_cmp = str(st.session_state.get("crrem_target_select", "1.5°C"))
                            _target_id_cmp = "1.5C" if _target_label_cmp.startswith("1.5") else "2C"

                            _pt_df_cmp = _crrem_cmp.get("property_types", pd.DataFrame()).copy()
                            _use_options_cmp = _pt_df_cmp["app_use"].dropna().astype(str).tolist() if "app_use" in _pt_df_cmp.columns else []

                            # Prefer the currently active CRREM use-type. Fallback to the active scenario payload, then Office.
                            _crrem_use_cmp = str(st.session_state.get("crrem_use_type", "") or "")
                            if _crrem_use_cmp not in _use_options_cmp:
                                try:
                                    _active_sc_cmp = str(st.session_state.get("active_scenario", ""))
                                    _crrem_use_cmp = str((scenarios.get(_active_sc_cmp, {}) or {}).get("crrem_use_type", _crrem_use_cmp) or _crrem_use_cmp)
                                except Exception:
                                    pass
                            if _crrem_use_cmp not in _use_options_cmp:
                                _crrem_use_cmp = "Office" if "Office" in _use_options_cmp else (_use_options_cmp[0] if _use_options_cmp else "")

                            _pc_cmp = _crrem_cmp.get("pathways_carbon", pd.DataFrame()).copy()
                            if (not _pc_cmp.empty) and {"target", "year", "property_type_code", "kgco2e_per_m2_yr"}.issubset(_pc_cmp.columns):
                                _pc_t_cmp = _pc_cmp.loc[_pc_cmp["target"].astype(str) == _target_id_cmp]
                                _carbon_pivot_cmp = _pc_t_cmp.pivot_table(
                                    index="year",
                                    columns="property_type_code",
                                    values="kgco2e_per_m2_yr",
                                )
                                _years_crrem_cmp = [int(y) for y in lcc_years_cmp if int(y) in _carbon_pivot_cmp.index]

                                _carbon_limit_cmp = pd.Series(dtype=float)
                                if _years_crrem_cmp and _crrem_use_cmp and _crrem_use_cmp != "Mixed Use":
                                    _code_row_cmp = _pt_df_cmp.loc[_pt_df_cmp["app_use"].astype(str) == str(_crrem_use_cmp)]
                                    if not _code_row_cmp.empty:
                                        _p_code_cmp = str(_code_row_cmp.iloc[0]["crrem_code"])
                                        if _p_code_cmp in _carbon_pivot_cmp.columns:
                                            _carbon_limit_cmp = _carbon_pivot_cmp[_p_code_cmp].reindex(_years_crrem_cmp).astype(float)

                                elif _years_crrem_cmp and _crrem_use_cmp == "Mixed Use":
                                    # Reuse the current mixed-use definition where available.
                                    _mixed_df_cmp = st.session_state.get("crrem_mixed_use_df")
                                    if not isinstance(_mixed_df_cmp, pd.DataFrame) or _mixed_df_cmp.empty:
                                        try:
                                            _active_sc_cmp = str(st.session_state.get("active_scenario", ""))
                                            _mixed_df_cmp = _mixed_use_records_to_df((scenarios.get(_active_sc_cmp, {}) or {}).get("crrem_mixed_use", []))
                                        except Exception:
                                            _mixed_df_cmp = pd.DataFrame()

                                    _components_cmp = []
                                    try:
                                        for _, _r_cmp in _mixed_df_cmp.iterrows():
                                            _u_cmp = str(_r_cmp.get("Use Type", "")).strip()
                                            _share_cmp = float(_r_cmp.get("Area Share %", 0.0) or 0.0)
                                            if _u_cmp and _share_cmp > 0:
                                                _components_cmp.append((_u_cmp, _share_cmp))
                                    except Exception:
                                        _components_cmp = []

                                    if _components_cmp:
                                        _tot_share_cmp = sum(_w_cmp for _, _w_cmp in _components_cmp)
                                        _use_to_code_cmp = dict(zip(_pt_df_cmp["app_use"].astype(str), _pt_df_cmp["crrem_code"].astype(str)))
                                        _carbon_limit_cmp = pd.Series(0.0, index=_years_crrem_cmp, dtype=float)
                                        for _u_cmp, _w_cmp in _components_cmp:
                                            _code_cmp = _use_to_code_cmp.get(str(_u_cmp))
                                            if _code_cmp and _code_cmp in _carbon_pivot_cmp.columns and _tot_share_cmp > 0:
                                                _carbon_limit_cmp = _carbon_limit_cmp + (float(_w_cmp) / float(_tot_share_cmp)) * _carbon_pivot_cmp[_code_cmp].reindex(_years_crrem_cmp).astype(float)

                                if _carbon_limit_cmp is not None and not _carbon_limit_cmp.empty:
                                    _annual_crrem_limit_t_cmp = (_carbon_limit_cmp.astype(float) * float(_area)) / 1000.0
                                    crrem_baseline_x_cmp = list(_annual_crrem_limit_t_cmp.index.astype(int))
                                    crrem_baseline_annual_y_cmp = _annual_crrem_limit_t_cmp.values.tolist()
                                    crrem_baseline_cumulative_y_cmp = _annual_crrem_limit_t_cmp.cumsum().values.tolist()
                    except Exception:
                        crrem_baseline_x_cmp = []
                        crrem_baseline_annual_y_cmp = []
                        crrem_baseline_cumulative_y_cmp = []

                    fig_emis_annual_cmp = go.Figure()
                    fig_emis_cum_cmp = go.Figure()
                    if crrem_baseline_x_cmp and crrem_baseline_annual_y_cmp:
                        fig_emis_annual_cmp.add_trace(go.Scatter(
                            x=crrem_baseline_x_cmp,
                            y=crrem_baseline_annual_y_cmp,
                            mode="lines+markers",
                            name="CRREM-Baseline",
                            line=dict(color=CRREM_COLOR_LIMIT, width=3, dash="dash"),
                            marker=dict(color=CRREM_COLOR_LIMIT, size=6),
                        ))
                    if crrem_baseline_x_cmp and crrem_baseline_cumulative_y_cmp:
                        fig_emis_cum_cmp.add_trace(go.Scatter(
                            x=crrem_baseline_x_cmp,
                            y=crrem_baseline_cumulative_y_cmp,
                            mode="lines+markers",
                            name="CRREM-Baseline",
                            line=dict(color=CRREM_COLOR_LIMIT, width=3, dash="dash"),
                            marker=dict(color=CRREM_COLOR_LIMIT, size=6),
                        ))

                    emissions_summary_rows = []
                    emissions_annual_rows = []
                    for _idx_sc, _sc_name in enumerate(scenario_order):
                        _color_sc = scenario_color_map.get(_sc_name, SCENARIO_COLOR_PALETTE[_idx_sc % len(SCENARIO_COLOR_PALETTE)])
                        _payload_sc = scenarios.get(_sc_name, {}) or {}
                        try:
                            _df_energy_em_sc = get_energy_balance_df(
                                uploaded_file.getvalue(),
                                uploaded_file.name,
                                scenario_name=str(_sc_name),
                            )
                            _annual_emissions_series_t = compute_crrem_like_scenario_emissions_series(
                                _df_energy_em_sc,
                                _payload_sc,
                                _crrem_cmp,
                                project_year_lcc_cmp,
                                lcc_years_cmp,
                            ).reindex(lcc_years_cmp).fillna(0.0)
                        except Exception:
                            _annual_emissions_series_t = pd.Series({int(y): 0.0 for y in lcc_years_cmp}, dtype=float)

                        _cum_emissions_series_t = _annual_emissions_series_t.cumsum()

                        fig_emis_annual_cmp.add_trace(go.Scatter(
                            x=lcc_years_cmp,
                            y=_annual_emissions_series_t.values,
                            mode="lines+markers",
                            name=str(_sc_name),
                            line=dict(color=_color_sc, width=2.5),
                            marker=dict(color=_color_sc, size=5),
                        ))
                        fig_emis_cum_cmp.add_trace(go.Scatter(
                            x=lcc_years_cmp,
                            y=_cum_emissions_series_t.values,
                            mode="lines+markers",
                            name=str(_sc_name),
                            line=dict(color=_color_sc, width=2.5),
                            marker=dict(color=_color_sc, size=5),
                        ))
                        emissions_summary_rows.append({
                            "Scenario": _sc_name,
                            "Annual Net CO₂ first year (t/a)": float(_annual_emissions_series_t.iloc[0]) if len(_annual_emissions_series_t) else 0.0,
                            "Annual Net CO₂ final year (t/a)": float(_annual_emissions_series_t.iloc[-1]) if len(_annual_emissions_series_t) else 0.0,
                            "Cumulative Net CO₂ (t)": float(_cum_emissions_series_t.iloc[-1]) if len(_cum_emissions_series_t) else 0.0,
                        })
                        for _y_cmp, _v_cmp in _annual_emissions_series_t.items():
                            emissions_annual_rows.append({
                                "Scenario": _sc_name,
                                "Year": int(_y_cmp),
                                "Annual Net CO₂ (t/a)": float(_v_cmp),
                                "Cumulative Net CO₂ (t)": float(_cum_emissions_series_t.loc[_y_cmp]),
                            })

                    # First row: LCC diagrams (annual first, cumulative second)
                    lc1, lc2 = st.columns(2)
                    with lc1:
                        st.subheader("Annual Energy Cost")
                        if fig_energy_cost_annual_cmp.data:
                            fig_energy_cost_annual_cmp.update_layout(
                                height=600,
                                xaxis_title="Year",
                                yaxis_title=f"Annual energy cost ({_curr}/a)",
                                legend_title_text="Scenario",
                                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                                margin=dict(l=40, r=20, t=50, b=120),
                            )
                            fig_energy_cost_annual_cmp.update_yaxes(rangemode="tozero")
                            st_plotly_chart(fig_energy_cost_annual_cmp, use_container_width=True, key="scenario_energy_cost_annual_all")
                        else:
                            st.info("No annual energy cost data available for comparison.")

                    with lc2:
                        st.subheader("Cumulative LCC")
                        if fig_lcc_cmp.data:
                            fig_lcc_cmp.update_layout(
                                height=600,
                                xaxis_title="Year",
                                yaxis_title=f"Cumulative cost ({_curr})",
                                legend_title_text="Scenario / cost basis",
                                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                                margin=dict(l=40, r=20, t=50, b=120),
                            )
                            fig_lcc_cmp.update_yaxes(rangemode="tozero")
                            st_plotly_chart(fig_lcc_cmp, use_container_width=True, key="scenario_lcc_cumulative_all")
                        else:
                            st.info("No LCC cash flows available. Add LCC inputs in the LCC-Analysis tab.")

                    # Second row: emission diagrams
                    em1, em2 = st.columns(2)
                    with em1:
                        st.subheader("Annual Emissions")
                        if fig_emis_annual_cmp.data:
                            fig_emis_annual_cmp.update_layout(
                                height=600,
                                xaxis_title="Year",
                                yaxis_title="Annual net emissions (tCO₂e/a)",
                                legend_title_text="Scenario",
                                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                                margin=dict(l=40, r=20, t=50, b=120),
                            )
                            fig_emis_annual_cmp.update_yaxes(rangemode="tozero")
                            st_plotly_chart(fig_emis_annual_cmp, use_container_width=True, key="scenario_emissions_annual_all")
                        else:
                            st.info("No emissions data available for annual comparison.")

                    with em2:
                        st.subheader("Cumulative Emissions")
                        if fig_emis_cum_cmp.data:
                            fig_emis_cum_cmp.update_layout(
                                height=600,
                                xaxis_title="Year",
                                yaxis_title="Cumulative net emissions (tCO₂e)",
                                legend_title_text="Scenario",
                                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
                                margin=dict(l=40, r=20, t=50, b=120),
                            )
                            fig_emis_cum_cmp.update_yaxes(rangemode="tozero")
                            st_plotly_chart(fig_emis_cum_cmp, use_container_width=True, key="scenario_emissions_cumulative_all")
                        else:
                            st.info("No emissions data available for cumulative comparison.")

                    show_life_cycle_data = st.checkbox(
                        "Show Life Cycle comparison data tables",
                        value=False,
                        key="scenario_life_cycle_data_show",
                    )
                    if show_life_cycle_data:
                        if lcc_summary_rows:
                            st.write("#### Cumulative LCC summary")
                            st.dataframe(pd.DataFrame(lcc_summary_rows), use_container_width=True)
                        if energy_cost_annual_rows:
                            st.write("#### Annual energy cost by scenario and year")
                            st.dataframe(pd.DataFrame(energy_cost_annual_rows), use_container_width=True)
                        if emissions_summary_rows:
                            st.write("#### Cumulative emissions summary")
                            st.dataframe(pd.DataFrame(emissions_summary_rows), use_container_width=True)
                        if emissions_annual_rows:
                            st.write("#### Annual emissions by scenario and year")
                            st.dataframe(pd.DataFrame(emissions_annual_rows), use_container_width=True)


            with st.expander("Static Comparission", expanded=True):
                show_static_raw_data = st.checkbox(
                    "Show static comparison raw data table",
                    value=False,
                    key="scenario_static_raw_data_show",
                )
                if show_static_raw_data:
                    st.dataframe(df_cmp_display, use_container_width=True)

                # Net KPI charts (incl. On-site_Generation) — values printed on bars
                if _area and _area > 0:
                    df_kpi = df_cmp.copy()
                    df_kpi["Scenario"] = df_kpi["Scenario"].astype(str)
                    df_kpi["Net Emissions (kgCO₂e/m²·a)"] = (df_kpi["Net CO2 (t/a)"] * 1000.0) / _area

                    net_cost_col_a = f"Net Cost ({_curr}/a)"
                    net_cost_col_m2 = f"Net Cost ({_curr}/m²·a)"
                    if net_cost_col_a in df_kpi.columns:
                        df_kpi[net_cost_col_m2] = df_kpi[net_cost_col_a] / _area

                    st.markdown("### Net KPI comparison (incl. On-site Generation)")

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
                        st_plotly_chart(fig_net_eui, use_container_width=True, key="scenario_net_eui")

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
                        st_plotly_chart(fig_net_emis, use_container_width=True, key="scenario_net_emissions")

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
                            st_plotly_chart(fig_net_cost, use_container_width=True, key="scenario_net_cost")

                    # Gross KPI charts (excl. On-site_Generation)
                    df_kpi["Gross Emissions (kgCO₂e/m²·a)"] = (df_kpi["Gross CO2 (t/a)"] * 1000.0) / _area

                    gross_cost_col_a = f"Gross Cost ({_curr}/a)"
                    gross_cost_col_m2 = f"Gross Cost ({_curr}/m²·a)"
                    if gross_cost_col_a in df_kpi.columns:
                        df_kpi[gross_cost_col_m2] = df_kpi[gross_cost_col_a] / _area

                    st.markdown("### Gross KPI comparison (excl. On-site Generation)")

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
                        st_plotly_chart(fig_gross_eui, use_container_width=True, key="scenario_gross_eui")

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
                        st_plotly_chart(fig_gross_emis, use_container_width=True, key="scenario_gross_emissions")

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
                            st_plotly_chart(fig_gross_cost, use_container_width=True, key="scenario_gross_cost")
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
                            title="End Energy /m² by Energy Source and Scenario (Net)",
                            category_orders={"Scenario": scenario_order},
                            color_discrete_map=color_map_sources,
                            text_auto=".1f",
                            height=600,
                        )
                        fig_end_energy.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="kWh/m²·a",
                            legend_title_text="Energy Source",
                        )
                        fig_end_energy.update_traces(textfont_size=14, textfont_color="white")
                        fig_end_energy.update_xaxes(type="category")
                        st_plotly_chart(fig_end_energy, use_container_width=True, key="scenario_end_energy_m2_by_source")

                    # 2) Energy Emissions /m² (factored) by energy source
                    df_emis_src = pd.DataFrame(emissions_rows)
                    if not df_emis_src.empty:
                        df_emis_src["Scenario"] = df_emis_src["Scenario"].astype(str)
                        fig_emis = px.bar(
                            df_emis_src,
                            x="Scenario",
                            y="Emissions (kgCO₂e/m²·a)",
                            color="Energy_Source",
                            barmode="relative",
                            title="Energy Emissions /m² by Energy Source and Scenario (Net)",
                            category_orders={"Scenario": scenario_order},
                            color_discrete_map=color_map_sources,
                            text_auto=".1f",
                            height=600,
                        )
                        fig_emis.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="kgCO₂e/m²·a",
                            legend_title_text="Energy Source",
                        )
                        fig_emis.update_traces(textfont_size=14, textfont_color="white")
                        fig_emis.update_xaxes(type="category")
                        st_plotly_chart(fig_emis, use_container_width=True, key="scenario_emissions_m2_by_source")

                    # 3) Energy Cost /m² (factored) by energy source
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
                            title=f"Energy Cost /m² by Energy Source and Scenario [{_curr}] (Net)",
                            category_orders={"Scenario": scenario_order},
                            color_discrete_map=color_map_sources,
                            text_auto=".1f",
                            height=600,
                        )
                        fig_cost.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title=f"{_curr}/m²·a",
                            legend_title_text="Energy Source",
                        )
                        fig_cost.update_traces(textfont_size=14, textfont_color="white")
                        fig_cost.update_xaxes(type="category")
                        st_plotly_chart(fig_cost, use_container_width=True, key="scenario_cost_m2_by_source")


                    # Scenario comparison charts (factored values, stacked by End Use)
                    df_energy_eu = pd.DataFrame(energy_use_rows)
                    if not df_energy_eu.empty:
                        df_energy_eu["Scenario"] = df_energy_eu["Scenario"].astype(str)
                        fig_end_energy_eu = px.bar(
                            df_energy_eu,
                            x="Scenario",
                            y="End Energy (kWh/m²·a)",
                            color="End_Use",
                            barmode="relative",
                            title="End Energy /m² by End Use and Scenario (Gross)",
                            category_orders={"Scenario": scenario_order, "End_Use": END_USE_ORDER},
                            color_discrete_map=color_map,
                            text_auto=".1f",
                            height=600,
                        )
                        fig_end_energy_eu.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="kWh/m²·a",
                            legend_title_text="End Use",
                        )
                        fig_end_energy_eu.update_traces(textfont_size=14, textfont_color="white")
                        fig_end_energy_eu.update_xaxes(type="category")
                        st_plotly_chart(fig_end_energy_eu, use_container_width=True, key="scenario_end_energy_m2_by_enduse")



                    df_emis_eu = pd.DataFrame(emissions_use_rows)
                    if not df_emis_eu.empty:
                        df_emis_eu["Scenario"] = df_emis_eu["Scenario"].astype(str)
                        fig_emis_eu = px.bar(
                            df_emis_eu,
                            x="Scenario",
                            y="Emissions (kgCO₂e/m²·a)",
                            color="End_Use",
                            barmode="relative",
                            title="Energy Emissions /m² by End Use and Scenario (Gross)",
                            category_orders={"Scenario": scenario_order, "End_Use": END_USE_ORDER},
                            color_discrete_map=color_map,
                            text_auto=".1f",
                            height=600,
                        )
                        fig_emis_eu.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title="kgCO₂e/m²·a",
                            legend_title_text="End Use",
                        )
                        fig_emis_eu.update_traces(textfont_size=14, textfont_color="white")
                        fig_emis_eu.update_xaxes(type="category")
                        st_plotly_chart(fig_emis_eu, use_container_width=True, key="scenario_emissions_m2_by_enduse")

                    df_cost_eu = pd.DataFrame(cost_use_rows)
                    if not df_cost_eu.empty and cost_col in df_cost_eu.columns:
                        df_cost_eu["Scenario"] = df_cost_eu["Scenario"].astype(str)
                        fig_cost_eu = px.bar(
                            df_cost_eu,
                            x="Scenario",
                            y=cost_col,
                            color="End_Use",
                            barmode="relative",
                            title=f"Energy Cost /m² by End Use and Scenario [{_curr}] (Gross)",
                            category_orders={"Scenario": scenario_order, "End_Use": END_USE_ORDER},
                            color_discrete_map=color_map,
                            text_auto=".1f",
                            height=600,
                        )
                        fig_cost_eu.update_layout(
                            xaxis_title="Scenario",
                            yaxis_title=f"{_curr}/m²·a",
                            legend_title_text="End Use",
                        )
                        fig_cost_eu.update_traces(textfont_size=14, textfont_color="white")
                        fig_cost_eu.update_xaxes(type="category")
                        st_plotly_chart(fig_cost_eu, use_container_width=True, key="scenario_cost_m2_by_enduse")




    if not uploaded_file:
        st.write("### ← Please upload data on sidebar")

# =========================
# Tab 1b — Energy Balance with Factors (Energy Balance with Factors Tab)
# =========================
with tab1_factors:
    if uploaded_file:
        # ---- Load data
        df_eff = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name)

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
        st.metric("Active Scenario", active_selected)

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

        totals_eff_clean = totals_eff[(totals_eff["End_Use"] != "On-site_Generation")]

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
                text=f"{net_eui_eff:,.1f}<br>kWh/m²·a",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=50, color="black"),
            )],
            showlegend=True,
        )
        energy_intensity_chart_per_source_eff.update_traces(textinfo="value+percent", textfont_size=18,
                                                            textfont_color="white")

        # ---- On-site Generation coverage (share of on-site generation vs consumption-only EUI)
        totals_indexed_eff = totals_eff.set_index("End_Use")
        pv_value_eff = totals_indexed_eff.loc[
            "On-site_Generation", "kWh_per_m2"] if "On-site_Generation" in totals_indexed_eff.index else 0.0
        pv_coverage_eff = abs((pv_value_eff / eui_eff) * 100) if eui_eff != 0 else 0.0

        # ---- Layout: charts and KPIs
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy")
            st_plotly_chart(monthly_chart_eff, use_container_width=True, key="ebf_monthly_enduse")
        with col2:
            st.subheader("Annual Energy")
            st_plotly_chart(annual_chart_eff, use_container_width=True, key="ebf_annual_enduse")

        # KPI calculations (kept identical logic)
        monthly_avr_eff = (totals_eff["kWh"].sum()) / 12
        net_total_eff = totals_eff["kWh"].sum()
        total_energy_eff = totals_eff.loc[totals_eff["kWh"] > 0, "kWh"].sum()
        pv_total_eff = abs(df_melted_eff.groupby("End_Use")["kWh"].sum().get("On-site_Generation", 0.0))

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st_plotly_chart(energy_intensity_chart_eff, use_container_width=True, key="ebf_eui_enduse")
        with col2:
            st.subheader("Energy KPI's")
            st.metric(label="Monthly Average Energy Consumption", value=f"{monthly_avr_eff:,.0f} kWh")
            st.metric(label="Total Annual Energy Consumption", value=f"{total_energy_eff:,.0f} kWh")
            st.metric(label="Net Annual Energy Consumption", value=f"{net_total_eff:,.0f} kWh")
            st.metric(label="EUI", value=f"{eui_eff:,.1f} kWh/m2.a")
            st.metric(label="Net EUI", value=f"{net_eui_eff:,.1f} kWh/m2.a")
            st.metric(label="On-site Generation Production", value=f"{pv_total_eff:,.1f} kWh")
            st.metric(label="On-site Generation Coverage", value=f"{pv_coverage_eff:,.1f} %")

        st.markdown("---")
        st.write("## Energy Balance with Factors (per Energy Source)")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Monthly Energy Demand")
            st_plotly_chart(monthly_chart_source_eff, use_container_width=True, key="ebf_monthly_source")
        with col2:
            st.subheader("Annual Energy Demand")
            st_plotly_chart(annual_chart_per_source_eff, use_container_width=True, key="ebf_annual_source")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Energy Use Intensity (kWh/m2.a)")
            st_plotly_chart(energy_intensity_chart_per_source_eff, use_container_width=True, key="ebf_eui_source")
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
        df = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name)
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
            (totals_co2_use["End_Use"] != "On-site_Generation")]

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

        co2_intensity_gross = totals_co2_use.loc[totals_co2_use["kgCO2_per_m2"] > 0, "kgCO2_per_m2"].sum()
        # Center annotations (show total intensity in donut centers)
        co2_intensity_pie_use.update_layout(
            annotations=[dict(
                text=f"{co2_intensity_gross:,.1f}<br>kgCO₂/m²·a",
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
        st.metric("Active Scenario", active_selected)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Monthly CO₂")
            st_plotly_chart(monthly_chart_co2_use, use_container_width=True)
        with c2:
            st.subheader("Annual CO₂")
            st_plotly_chart(annual_chart_co2_use, use_container_width=True)

        c3, c4 = st.columns([3, 1])
        with c3:
            st.subheader("CO₂ Intensity (kgCO₂/m²·a)")
            st_plotly_chart(co2_intensity_pie_use, use_container_width=True)
        with c4:
            st.subheader("CO₂ KPI's")
            st.metric("Monthly Average CO₂", f"{monthly_avg_co2:,.0f} kgCO₂")
            st.metric("Total Annual CO₂", f"{annual_total_co2:,.0f} kgCO₂")
            st.metric("CO₂ Intensity (Net)", f"{co2_intensity_total:,.1f} kgCO₂/m²·a")
            st.metric("CO₂ Intensity (Gross)", f"{co2_intensity_gross:,.1f} kgCO₂/m²·a")

        st.markdown("---")
        st.write("## CO₂ Emissions (per Energy Source)")
        c5, c6 = st.columns([3, 1])
        with c5:
            st.subheader("Monthly CO₂")
            st_plotly_chart(monthly_chart_co2_source, use_container_width=True)
        with c6:
            st.subheader("Annual CO₂")
            st_plotly_chart(annual_chart_co2_source, use_container_width=True)

        c7, c8 = st.columns([3, 1])
        with c7:
            st.subheader("CO₂ Intensity (kgCO₂/m²·a)")
            st_plotly_chart(co2_intensity_pie_source, use_container_width=True)
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
        df_cost_base = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name).copy()
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
                            "On-site_Generation"]},
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
            (totals_cost_use["End_Use"] != "On-site_Generation")]

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
        cost_intensity_gross = totals_cost_use.loc[totals_cost_use["cost_per_m2"] > 0, "cost_per_m2"].sum()
        cost_intensity_pie_use.update_layout(
            showlegend=True,
            annotations=[dict(
                text=f"{currency_symbol} {cost_intensity_gross:,.2f}<br>per m²·a",
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
        st.metric("Active Scenario", active_selected)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Monthly Cost")
            st_plotly_chart(monthly_chart_cost_use, use_container_width=True)
        with c2:
            st.subheader("Annual Cost")
            st_plotly_chart(annual_chart_cost_use, use_container_width=True)

        c3, c4 = st.columns([3, 1])
        with c3:
            st.subheader(f"Cost Intensity ( {currency_symbol}/m²·a )")
            st_plotly_chart(cost_intensity_pie_use, use_container_width=True)
        with c4:
            st.subheader("Cost KPI's")
            st.metric("Monthly Average Cost", f"{currency_symbol} {monthly_avg_cost:,.0f}")
            st.metric("Total Annual Cost", f"{currency_symbol} {annual_total_cost:,.0f}")
            st.metric("Cost Intensity (Net)", f"{currency_symbol} {cost_intensity_total:,.2f} /m²·a")
            st.metric("Cost Intensity (Gross)", f"{currency_symbol} {cost_intensity_gross:,.2f} /m²·a")

        st.markdown("---")
        st.write(f"## Energy Cost {currency_symbol} (per Energy Source)")
        c5, c6 = st.columns([3, 1])
        with c5:
            st.subheader("Monthly Cost")
            st_plotly_chart(monthly_chart_cost_source, use_container_width=True)
        with c6:
            st.subheader("Annual Cost")
            st_plotly_chart(annual_chart_cost_source, use_container_width=True)

        c7, c8 = st.columns([3, 1])
        with c7:
            st.subheader(f"Cost Intensity ( {currency_symbol}/m²·a )")
            st_plotly_chart(cost_intensity_pie_source, use_container_width=True)
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
        df_loads = get_loads_balance_df(uploaded_file.getvalue(), uploaded_file.name)

        # columns that are load metrics
        load_cols = [c for c in df_loads.columns if c not in ["hoy", "doy", "day", "month", "weekday", "hour"]]

        # (optional) ensure doy/hour are numeric
        df_loads["doy"] = pd.to_numeric(df_loads["doy"], errors="coerce")
        df_loads["hour"] = pd.to_numeric(df_loads["hour"], errors="coerce")

        st.write("## Load Analysis")
        st.metric("Active Scenario", active_selected)
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
        bar_color = color_map_loads.get(key, color_map.get(key, "#c02419"))  # fallback color

        monthly_total_load_bar.update_traces(textfont_size=14, textfont_color="white")

        monthly_total_load_bar.update_traces(marker_color=bar_color, name=selected_load, showlegend=True)
        monthly_total_load_bar.update_layout(showlegend=True, legend=dict(title=""))

        col1, col2 = st.columns([3, 1])

        with col1:

            st.subheader(f"Monthly Load Sum — {selected_load} (kWh)")
            st_plotly_chart(monthly_total_load_bar, use_container_width=True)

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
        st_plotly_chart(load_heatmap, use_container_width=True)

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
        st_plotly_chart(exceed_heatmap, use_container_width=True)

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
        st_plotly_chart(peak_day_fig, use_container_width=True)
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
        st_plotly_chart(ldc_fig, use_container_width=True)

        # -------------------------
        # On-site Generation Self-Consumption (hourly) — uses On-site_Generation from Loads_Balance
        # -------------------------
        st.subheader("On-site Generation Self-Consumption (hourly) — On-site Generation")
        pv_col = "On-site_Generation" if "On-site_Generation" in df_loads.columns else None
        if pv_col is None:
            st.warning(
                "No hourly On-site Generation column 'On-site_Generation' found in Loads_Balance. Add it to enable on-site generation self-consumption.")
        else:
            pv_enabled = st.checkbox(
                "Enable on-site generation self-consumption using On-site Generation",
                value=bool(st.session_state.get("pv_sc_enabled", False)),
                key="pv_sc_enabled",
            )
            pv_scale = numeric_input(
                "On-site Generation scale factor (dimensionless)",
                float(st.session_state.get("pv_scale", 1.0)),
                key="pv_scale",
                min_value=0.0,
                max_value=1000.0,
                fmt="{:.3f}",
                help="Scales the On-site Generation profile (e.g., 0.5 = half size, 2.0 = double size)."
            )

            # Persist On-site generation settings into the active scenario (without touching other scenario fields)
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
                k1.metric("On-site Generation", f"{pv_total:,.0f} kWh")
                k2.metric("Self-consumed On-site Generation", f"{self_total:,.0f} kWh")
                k3.metric("On-site Generation export", f"{export_total:,.0f} kWh")
                k4.metric("Grid import after On-site Generation", f"{import_total:,.0f} kWh")
                k5.metric("Self-consumption ratio", f"{sc_ratio * 100:,.1f} %")
                k6.metric("On-site Generation coverage of load", f"{coverage_ratio * 100:,.1f} %")

                # Peak-day overlay: load vs on-site generation vs net import
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
                                            name=ONSITE_GENERATION_LABEL,
                                            line=dict(color=color_map.get("On-site_Generation", "#a9c724"), width=5,
                                                      dash="dash")))
                fig_pv.add_trace(go.Scatter(x=day_profile["hour"], y=net_day, mode="lines",
                                            name="Net import", line=dict(color="black", width=5)))
                fig_pv.update_layout(
                    title=f"On-site Generation Matching on Peak Day — {selected_load} | {date_label}",
                    xaxis_title="Hour of Day",
                    yaxis_title="kW",
                    xaxis=dict(dtick=1),
                    height=700,
                )
                st_plotly_chart(fig_pv, use_container_width=True,
                                key=f"pv_match_peak_{st.session_state.get('active_scenario', '')}_{selected_load}")

                # Annual split: load covered by on-site generation vs grid import
                split_df = pd.DataFrame({
                    "Component": ["Covered by On-site Generation (self-consumed)", "Grid import"],
                    "kWh": [self_total, import_total]
                })
                split_fig = px.pie(split_df, names="Component", values="kWh",
                                   title="Annual Electricity Supply Split (selected load)")
                st_plotly_chart(split_fig, use_container_width=True,
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
            df_energy = get_energy_balance_df(uploaded_file.getvalue(), uploaded_file.name)
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
            st.metric("Active Scenario", active_selected)
            a1, a2 = st.columns([3, 1])
            with a1:
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.metric("Building Use", building_use, help="User input (sidebar)")
                with b2:
                    st.metric("Building Area", f"{project_area:,.0f} m²", help="User input (sidebar)")
                with b3:
                    st.metric("On-site generation share", f"{pv_coverage * 100:.0f} %",
                              help="Derived from negative energy balance entries (e.g., on-site generation)")
                with b3:
                    st.metric("EUI (Net)", f"{eui_net:.1f} kWh/m²·a")
                with b2:
                    st.metric("Energy Cost Intensity (Net)", f"{cost_intensity_net:.1f} €/m²·a")
                with b1:
                    st.metric("CO₂ Intensity (Net)", f"{co2_intensity_net:.1f} kgCO₂/m²·a")
                with b1:
                    st.metric("CO₂ Intensity (Gross)", f"{co2_intensity_gross:.1f} kgCO₂/m²·a")
                with b2:
                    st.metric("Energy Cost Intensity (Gross)", f"{cost_intensity_gross:.1f} €/m²·a")
                with b3:
                    st.metric("EUI (Gross)", f"{eui_gross:.1f} kWh/m²·a")


            with a2:
                try:
                    latitude_map = float(latitude)
                    longitude_map = float(longitude)
                    df_map = pd.DataFrame({"lat": [latitude_map], "lon": [longitude_map]})
                    st.metric("Project Location", "", help="User input (sidebar)")
                    st.map(data=df_map, latitude="lat", longitude="lon", height=220, zoom=9)
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
                    fig.add_vline(x=excellent_thr, line_width=2, line_dash="dot",
                                  line_color=get_benchmark_color("Excellent"))
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
                    marker=dict(size=40, symbol="square-open", color=MARKER_GROSS_COLOR,
                                line=dict(width=2, color=MARKER_GROSS_COLOR)),
                    name="Gross",
                    hovertemplate=f"Gross: %{{x:.2f}} {unit}<extra></extra>",
                ))

                fig.add_trace(go.Scatter(
                    x=[value_net], y=[0.7],
                    mode="markers",
                    marker=dict(size=40, symbol="square", color=MARKER_NET_COLOR,
                                line=dict(width=2, color=MARKER_NET_COLOR)),
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
                    st_plotly_chart(fig_band, use_container_width=True, key=f"bm_band_{tkey}")

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
            with st.expander(label="Validation Diagrams (under development)", expanded=False):
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
                    st_plotly_chart(fig_water, use_container_width=True, key="bm_waterfall_eui")
                with a2:
                    st_plotly_chart(fig_end_use, use_container_width=True, key="bm_enduse_energy")

                # Source split (energy & CO2) — handle negative entries explicitly as on-site generation
                df_src = df_melted.copy()
                df_src["Energy_Source_BM"] = df_src.apply(
                    lambda r: "On-site generation" if r["kWh"] < 0 else r["Energy_Source"],
                    axis=1,
                )

                _src_labels = list(pd.unique(df_src["Energy_Source_BM"]))
                _src_cmap = {s: color_map_sources.get(s, color_map.get(s, "#999999")) for s in _src_labels}
                if "On-site generation" in _src_cmap:
                    _src_cmap["On-site generation"] = color_map.get("On-site_Generation", CRREM_COLOR_MEASURES)

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
                    st_plotly_chart(fig_src_energy, use_container_width=True, key="bm_source_energy")
                with b2:
                    st_plotly_chart(fig_src_co2, use_container_width=True, key="bm_source_co2")

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
                    st_plotly_chart(fig_src_cost, use_container_width=True, key="bm_source_cost")

                    # Optional: show raw numbers for transparency
                    st.dataframe(
                        src_cost.rename(columns={"Energy_Source_BM": "Energy Source",
                                                 "cost_per_m2": f"Cost intensity ({_curr}/m²·a)"}),
                        use_container_width=True,
                        hide_index=True,
                    )

    if not uploaded_file:
        st.write("Please upload the project Excel file to see benchmark results.")


# =========================
# Tab 8 — Raw Data (editable Energy_Balance + Loads_Balance)
# =========================
with tab8:
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        wb_hash = hashlib.md5(file_bytes).hexdigest()[:10]

        st.write("## Raw Data")
        st.caption(
            "Edit raw sheets using the editors below. Energy_Balance can be updated globally or only for the active scenario. "
            "Energy_Balance table edits are buffered in the editor and are committed only when **Update Data** is clicked. "
            "Committed scenario-specific overrides survive scenario switching and are exported with **Save Project**."
        )

        # Ensure drafts exist for this workbook
        if _RAW_ENERGY_DRAFT_KEY not in st.session_state:
            st.session_state[_RAW_ENERGY_DRAFT_KEY] = get_global_energy_balance_df(file_bytes, uploaded_file.name).copy(deep=True)
        if _RAW_LOADS_DRAFT_KEY not in st.session_state:
            st.session_state[_RAW_LOADS_DRAFT_KEY] = get_loads_balance_df(file_bytes, uploaded_file.name).copy(deep=True)

        # ---------- Energy_Balance ----------
        with st.expander("Energy_Balance (monthly, kWh)", expanded=True):
            active_raw_scenario = str(st.session_state.get("active_scenario", "Base") or "Base")
            raw_energy_scope = st.radio(
                "Energy_Balance update scope",
                options=["Global/base data", "Active scenario only"],
                index=0,
                horizontal=True,
                key=f"raw_energy_scope_{wb_hash}",
                help=(
                    "Global/base data updates the workbook's base Energy_Balance sheet. "
                    "Active scenario only creates or updates an Energy_Balance override used only by the selected scenario."
                ),
            )
            use_scenario_energy_scope = raw_energy_scope == "Active scenario only"
            scope_suffix = "global" if not use_scenario_energy_scope else f"scenario_{_safe_state_key(active_raw_scenario)}"

            energy_editor_key = f"raw_energy_editor_{wb_hash}_{scope_suffix}"
            energy_flash_key = f"_raw_energy_flash_{wb_hash}_{scope_suffix}"
            energy_rename_key = f"raw_energy_rename_{wb_hash}_{scope_suffix}"

            if use_scenario_energy_scope:
                has_override = get_scenario_energy_balance_override(active_raw_scenario) is not None
                st.caption(
                    f"Editing scope: **{active_raw_scenario}** only. "
                    + ("This scenario already has its own Energy_Balance override." if has_override else "No committed override exists yet; the draft starts from the global/base Energy_Balance.")
                )
            else:
                st.caption("Editing scope: **global/base Energy_Balance**. Scenario-specific overrides, if any, are not changed.")

            # Flash messages (shown after rerun)
            if st.session_state.get(energy_flash_key) == "updated":
                if use_scenario_energy_scope:
                    st.success(f"Energy_Balance override updated for scenario '{active_raw_scenario}'.")
                else:
                    st.success("Global Energy_Balance updated and applied to calculations without scenario override.")
                del st.session_state[energy_flash_key]
            elif st.session_state.get(energy_flash_key) == "reverted":
                st.info("Energy_Balance edits reverted to the last applied version for the selected scope.")
                del st.session_state[energy_flash_key]
            elif st.session_state.get(energy_flash_key) == "renamed":
                st.info("Energy_Balance columns renamed in draft. Click **Update Data** to apply to the selected scope.")
                del st.session_state[energy_flash_key]
            elif st.session_state.get(energy_flash_key) == "removed_override":
                st.info(f"Scenario-specific Energy_Balance override removed for '{active_raw_scenario}'. This scenario now uses the global/base data.")
                del st.session_state[energy_flash_key]

            def _get_energy_draft_for_scope() -> pd.DataFrame:
                if use_scenario_energy_scope:
                    drafts = _scenario_energy_drafts()
                    if active_raw_scenario not in drafts or not isinstance(drafts.get(active_raw_scenario), pd.DataFrame):
                        src = get_scenario_energy_balance_override(active_raw_scenario)
                        if src is None:
                            src = get_global_energy_balance_df(file_bytes, uploaded_file.name)
                        drafts[active_raw_scenario] = sanitize_energy_balance_df(src).copy(deep=True)
                        st.session_state[_RAW_ENERGY_SCENARIO_DRAFTS_KEY] = drafts
                    return sanitize_energy_balance_df(drafts.get(active_raw_scenario, pd.DataFrame())).copy(deep=True)
                return sanitize_energy_balance_df(st.session_state.get(_RAW_ENERGY_DRAFT_KEY, pd.DataFrame())).copy(deep=True)

            def _set_energy_draft_for_scope(df_in: pd.DataFrame, mark_dirty: bool = False) -> None:
                clean = sanitize_energy_balance_df(df_in)
                if use_scenario_energy_scope:
                    drafts = _scenario_energy_drafts()
                    drafts[active_raw_scenario] = clean.copy(deep=True)
                    st.session_state[_RAW_ENERGY_SCENARIO_DRAFTS_KEY] = drafts
                    if mark_dirty:
                        _mark_scenario_energy_draft_dirty(active_raw_scenario, True)
                else:
                    st.session_state[_RAW_ENERGY_DRAFT_KEY] = clean.copy(deep=True)

            def _last_applied_energy_for_scope() -> pd.DataFrame:
                if use_scenario_energy_scope:
                    src = get_scenario_energy_balance_override(active_raw_scenario)
                    if src is None:
                        src = get_global_energy_balance_df(file_bytes, uploaded_file.name)
                    return sanitize_energy_balance_df(src).copy(deep=True)
                return sanitize_energy_balance_df(st.session_state.get(_RAW_ENERGY_KEY, pd.DataFrame())).copy(deep=True)

            # Work on draft copy (applied to calculations only after Update Data)
            df_energy_raw = _get_energy_draft_for_scope()

            c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
            with c1:
                new_col_name = st.text_input("Add new End Use column", value="", key=f"raw_add_energy_col_{wb_hash}_{scope_suffix}")
            with c2:
                new_col_default = numeric_input("Default value", 0.0, key=f"raw_add_energy_default_{wb_hash}_{scope_suffix}", fmt="{:.3f}")
            with c3:
                if st.button("Add column", key=f"raw_add_energy_btn_{wb_hash}_{scope_suffix}", use_container_width=True):
                    if new_col_name and str(new_col_name).strip():
                        col = str(new_col_name).strip()
                        if col not in df_energy_raw.columns:
                            df_energy_raw[col] = float(new_col_default)
                            _set_energy_draft_for_scope(df_energy_raw, mark_dirty=True)
                    # Force editor widget to rebuild so schema changes (new columns) are reflected immediately.
                    st.session_state.pop(energy_editor_key, None)
                    st.session_state.pop(energy_rename_key, None)
                    st.rerun()

            with c4:
                if st.button(
                    "Revert",
                    key=f"raw_revert_energy_{wb_hash}_{scope_suffix}",
                    use_container_width=True,
                    help="Discard unsaved edits and revert to the last applied data for the selected scope.",
                ):
                    _set_energy_draft_for_scope(_last_applied_energy_for_scope(), mark_dirty=False)
                    if use_scenario_energy_scope:
                        _mark_scenario_energy_draft_dirty(active_raw_scenario, False)
                    st.session_state.pop(energy_editor_key, None)
                    st.session_state.pop(energy_rename_key, None)
                    st.session_state[energy_flash_key] = "reverted"
                    st.rerun()

            with c5:
                if use_scenario_energy_scope:
                    if st.button(
                        "Remove override",
                        key=f"raw_remove_energy_override_{wb_hash}_{scope_suffix}",
                        use_container_width=True,
                        help="Delete this scenario's Energy_Balance override and use the global/base data again.",
                    ):
                        delete_scenario_energy_balance_override(active_raw_scenario)
                        st.session_state.pop(energy_editor_key, None)
                        st.session_state.pop(energy_rename_key, None)
                        st.session_state[energy_flash_key] = "removed_override"
                        st.rerun()
                else:
                    st.caption("Scenario override controls are available when 'Active scenario only' is selected.")

            # --- Rename columns (End Uses) ---
            with st.expander("Rename columns (End Uses)", expanded=False):
                st.caption(
                    "Rename End Use columns in the selected editing scope. Tip: the app uses End Use names without the `_kWh` suffix. If you enter `_kWh`, it will be removed; the suffix is added back automatically when saving to Excel."
                )
                renamable_cols = [c for c in df_energy_raw.columns if c != "Month"]
                if len(renamable_cols) == 0:
                    st.info("No End Use columns available to rename.")
                else:
                    with st.form(f"raw_energy_rename_form_{wb_hash}_{scope_suffix}", clear_on_submit=False):
                        rename_df = pd.DataFrame({"Current": renamable_cols, "New": renamable_cols})
                        edited_rename_df = st.data_editor(
                            rename_df,
                            num_rows="fixed",
                            use_container_width=True,
                            key=energy_rename_key,
                            disabled=["Current"],
                        )
                        apply_rename_energy = st.form_submit_button("Apply renaming to draft", use_container_width=True)

                    if apply_rename_energy:
                        def _norm_enduse_name(_s: str) -> str:
                            s_ = str(_s or "").strip()
                            # Match template convention: use '_' as separator
                            s_ = re.sub(r"\s+", "_", s_)
                            # App logic uses End Use names without suffix; strip if user typed it
                            s_ = re.sub(r"(?i)_kwh$", "", s_)
                            return s_

                        mapping = {}
                        final_cols = ["Month"]

                        for _, r in edited_rename_df.iterrows():
                            old = str(r["Current"]).strip()
                            raw_new = str(r["New"]).strip()
                            if not raw_new:
                                raw_new = old

                            new = _norm_enduse_name(raw_new)

                            # Prevent blank / reserved names
                            if not new or new == "Month":
                                new = old

                            mapping[old] = new
                            final_cols.append(new)

                        if len(set(final_cols)) != len(final_cols):
                            st.error("Duplicate column names detected. Please use unique End Use names.")
                        else:
                            # Preserve End Use colors on rename
                            try:
                                cmap = st.session_state.get("color_map_enduse")
                                if isinstance(cmap, dict):
                                    for _old, _new in mapping.items():
                                        if _old != _new and _old in cmap and _new not in cmap:
                                            cmap[_new] = cmap[_old]
                                    st.session_state["color_map_enduse"] = cmap
                            except Exception:
                                pass

                            # Preserve On-site Generation tagging on rename (so NET logic follows the renamed column)
                            try:
                                onsite_lst = st.session_state.get(_ONSITE_ENDUSES_KEY, [ONSITE_GENERATION_ENDUSE])
                                if not isinstance(onsite_lst, list) or len(onsite_lst) == 0:
                                    onsite_lst = [ONSITE_GENERATION_ENDUSE]
                                # normalize legacy token
                                onsite_lst = [ONSITE_GENERATION_ENDUSE if str(x) == LEGACY_PV_ENDUSE else str(x) for x in onsite_lst]
                                updated_lst = [mapping.get(x, x) for x in onsite_lst]
                                uniq = []
                                for x in updated_lst:
                                    if x not in uniq:
                                        uniq.append(x)
                                st.session_state[_ONSITE_ENDUSES_KEY] = uniq
                            except Exception:
                                pass

                            df_renamed = df_energy_raw.rename(columns=mapping)
                            df_renamed = sanitize_energy_balance_df(df_renamed)
                            _set_energy_draft_for_scope(df_renamed, mark_dirty=True)

                            # Reset widget state so editors rebuild with the new schema immediately
                            st.session_state.pop(energy_editor_key, None)
                            st.session_state.pop(energy_rename_key, None)

                            st.session_state[energy_flash_key] = "renamed"
                            st.rerun()

            # Draft editor: the data_editor is inside a form to prevent Streamlit from
            # rerunning the full app on every cell edit. Edits are sent to Python only
            # when the user clicks **Update Data**. Once committed, scenario-specific
            # overrides survive scenario switching and are included in Save Project.
            editor_kwargs = {
                "num_rows": "dynamic",
                "use_container_width": True,
                "key": energy_editor_key,
            }
            if hasattr(st, "column_config"):
                col_cfg = {"Month": st.column_config.TextColumn("Month", required=True)}
                for c in df_energy_raw.columns:
                    if c == "Month":
                        continue
                    col_cfg[c] = st.column_config.NumberColumn(c, format="%.3f")
                editor_kwargs["column_config"] = col_cfg

            with st.form(f"raw_energy_update_form_{wb_hash}_{scope_suffix}", clear_on_submit=False):
                edited_energy = st.data_editor(df_energy_raw, **editor_kwargs)
                apply_energy = st.form_submit_button("Update Data", use_container_width=True)

            if apply_energy:
                committed_energy = sanitize_energy_balance_df(edited_energy)
                if use_scenario_energy_scope:
                    set_scenario_energy_balance_override(active_raw_scenario, committed_energy)
                    _set_energy_draft_for_scope(committed_energy, mark_dirty=False)
                    _mark_scenario_energy_draft_dirty(active_raw_scenario, False)
                else:
                    st.session_state[_RAW_ENERGY_KEY] = committed_energy
                    st.session_state[_RAW_ENERGY_DRAFT_KEY] = committed_energy.copy(deep=True)
                st.session_state[_RAW_COMMIT_VERSION_KEY] = st.session_state.get(_RAW_COMMIT_VERSION_KEY, 0) + 1

                # Reset the editor widget state so it always reflects committed data after apply.
                st.session_state.pop(energy_editor_key, None)
                st.session_state.pop(energy_rename_key, None)

                # Force a full rerun only after an explicit user commit.
                st.session_state[energy_flash_key] = "updated"
                st.rerun()

# ---------- Loads_Balance ----------
        with st.expander("Loads_Balance (hourly, kW)", expanded=False):
            loads_editor_key = f"raw_loads_editor_{wb_hash}"
            loads_flash_key = f"_raw_loads_flash_{wb_hash}"


            loads_rename_key = f"raw_loads_rename_{wb_hash}"
            # Flash messages (shown after rerun)
            if st.session_state.get(loads_flash_key) == "updated":
                st.success("Loads_Balance updated and applied to all calculations.")
                del st.session_state[loads_flash_key]
            elif st.session_state.get(loads_flash_key) == "reverted":
                st.info("Loads_Balance edits reverted to the last applied version.")
                del st.session_state[loads_flash_key]
            elif st.session_state.get(loads_flash_key) == "renamed":
                st.info("Loads_Balance columns renamed in draft. Click **Update Data** to apply to all calculations.")
                del st.session_state[loads_flash_key]

            df_loads_raw = sanitize_loads_balance_df(st.session_state.get(_RAW_LOADS_DRAFT_KEY, pd.DataFrame())).copy(deep=True)

            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                new_load_col = st.text_input(
                    "Add new Load column",
                    value="",
                    key=f"raw_add_load_col_{wb_hash}",
                    help="Column will be treated as a load profile (kW).",
                )
            with c2:
                new_load_default = numeric_input("Default value", 0.0, key=f"raw_add_load_default_{wb_hash}", fmt="{:.3f}")
            with c3:
                if st.button("Add column", key=f"raw_add_load_btn_{wb_hash}", use_container_width=True):
                    if new_load_col and str(new_load_col).strip():
                        col = str(new_load_col).strip()
                        if col not in df_loads_raw.columns:
                            df_loads_raw[col] = float(new_load_default)
                            st.session_state[_RAW_LOADS_DRAFT_KEY] = sanitize_loads_balance_df(df_loads_raw)
                    st.session_state.pop(loads_editor_key, None)
                    st.session_state.pop(loads_rename_key, None)
                    st.rerun()

            with c4:
                if st.button(
                    "Revert",
                    key=f"raw_revert_loads_{wb_hash}",
                    use_container_width=True,
                    help="Discard unsaved edits and revert to last applied data.",
                ):
                    st.session_state[_RAW_LOADS_DRAFT_KEY] = st.session_state.get(_RAW_LOADS_KEY, pd.DataFrame()).copy(deep=True)
                    st.session_state.pop(loads_editor_key, None)
                    st.session_state.pop(loads_rename_key, None)
                    st.session_state[loads_flash_key] = "reverted"
                    st.rerun()

            # --- Rename columns (Loads) ---
            with st.expander("Rename columns (Loads)", expanded=False):
                fixed_cols = ["hoy", "doy", "day", "month", "weekday", "hour", "Grid_Injection"]
                fixed_in_df = [c for c in fixed_cols if c in df_loads_raw.columns]
                renamable_cols = [c for c in df_loads_raw.columns if c not in fixed_in_df]

                st.caption(
                    "Rename load columns. Tip: the app uses Load names without the `_load` suffix. If you enter `_load`, it will be removed; the suffix is added back automatically when saving to Excel. The time/meta columns "
                    f"({', '.join(fixed_in_df)}) are fixed and cannot be renamed."
                )

                if len(renamable_cols) == 0:
                    st.info("No load columns available to rename.")
                else:
                    with st.form(f"raw_loads_rename_form_{wb_hash}", clear_on_submit=False):
                        rename_df = pd.DataFrame({"Current": renamable_cols, "New": renamable_cols})
                        edited_rename_df = st.data_editor(
                            rename_df,
                            num_rows="fixed",
                            use_container_width=True,
                            key=loads_rename_key,
                            disabled=["Current"],
                        )
                        apply_rename_loads = st.form_submit_button("Apply renaming to draft", use_container_width=True)

                    if apply_rename_loads:
                        def _norm_load_name(_s: str) -> str:
                            s_ = str(_s or "").strip()
                            # Match template convention: use '_' as separator
                            s_ = re.sub(r"\s+", "_", s_)
                            # App logic uses Load names without suffix; strip if user typed it
                            s_ = re.sub(r"(?i)_load$", "", s_)
                            return s_

                        mapping = {}
                        final_cols = list(fixed_in_df)

                        # Validate + build mapping
                        for _, r in edited_rename_df.iterrows():
                            old = str(r["Current"]).strip()
                            raw_new = str(r["New"]).strip()
                            if not raw_new:
                                raw_new = old

                            new = _norm_load_name(raw_new)

                            if not new:
                                new = old

                            if new in fixed_in_df:
                                st.error(f"'{new}' is reserved for time/meta columns and cannot be used as a load name.")
                                mapping = None
                                break

                            mapping[old] = new
                            final_cols.append(new)

                        if mapping is not None:
                            if len(set(final_cols)) != len(final_cols):
                                st.error("Duplicate column names detected. Please use unique load names.")
                            else:
                                # Preserve Load colors on rename
                                try:
                                    cmap = st.session_state.get("color_map_loads")
                                    if isinstance(cmap, dict):
                                        for _old, _new in mapping.items():
                                            if _old != _new and _old in cmap and _new not in cmap:
                                                cmap[_new] = cmap[_old]
                                        st.session_state["color_map_loads"] = cmap
                                except Exception:
                                    pass

                                # Preserve On-site Generation tagging on rename (loads)
                                try:
                                    onsite_lst = st.session_state.get(_ONSITE_ENDUSES_KEY, [ONSITE_GENERATION_ENDUSE])
                                    if not isinstance(onsite_lst, list) or len(onsite_lst) == 0:
                                        onsite_lst = [ONSITE_GENERATION_ENDUSE]
                                    onsite_lst = [ONSITE_GENERATION_ENDUSE if str(x) == LEGACY_PV_ENDUSE else str(x) for x in onsite_lst]
                                    updated_lst = [mapping.get(x, x) for x in onsite_lst]
                                    uniq = []
                                    for x in updated_lst:
                                        if x not in uniq:
                                            uniq.append(x)
                                    st.session_state[_ONSITE_ENDUSES_KEY] = uniq
                                except Exception:
                                    pass

                                df_renamed = df_loads_raw.rename(columns=mapping)
                                df_renamed = sanitize_loads_balance_df(df_renamed)
                                st.session_state[_RAW_LOADS_DRAFT_KEY] = df_renamed

                                # Reset widget state so editors rebuild with the new schema immediately
                                st.session_state.pop(loads_editor_key, None)
                                st.session_state.pop(loads_rename_key, None)

                                st.session_state[loads_flash_key] = "renamed"
                                st.rerun()


            with st.form(f"raw_loads_form_{wb_hash}", clear_on_submit=False):
                editor_kwargs = {
                    "num_rows": "dynamic",
                    "use_container_width": True,
                    "key": loads_editor_key,
                }
                if hasattr(st, "column_config") and not df_loads_raw.empty:
                    col_cfg = {}
                    for c in df_loads_raw.columns:
                        if c == "weekday":
                            col_cfg[c] = st.column_config.TextColumn(c)
                        else:
                            col_cfg[c] = st.column_config.NumberColumn(c, format="%.3f")
                    editor_kwargs["column_config"] = col_cfg

                edited_loads = st.data_editor(df_loads_raw, **editor_kwargs)

                # Persist edits into draft on every rerun (do not apply to calculations yet).
                st.session_state[_RAW_LOADS_DRAFT_KEY] = sanitize_loads_balance_df(edited_loads)

                apply_loads = st.form_submit_button("Update Data", use_container_width=True)

            if apply_loads:
                committed_loads = sanitize_loads_balance_df(edited_loads)
                st.session_state[_RAW_LOADS_KEY] = committed_loads
                st.session_state[_RAW_LOADS_DRAFT_KEY] = committed_loads.copy(deep=True)
                st.session_state[_RAW_COMMIT_VERSION_KEY] = st.session_state.get(_RAW_COMMIT_VERSION_KEY, 0) + 1

                st.session_state.pop(loads_editor_key, None)
                st.session_state.pop(loads_rename_key, None)

                st.session_state[loads_flash_key] = "updated"
                st.rerun()

    else:
        st.write("### ← Please upload data on side bar")
