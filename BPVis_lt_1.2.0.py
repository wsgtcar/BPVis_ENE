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
    page_title="WSGT_BPVis_ENE 1.4.11",
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

# --- NEW: ensure a default project name exists before rendering the title
if "project_name" not in st.session_state:
    st.session_state["project_name"] = "Building Performance Dashboard"

# =========================
# Sidebar — template download & file upload
# =========================
st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## BPVis ENE")
st.sidebar.write("Version 1.4.11")

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



def parse_color_settings_df(df: Optional[pd.DataFrame]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Parse Color_Settings sheet into (End_Use colors, Energy_Source colors, Load colors)."""
    end_use_map: Dict[str, str] = {}
    source_map: Dict[str, str] = {}
    load_map: Dict[str, str] = {}

    if df is None or df.empty:
        return end_use_map, source_map, load_map

    # Preferred schema: Type | Name | Color
    if {"Type", "Name", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            typ = str(row.get("Type", "")).strip()
            name = str(row.get("Name", "")).strip()
            col = str(row.get("Color", "")).strip()
            if not name or not col:
                continue
            if not col.startswith("#"):
                col = f"#{col}"
            typ_l = typ.lower()

            if typ_l in ["end_use", "end use", "enduse", "end-use"]:
                end_use_map[_canon_enduse_name(name)] = col
            elif typ_l in ["energy_source", "energy source", "energysource", "energy-source", "source"]:
                source_map[name] = col
            elif typ_l in ["load", "loads"]:
                load_map[_canon_enduse_name(name)] = col

        return end_use_map, source_map, load_map

    # Fallback schema(s)
    if {"End_Use", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("End_Use", "")).strip()
            col = str(row.get("Color", "")).strip()
            if not name or not col:
                continue
            if not col.startswith("#"):
                col = f"#{col}"
            end_use_map[_canon_enduse_name(name)] = col

    if {"Energy_Source", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("Energy_Source", "")).strip()
            col = str(row.get("Color", "")).strip()
            if not name or not col:
                continue
            if not col.startswith("#"):
                col = f"#{col}"
            source_map[name] = col

    if {"Load", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("Load", "")).strip()
            col = str(row.get("Color", "")).strip()
            if not name or not col:
                continue
            if not col.startswith("#"):
                col = f"#{col}"
            load_map[_canon_enduse_name(name)] = col

    return end_use_map, source_map, load_map


    # Preferred schema: Type | Name | Color
    if {"Type", "Name", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            typ = str(row.get("Type", "")).strip()
            name = str(row.get("Name", "")).strip()
            col = str(row.get("Color", "")).strip()
            if not name or not col:
                continue
            if not col.startswith("#"):
                col = f"#{col}"
            typ_l = typ.lower()
            if typ_l in ["end_use", "end use", "enduse", "end-use"]:
                end_use_map[_canon_enduse_name(name)] = col
            elif typ_l in ["energy_source", "energy source", "energysource", "energy-source", "source"]:
                source_map[name] = col
        return end_use_map, source_map

    # Fallback schema(s)
    if {"End_Use", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("End_Use", "")).strip()
            col = str(row.get("Color", "")).strip()
            if not name or not col:
                continue
            if not col.startswith("#"):
                col = f"#{col}"
            end_use_map[_canon_enduse_name(name)] = col

    if {"Energy_Source", "Color"}.issubset(df.columns):
        for _, row in df.iterrows():
            name = str(row.get("Energy_Source", "")).strip()
            col = str(row.get("Color", "")).strip()
            if not name or not col:
                continue
            if not col.startswith("#"):
                col = f"#{col}"
            source_map[name] = col

    return end_use_map, source_map



def build_color_settings_df(
        color_map_end_use: Dict[str, str],
        color_map_sources_in: Dict[str, str],
        color_map_loads_in: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Build Color_Settings sheet from the current color maps."""
    rows = []
    for k, v in (color_map_end_use or {}).items():
        rows.append({"Type": "End_Use", "Name": str(k), "Color": str(v)})
    for k, v in (color_map_sources_in or {}).items():
        rows.append({"Type": "Energy_Source", "Name": str(k), "Color": str(v)})
    for k, v in (color_map_loads_in or {}).items():
        rows.append({"Type": "Load", "Name": str(k), "Color": str(v)})
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
    saved_colors_enduse, saved_colors_sources, saved_colors_loads = parse_color_settings_df(cfg_saved.get("colors"))
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
        "scenarios_df": cfg_saved.get("scenarios"),
        "lcc_global_df": cfg_saved.get("lcc_global"),
        "lcc_investments_df": cfg_saved.get("lcc_investments"),
        "scenario_energy_df": cfg_saved.get("scenario_energy"),
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
            st.session_state["color_map_enduse"] = enduse_base
            st.session_state["color_map_sources"] = source_base
            st.session_state["color_map_loads"] = loads_base
        except Exception:
            st.session_state["color_map_enduse"] = dict(DEFAULT_COLOR_MAP)
            st.session_state["color_map_sources"] = dict(DEFAULT_COLOR_MAP_SOURCES)
            st.session_state["color_map_loads"] = dict(DEFAULT_COLOR_MAP_LOADS)

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
tab1, tab1_factors, tab2, tab3, tab4, tab5, tab6, tab_lcc, tab7, tab8 = st.tabs(
    ["Energy Balance (without Factors)", "Energy Balance (with Factors)", "CO2 Emissions (with Factors)", "Energy Cost (with Factors)", "Loads Analysis", "Benchmark",
     "CRREM-Analysis", "LCC-Analysis", "Scenarios", "Raw Data"])

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

                    scenarios.pop(current_active, None)
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

                # Clear color-picker widget state so new defaults are reflected immediately
                for _k in list(st.session_state.keys()):
                    if str(_k).startswith(("cp_eu_", "cp_src_", "cp_ld_")):
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


        # ---- Apply current color settings to plotting maps
        color_map = st.session_state.get("color_map_enduse", color_map)
        color_map_sources = st.session_state.get("color_map_sources", color_map_sources)
        color_map_loads = st.session_state.get("color_map_loads", DEFAULT_COLOR_MAP_LOADS)

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
            with st.expander("Raw Data", expanded=False):
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
