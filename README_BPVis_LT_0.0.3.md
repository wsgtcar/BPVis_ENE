# BPVis LT — Building Performance Visualizer
**Version:** 0.0.3  
**Type:** Streamlit dashboard  
**Scope:** Energy Balance, CO₂ Emissions, Loads Analysis, Energy Cost

---

## 1) What this tool does (high level)
BPVis LT ingests a simple Excel workbook and gives you a clean, repeatable analysis of a building’s performance:

- **Energy Balance** (kWh): monthly + annual, End Use vs Energy Source, EUI (kWh/m²·a), PV coverage
- **CO₂ Emissions** (kgCO₂): same visuals, using per‑source emission factors
- **Loads Analysis** (kW / W/m²): heatmaps (doy × hour), monthly totals, peak day profile, duration curve, percentiles, threshold exceedance
- **Energy Cost** (currency): same visuals, using per‑source cost/kWh (tariffs)

All charts share consistent color palettes and fixed category orders for readability and comparability.

---

## 2) Project structure (typical)
```
BPVis/
├─ BPVis_LT.py                    # Streamlit app (main entry point)
├─ requirements.txt
├─ README.md                      # this file
├─ templates/
│  └─ energy_database_complete_template.xlsx
└─ assets/
   └─ WS_Logo.jpg                 # optional branding
```
> Your actual filenames may differ slightly; the concepts and blocks below map directly to the code in `BPVis_LT.py`.

---

## 3) Running the app
### 3.1. Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3.2. Launch
```bash
streamlit run BPVis_LT.py
```
Your browser will open to `http://localhost:8501`.

---

## 4) Data model — Excel template
The app expects a workbook with at least these two sheets:

### 4.1. `Energy_Balance`
- **Columns**
  - `Month` — text: `January, February, …, December`
  - One column per **End Use**, with suffix **`_kWh`**:
    - `Heating_kWh`, `Cooling_kWh`, `Ventilation_kWh`, `Lighting_kWh`,
      `Equipment_kWh`, `HotWater_kWh`, `Pumps_kWh`, `Other_kWh`, `PV_Generation_kWh`
- **Semantics**
  - Values per month (kWh). `PV_Generation_kWh` is **negative** (generation offsets demand).

### 4.2. `Loads_Balance`
- **Time keys**
  - `hoy` (hour of year), `doy` (day of year), `day`, `month`, `weekday`, `hour`
- **End‑use loads** (kW), each with suffix **`_load`**:
  - `Heating_load`, `Cooling_load`, `Ventilation_load`, `Lighting_load`, `Equipment_load`, `HotWater_load`, `Pumps_load`, `Other_load`, `PV_Generation_load`
- **Source loads** (kW), also `_load`‑suffixed:
  - `electricity_load`, `gas_load`, `district_heating_load`, `district_cooling_load`

> The app automatically removes `_kWh`/`_load` suffixes only where needed for display/aggregation; internally it keeps enough structure to locate and chart the correct fields.

---

## 5) UI walkthrough (user manual)
### 5.1. Sidebar
1. **Download Template** — saves `templates/energy_database_complete_template.xlsx`
2. **Upload Data** — select your filled template (`.xlsx`)
3. **Project Data**
   - *Project Name*: a label for your run
   - *Project Area (m²)*: used to compute intensities (EUI, kgCO₂/m²·a, cost/m²·a, W/m²)
4. **Emission Factors** (kgCO₂/kWh) — per Energy Source
5. **Energy Tariffs** (currency/kWh) — per Energy Source, plus currency selector
6. **Assign Energy Sources** — for each detected End Use, pick its supply source  
   *(e.g., map `Heating`→`District Heating`, `Cooling`→`Electricity`, `PV_Generation`→`Electricity`)*

### 5.2. Tabs
- **Energy Balance** — kWh charts, EUI donut, PV coverage & energy KPIs
- **CO₂ Emissions** — kgCO₂ charts, intensity donut & KPIs
- **Loads Analysis** — heatmap, monthly totals, KPIs, peak day, duration curve, profiles, exceedance
- **Energy Cost** — currency charts, intensity donut & KPIs

---

## 6) Code trace back & block‑by‑block guide
Below is a detailed explanation of code blocks (names may vary slightly). Each block shows **inputs → process → outputs**.

### 6.1. Page setup & constants
**Inputs:** none  
**Process:** `st.set_page_config`, title, branding; define `MONTH_ORDER`, `END_USE_ORDER`, `ENERGY_SOURCE_ORDER`, and the color maps (`color_map`, `color_map_sources`).  
**Outputs:** UI header; global constants for consistent chart ordering and colors.

### 6.2. Cached loaders
```python
@st.cache_data
def load_energy_balance_sheet(file_bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df = pd.read_excel(xls, sheet_name="Energy_Balance")
    df.columns = df.columns.str.replace("_kWh", "", regex=False)
    return df

def load_loads_balance(file_bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df = pd.read_excel(xls, sheet_name="Loads_Balance")
    # keep `_load` suffix for loads analysis (or strip with .str.replace as needed)
    return df
```
**Inputs:** raw uploaded file bytes  
**Process:** read the two sheets, optionally strip suffixes (Energy_Balance)  
**Outputs:** `df` DataFrames for further transforms

### 6.3. Sidebar inputs
- **Project Data** → `project_name: str`, `project_area: float`
- **Emission Factors** → `co2_Emissions_Electricity`, `co2_emissions_dh`, `co2_emissions_dc`, `co2_emissions_gas`
- **Energy Tariffs** → `cost_electricity`, `cost_dh`, `cost_dc`, `cost_gas`, `currency_symbol`
- **Assign Energy Sources**  
  **Inputs:** `df_melted["End_Use"].unique()`  
  **Process:** for each End Use, `st.selectbox(key=f"source_{use}")` and store in `mapping_dict`  
  **Outputs:** `mapping_dict: {End_Use → Energy_Source}` and session_state persistence

### 6.4. Energy Balance tab
**Data prep**
```python
df = load_energy_balance_sheet(uploaded_file.getvalue())
df_melted = df.melt(id_vars="Month", var_name="End_Use", value_name="kWh")
df_melted["Energy_Source"] = df_melted["End_Use"].map(mapping_dict)
```
- **Inputs:** `df`, `mapping_dict`, `project_area`
- **Outputs:** `df_melted` with columns `Month, End_Use, kWh, Energy_Source`

**Monthly net line overlay**
```python
monthly_totals = (df_melted.groupby("Month", as_index=False)["kWh"].sum())
# categorize/sort Month; create px.bar; add px.line(trace) with dashed style
```
- **Inputs:** `df_melted`
- **Outputs:** `monthly_chart` (stacked bars by End Use + net line)

**Monthly by source (clean hovers)**
```python
monthly_by_source = df_melted.groupby(["Month","Energy_Source"], as_index=False)["kWh"].sum()
monthly_chart_source = px.bar(monthly_by_source, ...)
```
- **Outputs:** `monthly_chart_source`

**Annual totals & intensities**
```python
totals = df_melted.groupby("End_Use", as_index=False)["kWh"].sum()
totals["Per Use"] = "Total"
totals["kWh_per_m2"] = (totals["kWh"] / project_area).round(1)

eui       = totals.loc[totals["kWh_per_m2"] > 0, "kWh_per_m2"].sum()
net_eui   = totals["kWh_per_m2"].sum()
net_energy= totals["kWh"].sum()                  # includes PV
```
- **Outputs:** `totals`, `eui`, `net_eui`, `net_energy`

**Annual charts**
- End Use: stacked bar + dashed `add_hline(y=net_energy)` + annotation
- Source: stacked bar from `totals_per_source = df_melted.groupby("Energy_Source")["kWh"].sum()`

**Intensity donuts (kWh/m²·a)**
```python
energy_intensity_chart = px.pie(totals, names="End_Use", values="kWh_per_m2", hole=0.5, ...)
# center annotation = total EUI (consumption-only)
```
- **Outputs:** two donut charts (End Use / Source), with `textinfo="value+percent"` and center label

**KPIs & PV coverage**
```python
pv_value    = totals.set_index("End_Use").loc["PV_Generation","kWh_per_m2"]
pv_coverage = abs(pv_value / eui) * 100
monthly_avr = totals["kWh"].sum() / 12
total_energy= totals.loc[totals["kWh"] > 0, "kWh"].sum()
```
- **Outputs:** Streamlit `st.metric` widgets

### 6.5. CO₂ Emissions tab (mirrors Energy)
**Inputs:** `df_melted`, `mapping_dict`, emission factors, `project_area`  
**Process:**
```python
factor_map = {
    "Electricity": co2_Emissions_Electricity,
    "Green Electricity": co2_Emissions_Green_Electricity,
    "Gas": co2_emissions_gas,
    "District Heating": co2_emissions_dh,
    "District Cooling": co2_emissions_dc,
}
df_co2 = df_melted.assign(
  CO2_factor_kg_per_kWh = lambda d: d["Energy_Source"].map(factor_map).fillna(0.0),
  kgCO2 = lambda d: d["kWh"] * d["CO2_factor_kg_per_kWh"]  # PV (negative kWh) => negative kgCO2 (offset)
)
```
**Outputs:** Charts identical to Energy tab but for `kgCO2`; pies show `kgCO2_per_m2`; KPIs for monthly avg, annual total, intensity total.

### 6.6. Loads Analysis tab
**Inputs:** `df_loads = load_loads_balance(...)` (keeps `_load` suffix), `project_area`, `selected_load`  
**Blocks:**
- **2D Heatmap** (doy × hour; `px.density_heatmap(z=selected_load, histfunc="avg" or "sum")`)
- **Monthly totals bar** (group by `month`, calendar order, value labels)
- **KPIs** (total, min/max kW, specific max/min W/m², 95th/80th percentiles)
- **Peak day profile** (find `doy` with max daily sum; plot `hour` line+markers; optional area fill)
- **Load duration curve** (sorted descending values vs % of hours)
- **Profiles by Month/Weekday** (average daily shape by group)
- **Threshold exceedance heatmap** (count hours > threshold per (doy,hour))

**Outputs:** plotly figures and metrics for operational insight.

### 6.7. Energy Cost tab (mirrors CO₂)
**Inputs:** `df_melted`, `mapping_dict`, `project_area`, tariffs from sidebar, `currency_symbol`  
**Process:**
```python
cost_map = {
  "Electricity": cost_electricity,
  "Gas": cost_gas,
  "District Heating": cost_dh,
  "District Cooling": cost_dc,
  "Green Electricity": cost_green_electricity,
}
df_cost = df_melted.assign(
  cost_per_kWh = lambda d: d["Energy_Source"].map(cost_map).fillna(0.0),
  cost = lambda d: d["kWh"] * d["cost_per_kWh"]  # PV (negative) => negative cost (saves money)
)
```
**Outputs:** Monthly/annual charts by End Use & Source; donuts of `cost_per_m2`; KPIs for monthly avg, annual total, and total cost intensity.

---

## 7) Inputs & outputs (reference tables)

### 7.1. Mapping (sidebar “Assign Energy Sources”)
- **Input:** `End_Use` list derived from uploaded file
- **Output:** `mapping_dict: dict[str, str]` (e.g., `{"Heating": "District Heating"}`)

### 7.2. Energy tab
- **Input:** `df_melted[Month, End_Use, kWh, Energy_Source]`, `project_area`
- **Output:**
  - **Charts:** Monthly (End Use + net line), Monthly (Source), Annual (End Use + net line), Annual (Source), EUI pies (End Use/Source)
  - **Metrics:** Monthly avg kWh, Total annual (consumption-only), Net annual, EUI, Net EUI, PV Coverage

### 7.3. CO₂ tab
- **Input:** Energy tab + emission `factor_map`
- **Output:**
  - **Charts:** Monthly/Annual per End Use & Source in `kgCO2`
  - **Pies:** `kgCO2_per_m2`
  - **Metrics:** Monthly avg CO₂, Total annual CO₂, Total CO₂ intensity

### 7.4. Loads tab
- **Input:** `df_loads` with `_load` columns, `project_area`, `selected_load`
- **Output:**
  - **Charts:** 2D heatmap, monthly bar, peak‑day profile, duration curve, profiles by month/weekday, exceedance heatmap
  - **Metrics:** totals, min/max, percentiles, specific loads

### 7.5. Cost tab
- **Input:** Energy tab + `cost_map`, `currency_symbol`
- **Output:**
  - **Charts:** Monthly/Annual cost by End Use & Source; cost intensity pies
  - **Metrics:** Monthly avg cost, Total annual cost, Cost intensity total; per‑source cost intensities

---

## 8) Color & category control
- **Colors:** fixed `color_map` for End Uses, `color_map_sources` for Energy Sources
- **Orders:** `MONTH_ORDER` + explicit `category_orders` in all PX figures to avoid random legend/axis reorders
- **Text labels:** `text_auto`, `textfont_size`, and `textfont_color` set for clarity

---

## 9) PV handling
- **Energy:** negative kWh lowers net totals
- **CO₂:** negative kWh × positive factor ⇒ negative kgCO₂ (offset)
- **Cost:** negative kWh × positive cost ⇒ negative currency (savings)
- **Best practice:** map `PV_Generation` → `Electricity` in the sidebar; this offsets electricity emissions/costs.

---

## 10) Performance & robustness
- **Caching:** `@st.cache_data` for Excel loading to speed up UI changes
- **Type safety:** numeric coercion (`pd.to_numeric(..., errors="coerce")`) before math
- **Zero‑division guards:** e.g., PV coverage if `eui == 0`
- **Missing PV:** index checks before accessing `PV_Generation`

---

## 11) Common pitfalls & fixes
- **Wrong month order** → ensure exact names; set `category_orders={"Month": MONTH_ORDER}`
- **Legend title tweaks** → `fig.update_layout(legend=dict(title=dict(text="")))`
- **Annotation errors** → put only annotation fields in `annotations=[...]`; never legend keys
- **KeyError on indexing** → don’t use Series values as index labels; use `.sort_values` or `.isin`

---

## 12) Extensibility (future‑proof ideas)
- Export all computed tables (CSV/XLSX)
- Filters (End Uses / Sources) that also update KPIs
- Toggles: show **net vs. consumption‑only** lines
- Validation panel (missing months, non‑numeric cells, PV sanity checks)
- Scenario compare (A/B cost or factor sets)

---

## 13) Glossary
- **End Use** — functional demand (Heating, Cooling, Lighting, …)
- **Energy Source** — supply type (Electricity, Gas, District Heating, District Cooling)
- **EUI** — Energy Use Intensity (kWh/m²·a), consumption‑only
- **CO₂ intensity** — kgCO₂/m²·a
- **Net** — includes PV offsets
- **Peak day** — day with max daily sum of a selected load

---

## 14) License
© Werner Sobek Green Technologies GmbH. All rights reserved.  
(Replace with your preferred license if needed.)
