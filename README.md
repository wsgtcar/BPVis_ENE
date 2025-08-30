
# BPVis LT — Building Performance Visualization (v1.1.3 · Optimized)

**BPVis LT** is a Streamlit app for exploring a building’s **energy balance**, **CO₂ emissions**, **energy cost**, **loads**, and **benchmarking** from a single Excel workbook.  
It lets you map **End Uses → Energy Sources**, enter **CO₂ factors** and **tariffs**, and **save your inputs back** to the workbook for future sessions.

This release is optimized to avoid full-page reruns on every keystroke using **forms**, **`st.session_state`**, and **`@st.cache_data`**.

---

## ✨ Features

- **Tabs**
  - **Energy Balance**: Monthly and annual stacked bars by **End Use** and by **Energy Source**; EUI donuts; KPI metrics; PV treated as negative generation.
  - **CO₂ Emissions**: Mirrors Energy Balance but values are computed from user-defined **kgCO₂/kWh** factors.
  - **Energy Cost**: Mirrors Energy Balance but values use **tariffs per kWh** in the selected currency (€, $, £).
  - **Loads Analysis**: 2D heatmap by Day-of-Year × Hour; monthly totals; exceedance heatmap; peak-day profile; key percentiles.
  - **Benchmark**: Compares **Energy Density (EUI)**, **CO₂ Intensity**, and **Cost Intensity** with thresholds **per Building Use** from a benchmark workbook (gauge + vertical “label” charts + thresholds table).
- **Sidebar input forms** (apply-on-click → no constant reruns): **Project Data**, **Emission Factors**, **Energy Tariffs**, **Assign Energy Sources**.
- **Save Project** writes/updates configuration sheets in the uploaded workbook:
  - `Project_Data`, `Emission_Factors`, `Energy_Tariffs`, `EndUse_to_Source`
- **Benchmarking** from `templates/benchmark_template.xlsx` (one sheet per Building Use). You can also upload a custom benchmark file.
- **Consistent colors & category ordering** across charts.
- **Caching** of Excel I/O and transformations for faster UI.

---

## ⚙️ Installation

1. Ensure Python **3.9+** (3.10/3.11 recommended).
2. Create/activate a virtual environment (optional, recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> On Streamlit Community Cloud, the platform installs from `requirements.txt` automatically.

---

## ▶️ Run the App

From the project directory:

```bash
streamlit run BPVis_lt_1.1.3_optimized.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 📦 Input Data Format (Excel)

BPVis LT expects an `.xlsx` with these sheets. You can start from your own workbook or the provided template.

### 1) `Energy_Balance` (required)

Wide monthly table. End-Use columns may include a `_kWh` suffix (it will be stripped automatically).

| Month | Heating_kWh | Cooling_kWh | Ventilation_kWh | Lighting_kWh | Equipment_kWh | HotWater_kWh | Pumps_kWh | Other_kWh | PV_Generation_kWh |
|------|-------------:|------------:|----------------:|-------------:|--------------:|-------------:|----------:|----------:|------------------:|

- **Month** values must be English labels: `January … December` (ordering is enforced).
- **PV_Generation** should be **negative** if it offsets demand (so it subtracts in “net” values).

### 2) `Loads_Balance` (optional; required for Loads tab)

Hourly loads. Load columns end with `_load` (suffix is stripped).

| hoy | doy | day | month | weekday | hour | Heating_load | Cooling_load | … | PV_Generation_load |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|

- `hoy`: hour-of-year (1–8760)  
- `doy`: day-of-year (1–365)  
- `hour`: 0–23

### 3) **Saved Inputs** (created/updated by **Save Project** button)

- `Project_Data` (`Key`, `Value`) — includes:
  - `Project_Name`, `Project_Area`, `Currency` (€, $, £), `Building_Use`, `Project_Latitude`, `Project_Longitude`
- `Emission_Factors` — columns: `Energy_Source`, `Factor_kgCO2_per_kWh`
- `Energy_Tariffs` — columns: `Energy_Source`, `Tariff_per_kWh`
- `EndUse_to_Source` — columns: `End_Use`, `Energy_Source`

> When you upload a file, if these sheets exist, values are pre-loaded into the app; otherwise defaults are used.

### 4) `templates/benchmark_template.xlsx` (read at runtime)

One **sheet per Building Use** (`Office`, `Hospitality`, `Retail`, `Residential`, `Industrial`, `Education`, `Leisure`, `Healthcare`, …). Each sheet must contain **exactly** these columns:

| KPI            | Unit       | Excellent_Max | Good_Max | Poor_Max |
|----------------|------------|--------------:|---------:|---------:|
| Energy_Density | kWh/m²·a   |            50 |      100 |      150 |
| CO2_Intensity  | kgCO₂/m²·a |            10 |       20 |       30 |
| Cost_Intensity | €/m²·a     |            15 |       25 |       40 |

**Classification rule (lower is better):**  
`value ≤ Excellent_Max → Excellent` → else if `≤ Good_Max → Good` → else if `≤ Poor_Max → Poor` → else **Very Poor**.

---

## 🧰 Sidebar Controls (all use forms)

- **Project Data**
  - Project Name, Area (m²), Latitude, Longitude
  - **Building Use** (default: Office; saved to `Project_Data`)
  - **Apply Project Data** button
- **Emission Factors** (kgCO₂/kWh)
  - Electricity, **Green Electricity**, District Heating, District Cooling, Gas
  - **Apply Emission Factors**
- **Energy Tariffs**
  - Currency (€, $, £) + per-kWh tariffs for Electricity, **Green Electricity**, District Heating/Cooling, Gas
  - **Apply Energy Tariffs**
- **Assign Energy Sources**
  - For each **End Use** (Heating, Cooling, …, PV_Generation), pick an **Energy Source** from:
    `["Electricity", "Green Electricity", "Gas", "District Heating", "District Cooling"]`
  - **Apply Energy Sources**
- **Save Project**
  - Writes/updates the four configuration sheets and offers the **download** of the updated workbook.
- **Benchmark Settings**
  - Upload a custom `benchmark_template.xlsx` (optional; overrides default for this session).

> Because inputs are in **forms**, the app **does not rerun** on every keystroke. Changes take effect after clicking **Apply**.

---

## 📊 Visuals & KPI’s

### Energy Balance
- **Monthly stacked bar** by *End Use* (`barmode="relative"`). A **dashed line** overlays monthly **net total** (PV subtracts).
- **Monthly by Source** stacked bar by *Energy Source*.
- **Annual** stacked bars (single column) by Use/Source with net total reference line.
- **EUI Donuts** (per Use, per Source): center shows **total EUI** (kWh/m²·a).
- **KPIs**
  - Monthly Average Energy Consumption (kWh)
  - Total Annual Energy Consumption (kWh) — positive-only, excludes PV
  - Net Annual Energy Consumption (kWh) — includes PV
  - **EUI** and **Net EUI** (kWh/m²·a)
  - **PV Production** (kWh) and **PV Coverage** (% = |PV EUI| / EUI × 100)

### CO₂ Emissions
- Same visuals as Energy Balance, **values in kgCO₂**; donuts show **kgCO₂/m²·a**.

### Energy Cost
- Same visuals, **values in selected currency**; donuts show **currency/m²·a**.

### Loads Analysis
- **2D heatmap** (Day-of-Year × Hour) for selected load.
- **Monthly totals** bar chart.
- **KPIs**: total (kWh), max/min (kW), specific loads (W/m²), 95th & 80th percentiles.
- **Exceedance heatmap** above an adjustable threshold.
- **Peak-day profile**: hourly line with filled area; title shows date or DOY and daily total.

### Benchmark
- **Gauge** (speedometer) + **vertical band** (“energy label”) per KPI: EUI, CO₂ Intensity, Cost Intensity.
- **Thresholds table** (Excellent / Good / Poor / Very Poor breakpoints) + **Project KPIs** with class labels.

---

## 🧮 Calculations

Let:
- `A` = Project Area (m²)  
- `kWh_eu_m` = monthly energy for a given **End Use**  
- `Factor_src` = CO₂ factor (kgCO₂/kWh) of an **Energy Source**  
- `Tariff_src` = cost (currency/kWh) of an **Energy Source**

**Energy**
- Annual kWh per End Use = `∑ₘ kWh_eu_m`
- **EUI (kWh/m²·a)** = `∑ EndUses max(0, Annual_kWh_eu) / A` (PV excluded from EUI total)  
- **Net EUI** = `∑ EndUses (Annual_kWh_eu) / A` (PV included; can reduce net)

**CO₂**
- `kgCO₂ = kWh * Factor_src`  
- **CO₂ Intensity (kgCO₂/m²·a)** = `∑ kgCO₂ / A`

**Cost**
- `cost = kWh * Tariff_src`  
- **Cost Intensity (currency/m²·a)** = `∑ cost / A`

**PV Coverage (%)** = `|PV_EUI| / EUI × 100`

**Loads**
- Specific load (W/m²) = `kW / A × 1000`  
- Percentiles computed over hourly `W/m²` series.

**Benchmark Classes** (lower is better):  
`Excellent` if `value ≤ Excellent_Max`; else `Good` if `≤ Good_Max`; else `Poor` if `≤ Poor_Max`; else **Very Poor**.

---

## 🎨 Colors and Ordering

**End Uses**
```
Heating: #c02419
Cooling: #5a73a5
Ventilation: #42b38d
Lighting: #d3b402
Equipment: #833fd1
HotWater: #ff9a0a
Pumps: #06b6d1
Other: #d0448c
PV_Generation: #a9c724
```

**Energy Sources**
```
Electricity: #42b360
Green Electricity: #64c423
Gas: #c9d302
District Heating: #ec6939
District Cooling: #5a5ea5
```

**Category Ordering**
```
Months: January … December
End Uses: Heating, Cooling, Ventilation, Lighting, Equipment, HotWater, Pumps, Other, PV_Generation
Sources: Electricity, Green Electricity, Gas, District Heating, District Cooling
```

---

## ⚡ Performance

- **Forms** prevent reruns while typing; changes apply on **Apply** buttons.
- **`@st.cache_data`** caches Excel loading and wide→long transforms.
- **`st.session_state`** holds the applied values across reruns.
- Tips:
  - Keep only necessary sheets/columns in large files.
  - Ensure numeric columns are numeric (no stray text).  
  - Clear cache via the Streamlit menu if needed.

---

## 🛠 Troubleshooting

- **Page reruns while typing** → Use the **Apply** buttons; do not expect instant updates.
- **Wrong month order** → Month names must match exactly; the app enforces the order.
- **Benchmark not found** → Add `templates/benchmark_template.xlsx` or upload via **Benchmark Settings**.
- **PV positive** → To subtract from demand, PV should be **negative** in `Energy_Balance`.
- **Green Electricity missing** → Ensure it exists in Emission Factors & Tariffs if you preload from a workbook.
- **Save Project disabled** → You must upload a workbook first.

---

## 🗂 Project Structure (suggested)

```
.
├── BPVis_lt_1.1.3_optimized.py
├── requirements.txt
├── templates/
│   └── benchmark_template.xlsx
├── data/                     # optional: place your workbooks here
└── README.md
```

---

## 🧾 Changelog

### v1.1.3 (Optimized)
- Input **forms** (Apply buttons) + **session_state** → no reruns on every keystroke.
- **Caching** for Excel I/O and transformations.
- **Benchmark** tab with gauge + vertical “label” charts; sheet-per-building-use benchmark file.
- **Save Project** writes/updates `Project_Data`, `Emission_Factors`, `Energy_Tariffs`, `EndUse_to_Source`.
- **Green Electricity** included across factors, tariffs, mapping, and charts.
- All visuals, KPIs, and color schemes retained.

---

## 📬 Feedback / Ideas

Open an issue or PR with improvements. Common asks: new building uses in the benchmark, exporting images, custom color palettes, or additional KPIs.
