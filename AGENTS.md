# Repository Guidelines

## Project Structure & Module Organization

`BPVis.py` is the Streamlit application and contains UI, Excel parsing, scenario calculations, charts, authentication, and PDF reporting. Runtime dependencies are listed in `requirements.txt`. Excel schemas and defaults live in `templates/`; root-level `.xlsx` files are sample or reference data, while `External/` contains viewer-assigned project workbooks. Treat `backup/` as historical material, not active source. Branding assets are the `Pamo_Icon_*.png` and `WS_Logo.*` files. There is currently no automated `tests/` directory.

## Build, Test, and Development Commands

Use Python 3.13, as documented in `requirements.txt`, preferably in an isolated environment.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run BPVis.py
python -m py_compile BPVis.py
```

The first command installs dependencies, the second starts the local app, and the third performs a fast syntax check. Streamlit normally serves the app at `http://localhost:8501`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, `UPPER_CASE` for constants, and leading underscores for internal helpers. Add type hints to reusable calculation and parsing functions. Keep Streamlit widget keys stable and descriptive because session state and workbook reloads depend on them. Prefer small pure helpers for data transformations; isolate UI rendering and file I/O where practical. No formatter or linter is configured, so keep imports organized and follow PEP 8.

## Testing Guidelines

Until an automated suite is added, run `python -m py_compile BPVis.py` and manually exercise affected Streamlit flows. For workbook changes, test both provided templates, verify required sheets and columns, and confirm exported files reopen successfully. Pay special attention to scenario switching, saved configuration, negative PV values, authentication modes, charts, and PDF export. New automated tests should go in `tests/` and use names such as `test_energy_balance.py` with `test_<behavior>` functions.

## Commit & Pull Request Guidelines

Existing history mostly uses brief imperative messages such as `Update BPVis.py`; use a more informative equivalent, for example `Add scenario export validation`. Keep each commit focused and avoid committing credentials or generated local workbooks. Pull requests should summarize user-visible behavior, identify affected workbook sheets, list validation performed, and link relevant issues. Include screenshots for UI or chart changes and note any template or dependency changes.

## Security & Data Handling

`users.xlsx` and files under `External/` may contain credentials, assignments, or project data. Do not expose their contents in logs, screenshots, fixtures, or commits. Use anonymized workbooks when reproducing bugs.
