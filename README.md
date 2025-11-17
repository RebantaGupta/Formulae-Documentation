# Parameter-file comparer (Streamlit)

This Streamlit app scans a folder of text files named like `<param>_<value>.txt` (for example `V_rf_300.txt`), extracts the largest numeric table found in each file, and provides interactive visualizations showing how table columns change with experimental parameters.

Features added in this branch
- Parameter-file discovery: finds files matching the pattern `<param>_<value>.txt` and groups them by parameter.
- Numeric table extraction: heuristically finds the largest contiguous numeric block in a text file and parses it into a table (supports header detection when present).
- Column values vs parameter: aggregate per-file columns and plot them vs a chosen X-axis parameter (supports mean/first/min/max aggregation for multi-row tables).
- Fit vs parameter tab: scatter plot for a chosen column with a linear best-fit line, slope, intercept, and Pearson correlation (r).
- X-axis units: axis labels include units when available (e.g., `V_rf (V)`, `f (MHz)`).
- Live watch (optional): an optional `watchdog`-based auto-refresh when files change in the scanned folder (install `watchdog` to enable).
- Theme & customization: sidebar color scheme choices (Dark/Light/Custom) and color pickers for accent/background/text.
- Per-parameter subfolders: detects directories like `V_rf_files` and can scan them automatically.

Quick start (PowerShell)

1. Create or activate the workspace venv (the repo contains a venv at `\.venv` by default):

```powershell
.\.venv\Scripts\Activate.ps1  # may be blocked by PowerShell policy
# or run Streamlit directly with the venv python:
.\.venv\Scripts\python.exe -m streamlit run website\streamlit_app.py
```

2. Open http://localhost:8501 in your browser.

Notes
- The app ignores `.mph` files by design.
- If you enable live watching, install `watchdog` in the venv: `pip install watchdog`.
- The app uses heuristic parsing — if a file's table isn't detected, inspect the file or adjust file naming/header style.

Development & Git
- The new UI and features are implemented on branch `streamlit-updates`; this branch has been merged to `main` in this commit.
- To create a PR or view the branch on GitHub: https://github.com/jackfrostbob333/Hard_Nanos_HardHaq/pull/new/streamlit-updates

Contact
If you'd like further improvements (SI-prefix-aware tick formatting, robust regressions, CSV export of fit parameters, etc.), open an issue or tell me what to add.

# Parameter-based Text File Comparer (Streamlit)

This Streamlit app scans a folder for parameterized text files following the pattern `<param>_<value>.txt` (for example `V_rf_300.txt`) and helps you inspect how numeric table values inside those files change with the parameter.

## Features
- Scans a folder for files named like `PARAM_VALUE.txt` and groups them by `PARAM` sorted by numeric `VALUE`.
- Ignores `.mph` files completely (won't open or modify them).
- Extracts the largest contiguous numeric table from each text file (heuristic-based), converts it to a `pandas.DataFrame`, and assigns sensible column names when a header is missing.
- Default column name mapping (used when a header isn't found and 8 columns are present):
  - `depth_eV`, `minU_eV`, `maxU_eV`, `trap_x`, `trap_y`, `trap_z`, `offset_mm`, `P_est_mW`
- Multi-file comparisons: choose multiple `VALUE`s and a baseline `VALUE` to compare against.
- Unified text diffs and per-file CSV downloads for parsed tables.
- Column-wise plotting: aggregate table rows per file (mean/first/min/max) and plot selected columns versus the parameter.
- Column change summary table: shows whether each column changes across the selected values, fitted slope vs parameter, Pearson r vs parameter, and top correlations with other columns.
- Visual cues: plots use a dark background; series that show no change are dimmed and labeled `(no change)`.

## Run (Windows PowerShell)

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

## Usage notes
- In the app sidebar: set the folder to scan, enable live watching if desired, pick the parameter (e.g., `V_rf`), then select values to compare and the baseline.
- Use the "Columns to plot vs parameter" control to choose which numeric columns to plot. Aggregation controls choose how multi-row tables are reduced per-file (mean/first/min/max).
- The column summary table is shown above the plots and highlights which columns change with the parameter. Plots use a dark theme and annotate series with no change.

## Live watching

- Enable live watching from the sidebar. When enabled the app will monitor the chosen folder for `.txt` changes (matching the filename pattern) and auto-refresh. This requires the `watchdog` package (included in `requirements.txt`). The watcher ignores `.mph` files.

## Extensibility
- Table parsing is heuristic-based. If your files use special separators, multi-line headers, or other formats, I can extend the parser to handle them (or accept a user-provided parser).
- Optional improvements: align columns by header name across files, plot error bars (std dev), or export per-column change summaries as CSV.

If you want, I can update the README with screenshots or add a quick demo dataset and a single-command run script.
# Parameter-based Text File Comparer (Streamlit)

This small Streamlit app scans a folder for text files whose filenames follow the pattern `<param>_<value>.txt` (for example `V_rf_300.txt`) and provides:

- Side-by-side viewing of two selected files.
- A unified diff between them and a download button.
- Simple numeric metrics (line count, character count, mean of numbers found) plotted vs parameter value.
- Optional diffs of all files vs a selected baseline value.

## Run (Windows PowerShell)

1. Create a venv and activate it (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run Streamlit:

```powershell
streamlit run streamlit_app.py
```

4. In the Streamlit UI, set the folder to the folder containing your files (by default it uses the current working directory). The app looks for files like `V_rf_200.txt`, `V_rf_250.txt`, etc.

## Notes / Next steps

- Filenames must match the regex `(?P<param>[A-Za-z0-9]+)_(?P<value>-?\d+(?:\.\d+)?)\.txt`.
- I implemented simple metrics; if you want the app to parse `.mph` files or use richer domain-specific parsing, I can add that.
- The app intentionally ignores `.mph` files and will not open or modify them. If you need `.mph` parsing, I can add an opt-in feature.

## Live file watching

- The app can optionally enable a live file-watching mode (via the sidebar) which automatically refreshes the Streamlit UI when matching `.txt` files in the selected folder change.
- This feature uses the `watchdog` package. Install dependencies with `pip install -r requirements.txt` to ensure `watchdog` is available.
- The watcher ignores `.mph` files and will only react to `.txt` files that match the filename pattern.

Enable live watching from the sidebar; when a change is detected the app will auto-refresh.
