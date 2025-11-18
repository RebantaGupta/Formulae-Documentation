import os
import re
import difflib
import threading
import time
from pathlib import Path
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
PARAM_FILENAME_RE = re.compile(r"(?P<param>[A-Za-z0-9_]+)_(?P<value>-?\d+(?:\.\d+)?)(?:.*)?\.txt$", re.IGNORECASE)

# file extensions to ignore (explicitly do not open/parse these)
IGNORED_EXTS = {".mph"}

# Default column names (used when a header row isn't found and the column count matches)
DEFAULT_COL_NAMES = [
    "depth_eV",
    "minU_eV",
    "maxU_eV",
    "trap_x",
    "trap_y",
    "trap_z",
    "offset_mm",
    "P_est_mW",
]

# Default parameter values (from provided table/image). Used when selected X-axis parameter
# does not match the parameter extracted from filenames.
DEFAULT_PARAM_VALUES = {
    "rod_radius": 0.002,        # 2 mm
    "rod_length": 0.04,         # 40 mm
    "rod_spacing": 0.005,       # 5 mm
    "V_rf": 300.0,              # 300 V
    "V_dc": 50.0,               # 50 V
    "endcap_rad": 0.006,        # 6 mm
    "endcap_thickness": 0.0005, # 0.5 mm
    "endcap_outer": 0.001,      # 1 mm (if present)
    "V_endcap": 10.0,           # 10 V
    "f": 1e7,                   # 10 MHz
}

# Display unit and scale mapping for parameters (for nicer axis labels)
PARAM_DISPLAY_UNITS = {
    "rod_radius": "mm",
    "rod_length": "mm",
    "rod_spacing": "mm",
    "V_rf": "V",
    "V_dc": "V",
    "endcap_rad": "mm",
    "endcap_thickness": "mm",
    "endcap_outer": "mm",
    "V_endcap": "V",
    "f": "MHz",
}
PARAM_DISPLAY_SCALE = {
    "rod_radius": 1e3,  # m -> mm
    "rod_length": 1e3,
    "rod_spacing": 1e3,
    "V_rf": 1.0,
    "V_dc": 1.0,
    "endcap_rad": 1e3,
    "endcap_thickness": 1e3,
    "endcap_outer": 1e3,
    "V_endcap": 1.0,
    "f": 1e-6,  # Hz -> MHz
}

# Column units mapping for y-axis labeling
COLUMN_UNITS = {
    "depth_eV": "eV",
    "minU_eV": "eV",
    "maxU_eV": "eV",
    "trap_x": "m",
    "trap_y": "m",
    "trap_z": "m",
    "offset_mm": "mm",
    "P_est_mW": "mW",
}


def format_xlabel(param_name: str):
    """Return a nicely formatted x-axis label including units when available."""
    if not param_name:
        return "parameter"
    unit = PARAM_DISPLAY_UNITS.get(param_name, "")
    if unit:
        return f"{param_name} ({unit})"
    return str(param_name)


def find_param_files(folder: Path):
    results = {}
    unmatched = []
    for p in folder.glob("*.txt"):
        m = PARAM_FILENAME_RE.match(p.name)
        if m:
            param = m.group("param")
            try:
                val = float(m.group("value"))
            except Exception:
                # if value cannot be parsed, treat as unmatched so user can inspect
                unmatched.append(p.name)
                continue
            results.setdefault(param, []).append((val, p))
        else:
            unmatched.append(p.name)
    # sort value lists
    for k in results:
        results[k] = sorted(results[k], key=lambda x: x[0])
    return results, unmatched


def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return "(unable to read file)"


def simple_metrics(text: str):
    lines = text.splitlines()
    num_lines = len(lines)
    num_chars = len(text)
    # extract numbers in file for a rough numeric summary
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    nums = [float(n) for n in nums] if nums else []
    mean_num = sum(nums) / len(nums) if nums else None
    return {"lines": num_lines, "chars": num_chars, "mean_number": mean_num}


def extract_numeric_table(text: str):
    """Try to find the largest contiguous block of lines that look like a numeric table.
    Returns (DataFrame, (start_line, end_line)) or (None, None) if no table found.
    """
    lines = text.splitlines()
    # tokenize helper: split by commas or whitespace
    def tokenize(line):
        # replace commas with space then split
        return re.split(r"[\t,\s]+", line.strip()) if line.strip() else []

    is_numeric_token = lambda tok: re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", tok) is not None

    blocks = []  # list of (start, end) indices inclusive
    cur_start = None
    for i, line in enumerate(lines):
        toks = tokenize(line)
        if not toks:
            if cur_start is not None:
                blocks.append((cur_start, i - 1))
                cur_start = None
            continue
        # consider this line part of a numeric table row if at least half tokens are numeric
        num_toks = len(toks)
        if num_toks == 0:
            if cur_start is not None:
                blocks.append((cur_start, i - 1))
                cur_start = None
            continue
        numeric_count = sum(1 for t in toks if is_numeric_token(t))
        if numeric_count >= max(1, num_toks // 2):
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                blocks.append((cur_start, i - 1))
                cur_start = None
    if cur_start is not None:
        blocks.append((cur_start, len(lines) - 1))

    if not blocks:
        return None, None

    # pick the longest block
    best = max(blocks, key=lambda x: x[1] - x[0])
    s, e = best
    rows = []
    for i in range(s, e + 1):
        toks = [t for t in tokenize(lines[i]) if t != ""]
        rows.append(toks)

    # find max columns and normalize row lengths by padding with NaN
    max_cols = max(len(r) for r in rows) if rows else 0
    norm_rows = [r + [""] * (max_cols - len(r)) for r in rows]

    # decide if first row is header: if first row has any non-numeric tokens while most subsequent rows are numeric
    first_row = norm_rows[0] if norm_rows else []
    rest_rows = norm_rows[1:]
    def row_numeric_frac(row):
        cnt = sum(1 for t in row if is_numeric_token(t))
        return cnt / max(1, len(row))

    first_frac = row_numeric_frac(first_row)
    rest_fracs = [row_numeric_frac(r) for r in rest_rows] if rest_rows else []
    header = False
    if rest_fracs and first_frac < 0.5 and (sum(1 for f in rest_fracs if f > 0.5) >= max(1, len(rest_fracs) // 2)):
        header = True

    import numpy as np
    data = []
    for r in (norm_rows[1:] if header else norm_rows):
        row_vals = []
        for tok in r:
            if is_numeric_token(tok):
                try:
                    row_vals.append(float(tok))
                except Exception:
                    row_vals.append(np.nan)
            else:
                row_vals.append(np.nan)
        data.append(row_vals)

    cols = []
    if header:
        cols = [c if c != "" else f"col{i}" for i, c in enumerate(first_row)]
    else:
        # if table has the same number of columns as our default names, use those
        if max_cols == len(DEFAULT_COL_NAMES):
            cols = DEFAULT_COL_NAMES[:max_cols]
        else:
            cols = [f"col{i}" for i in range(max_cols)]

    try:
        df = pd.DataFrame(data, columns=cols)
        return df, (s + 1, e + 1)  # 1-based line numbers
    except Exception:
        return None, None


st.set_page_config(page_title="Parameter-file comparer", layout="wide")
st.title("Parameter-based Text File Comparer")

st.sidebar.header("Settings")
# Theme / appearance controls
theme_choice = st.sidebar.selectbox("Color scheme", options=["Streamlit default", "Dark", "Light", "Custom"], index=1, key="theme_choice")
accent_color = "#1f77b4"
bg_color = "#0e1117" if theme_choice == "Dark" else "#ffffff"
text_color = "#ffffff" if theme_choice == "Dark" else "#111111"
sidebar_bg = "#0b0d10" if theme_choice == "Dark" else "#f5f5f5"
sidebar_text = text_color
if theme_choice == "Custom":
    accent_color = st.sidebar.color_picker("Accent color", value="#1f77b4", key="accent_color")
    bg_color = st.sidebar.color_picker("Background color", value="#ffffff", key="bg_color")
    text_color = st.sidebar.color_picker("Text color", value="#111111", key="text_color")
    sidebar_bg = st.sidebar.color_picker("Sidebar background", value="#f5f5f5", key="sidebar_bg")
    sidebar_text = st.sidebar.color_picker("Sidebar text color", value="#111111", key="sidebar_text")

_css = f"""
<style>
body {{ background-color: {bg_color}; color: {text_color}; }}
.stApp, .reportview-container, .main, .block-container {{ background-color: {bg_color} !important; color: {text_color} !important; }}
.sidebar .sidebar-content {{ background-color: {sidebar_bg} !important; color: {sidebar_text} !important; }}
.stButton>button, .stCheckbox>div, .stSelectbox>div {{ color: {accent_color} !important; }}
/* Center common Streamlit headings */
h1, h2, h3, h4 {{ text-align: center !important; }}
.sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {{ text-align: center !important; }}
.streamlit-expanderHeader {{ text-align: center !important; }}
/* Top tab styling: make tabs larger, centered, and give active tab a colored background */
div[role="tablist"] > button {{
    font-size: 16px !important;
    padding: 8px 18px !important;
    margin: 0 6px !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: {text_color} !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    transition: background-color 220ms ease, transform 160ms ease, box-shadow 220ms ease;
    will-change: transform, background-color;
}}
div[role="tablist"] > button:hover {{
    transform: translateY(-3px) scale(1.02);
    filter: brightness(1.06);
}}
div[role="tablist"] > button[aria-selected="true"] {{
    background: {accent_color} !important;
    color: #fff !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25) !important;
    transform: translateY(-2px) scale(1.02);
}}
/* make the tab container centered and visually separated */
div[role="tablist"] {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 6px 0 12px 0 !important;
}}
/* compact the sidebar caption so it doesn't draw attention */
.sidebar .stCaption {{ opacity: 0.8; font-size: 12px; }}
</style>
"""
st.markdown(_css, unsafe_allow_html=True)
# default to the V_rf_files subfolder if present/desired
default_folder = os.path.join(os.getcwd(), "V_rf_files")
base_folder = st.sidebar.text_input("Folder to scan", value=default_folder)
folder = Path(base_folder)

# Live watch controls
live_watch = st.sidebar.checkbox("Enable live file watching (auto-refresh on changes)", value=False)
if live_watch and not WATCHDOG_AVAILABLE:
    st.sidebar.warning("`watchdog` package is not installed. Install from `requirements.txt` and restart the app to enable live watching.")

if not folder.exists():
    st.error(f"Folder does not exist: {folder}")
    st.stop()

# initial scan of the configured folder to discover any parameter files
base_param_files, base_unmatched_files = find_param_files(folder)

# detect per-parameter subfolders named like `<param>_files` in the same parent directory
base_root = folder.parent if folder.name.lower().endswith("_files") else folder
param_subfolders = {}
try:
    for d in base_root.iterdir():
        if d.is_dir() and d.name.lower().endswith("_files"):
            pname = d.name[:-6]  # strip trailing '_files'
            param_subfolders[pname] = d
except Exception:
    param_subfolders = {}

# available params are those discovered either in the scanned folder or via subfolder names
available_params = sorted(set(list(base_param_files.keys()) + list(param_subfolders.keys())))

if not available_params:
    st.info("No files matching pattern like `V_rf_200.txt` found in the folder or subfolders.")
    st.write("Files found (.txt):")
    for p in folder.glob("*.txt"):
        st.write(p.name)
    if base_unmatched_files:
        st.write("Unmatched .txt filenames (these did not match the expected pattern):")
        for n in base_unmatched_files:
            st.write(f"- {n}")
    st.stop()

if base_unmatched_files:
    st.sidebar.warning(f"{len(base_unmatched_files)} .txt file(s) were found in the configured folder but did not match the `<param>_<value>.txt` pattern.")

# Live-watching trigger file (watcher writes timestamp here when a relevant change occurs)
trigger_file = folder / ".streamlit_watch_trigger"

# Start a watchdog observer in the background (one per Streamlit session) when requested
if live_watch and WATCHDOG_AVAILABLE:
    sess_flag = f"watcher_started_{str(folder)}"
    if not st.session_state.get(sess_flag, False):
        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return
                src_path = event.src_path if isinstance(event.src_path, str) else (event.src_path.decode('utf-8') if isinstance(event.src_path, bytes) else str(event.src_path))
                p = Path(src_path)
                if p.suffix.lower() in IGNORED_EXTS:
                    return
                # only react to text files (filename pattern detected by regex)
                if p.suffix.lower() != ".txt":
                    return
                try:
                    trigger_file.write_text(str(time.time()))
                except Exception:
                    pass

        def _start_observer():
            observer = Observer()
            observer.schedule(_Handler(), str(folder), recursive=False)
            observer.daemon = True
            observer.start()
            # keep thread alive
            try:
                while True:
                    time.sleep(1)
            except Exception:
                try:
                    observer.stop()
                except Exception:
                    pass

        t = threading.Thread(target=_start_observer, daemon=True)
        t.start()
        st.session_state[sess_flag] = True

# If trigger file doesn't exist, create it so we have a baseline timestamp
try:
    if not trigger_file.exists():
        trigger_file.write_text(str(time.time()))
except Exception:
    pass

# If live watch is enabled, compare trigger file mtime to the last seen value and rerun when changed
if live_watch:
    try:
        current_trigger_mtime = trigger_file.stat().st_mtime
    except Exception:
        current_trigger_mtime = None
    last_mtime_key = f"last_trigger_mtime_{str(folder)}"
    last = st.session_state.get(last_mtime_key)
    if last is None:
        st.session_state[last_mtime_key] = current_trigger_mtime
    else:
        if current_trigger_mtime is not None and current_trigger_mtime != last:
            st.session_state[last_mtime_key] = current_trigger_mtime
            st.rerun()

# detect ignored files (e.g., COMSOL .mph) and inform the user; we do not open or parse them
mph_files = list(folder.glob("*.mph"))
if mph_files:
    st.sidebar.info(f"Found {len(mph_files)} `.mph` file(s) in the folder — these are ignored by the app.")

if not base_param_files:
    st.info("No files matching pattern like `V_rf_200.txt` found in the configured folder.")
    st.write("Files found (.txt):")
    for p in folder.glob("*.txt"):
        st.write(p.name)
    if base_unmatched_files:
        st.write("Unmatched .txt filenames (these did not match the expected pattern):")
        for n in base_unmatched_files:
            st.write(f"- {n}")
    st.stop()

if base_unmatched_files:
    st.sidebar.warning(f"{len(base_unmatched_files)} .txt file(s) were found in the configured folder but did not match the `<param>_<value>.txt` pattern.")

# Parameter selection: choose from available params (detected via files or subfolders)
param = st.sidebar.selectbox("Parameter", available_params, key="param_select")

# allow toggling use of per-parameter subfolders (if present)
use_subfolders = False
if param_subfolders:
    use_subfolders = st.sidebar.checkbox("Use per-parameter subfolders when available", value=True)

# decide which folder to scan for the selected parameter
if use_subfolders and param in param_subfolders:
    scan_folder = param_subfolders[param]
else:
    scan_folder = folder

# perform the file discovery in the selected scan folder
param_files, unmatched_files = find_param_files(scan_folder)

# pick the matching param key inside the scanned folder (filenames may include the param name)
def _pick_param_key(pfiles, desired):
    if not pfiles:
        return None
    for k in pfiles.keys():
        if k.lower() == desired.lower():
            return k
    return next(iter(pfiles.keys()))

param_used = _pick_param_key(param_files, param)
if param_used is None:
    st.error(f"No parameter files found in {scan_folder}")
    st.stop()

# minimize scan-folder display: show a small caption and allow expanding only when requested
st.sidebar.caption(f"Scanning: `{param_used}`")
show_scan = st.sidebar.checkbox("Show scan folder details", value=False, key="show_scan_folder")
if show_scan:
    with st.sidebar.expander("Scan folder details", expanded=True):
        st.write(f"Folder: `{scan_folder}`")
        st.write(f"Parameter key used: `{param_used}`")
# Page tabs at the top: Home, Column-values and Fit
tab_home, tab_col, tab_fit = st.tabs(["Home", "Column values vs parameter", "Fit vs parameter"]) 
# allow the user to choose which parameter to measure by (X-axis)
# options: union of detected filename params and known default params
_x_options = sorted(set(list(param_files.keys()) + list(DEFAULT_PARAM_VALUES.keys())))
# default X selection should follow the selected parameter (param_used) when available
if param_used in _x_options:
    default_x = param_used
else:
    default_x = "V_rf" if "V_rf" in _x_options else _x_options[0]

# If the Parameter selection changed since the last run, force the X-axis selection
# to follow the newly selected parameter (this overrides any previous X selection).
prev_param = st.session_state.get("_last_param_selected")
if prev_param != param:
    if param_used in _x_options:
        st.session_state["x_param_select"] = param_used
    else:
        st.session_state["x_param_select"] = default_x

x_param = st.sidebar.selectbox("X-axis parameter", options=_x_options, index=_x_options.index(default_x), key="x_param_select")
# remember the currently selected parameter to detect changes on next run
st.session_state["_last_param_selected"] = param
# Optional override: allow using a single custom X value instead of per-file or default values
override_enabled = st.sidebar.checkbox("Use custom X-axis value (override)", value=False, key="x_override")
override_value = None
if override_enabled:
    # default the override input to the selected x_param default if available
    default_override = DEFAULT_PARAM_VALUES.get(x_param, 0.0) if x_param is not None else 0.0
    override_value = st.sidebar.number_input("Custom X-axis value", value=float(default_override), key="x_override_value", format="%.6g")
values = [v for v, _ in param_files[param_used]]
paths = [p for _, p in param_files[param_used]]

st.sidebar.write(f"Found {len(values)} values for parameter `{param_used}` (selected: `{param}`)")
st.sidebar.write(f"X-axis parameter: `{x_param}`")

with tab_home:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Select values to compare")
        selected_values = st.multiselect("Values (choose 2+)", options=values, default=values, format_func=lambda x: str(x))
        baseline_val = None
        if selected_values:
            baseline_val = st.selectbox("Baseline value (compare others to this)", options=selected_values, index=0, format_func=lambda x: str(x), key="baseline_select")
        # convenience: require at least two values to compare
        if not selected_values or len(selected_values) < 2:
            st.info("Select at least two values to enable comparison.")


    # map value -> path
    value_to_path = {v: p for v, p in param_files[param_used]}

    if selected_values and len(selected_values) >= 2:
        baseline_path = value_to_path[baseline_val]
        baseline_text = read_text(baseline_path)
        baseline_table, baseline_range = extract_numeric_table(baseline_text)
    else:
        baseline_path = None
        baseline_text = None
        baseline_table = None
        baseline_range = None

    # (table extraction for baseline and others is performed later per-file)

    # show Files and Unified Diff panel in Home tab (right column)
    with col2:
        st.subheader("Files and Unified Diff")
        if baseline_path is not None:
            st.markdown(f"**Baseline:** `{baseline_path.name}` — `{baseline_path}`")
        else:
            st.markdown("**Baseline:** (none selected)")


# --- Column vs Parameter plots (aggregate numeric table columns per file) ---
with tab_col:
    st.write("---")
    st.subheader("Column values vs parameter")

    # Build per-file column aggregates from parsed numeric tables
    col_values_rows = []
    all_found_cols = set()
    for v, p in param_files[param_used]:
        txt = read_text(p)
        tbl, rng = extract_numeric_table(txt)
        # determine x-axis value for this file: if user selected the same param as filenames, use the parsed value;
        # otherwise, fall back to a default value from DEFAULT_PARAM_VALUES (from provided image/table).
        if override_enabled and override_value is not None:
            xval = float(override_value)
        elif x_param is not None and param_used is not None and x_param.lower() == param_used.lower():
            xval = v
        else:
            # fallback to DEFAULT_PARAM_VALUES mapping if available, otherwise use the file's parsed value
            xval = float(DEFAULT_PARAM_VALUES[x_param]) if x_param in DEFAULT_PARAM_VALUES else v
        row = {"value": xval, "file": p.name, "file_param_value": v}
        if tbl is not None:
            for c in tbl.columns:
                all_found_cols.add(c)
                # compute default aggregate as mean
                try:
                    row[c] = tbl[c].mean(skipna=True)
                except Exception:
                    row[c] = None
        col_values_rows.append(row)

    col_values_df = pd.DataFrame(col_values_rows)

    # candidate columns: defaults first, then any found in tables
    candidate_cols = [c for c in DEFAULT_COL_NAMES if c in all_found_cols] + [c for c in sorted(all_found_cols) if c not in DEFAULT_COL_NAMES]
    if not candidate_cols:
        st.info("No numeric table columns detected across files to plot. Try files with numeric tables or adjust parsing settings.")
    else:
        agg_method = st.selectbox("Aggregation for multi-row tables", options=["mean", "first", "min", "max"], index=0, key="agg_method_select")
        if agg_method != "mean":
            # recompute according to selection
            for i, (v, p) in enumerate(param_files[param_used]):
                txt = read_text(p)
                tbl, rng = extract_numeric_table(txt)
                if tbl is None:
                    continue
                for c in tbl.columns:
                    try:
                        if agg_method == "first":
                            col_values_df.loc[col_values_df["file"] == p.name, c] = tbl[c].iloc[0]
                        elif agg_method == "min":
                            col_values_df.loc[col_values_df["file"] == p.name, c] = tbl[c].min(skipna=True)
                        elif agg_method == "max":
                            col_values_df.loc[col_values_df["file"] == p.name, c] = tbl[c].max(skipna=True)
                    except Exception:
                        pass

        # column selection
        default_select = []
        if "depth_eV" in candidate_cols:
            default_select.append("depth_eV")
        if "P_est_mW" in candidate_cols:
            default_select.append("P_est_mW")
        selected_plot_cols = st.multiselect("Columns to plot vs parameter", options=candidate_cols, default=default_select)
        combined_plot = st.checkbox("Plot selected columns on one combined graph", value=False)

        if selected_plot_cols:
            plot_df = col_values_df[["value", "file"] + selected_plot_cols].copy()
            # drop rows where all selected cols are NaN
            plot_df = plot_df.dropna(subset=selected_plot_cols, how="all")
            if plot_df.empty:
                st.info("No numeric values available for the selected columns.")
            else:
                # ensure rows sorted by parameter value for consistent plotting
                plot_df = plot_df.sort_values(by="value")
                xvals = plot_df["value"].to_numpy()
                # compute display scaling and units for selected X parameter
                display_scale = PARAM_DISPLAY_SCALE.get(x_param if x_param is not None else "", 1.0)
                display_unit = PARAM_DISPLAY_UNITS.get(x_param if x_param is not None else "", "")
                display_xvals = xvals * display_scale

                # compute simple stats: detect which columns change, slope vs parameter, and Pearson r
                def compute_stats(col_series, xarr):
                    y = col_series.to_numpy()
                    # drop nan pairs
                    mask = ~np.isnan(xarr) & ~np.isnan(y)
                    x = xarr[mask]
                    yy = y[mask]
                    if len(x) < 2:
                        return {
                            "changing": False,
                            "slope": float("nan"),
                            "r": float("nan"),
                        }
                    changing = not np.isclose(np.nanmax(yy), np.nanmin(yy))
                    # slope via linear fit
                    try:
                        slope, intercept = np.polyfit(x, yy, 1)
                    except Exception:
                        slope = float("nan")
                    # Pearson r
                    try:
                        if np.nanstd(x) == 0 or np.nanstd(yy) == 0:
                            r = float("nan")
                        else:
                            r = float(np.corrcoef(x, yy)[0, 1])
                    except Exception:
                        r = float("nan")
                    return {"changing": bool(changing), "slope": float(slope), "r": float(r)}

                stats = {}
                for c in selected_plot_cols:
                    if c in plot_df:
                        # compute stats using display-scaled x values so slopes/r reflect displayed axis
                        stats[c] = compute_stats(plot_df[c], display_xvals)
                    else:
                        stats[c] = {"changing": False, "slope": float("nan"), "r": float("nan")}

                # pairwise correlations among selected columns
                pairwise_r = {}
                for i, ca in enumerate(selected_plot_cols):
                    pairwise_r[ca] = {}
                    for j, cb in enumerate(selected_plot_cols):
                        if ca == cb:
                            continue
                        a = plot_df[ca].to_numpy()
                        b = plot_df[cb].to_numpy()
                        mask = ~np.isnan(a) & ~np.isnan(b)
                        if np.sum(mask) < 2:
                            rr = float("nan")
                        else:
                            try:
                                if np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
                                    rr = float("nan")
                                else:
                                    rr = float(np.corrcoef(a[mask], b[mask])[0, 1])
                            except Exception:
                                rr = float("nan")
                        pairwise_r[ca][cb] = rr

                # order columns: any-changing first, then by |r with parameter desc
                def sort_key(c):
                    s = stats.get(c, {})
                    changing = 1 if s.get("changing") else 0
                    r_abs = abs(s.get("r") or 0.0)
                    return (-int(changing), -r_abs)

                ordered_cols = sorted(selected_plot_cols, key=sort_key)

                # build a summary DataFrame and display as a table
                rows = []
                for c in ordered_cols:
                    s = stats.get(c, {})
                    slope = s.get("slope")
                    r = s.get("r")
                    # top correlated other columns
                    others = pairwise_r.get(c, {})
                    top_others = sorted(others.items(), key=lambda kv: -abs(kv[1]) if (kv[1] is not None and not np.isnan(kv[1])) else 0)[:3]
                    top_strs = []
                    for oc, rr in top_others:
                        if rr is None or np.isnan(rr):
                            continue
                        top_strs.append(f"{oc} (r={rr:.3f})")
                    topcorr = ", ".join(top_strs) if top_strs else "none"
                    rows.append({
                        "column": c,
                        "changing": bool(s.get("changing")),
                        "slope": (None if slope is None or np.isnan(slope) else float(slope)),
                        "r_vs_param": (None if r is None or np.isnan(r) else float(r)),
                        "top_correlations": topcorr,
                    })

                stats_df = pd.DataFrame(rows)
                # nicer ordering columns
                stats_df = stats_df[["column", "changing", "slope", "r_vs_param", "top_correlations"]]
                st.markdown("**Column change summary (table)**")
                try:
                    # style rows that have no change as muted/gray
                    def row_style(row):
                        return ["color: gray; opacity: 0.7;" if (not row["changing"]) else "" for _ in row.index]

                    st.dataframe(stats_df.style.format({"slope": "{:.3g}", "r_vs_param": "{:.3f}"}).apply(row_style, axis=1), height=200)
                except Exception:
                    st.dataframe(stats_df, height=200)

                # plotting: choose matplotlib style and palette according to theme
                style_ctx = 'dark_background' if theme_choice == 'Dark' else 'default'
                # build palette: use accent color as first color for custom theme, then tab10
                cmap = plt.get_cmap('tab10')
                palette = []
                for i in range(len(ordered_cols)):
                    if theme_choice == 'Custom' and i == 0:
                        palette.append(accent_color)
                    else:
                        palette.append(mcolors.to_hex(cmap(i % 10)))

                # use display-scaled x values for plotting and label axes with units
                xlabel = format_xlabel(x_param if x_param is not None else param_used)
                with plt.style.context(style_ctx):
                    if combined_plot:
                        fig3, ax3 = plt.subplots(figsize=(10, 5))
                        for i, c in enumerate(ordered_cols):
                            if c not in plot_df:
                                continue
                            s = stats.get(c, {})
                            color = palette[i]
                            unit = COLUMN_UNITS.get(c, "")
                            label_unit = f" ({unit})" if unit else ""
                            if not s.get("changing"):
                                ax3.plot(display_xvals, plot_df[c], marker="o", label=f"{c}{label_unit} (no change)", linewidth=2, color="gray", linestyle="--", alpha=0.5)
                            else:
                                ax3.plot(display_xvals, plot_df[c], marker="o", label=f"{c}{label_unit}", linewidth=2, color=color, alpha=0.95)
                        # ensure figure and axes use chosen background/text colors for custom theme
                        if theme_choice == 'Custom':
                            fig3.patch.set_facecolor(bg_color)
                            ax3.set_facecolor(bg_color)
                        ax3.set_xlabel(xlabel)
                        ax3.set_ylabel("Column value")
                        ax3.set_title(f"Selected columns vs {xlabel}")
                        ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                        ax3.grid(True, alpha=0.3)
                        fig3.tight_layout()
                        st.pyplot(fig3)
                    else:
                        n = len(ordered_cols)
                        fig4, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
                        if n == 1:
                            axes = [axes]
                        for i, c in enumerate(ordered_cols):
                            axc = axes[i]
                            s = stats.get(c, {})
                            color = palette[i]
                            unit = COLUMN_UNITS.get(c, "")
                            if c in plot_df:
                                if not s.get("changing"):
                                    axc.plot(display_xvals, plot_df[c], marker="o", linewidth=2, color="gray", linestyle="--", alpha=0.5)
                                else:
                                    axc.plot(display_xvals, plot_df[c], marker="o", linewidth=2, color=color, alpha=0.95)
                            ylabel = f"{c} ({unit})" if unit else c
                            axc.set_ylabel(ylabel)
                            title_suffix = " (no change)" if not s.get("changing") else ""
                            axc.set_title(f"{c} vs {xlabel}{title_suffix}")
                            if theme_choice == 'Custom':
                                fig4.patch.set_facecolor(bg_color)
                                axc.set_facecolor(bg_color)
                            axc.grid(True, alpha=0.3)
                        axes[-1].set_xlabel(xlabel)
                        fig4.tight_layout()
                        st.pyplot(fig4)

st.write("---")
st.caption("Pattern used to detect parameter files: `<param>_<value>.txt` (e.g., `V_rf_300.txt`). If you'd like different filename parsing or support for .mph files, tell me and I can extend this.")

# (removed separate metrics plot — only the Column values vs parameter plot is shown)


# --- Fit tab: scatter + best-fit line for a selected column vs parameter ---
with tab_fit:
    st.write("---")
    st.subheader("Line of best fit vs parameter")

    # Rebuild per-file aggregates (lightweight) — same logic as Column-values tab
    fit_rows = []
    fit_found_cols = set()
    for v, p in param_files[param_used]:
        txt = read_text(p)
        tbl, rng = extract_numeric_table(txt)
        if override_enabled and override_value is not None:
            xval = float(override_value)
        elif x_param is not None and param_used is not None and x_param.lower() == param_used.lower():
            xval = v
        else:
            xval = float(DEFAULT_PARAM_VALUES[x_param]) if x_param in DEFAULT_PARAM_VALUES else v
        row = {"value": xval, "file": p.name, "file_param_value": v}
        if tbl is not None:
            for c in tbl.columns:
                fit_found_cols.add(c)
                try:
                    row[c] = tbl[c].mean(skipna=True)
                except Exception:
                    row[c] = None
        fit_rows.append(row)

    fit_df = pd.DataFrame(fit_rows)
    fit_candidate_cols = [c for c in DEFAULT_COL_NAMES if c in fit_found_cols] + [c for c in sorted(fit_found_cols) if c not in DEFAULT_COL_NAMES]

    if not fit_candidate_cols:
        st.info("No numeric table columns detected across files to fit. Try files with numeric tables or adjust parsing settings.")
    else:
        default_fit = fit_candidate_cols[0] if fit_candidate_cols else None
        fit_col = st.selectbox("Column to fit", options=fit_candidate_cols, index=0, key="fit_col_select")

        # prepare x and y
        fd = fit_df.dropna(subset=[fit_col, "value"], how="any")
        if fd.empty:
            st.info("No data available for the selected column to fit.")
        else:
            fd = fd.sort_values(by="value")
            x_raw = fd["value"].to_numpy()
            display_scale = PARAM_DISPLAY_SCALE.get(x_param if x_param is not None else "", 1.0)
            display_unit = PARAM_DISPLAY_UNITS.get(x_param if x_param is not None else "", "")
            x = x_raw * display_scale
            y = fd[fit_col].to_numpy()

            # compute linear fit
            try:
                coef = np.polyfit(x, y, 1)
                slope, intercept = float(coef[0]), float(coef[1])
                y_fit = slope * x + intercept
            except Exception:
                slope, intercept = float('nan'), float('nan')
                y_fit = np.full_like(x, np.nan)

            # Pearson r
            try:
                if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                    r = float('nan')
                else:
                    r = float(np.corrcoef(x, y)[0, 1])
            except Exception:
                r = float('nan')

            # plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x, y, label='data', color=accent_color if theme_choice == 'Custom' else 'tab:blue')
            ax.plot(x, y_fit, label=f'fit: y={slope:.3g}x+{intercept:.3g}', color='red')
            ax.set_xlabel(format_xlabel(x_param if x_param is not None else param_used))
            unit = COLUMN_UNITS.get(fit_col, "")
            ax.set_ylabel(f"{fit_col} ({unit})" if unit else fit_col)
            ax.set_title(f"Best-fit for {fit_col} vs {x_param}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

            st.markdown(f"**Slope:** {slope:.6g} — **Intercept:** {intercept:.6g} — **Pearson r:** {('nan' if r is None or np.isnan(r) else f'{r:.3f}')} ")
