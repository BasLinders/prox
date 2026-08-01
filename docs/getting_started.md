# Getting Started with PRoX

This guide covers everything you need to set up, configure, and run **PRoX (Process Excavator)**.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.9+** ([Download](https://www.python.org/downloads/))
2.  **GraphViz** (Required for generating process maps)
    * **Windows:** [Download Installer](https://graphviz.org/download/). Run it and **select "Add GraphViz to the system PATH for all users"** during installation.
    * **Mac:** Run `brew install graphviz` in your terminal.

---

## Installation

### 1. Set up a Virtual Environment (Recommended)
It is best practice to run this tool in a clean environment to avoid conflicts.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

Install all required libraries (pandas, numpy, matplotlib, seaborn, pm4py, streamlit):

```bash
pip install -r requirements.txt
```

No compilation step is needed — PRoX is pure Python.

---

## Running the Tool

Start the Streamlit app:

```bash
streamlit run main.py
```

This opens PRoX in your browser. From there:

1. **Upload** a CSV event log using the sidebar file uploader.
2. **Configure** the discovery algorithm, noise threshold, conformance method, and sample size in the sidebar.
3. Click **Run Analysis**.
4. Explore results across seven tabs: Process Maps, Variants, Bottlenecks, Conformance, Funnel, Business Insights, and Segment Comparison.

### Required Columns

Your CSV must contain at least these three columns (names are auto-detected, see `COLUMN_MAPPINGS` in `prox/config.py`):

* **Case ID:** (e.g., `session_id`, `case_id`, `trace_id`)
* **Activity Name:** (e.g., `event_name`, `activity`, `action`)
* **Timestamp:** (e.g., `timestamp`, `created_at`, `datetime`)

Optional columns (`price`/`revenue`, `purchase`/`transaction`, `add_to_cart`, `page_type`, `category`) unlock additional analytics — see `README.md` for the full table.

---

## Configuration

All defaults are managed in `prox/config.py`. You do not need to change code logic to adapt the tool to new datasets — the sidebar controls the most common settings, and `create_analysis_config()` exposes the rest for scripted use.

### 1. Column Mappings

Map your CSV column headers to PRoX's standard names in `COLUMN_MAPPINGS`:

```python
# prox/config.py
COLUMN_MAPPINGS = {
    'case:concept:name': frozenset(['session_id', 'case_id', 'trace_id', ...]),
    'concept:name': frozenset(['event_name', 'activity', 'action', ...]),
    'time:timestamp': frozenset(['timestamp', 'created_at', 'datetime', ...]),
}
```

### 2. Analysis Parameters

Tune performance and depth via `create_analysis_config()` or by editing `CONFIG` directly:

* **`sample_size`** — cases used for conformance checking. Lower (e.g. 100) for faster results.
* **`strata_col`** — set to a column (e.g. `'purchase'`) to ensure rare cases are included when sampling.
* **`filter_steps`** — list of filter dicts applied before discovery (see `README.md` for the full spec).

---

## Output Locations

* **Browser (Streamlit):** All results — metrics, tables, and diagrams — are displayed live in the app tabs.
* **`output/` folder:** Generated artifacts referenced by the UI, including process map and happy-path PNGs.

---

## Troubleshooting

| Issue | Cause | Solution |
| --- | --- | --- |
| **ExecutableNotFound:** failed to execute `dot` | GraphViz is missing or not in PATH. | Install GraphViz and add the `/bin` folder to your System PATH. |
| **System Freeze / Memory Error** | Dataset is too large for RAM. | Reduce `sample_size` / `total_sample_size`, or lower the sidebar sample size. |
| **"Failed to load data"** | Required columns (Case ID, Activity, Timestamp) not found. | Check your CSV headers against `COLUMN_MAPPINGS` in `prox/config.py`, or rename them. |
| **No process map images shown** | GraphViz not installed/on PATH. | See above. |
