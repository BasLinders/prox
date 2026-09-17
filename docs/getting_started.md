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
It is best practice to run this tool in a clean environment to avoid conflicts. You can give the virtual environment any name.

**Windows:**
```bash
python -m venv .venv-prox
source .venv-prox\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv-prox
source .venv-prox/bin/activate
```

**Deactivating the virtual environment**
```bash
deactivate
```

### 2. Clone the repository to your disk
There's no install package for ProX yet. Both installation and updates have to be done manually for the time being. By default, the command below installs the program into C://prox. If you want to install it into a specific directory, navigate to it first in your Bash window.

**Install ProX**
```bash
git clone "https://github.com/BasLinders/prox.git"
```

### 3. Installing updates
Updates can be pulled down from the repository in a Bash window. Navigate to the ProX directory first, then execute the command below.

**Pulling updates down**
```bash
git pull
```


### 4. Install Python Dependencies

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

### Keeping the App Running During Long Analyses

PRoX runs as a local process, not a background service — if your machine goes to sleep, Windows/macOS suspends every process, including Streamlit, and the browser tab loses its connection. This is separate from screen lock/dimming, which doesn't affect background processes; it's full system sleep that does. Worth preventing this during a long BigQuery pull or a large event log merge:

* **Windows:** Install [PowerToys](https://learn.microsoft.com/en-us/windows/powertoys/) and enable the **Awake** module — set it to keep the machine awake indefinitely while PRoX is working (the screen can still turn off; only actual sleep kills the process). Without extra tools, go to Settings → System → Power & sleep and set "Sleep" to "Never" for the duration, or run `powercfg /change standby-timeout-ac 0` from a terminal (revert afterward, e.g. `powercfg /change standby-timeout-ac 30`).
* **Mac:** Prefix the run command with `caffeinate -i`, e.g. `caffeinate -i streamlit run main.py`. This keeps the Mac awake only for as long as that process runs, then lets it sleep normally again once you quit it.

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
| **Browser shows "Connection lost"** | Machine went to sleep, suspending the PRoX process. | See [Keeping the App Running During Long Analyses](#keeping-the-app-running-during-long-analyses) above. |
| **"Failed to load data"** | Required columns (Case ID, Activity, Timestamp) not found. | Check your CSV headers against `COLUMN_MAPPINGS` in `prox/config.py`, or rename them. |
| **No process map images shown** | GraphViz not installed/on PATH. | See above. |
