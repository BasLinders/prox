# Getting Started with Process Miner Pro

This guide covers everything you need to set up, configure, and run the **Process Miner Pro** tool.

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

Install all required libraries (Pandas, PM4Py, Cython, etc.):

```bash
pip install -r requirements.txt

```

### 3. Activate the Cython Module (Critical Step)

This tool uses Cython to accelerate heavy calculations (fitness & alignments) by compiling Python code into C. You must perform this step before running the tool, or you will encounter import errors.

Run this command in the project root folder:

```bash
python setup.py build_ext --inplace

```

**Verification:**

* If successful, you will see a new file in your directory ending in `.so` (Linux/Mac) or `.pyd` (Windows), such as `conformance.cp39-win_amd64.pyd`.
* You only need to run this command once (or whenever you modify `conformance.pyx`).

---

## Data Setup

1. **Export Data:** Export your event log as a CSV file.
2. **Placement:** Place the CSV file inside the project folder (e.g., named `input_data.csv`).
3. **Update Path:** Open `main.py` and update the `uploaded_file` variable to match your filename:

```python
# main.py
uploaded_file = 'input_data.csv' 

```

4. **Required Columns:** Your CSV must contain at least these three columns:
* **Case ID:** (e.g., Session ID, Order ID)
* **Activity Name:** (e.g., Page View, Button Click)
* **Timestamp:** (e.g., 2023-10-27 14:30:00)

---

## Configuration

All settings are managed in `config.py`. You do not need to change code logic to adapt the tool to new datasets.

### 1. Column Mappings

Map your CSV column headers to the tool's standard names in `COLUMN_MAPPINGS`.

```python
# config.py
COLUMN_MAPPINGS = {
    'case:concept:name': frozenset(['session_id', 'order_id', 'trace_id']),  # Your Case ID column
    'concept:name': frozenset(['event_name', 'activity', 'action']),         # Your Activity column
    'time:timestamp': frozenset(['timestamp', 'created_at', 'datetime'])     # Your Timestamp column
}

```

### 2. Analysis Parameters

Tune performance and depth in the `CONFIG` dictionary:

* **speed_params:**
* `max_align`: Set lower (e.g., 15) for faster results.

* **sampling_config:**
* `strata_col`: Set this to a column (e.g., `'has_purchase'`) to ensure rare cases are included.

---

## Running the Tool

Once setup is complete, run the analysis:

```bash
python main.py

```

### Output Locations

* **Console:** Displays real-time analysis logs and fitness scores.
* **`output/` Folder:** Contains generated artifacts:
* `process_model.png`: The visual Petri Net.
* `happy_path_model.png`: BPMN of the most frequent variant.
* `process_variant_analysis.csv`: Detailed statistics on process paths.

---

## Troubleshooting

| Issue | Cause | Solution |
| --- | --- | --- |
| **ModuleNotFoundError:** No module named 'conformance' | Cython module hasn't been compiled. | Run `python setup.py build_ext --inplace`. |
| **ExecutableNotFound:** failed to execute `dot` | GraphViz is missing or not in PATH. | Install GraphViz and add the `/bin` folder to your System PATH. |
| **System Freeze / Memory Error** | Dataset is too large for RAM. | Reduce `total_sample_size` in `config.py`. |
| **Visual C++ Build Tools Missing** | Windows requires a C compiler for Cython. | Install "Desktop development with C++" via Visual Studio Build Tools. |
