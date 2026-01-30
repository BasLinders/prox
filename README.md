# Process Miner Pro

A modular, high-performance Process Mining tool built in Python. It leverages **PM4Py** for mining algorithms and **Cython** for accelerated conformance checking. Designed to handle medium-to-large datasets on standard laptops through memory optimization and chunked loading.

## Features

* **Automated Discovery:** Supports Inductive, Heuristics, and Alpha Miners.
* **High-Performance Conformance:** Custom Cython implementation for fast fitness and alignment calculation.
* **Business Analytics:** Automated detection of bottlenecks, repeat purchase rates, and process deviations.
* **Memory Efficient:** Optimized data types and chunked CSV loading for constrained environments.
* **Visual Insights:** Auto-generates BPMN models for "Happy Paths" and "Main Process Flows".

---

## Installation & Setup

### 1. Prerequisites
* Python 3.9+
* [GraphViz](https://graphviz.org/download/) (Required for visualizations)
    * *Windows:* Download installer, run it, and **add GraphViz to System PATH**.
    * *Mac:* `brew install graphviz`

### 2. Install Python Dependencies
Run the following command in your terminal:
```bash
pip install -r requirements.txt
```
 - Bottlenecks: **Active**. Calculated from timestamps in the dataframe.
 - Trace deviations: **Active**. Calculated by Alignments (A*). See exactly which steps were skipped.
 - Variant analysis: **Active**. Statistical count of paths.
 - Fitness score: **Active**. Derived/Back-calculated from the Alignment Cost.
 - Preliminary fitness score (Token replay): **Inactive**. Too expensive for up to 16GB of RAM. Alignment cost fitness is more accurate.
 - Precision score: **Inactive**. Too expensive for up to 16GB of RAM. The only metric truly lost.
