# PRoX - Process Excavator

A modular process mining tool for analysing customer journeys from any event log — not just e-commerce. Upload an event log, and PRoX automatically discovers the paths users take, identifies where they drop off or deviate, and surfaces business intelligence like repeat purchase rates, revenue lift, and funnel conversion. Built on [PM4Py](https://pm4py.fit.fraunhofer.de/), designed to run locally on a standard laptop.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Data Format](#input-data-format)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Metrics Reference](#metrics-reference)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [License](#license)

---

## Features

- **Golden Path Discovery** - Inductive Miner (sound, robust) and Heuristics Miner (noisy/large logs) produce Petri net models; DFG (Directly-Follows Graph) gives a fast first look. The most frequent variant is rendered as a "Happy Path" BPMN diagram.
- **Conformance Checking** - Token Replay for fast fitness and precision scores; State Equation A\* for exact per-trace deviations (skipped and unsolicited activities), optionally parallelized across CPU cores.
- **Bottleneck Analysis** - Activity and transition durations ranked by impact score; overall process health score.
- **Variant Analysis** - Top-20 variants with frequency, coverage, and duration statistics.
- **Business Insights** - Repeat buyer detection, inter-purchase timing, revenue multiplier (repeat vs. one-time buyers), average order value, cart abandonment rate, category-level revenue breakdown, and revenue-over-time trend.
- **Funnel Analysis** - Conversion/drop-off rate across any sequence of activities you define, for any process in any industry — or let PRoX auto-derive a likely order from the data as a starting point.
- **Segment Comparison** - Split the log by any low-cardinality column (e.g. device, traffic source) and compare health score, fitness, precision, repeat rate, and happy path side by side, optionally run in parallel across CPU cores.
- **Self-Contained HTML Reports** - Download a standalone report (no external dependencies, images embedded) for the full analysis or a segment comparison. Opens with a plain-language Executive Summary — health verdict, translated fitness/precision, biggest bottleneck, business/funnel highlights — for non-technical stakeholders, ahead of the detailed technical tables. Process-map diagrams are click-to-zoom.
- **Fast, Cached, and Local** - Chunked CSV loading for files up to 500 MB, categorical downcasting to reduce RAM usage, and Streamlit-layer caching so re-running with unchanged inputs is near-instant. No cloud dependency and no GPU required — everything runs on your machine.
- **Streamlit UI** - All results presented across seven tabs; no notebook required.

---

## Installation

### Prerequisites

- Python 3.9+
- [Graphviz](https://graphviz.org/download/) - required for BPMN and Petri net visualisations
  - **Windows:** run the installer and add Graphviz to your system PATH
  - **macOS:** `brew install graphviz`

### Install dependencies

```bash
pip install -r requirements.txt
```

No compilation step is needed. The Cython conformance module from earlier versions has been replaced with pure Python.

---

## Usage

```bash
streamlit run main.py
```

This opens the app in your browser. From there:

1. **Upload** a CSV event log using the sidebar file uploader.
2. **Configure** the discovery algorithm, noise threshold, conformance method, precision, CPU cores, and sample size in the sidebar.
3. Click **Run Analysis**.
4. Explore results across seven tabs: Process Maps, Variants, Bottlenecks, Conformance, Funnel, Business Insights, and Segment Comparison.
5. **Download** a self-contained HTML report from the top of the page, or a segment comparison report from the Segment Comparison tab.

### Using the engine directly

The `prox` package is fully importable for scripting or notebook use:

```python
from prox import load_and_validate_csv, create_analysis_config, run_full_analysis

df, messages, has_category = load_and_validate_csv(open("my_log.csv", "rb"))

config = create_analysis_config(
    discovery_algo="inductive_miner",
    noise_threshold=0.2,
    conformance_algo="token_replay",
    sample_size=250,
)

results = run_full_analysis(df, config)
```

`results` is a plain dict with keys: `log_summary`, `model`, `conformance`, `performance`, `visualizations`, `repeat_purchase_analysis`, `funnel_analysis`.

---

## Input Data Format

PRoX expects a CSV file with at least three columns:

| Column | Description | Accepted names |
|---|---|---|
| Case ID | Unique session or journey identifier | `session_id`, `case_id`, `trace_id`, `ga_session_id`, … |
| Activity | Event or action name | `event_name`, `activity`, `action`, `event`, … |
| Timestamp | When the event occurred | `timestamp`, `event_timestamp`, `created_at`, `datetime`, … |
| User ID | Customer identifier (for composite key) | `user_id`, `user_pseudo_id`, `customer_id`, … |

Column names are auto-detected, including Dutch aliases (`gebruikers id`, `tijdstempel`, etc.) — see `COLUMN_MAPPINGS` in `prox/config.py` for the full list. If both a user ID and a session ID are present, PRoX creates a composite case key (`user_id + session_id`) to correctly scope sessions per user.

Optional columns that unlock additional analytics:

| Column | Unlocks |
|---|---|
| `price` / `revenue` / `event_value` | Revenue metrics in Business Insights (AOV, revenue trend, value multiplier) |
| `purchase` / `transaction` | Repeat buyer detection, crop filter, cart-abandonment outcome |
| `add_to_cart` | Cart abandonment rate |
| `page_type` / `screen_class` | Automatic `page_view` label refinement |
| `category` | Category-level filters and revenue breakdown |

None of the above is e-commerce-only by requirement — the Funnel tab works on any sequence of activities in your log, whether or not any of these columns are present.

---

## Configuration

`create_analysis_config()` builds the config dict `run_full_analysis()` expects. Parameters are grouped below by the pipeline stage they affect; see `prox/config.py` for the full defaults.

### Discovery

| Parameter | Default | Description |
|---|---|---|
| `discovery_algo` | `"inductive_miner"` | `"inductive_miner"`, `"heuristics_miner"`, or `"dfg"` |
| `noise_threshold` | `0.2` | Inductive Miner noise filter (0.0-0.8). Higher = simpler model. |
| `dependency_threshold` | `0.9` | Heuristics Miner dependency strength threshold. |
| `activity_threshold` | `0` | Heuristics Miner: minimum activity occurrences for inclusion. |

### Conformance & sampling

| Parameter | Default | Description |
|---|---|---|
| `conformance_algo` | `"token_replay"` | `"token_replay"` (fast) or `"state_equation_a_star"` (per-trace deviations) |
| `calculate_precision` | `True` | Include precision score in conformance output |
| `cores` | `1` | CPU cores for alignment computation (State Equation A\* only). `0` = all available minus one. |
| `sample_size` | `250` | Cases used for conformance checking |
| `enable_sampling` | `True` | Stratified sampling to preserve rare events (e.g. purchases) |
| `strata_col` | `"purchase"` | Column used to prioritise rare cases when sampling |
| `max_priority_ratio` | `0.5` | Max share of the sample reserved for priority (strata) cases |

### Performance analysis

| Parameter | Default | Description |
|---|---|---|
| `time_unit` | `"minutes"` | Duration unit: `"seconds"`, `"minutes"`, `"hours"`, `"days"` |
| `bottleneck_threshold_percentile` | `75` | Percentile above which an activity/transition counts as a bottleneck |
| `bottleneck_top_k` | `50` | Max activities considered for the bottleneck visualisation |
| `max_bottleneck_edges` | `2` | Max bottleneck transitions highlighted per activity in the diagram |

### Business insights & funnel

| Parameter | Default | Description |
|---|---|---|
| `business_params` | see below | Dict override: `user_col`, `revenue_col`, `purchase_values`, plus optional `cart_values` and `funnel_steps` |
| `filter_steps` | see below | List of filter dicts applied before discovery |

Default `business_params`: `{"user_col": "user_id", "revenue_col": "event_value", "purchase_values": ["purchase", "has_purchase"]}`. Set `cart_values` (default `["add_to_cart", "add_to_basket"]`) to change cart-abandonment detection, or `funnel_steps` (default `None`, auto-derived from the data) to define an explicit, ordered funnel — the same option exposed interactively in the Funnel tab.

### Data loading

| Parameter | Default | Description |
|---|---|---|
| `max_file_size_mb` | `500` | Reject uploads larger than this |
| `chunk_threshold_mb` | `50` | Files above this size are loaded in chunks |
| `chunk_size` | `50000` | Rows per chunk when chunked loading is used |

### Filter steps

Filters are applied in order before the discovery stage. Each step is a dict with a `type` key:

```python
filter_steps=[
    # Remove noise events at the row level
    {"type": "activity", "activities": ["scroll", "session_start"], "mode": "remove_events"},

    # Keep only traces that reached a purchase, then take the top 10 variants
    {"type": "crop", "activity": ["purchase", "has_purchase"], "top_n": 10},
]
```

Available filter types: `activity`, `crop`, `case_duration`, `endpoints`, `attribute`, `top_variants`.

---

## Project Structure

```
prox/               Engine package — import this from any Python script
├── __init__.py     Public API
├── config.py       Column mappings and default configuration
├── data_manager.py CSV loading, cleaning, filtering, sampling
├── discovery.py    Process discovery (Inductive, Heuristics Miners, DFG)
├── conformance.py  Fitness, precision, alignment-based trace deviations
├── analytics.py    Performance metrics, bottlenecks, business insights, funnel analysis
├── visualizer.py   BPMN and Petri net diagram generation
├── report.py       Self-contained HTML report export
├── segments.py     Segment comparison — runs the pipeline per segment, optionally in parallel
└── pipeline.py     Orchestrator — runs all stages in sequence

main.py             Streamlit app (UI layer only)
utility/            Standalone UI-layer tools main.py wires in - BigQuery data
                    source (utility/bigquery_source.py) and the modular PDF
                    report builder (utility/pdf_builder.py)
tests/              pytest suite covering the prox/ engine
scripts/            Dev tooling (e.g. pipeline profiling), not part of the installable package
docs/               Reference docs (see Documentation below) and development status
output/             Generated PNGs and CSVs (created on first run)
```

---

## Metrics Reference

| Metric | What it means |
|---|---|
| **Fitness** | How much of the log the model can replay (0-1). Low fitness means many cases deviate. |
| **Precision** | How much the model allows behaviour not seen in the log (0-1). Low precision means an overly permissive model. |
| **Health Score** | Composite score (0-100) penalising high duration variability and a large bottleneck ratio. |
| **Repeat Rate** | Percentage of identified buyers who made more than one purchase. |
| **Value Multiplier** | Average lifetime revenue: repeat buyers ÷ one-time buyers. |
| **Average Order Value** | Mean revenue per completed purchase. |
| **Cart Abandonment Rate** | Percentage of add-to-cart sessions that didn't go on to purchase. |
| **Funnel Drop-off** | Percentage of cases that reached a funnel stage but didn't continue to the next one. |

---

## Limitations

- Precision calculation uses ETC Conformance Token (fast approximation). Full alignment-based precision is disabled due to memory cost on standard hardware.
- Visualisations require Graphviz to be installed and on the system PATH.
- Very large logs (>10 000 events after filtering) will trigger a warning; enabling sampling is recommended.
- Funnel auto-derivation is a heuristic aimed at typical, roughly-linear processes — pass an explicit funnel definition (via the Funnel tab, or `funnel_steps` in `business_params`) for a guaranteed-correct funnel, especially on logs with a lot of optional/noise activity.

---

## Documentation

- [`docs/getting_started.md`](docs/getting_started.md) - setup, running the app, troubleshooting.
- [`docs/docs.md`](docs/docs.md) - process mining concepts (fitness, precision, alignment) explained for non-technical readers.
- [`docs/dev_roadmap.md`](docs/dev_roadmap.md) - current development status and roadmap, for anyone tracking or contributing to PRoX.

---

## License
This repository is not licensed for use, modification, or distribution.
All rights reserved.
