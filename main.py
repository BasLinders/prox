import io
import logging
import math
import os
import threading
import time

import pandas as pd
import streamlit as st

from utility.bigquery_source import render_bigquery_source
from utility.pdf_builder import render_pdf_builder
from prox import (
    load_and_validate_csv,
    refine_activity_labels,
    optimize_dataframe_memory,
    winsorize_series,
    create_analysis_config,
    run_full_analysis,
    format_business_report,
    generate_html_report,
    generate_segment_comparison_report,
    generate_reference_conformance_report,
    compare_segments,
    analyze_conversion_funnel,
    analyze_funnel_by_segment,
    check_data_quality,
    generate_mock_csv_bytes,
    filter_event_log,
    run_conformance_checking,
    build_structured_reference_model,
    import_reference_model_bpmn,
    diff_reference_model_coverage,
    render_petri_net,
    DISCOVERY_ALGORITHMS,
    CONFORMANCE_METHODS,
)

# Pre-selected as a starting point in the event-filter UI - GA4-style noise
# events that carry no process-mining signal. Only ones actually present in
# the uploaded log are pre-checked; the user can add/remove freely.
KNOWN_NOISE_ACTIVITIES = {
    "experience_impression", "view_cookie_bar", "javascript_error", "scroll",
    "view_item_list_empty", "user_engagement", "page_timestamp",
    "session_start", "first_visit",
}

def _analyzed_activities(raw_df: pd.DataFrame, config: dict) -> list:
    """Activity choices for standalone tabs (Funnel, Reference Model): applies
    the same event filter used in the main analysis to raw_df first, so
    activities removed as noise (e.g. session_start) don't reappear as
    selectable steps just because these tabs otherwise operate on raw_df."""
    df = raw_df
    for step_config in (config or {}).get("filter_steps") or []:
        params = step_config.copy()
        f_type = params.pop("type", None)
        if f_type is None:
            continue
        filtered_df, _ = filter_event_log(df, filter_type=f_type, **params)
        if filtered_df is not None and not filtered_df.empty:
            df = filtered_df
    return sorted(df["concept:name"].dropna().astype(str).unique().tolist())


def _describe_reference_stages(stages: list) -> str:
    """Renders an assembled reference-model stage list as a plain-English
    caption, e.g. 'a → (b or c) → [optional] d → e (repeatable)'."""
    def _describe(stage):
        acts = stage["activities"]
        t = stage["type"]
        if t == "optional":
            return f"[optional] {acts[0]}"
        if t == "repeatable":
            return f"{acts[0]} (repeatable)"
        if t == "choice":
            return "(" + " or ".join(acts) + ")"
        if t == "parallel":
            return "(" + " + ".join(acts) + ", any order)"
        return acts[0]

    return " → ".join(_describe(s) for s in stages)


# Above this many cases, running conformance checking unsampled (the default -
# see the Sampling step) risks becoming slow, especially with State Equation
# A*. 2,000 is well above PRoX's default sample size (250) - comfortably
# tractable on a laptop for most logs - while still being the point where
# unsampled exact alignments on a noisy/high-variant clickstream log start
# to noticeably drag, per profiling in docs/dev_optimization.md (conformance
# is the single biggest cost share even sampled at 250; discovery/viz already
# scale linearly with event count, so letting conformance scale too compounds
# that at real volume).
LARGE_CASE_COUNT_THRESHOLD = 2000

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

st.set_page_config(
    page_title="PRoX - Process Excavator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Shut down: stops the local server process for this session. Runs before
# anything else so the sidebar/results don't render once shutdown starts.
# ---------------------------------------------------------------------------
if st.session_state.get("shutdown_requested"):
    # Renders nothing - the page goes blank immediately, aside from CSS that
    # hides Streamlit's own "CONNECTING" status widget (it would otherwise
    # keep showing and retrying forever once the server dies below). Custom
    # component scripts are sandboxed against touching the top-level window,
    # but st.markdown renders straight into the main document, so a <style>
    # tag here isn't blocked the way <script> injection was. Killing the
    # server process is still what actually matters: once it's dead, this
    # tab can never reconnect to it - only a manually-restarted app produces
    # a live session again, which is the intended behaviour.
    st.markdown(
        "<style>[data-testid='stStatusWidget'] { display: none !important; }</style>",
        unsafe_allow_html=True,
    )
    if not st.session_state.get("_shutdown_timer_started"):
        st.session_state["_shutdown_timer_started"] = True
        threading.Timer(1.5, lambda: os._exit(0)).start()
    st.stop()


@st.cache_data(show_spinner=False)
def _cached_load_and_prepare(file_bytes, max_file_size_mb, chunk_threshold_mb, chunk_size, case_grouping):
    """Loads + validates the CSV and applies label refinement/memory optimization.
    Cached on file content and loader params so re-running with the same
    upload (e.g. only sidebar options changed) skips CSV parsing entirely.
    """
    df, messages, has_category = load_and_validate_csv(
        io.BytesIO(file_bytes),
        max_file_size_mb=max_file_size_mb,
        chunk_threshold_mb=chunk_threshold_mb,
        chunk_size=chunk_size,
        case_grouping=case_grouping,
    )
    if df is None:
        return None, None, messages, has_category

    df_ready = df.copy()
    for col in df_ready.select_dtypes(include=["category"]).columns:
        df_ready[col] = df_ready[col].astype("object")

    if "page_type" in df_ready.columns:
        df_ready = refine_activity_labels(df_ready, target_activity="page_view", context_column="page_type")

    optimize_dataframe_memory(df_ready)

    return df, df_ready, messages, has_category


@st.cache_resource(show_spinner=False)
def _cached_run_full_analysis(df_ready, config, output_folder="output"):
    """Runs the full pipeline. Cached on the input data + config, so re-running
    with identical settings (e.g. clicking Run Analysis again) is instant
    instead of redoing filtering, discovery, conformance, etc. from scratch.

    The progress bar is created *inside* this function - not passed in from
    the caller - because Streamlit's caching machinery records every element
    call made during a cache-missing run so it can replay them on later cache
    hits. A callback driving a bar created outside this function raises
    CacheReplayClosureError on replay, since that bar no longer exists by the
    time the hit happens; a bar whose whole lifecycle (create, update, empty)
    is recorded from inside replays cleanly - just near-instantly on a hit.

    The pipeline's own progress_callback only fires 6 times (once per major
    stage - see pipeline.py's _PROGRESS_STAGES), and one of those stages
    (Conformance, especially State Equation A*) can itself run for a long
    time with no callback in between - the bar would otherwise sit frozen
    for stretches at a time. So run_full_analysis executes on a background
    thread, while *this* (main) thread polls it in a sleep loop and drives
    the percentage itself from elapsed time - independent of the real
    callback, which only supplies the stage label text. That percentage is
    therefore a rough estimate, not a measurement: it climbs smoothly toward
    a 95% cap (via an asymptotic curve, so it visibly slows down rather than
    stalling outright on a longer-than-expected run) and only jumps to 100%
    once the background thread actually finishes. Only the main thread ever
    calls st.* here - the background thread just computes - since Streamlit
    element updates aren't safe to make from arbitrary threads.
    """
    progress_bar = st.progress(0, text="Starting analysis...")

    state = {"label": "Starting analysis...", "stage_num": 0, "total_stages": 0}
    state_lock = threading.Lock()

    def _update_progress(stage_num, total_stages, stage_label):
        with state_lock:
            state["label"] = stage_label
            state["stage_num"] = stage_num
            state["total_stages"] = total_stages

    outcome = {}

    def _run():
        try:
            outcome["results"] = run_full_analysis(
                df_ready, config=config, output_folder=output_folder, progress_callback=_update_progress
            )
        except Exception as e:
            outcome["error"] = e

    # Rough duration estimate to scale the ticking curve by dataset size -
    # not measured, just enough so a tiny log doesn't crawl and a huge one
    # doesn't rocket to the 95% cap in the first second.
    estimated_seconds = max(4.0, min(60.0, len(df_ready) / 3000))

    worker = threading.Thread(target=_run, daemon=True)
    start = time.monotonic()
    worker.start()
    while worker.is_alive():
        elapsed = time.monotonic() - start
        pct = min(1 - math.exp(-elapsed / estimated_seconds), 0.95)
        with state_lock:
            label, stage_num, total_stages = state["label"], state["stage_num"], state["total_stages"]
        stage_note = f" ({stage_num}/{total_stages})" if total_stages else ""
        progress_bar.progress(pct, text=f"{label}{stage_note} - {pct * 100:.0f}%")
        time.sleep(0.15)
    worker.join()

    if "error" in outcome:
        raise outcome["error"]

    progress_bar.progress(1.0, text="Done - 100%")
    progress_bar.empty()
    return outcome.get("results")

# ---------------------------------------------------------------------------
# Sidebar - configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("PRoX ⚙️")
    st.caption("Process Excavator")
    st.divider()

    st.header("Discovery")
    discovery_algo = st.selectbox(
        "Algorithm",
        list(DISCOVERY_ALGORITHMS.keys()),
        format_func=lambda key: DISCOVERY_ALGORITHMS[key]['label'],
        help="\n\n".join(
            f"**{v['label']}** - {v['help']}" for v in DISCOVERY_ALGORITHMS.values()
        )
    )
    noise_threshold = st.slider(
        "Noise Threshold", 0.0, 0.8, 0.2, 0.05,
        help="Higher values produce a simpler model by filtering rare paths. Used by Inductive Miner only.",
        disabled=(discovery_algo != "inductive_miner")
    )

    st.divider()
    st.header("Conformance")
    conformance_algo = st.selectbox(
        "Method",
        list(CONFORMANCE_METHODS.keys()),
        format_func=lambda key: CONFORMANCE_METHODS[key]['label'],
        help="\n\n".join(
            f"**{v['label']}** - {v['help']}" for v in CONFORMANCE_METHODS.values()
        )
    )
    calculate_precision = st.checkbox("Calculate Precision", value=True)

    cpu_count = os.cpu_count() or 1
    cores = st.number_input(
        "CPU Cores", min_value=0, max_value=cpu_count, value=1, step=1,
        help=(
            "Parallel alignment computation. 0 = use all available cores minus one. "
            "1 = sequential (default). Only used by State Equation A\\*, not Token Replay."
        ),
        disabled=(conformance_algo == "token_replay")
    )

    st.divider()
    if st.button("Clear Results", width='stretch'):
        for key in (
            "results", "df", "load_messages", "segment_result", "funnel_result",
            "funnel_segment_result", "reference_conformance_result",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    if st.session_state.get("confirm_shutdown"):
        st.warning("Shut down PRoX? This stops the local server and closes this tab.")
        cancel_col, confirm_col = st.columns(2)
        with cancel_col:
            if st.button("Cancel", width='stretch'):
                st.session_state["confirm_shutdown"] = False
                st.rerun()
        with confirm_col:
            if st.button("Confirm", type="primary", width='stretch'):
                st.session_state["shutdown_requested"] = True
                st.rerun()
    else:
        if st.button("Shut Down App", width='stretch'):
            st.session_state["confirm_shutdown"] = True
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Process Excavator")
st.caption("Upload a website event log to discover customer journeys and golden paths.")

with st.expander("What is process mining, and when should I use it?"):
    st.markdown(
        """
Process mining reconstructs the process *as it actually happened*, from an
event log - not the process as it's documented in a flowchart, and not just
an aggregate count of how often each step occurred. It needs three things
per event: which **case** it belongs to (an order, a session, a support
ticket, a claim), which **activity** occurred, and **when**. From that alone
it can reconstruct every individual path a case took, then aggregate those
paths into a process model.

That's the difference from two things people often reach for instead:

- **Process mapping** (whiteboard sessions, documented SOPs) describes the
  *intended* process. Process mining shows the process people and systems
  *actually execute* - which is very often different, and the gap itself is
  usually the interesting finding.
- **BI dashboards** report aggregate metrics (conversion rate, average order
  value) but throw away the *sequence* - they can tell you 30% of sessions
  converted, not that half of the drop-off happens at one specific step, or
  that a specific detour through `search` before `add_to_cart` correlates
  with a longer, less reliable path to purchase.

Process mining is generally framed as three activities, and PRoX covers all
three: **discovery** (what's the real process - Process Maps, Variants),
**conformance** (does reality match the intended process, and where does it
deviate - Conformance tab), and **enhancement** (how to improve it -
Bottlenecks, Funnel, Business Insights).

**Good fit:** you have timestamped events with a clear case concept, cases
typically involve several steps, and you want to see real paths and
bottlenecks rather than just aggregate counts - customer journeys, order
fulfillment, support ticket handling, loan/claims processing, CI/CD
pipelines.

**Poor fit:** single-step "events" with no real sequence, no reliable case
ID to group events by, too few cases to see a repeatable pattern, or a
question a simple funnel/BI report already answers just as well.
        """
    )

with st.expander("How the algorithms work: Miners, Token Replay, and State Equation A*"):
    st.markdown(
        """
PRoX's two pipeline stages - **discovery** (turning the log into a process
model) and **conformance** (measuring how well the log fits that model) -
each offer a choice of algorithm, trading speed against precision.

**Discovery: which "miner" builds the model**

- **Inductive Miner** (default) recursively splits the log into sub-logs by
  detected control-flow patterns (sequence, choice, parallelism, loop),
  building a process tree. It's the only one of the three that *guarantees*
  a sound model - no deadlocks, every case can always reach completion -
  which it achieves even on noisy data via the **Noise Threshold**: raising
  it filters out infrequent, non-representative behaviour before mining, at
  the cost of a simpler model that captures less of the log's real variety.
- **Heuristics Miner** builds a dependency graph from how often one activity
  is directly followed by another, using frequency/dependency thresholds to
  decide which of those direct-follows relationships are real causality
  versus coincidental noise. More tolerant of very large or very messy logs,
  but - unlike Inductive Miner - the resulting model isn't guaranteed sound.
- **DFG** (Directly-Follows Graph) is the simplest option: an edge between
  two activities whenever one is *ever* directly followed by the other,
  weighted by frequency. No AND/XOR split semantics, no soundness
  guarantee - it's a fast first look at the shape of the data, not a model
  precise enough for real conformance checking.

**Conformance: Token Replay vs. State Equation A\\***

- **Token Replay** simulates each trace by "playing tokens" through the
  discovered Petri net, consuming and producing tokens as each event
  occurs. When an event doesn't match a currently-enabled step, a token is
  borrowed and the gap is counted. Fitness is the fraction of tokens
  correctly accounted for. It's fast (near-linear in log size), but the
  diagnostics are approximate - it doesn't necessarily find the
  cheapest/most-likely explanation for a deviation.
- **State Equation A\\*** (alignments) frames conformance as a shortest-path
  search: for each trace, it finds the *provably optimal* alignment against
  the closest fully-compliant path through the model, using the A* search
  algorithm guided by a heuristic derived from the net's state equation (an
  LP relaxation used to estimate remaining cost and prune the search space).
  This gives exact fitness plus a precise list of skipped (the model
  expected it, it didn't happen) and unsolicited (it happened, the model
  didn't expect it) activities per case - the Conformance tab's per-case
  deviation table only exists in this mode. It's also the expensive one:
  every trace needs its own search, which is exactly why PRoX groups
  identical variants before aligning them, and why **Sampling** and
  **CPU Cores** exist as controls.
        """
    )

with st.expander("Data requirements and a SQL template to extract them"):
    st.markdown(
        """
PRoX auto-detects columns by name (English and Dutch aliases both work; see
`COLUMN_MAPPINGS` in `prox/config.py` for the full list), so exact column
naming isn't critical - what matters is that these concepts exist somewhere
in the file:

| Concept | Required? | Common names PRoX recognises |
|---|---|---|
| Case ID | Yes | `session_id`, `case_id`, `ga_session_id`, `trace_id` |
| Activity | Yes | `event_name`, `activity`, `action`, `event` |
| Timestamp | Yes | `timestamp`, `event_timestamp`, `created_at` |
| User ID | Yes (forms the composite case key) | `user_id`, `user_pseudo_id`, `customer_id` |
| Revenue | Optional - unlocks AOV, revenue trend | `price`, `revenue`, `event_value` |
| Purchase / conversion flag | Optional - unlocks repeat-buyer, cart abandonment | `purchase`, `transaction`, `conversion` |
| Category | Optional - unlocks category revenue breakdown | `category`, `product_category` |

If both a user ID and a session ID are present, PRoX groups cases by user by
default, so the same user's several sessions form one case - letting you see
their journey across sessions (e.g. a browsing session followed by a buying
session) in the Session Insights tab. Switch to per-session grouping below
if you want process discovery/conformance scoped to a single session
instead. Either way, session IDs are internally kept unique per user (so the
same session ID re-used by two different users never merges into one case).

**SQL template (GA4 BigQuery export)** - since PRoX's column aliases and the
mock-data generator are both modelled on GA4's event shape, this is the
most direct path from a GA4 property to a PRoX-ready CSV. Adjust the
project/dataset and date range, export the result, and upload it directly -
every output column below already matches a recognised alias.
        """
    )
    st.code(
        """\
-- PRoX-ready event export from a GA4 BigQuery export dataset.
-- Adjust `your_project.analytics_XXXXXX` and the date range below, then
-- export the query result to CSV and upload it directly - every column
-- here already matches a name PRoX auto-detects.

SELECT
  user_pseudo_id,
  (SELECT value.int_value FROM UNNEST(event_params)
   WHERE key = 'ga_session_id')                      AS ga_session_id,
  event_name,
  TIMESTAMP_MICROS(event_timestamp)                   AS event_timestamp,
  ecommerce.purchase_revenue                          AS price,
  (SELECT item_category FROM UNNEST(items) LIMIT 1)   AS category,
  IF(event_name = 'purchase', 1, 0)                   AS purchase,
  IF(event_name = 'add_to_cart', 1, 0)                AS add_to_cart
FROM
  `your_project.analytics_XXXXXX.events_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20260101' AND '20260131'
  AND user_pseudo_id IS NOT NULL
ORDER BY
  user_pseudo_id, ga_session_id, event_timestamp
""",
        language="sql",
    )

st.header("1. Load Data")
data_source = st.radio(
    "Data source", ["Upload CSV", "Connect to BigQuery"],
    horizontal=True, key="data_source_choice",
)

if data_source == "Connect to BigQuery":
    active_file_bytes = render_bigquery_source()
    if active_file_bytes is None:
        st.stop()
else:
    uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=["csv"])

    with st.expander("No data? Generate a mock event log"):
        st.caption(
            "Creates a synthetic e-commerce clickstream (funnel drop-off, "
            "repeat buyers, categories, revenue, device/traffic segments) "
            "so you can try PRoX without your own data."
        )
        mock_sessions = st.number_input(
            "Sessions", min_value=50, max_value=5000, value=400, step=50, key="mock_sessions"
        )
        mock_seed = st.number_input(
            "Seed", min_value=0, value=42, step=1, key="mock_seed",
            help="Same seed + session count always reproduces the same data."
        )
        if st.button("Generate Mock Data", width='stretch'):
            st.session_state["mock_csv_bytes"] = generate_mock_csv_bytes(
                n_sessions=int(mock_sessions), seed=int(mock_seed)
            )
            st.session_state["mock_csv_label"] = f"mock_event_log_{int(mock_sessions)}s_seed{int(mock_seed)}.csv"

        mock_csv_bytes = st.session_state.get("mock_csv_bytes")
        if mock_csv_bytes:
            st.success(f"Mock data ready: {st.session_state['mock_csv_label']}")
            dl_col, clear_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "Download CSV", data=mock_csv_bytes,
                    file_name=st.session_state["mock_csv_label"], mime="text/csv",
                    width='stretch',
                )
            with clear_col:
                if st.button("Clear", width='stretch'):
                    st.session_state.pop("mock_csv_bytes", None)
                    st.session_state.pop("mock_csv_label", None)
                    st.rerun()
            if uploaded_file is None:
                st.caption("Will be used for analysis (no file uploaded above).")
            else:
                st.caption("Uploaded file takes priority - remove it to use mock data instead.")

    active_file_bytes = uploaded_file.getvalue() if uploaded_file else st.session_state.get("mock_csv_bytes")

    if active_file_bytes is None:
        st.info("Upload a CSV event log above to get started, or generate a mock one.")
        st.stop()

case_grouping_label = st.radio(
    "Case grouping",
    ["By user (recommended)", "By session"],
    horizontal=True,
    help=(
        "By user: one case spans everything a user did across all their "
        "sessions - needed to see a user's sequence of sessions (e.g. a "
        "browsing session followed by a buying session) in the Session "
        "Insights tab. By session: one case per session, as before - use "
        "this if you want process discovery/conformance scoped to a single "
        "session instead of a user's full history."
    ),
)
case_grouping = "user" if case_grouping_label.startswith("By user") else "session"

loader_defaults = create_analysis_config()["data_loading"]
with st.spinner("Loading and validating data..."):
    raw_df, df_ready, load_messages, has_category = _cached_load_and_prepare(
        active_file_bytes,
        loader_defaults["max_file_size_mb"],
        loader_defaults["chunk_threshold_mb"],
        loader_defaults["chunk_size"],
        case_grouping,
    )

st.session_state["load_messages"] = load_messages

if raw_df is None:
    st.error("Failed to load data. See messages below.")
    for msg in load_messages:
        if "Critical" in msg or "Error" in msg:
            st.error(msg)
        else:
            st.warning(msg)
    st.stop()

# ---------------------------------------------------------------------------
# Winsorize revenue/price outliers - opt-in, applied right after the data is
# cached (both raw_df and df_ready) and before anything downstream reads
# 'price' (business insights' AOV/revenue trend/category breakdown, sampling
# strata, etc.), so a handful of extreme values don't dilute those reports.
# Caps values rather than dropping rows - see prox.winsorize_series.
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Handle Outliers")
if "price" not in raw_df.columns:
    st.caption("No revenue/price column detected - nothing to winsorize.")
else:
    winsorize_enabled = st.checkbox(
        "Winsorize Revenue/Price Outliers", value=False,
        help=(
            "Caps extreme values in the revenue/price column instead of "
            "removing those rows, so a handful of outlier orders don't "
            "dilute Average Order Value, revenue trend, or category "
            "revenue breakdown in Business Insights."
        )
    )
    if winsorize_enabled:
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            winsorize_method_label = st.radio(
                "Method", ["Standard Deviation", "Percentile"], horizontal=True,
                help=(
                    "Standard Deviation: caps at mean +/- N standard deviations. "
                    "Percentile: caps at the Nth/100-Nth percentile band."
                )
            )
        with w_col2:
            if winsorize_method_label == "Standard Deviation":
                winsorize_param = st.slider(
                    "Std deviations", 1.0, 5.0, 3.0, 0.5,
                    help="Values beyond mean +/- this many standard deviations are capped.",
                )
            else:
                winsorize_param = st.slider(
                    "Percentile cutoff", 0.5, 10.0, 1.0, 0.5,
                    help="Caps at this percentile and its mirror (e.g. 1 = 1st/99th percentile).",
                )

        winsorize_method = "std" if winsorize_method_label == "Standard Deviation" else "percentile"
        clipped, lower, upper = winsorize_series(raw_df["price"], method=winsorize_method, param=winsorize_param)
        n_capped = int(((raw_df["price"] < lower) | (raw_df["price"] > upper)).sum())

        raw_df = raw_df.copy()
        df_ready = df_ready.copy()
        raw_df["price"] = clipped
        df_ready["price"] = df_ready["price"].clip(lower, upper)

        if n_capped > 0:
            st.info(f"Capped {n_capped:,} value(s) to the range [{lower:,.2f}, {upper:,.2f}].")
        else:
            st.caption("No values fell outside the winsorization bounds - nothing was capped.")

# ---------------------------------------------------------------------------
# Data quality check - surfaced before filtering/analysis, so messy data is
# caught here instead of showing up as a confusing downstream result
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Data Quality Check")
data_quality = check_data_quality(raw_df)
if data_quality["issues"]:
    with st.expander(f"{len(data_quality['issues'])} data quality issue(s) found", expanded=True):
        for issue in data_quality["issues"]:
            st.warning(issue)
else:
    st.success("No data quality issues detected.")

# ---------------------------------------------------------------------------
# Filter events before analysis
# ---------------------------------------------------------------------------
st.divider()
st.header("4. Filter Events")
st.caption(
    "Remove noisy or irrelevant events before analysis, or narrow it down to "
    "just the events you care about. Optional - leave the list empty to "
    "analyse every event."
)

all_activities = sorted(raw_df["concept:name"].dropna().astype(str).unique().tolist())
default_noise_selection = [a for a in all_activities if a in KNOWN_NOISE_ACTIVITIES]

filter_col1, filter_col2 = st.columns([1, 2])
with filter_col1:
    filter_mode_label = st.radio(
        "Mode",
        ["Remove selected events", "Keep only selected events"],
        help=(
            "Remove: analyse everything except the events picked on the right. "
            "Keep: analyse only the events picked on the right."
        )
    )
with filter_col2:
    selected_events = st.multiselect(
        "Events",
        options=all_activities,
        default=default_noise_selection,
        help=(
            "Pre-checked with common non-process noise events (cookie banners, "
            "scroll, JS errors, etc.) found in this log - add or remove freely."
        )
    )

filter_steps = []
if selected_events:
    filter_mode = "remove_events" if filter_mode_label == "Remove selected events" else "keep_events"
    filter_steps.append({"type": "activity", "activities": selected_events, "mode": filter_mode})

default_purchase_activities = {"purchase", "has_purchase"}
endpoint_options = ["(no cropping - use full traces)"] + all_activities
default_endpoint = next(
    (a for a in all_activities if a.lower() in default_purchase_activities),
    endpoint_options[0],
)
endpoint_choice = st.selectbox(
    "Process end point",
    endpoint_options,
    index=endpoint_options.index(default_endpoint),
    help=(
        "Crops every case's trace at the first occurrence of this activity - "
        "later events in the same case are dropped, and cases that never "
        "reach it are removed entirely. Anchors the analysis to a specific "
        "outcome (e.g. purchase) instead of wherever the log happens to end. "
        "Pick '(no cropping)' to analyse full traces."
    ),
)
if endpoint_choice != endpoint_options[0]:
    filter_steps.append({"type": "crop", "activity": [endpoint_choice]})

preview_df = raw_df
for _step in filter_steps:
    _params = _step.copy()
    _f_type = _params.pop("type")
    _filtered, _ = filter_event_log(preview_df, filter_type=_f_type, **_params)
    if _filtered is not None and not _filtered.empty:
        preview_df = _filtered

post_cases = preview_df["case:concept:name"].nunique() if preview_df is not None and not preview_df.empty else 0
post_events = len(preview_df) if preview_df is not None else 0

st.caption(
    f"After filtering: **{post_cases:,} cases**, **{post_events:,} events** "
    f"(from {raw_df['case:concept:name'].nunique():,} cases, {len(raw_df):,} events)."
)

# ---------------------------------------------------------------------------
# Sampling - opt-in, with a warning above a "large" case-count threshold
# ---------------------------------------------------------------------------
st.divider()
st.header("5. Sampling")
enable_sampling = st.checkbox(
    "Enable Sampling", value=False,
    help=(
        "Off by default: conformance checking runs on every case. Turn this "
        "on to check only a representative subset instead, which is much "
        "faster on large logs."
    )
)
if enable_sampling:
    sample_size = st.number_input(
        "Sample Size (cases)", min_value=50, max_value=1000, value=250, step=50,
        help="Cases used for conformance. Higher = more accurate but slower."
    )

    # Stratification candidates: binary flag-style columns (e.g. 'purchase',
    # 'add_to_cart') where sample_log_stratified's priority_value=1 actually
    # means something - a non-binary or all-zero column would silently just
    # degrade to a random sample, so those aren't offered.
    exclude_cols = {"case:concept:name", "concept:name", "time:timestamp", "user_id", "session_id"}

    def _has_priority_value(series: pd.Series) -> bool:
        try:
            return (series == 1).any() or (series.astype(str).str.strip() == "1").any()
        except Exception:
            return False

    strata_candidates = [
        c for c in raw_df.columns
        if c not in exclude_cols and raw_df[c].nunique(dropna=True) == 2 and _has_priority_value(raw_df[c])
    ]

    strata_col1, strata_col2 = st.columns([2, 1])
    with strata_col1:
        strata_options = ["(none - plain random sample)"] + strata_candidates
        default_strata = next((c for c in strata_candidates if c.lower() in {"purchase", "has_purchase"}), strata_options[0])
        strata_choice = st.selectbox(
            "Prioritise a column when sampling",
            strata_options,
            index=strata_options.index(default_strata),
            help=(
                "Stratified sampling: reserves part of the sample for cases where "
                "this column = 1, so rare-but-important cases (e.g. purchases) "
                "aren't sampled away, instead of a plain random sample across all "
                "cases. Only binary (0/1-style) columns are offered here, since "
                "that's what stratification actually prioritises on."
            ),
        )
    with strata_col2:
        max_priority_ratio = st.slider(
            "Max priority share", 0.1, 1.0, 0.5, 0.05,
            help="Upper bound on how much of the sample can be priority cases.",
            disabled=(strata_choice == strata_options[0]),
        )
    # "(none)" needs an explicit non-purchase sentinel, not None/"" - those
    # are falsy, and run_conformance_checking's own fallback then re-checks
    # for a 'purchase' column regardless, silently reintroducing
    # stratification the user just opted out of. 'case:concept:name' is
    # unique per case, so priority-matching against it never hits - the same
    # sentinel that function already falls back to internally.
    strata_col = "case:concept:name" if strata_choice == strata_options[0] else strata_choice
else:
    sample_size = max(post_cases, 1)
    strata_col = "case:concept:name"
    max_priority_ratio = 0.5
    if post_cases > LARGE_CASE_COUNT_THRESHOLD:
        st.warning(
            f"{post_cases:,} cases will be analysed without sampling. Conformance "
            f"checking - especially State Equation A* - can get slow above "
            f"~{LARGE_CASE_COUNT_THRESHOLD:,} cases. Consider enabling sampling above, "
            "or switching to Token Replay in the sidebar."
        )

st.divider()
run_btn = st.button("Run Analysis", type="primary", width='stretch')

# ---------------------------------------------------------------------------
# Run analysis when button is pressed
# ---------------------------------------------------------------------------
if run_btn:
    config = create_analysis_config(
        discovery_algo=discovery_algo,
        noise_threshold=noise_threshold,
        conformance_algo=conformance_algo,
        calculate_precision=calculate_precision,
        sample_size=int(sample_size),
        cores=int(cores),
        enable_sampling=enable_sampling,
        strata_col=strata_col,
        max_priority_ratio=float(max_priority_ratio),
        filter_steps=filter_steps,
    )

    results = _cached_run_full_analysis(df_ready, config)

    if results is None:
        st.error("Analysis failed. Check the application logs for details.")
        st.stop()

    st.session_state["results"] = results
    st.session_state["df"] = raw_df
    st.session_state["config"] = config

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if not results:
    st.info("Configure the filter and sampling options above and click **Run Analysis**.")
    st.stop()

# Load messages expander
load_messages = st.session_state.get("load_messages", [])
if load_messages:
    with st.expander("Data loading messages", expanded=False):
        for msg in load_messages:
            st.text(msg)

# Top-level metrics strip
summary = results.get("log_summary", {})
if summary:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
    c1.metric("Cases", f"{summary.get('Number of Cases', 0):,}")
    c2.metric("Events", f"{summary.get('Number of Events', 0):,}")
    c3.metric("Activities", summary.get("Number of Unique Activities", 0))
    c4.metric("Duration (days)", summary.get("Total Duration (Days)", 0))
    with c5:
        st.download_button(
            "Download Full Report",
            data=generate_html_report(results),
            file_name="prox_report.html",
            mime="text/html",
            width='stretch',
        )

st.divider()

tab_map, tab_variants, tab_bottlenecks, tab_conf, tab_funnel, tab_biz, tab_sessions, tab_segments = st.tabs(
    [
        "Process Maps",
        "Variants",
        "Bottlenecks",
        "Conformance",
        "Funnel",
        "Business Insights",
        "Session Insights",
        "Segment Comparison",
    ],
    # on_change="rerun" makes the tabs a stateful widget: the active tab is
    # tracked via `key` and survives reruns triggered by other widgets (e.g.
    # clicking "Run Funnel Analysis" inside the Funnel tab). Without it,
    # tabs default to on_change="ignore", which doesn't track state at all -
    # every rerun snaps back to the first tab, regardless of what the user
    # was looking at. As a side effect, only the currently open tab's block
    # actually executes each rerun (lazy execution) instead of all seven -
    # each tab's content still renders correctly once you click into it.
    key="active_results_tab",
    on_change="rerun",
)

# ---------------------------------------------------------------------------
# Tab 1: Process Maps
# ---------------------------------------------------------------------------
with tab_map:
    viz = results.get("visualizations", {})
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Happy Path")
        st.caption("Most frequent variant - the intended customer journey.")
        hp = viz.get("happy_path")
        if hp and os.path.exists(hp):
            st.image(hp, width='stretch')
        else:
            st.info("Happy path image not available. Check that Graphviz is installed.")

    with col2:
        st.subheader("Main Process Flow")
        st.caption("Top-K variants combined, showing common deviations.")
        mf = viz.get("bottlenecks")
        if mf and os.path.exists(mf):
            st.image(mf, width='stretch')
        else:
            st.info("Main flow image not available.")

# ---------------------------------------------------------------------------
# Tab 2: Variants
# ---------------------------------------------------------------------------
with tab_variants:
    vp = results.get("performance", {}).get("variant_performance", {})
    if vp:
        total_v = vp.get("total_variants", 0)
        coverage = vp.get("variant_coverage", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Variants", total_v)
        c2.metric("Top 5 Coverage", f"{coverage.get('top_5_coverage', 0):.1f}%")
        c3.metric("Top 10 Coverage", f"{coverage.get('top_10_coverage', 0):.1f}%")

        top = vp.get("top_variants", {})
        if top:
            var_df = pd.DataFrame.from_dict(top, orient="index")
            var_df.index.name = "Variant"
            show_cols = [c for c in ["frequency", "percentage", "num_activities"] if c in var_df.columns]
            st.dataframe(var_df[show_cols], width='stretch')
    else:
        st.info("No variant data available.")

# ---------------------------------------------------------------------------
# Tab 3: Bottlenecks
# ---------------------------------------------------------------------------
with tab_bottlenecks:
    perf = results.get("performance", {})
    stats = perf.get("summary_statistics", {})
    case_stats = perf.get("case_performance", {}).get("duration_stats", {})
    time_unit = case_stats.get("unit", "")

    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Health Score", f"{stats.get('process_health_score', 0):.0f} / 100")
        c2.metric(f"Avg Lead Time ({time_unit})", f"{case_stats.get('mean', 0):.1f}")
        c3.metric(f"Median Lead Time ({time_unit})", f"{case_stats.get('median', 0):.1f}")
        c4.metric("Unique Variants", perf.get("variant_performance", {}).get("total_variants", 0))

    bns = perf.get("bottlenecks", {}).get("activity_bottlenecks", {})
    if bns:
        st.subheader("Activity Bottlenecks")
        bn_df = pd.DataFrame.from_dict(bns, orient="index")
        display_cols = [c for c in ["mean_duration", "frequency", "impact_score", "severity"] if c in bn_df.columns]
        st.dataframe(
            bn_df[display_cols].sort_values("impact_score", ascending=False),
            width='stretch'
        )
    else:
        st.info("No significant activity bottlenecks found.")

    max_bottleneck_edges = (
        st.session_state.get("config", {})
        .get("visualisation_params", {})
        .get("max_bottleneck_edges", 2)
    )
    tbns = perf.get("bottlenecks", {}).get("transition_bottlenecks", {})
    if tbns:
        st.subheader("Slowest Transitions")
        st.caption(f"Top {max_bottleneck_edges} step-to-step transitions by impact score.")
        tbn_df = pd.DataFrame.from_dict(tbns, orient="index")
        display_cols = [c for c in ["mean_duration", "frequency", "impact_score", "severity"] if c in tbn_df.columns]
        tbn_df.index.name = "Transition"
        st.dataframe(
            tbn_df[display_cols].sort_values("impact_score", ascending=False).head(max_bottleneck_edges),
            width='stretch'
        )

    res_metrics = perf.get("resource_performance", {}).get("resource_metrics", {})
    if res_metrics:
        st.subheader("Resource Performance")
        st.caption(
            "Who or what handled each step - detected from a 'resource', "
            "'org:resource', 'user', or 'operator' column. Processing time is "
            f"time since the previous event in the case ({time_unit})."
        )
        res_df = pd.DataFrame.from_dict(res_metrics, orient="index")
        res_df.index.name = "Resource"
        display_cols = [
            c for c in ["total_events", "unique_cases", "mean_proc_time", "median_proc_time"]
            if c in res_df.columns
        ]
        st.dataframe(
            res_df[display_cols].sort_values("total_events", ascending=False),
            width='stretch'
        )

    recs = stats.get("recommendations", [])
    if recs:
        st.subheader("Recommendations")
        for r in recs:
            st.write(f"- {r}")

# ---------------------------------------------------------------------------
# Tab 4: Conformance
# ---------------------------------------------------------------------------
with tab_conf:
    conf = results.get("conformance", {})
    overall = conf.get("overall_summary", {})

    if overall:
        c1, c2, c3 = st.columns(3)
        c1.metric("Fitness", f"{overall.get('fitness_score', 0):.1%}")
        c2.metric("Precision", f"{overall.get('precision_score', 0):.1%}")
        c3.metric("Quality", overall.get("quality_assessment", "N/A"))

    cases = conf.get("case_analysis", {}).get("cases", [])
    imperfect = sorted(
        [c for c in cases if c.get("fitness", 1.0) < 1.0],
        key=lambda x: x["fitness"]
    )

    if cases:
        st.caption(f"{len(imperfect)} deviant case(s) out of {len(cases)} sampled.")
        if imperfect:
            dev_rows = [
                {
                    "Case ID": c["case_id"],
                    "Fitness": f"{c['fitness']:.2%}",
                    "Skipped": ", ".join(c.get("deviations", {}).get("skipped", [])) or "-",
                    "Unsolicited": ", ".join(c.get("deviations", {}).get("unsolicited", [])) or "-"
                }
                for c in imperfect[:100]
            ]
            st.dataframe(pd.DataFrame(dev_rows), width='stretch')
        else:
            st.success("All sampled cases follow the model perfectly.")
    else:
        st.info(
            "No per-trace deviation data. "
            "Token Replay mode does not produce trace-level details - "
            "switch to State Equation A* for per-case deviations."
        )

    st.divider()
    st.subheader("Conformance vs. a Reference Model")
    st.caption(
        "The fitness/precision above measure self-consistency: how well a model "
        "*discovered from this same log* fits the log it came from. This checks "
        "something different - real behaviour against a model **you** define: "
        "the process as it's supposed to work, not the pattern PRoX found in "
        "what already happened."
    )

    ref_raw_df = st.session_state.get("df")
    if ref_raw_df is None or "concept:name" not in ref_raw_df.columns:
        st.info("Run an analysis first to enable reference-model conformance checking.")
    else:
        ref_mode = st.radio(
            "Reference model source",
            ["Define expected path", "Import a BPMN file"],
            horizontal=True,
            key="ref_mode",
        )

        ref_stages = []
        ref_uploaded_bpmn = None

        if ref_mode == "Define expected path":
            ref_activities = _analyzed_activities(ref_raw_df, st.session_state.get("config", {}))
            ref_selected_activities = st.multiselect(
                "Expected activities, in order",
                options=ref_activities,
                key="ref_selected_activities",
                help="Activities are added to the reference path in the order you select them."
            )

            if ref_selected_activities:
                st.caption("Configure each step:")
                skip_next = False
                for i, act in enumerate(ref_selected_activities):
                    if skip_next:
                        skip_next = False
                        continue

                    has_next = i + 1 < len(ref_selected_activities)
                    cols = st.columns([2, 2, 2, 2])
                    with cols[0]:
                        st.write(f"**{act}**")
                    with cols[1]:
                        step_type = st.selectbox(
                            "Type", ["Required", "Optional", "Repeatable"],
                            key=f"ref_type_{act}", label_visibility="collapsed"
                        )
                    with cols[2]:
                        combine_choice = st.checkbox(
                            "Choice with next", key=f"ref_choice_{act}", disabled=not has_next
                        )
                    with cols[3]:
                        combine_parallel = st.checkbox(
                            "Parallel with next", key=f"ref_parallel_{act}",
                            disabled=not has_next or combine_choice
                        )

                    if has_next and (combine_choice or combine_parallel):
                        next_act = ref_selected_activities[i + 1]
                        stage_type = "choice" if combine_choice else "parallel"
                        ref_stages.append({"activities": [act, next_act], "type": stage_type})
                        skip_next = True
                    else:
                        ref_stages.append({"activities": [act], "type": step_type.lower()})

                st.caption("Reference path: " + _describe_reference_stages(ref_stages))
        else:
            ref_uploaded_bpmn = st.file_uploader("BPMN file", type=["bpmn", "xml"], key="ref_bpmn_upload")

        run_ref_btn = st.button("Check Conformance Against Reference Model", width='stretch')

        if run_ref_btn:
            ref_model = None
            ref_build_errors = []

            if ref_mode == "Define expected path":
                if len(ref_stages) < 2:
                    st.warning("Select at least two activities to define a reference path.")
                else:
                    with st.spinner("Building reference model..."):
                        ref_model, ref_build_errors = build_structured_reference_model(ref_stages)
            else:
                if not ref_uploaded_bpmn:
                    st.warning("Upload a BPMN file first.")
                else:
                    with st.spinner("Importing BPMN reference model..."):
                        ref_model, ref_build_errors = import_reference_model_bpmn(ref_uploaded_bpmn.getvalue())

            for err in ref_build_errors:
                st.error(err)

            if ref_model:
                ref_net, ref_im, ref_fm = ref_model
                ref_config = st.session_state.get("config", {})
                ref_speed = ref_config.get("speed_params", {})
                ref_conf_cfg = ref_config.get("conformance_params", {})
                ref_sampling_cfg = ref_config.get("sampling_config", {})

                with st.spinner("Checking conformance against the reference model..."):
                    ref_conf_result = run_conformance_checking(
                        ref_raw_df, ref_net, ref_im, ref_fm,
                        max_align=ref_speed.get("max_align", 250),
                        max_prec_cases=ref_speed.get("max_prec_traces", 250),
                        cores=ref_speed.get("cores", 1),
                        alignment_variant=ref_conf_cfg.get("algorithm", "state_equation_a_star"),
                        enable_detailed_analysis=ref_conf_cfg.get("calculate_precision", True),
                        optimize_variants=ref_conf_cfg.get("optimize_variants", True),
                        perform_sampling=ref_sampling_cfg.get("enabled", True),
                        strata_col=ref_sampling_cfg.get("strata_col", "purchase"),
                        max_priority_ratio=ref_sampling_cfg.get("max_priority_ratio", 0.5),
                    )

                ref_img_path = render_petri_net(
                    ref_net, ref_im, ref_fm, os.path.join("output", "reference_model.png")
                )
                discovered_img_path = results.get("visualizations", {}).get("happy_path")
                coverage_diff = diff_reference_model_coverage(ref_net, ref_raw_df)

                st.session_state["reference_conformance_result"] = {
                    "conformance_result": ref_conf_result,
                    "discovered_img": discovered_img_path,
                    "reference_img": ref_img_path,
                    "coverage_diff": coverage_diff,
                }

        ref_state = st.session_state.get("reference_conformance_result")
        if ref_state:
            ref_conf_result = ref_state["conformance_result"]
            for err in ref_conf_result.get("errors", []):
                st.error(err)

            ref_overall = ref_conf_result.get("overall_summary", {})
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Fitness (vs. Reference)", f"{ref_overall.get('fitness_score', 0):.1%}")
            rc2.metric("Precision (vs. Reference)", f"{ref_overall.get('precision_score', 0):.1%}")
            rc3.metric("Quality", ref_overall.get("quality_assessment", "N/A"))

            ref_cases = ref_conf_result.get("case_analysis", {}).get("cases", [])
            ref_imperfect = sorted(
                [c for c in ref_cases if c.get("fitness", 1.0) < 1.0],
                key=lambda x: x["fitness"]
            )
            if ref_cases:
                st.caption(f"{len(ref_imperfect)} deviant case(s) out of {len(ref_cases)} sampled.")
                if ref_imperfect:
                    ref_dev_rows = [
                        {
                            "Case ID": c["case_id"],
                            "Fitness": f"{c['fitness']:.2%}",
                            "Skipped": ", ".join(c.get("deviations", {}).get("skipped", [])) or "-",
                            "Unsolicited": ", ".join(c.get("deviations", {}).get("unsolicited", [])) or "-"
                        }
                        for c in ref_imperfect[:100]
                    ]
                    st.dataframe(pd.DataFrame(ref_dev_rows), width='stretch')

            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.caption("Discovered from your data")
                if ref_state["discovered_img"] and os.path.exists(ref_state["discovered_img"]):
                    st.image(ref_state["discovered_img"], width='stretch')
                else:
                    st.info("Not available.")
            with img_c2:
                st.caption("Reference model")
                if ref_state["reference_img"] and os.path.exists(ref_state["reference_img"]):
                    st.image(ref_state["reference_img"], width='stretch')
                else:
                    st.info("Not available.")

            coverage_diff = ref_state["coverage_diff"]
            cov_c1, cov_c2 = st.columns(2)
            with cov_c1:
                st.markdown("**Happening in your data but not in the reference model**")
                if coverage_diff["unexpected_in_data"]:
                    for a in coverage_diff["unexpected_in_data"]:
                        st.write(f"- {a}")
                else:
                    st.write("None.")
            with cov_c2:
                st.markdown("**Expected by the reference model but never observed**")
                if coverage_diff["never_observed"]:
                    for a in coverage_diff["never_observed"]:
                        st.write(f"- {a}")
                else:
                    st.write("None.")

            st.download_button(
                "Download Reference Conformance Report",
                data=generate_reference_conformance_report(
                    ref_conf_result,
                    discovered_model_img=ref_state["discovered_img"],
                    reference_model_img=ref_state["reference_img"],
                    coverage_diff=coverage_diff,
                ),
                file_name="prox_reference_conformance_report.html",
                mime="text/html",
            )
        else:
            st.info("Configure a reference model above and click **Check Conformance Against Reference Model**.")

# ---------------------------------------------------------------------------
# Tab: Funnel
# ---------------------------------------------------------------------------
with tab_funnel:
    st.caption(
        "Define the funnel that matters for your process, in any industry - not just "
        "e-commerce. Pick the activities in the order they should occur, and PRoX shows "
        "how many cases reach each stage and where the biggest drop-off is."
    )

    raw_df = st.session_state.get("df")
    if raw_df is None or "concept:name" not in raw_df.columns:
        st.info("Run an analysis first to enable funnel analysis.")
    else:
        if "funnel_result" not in st.session_state:
            st.session_state["funnel_result"] = results.get("funnel_analysis")

        activities = _analyzed_activities(raw_df, st.session_state.get("config", {}))

        mode = st.radio(
            "Funnel definition",
            ["Define manually", "Auto-detect from data"],
            horizontal=True,
            help=(
                "Manual: pick activities in the order they should occur - full control, "
                "works for any process. Auto-detect: PRoX infers a likely order from each "
                "activity's typical position within a case; a rough starting point, not a "
                "substitute for defining the funnel yourself."
            )
        )

        funnel_steps = None
        if mode == "Define manually":
            funnel_steps = st.multiselect(
                "Funnel steps",
                options=activities,
                help="Activities are added to the funnel in the order you select them."
            )
            if funnel_steps:
                st.caption("Funnel order: " + " → ".join(funnel_steps))

        funnel_exclude_cols = {"case:concept:name", "concept:name", "time:timestamp", "user_id"}
        funnel_segment_candidates = [
            c for c in raw_df.columns
            if c not in funnel_exclude_cols and 2 <= raw_df[c].nunique(dropna=True) <= 20
        ]
        funnel_segment_col = None
        if funnel_segment_candidates:
            funnel_segment_choice = st.selectbox(
                "Split by segment (optional)",
                options=["None"] + funnel_segment_candidates,
                help=(
                    "Compare drop-off across segment values instead of just the "
                    "overall funnel - e.g. 'does mobile drop off earlier than "
                    "desktop?' Uses the same stage order as the overall funnel "
                    "above, so segments are directly comparable."
                )
            )
            if funnel_segment_choice != "None":
                funnel_segment_col = funnel_segment_choice

        run_funnel_btn = st.button("Run Funnel Analysis", width='stretch')

        if run_funnel_btn:
            if mode == "Define manually" and not funnel_steps:
                st.warning("Select at least one activity to define a funnel.")
            else:
                with st.spinner("Computing funnel..."):
                    if funnel_segment_col:
                        combined = analyze_funnel_by_segment(
                            raw_df, segment_col=funnel_segment_col, funnel_steps=funnel_steps
                        )
                        st.session_state["funnel_result"] = combined["overall"]
                        st.session_state["funnel_segment_result"] = combined
                    else:
                        st.session_state["funnel_result"] = analyze_conversion_funnel(
                            raw_df, funnel_steps=funnel_steps
                        )
                        st.session_state.pop("funnel_segment_result", None)

        funnel_result = st.session_state.get("funnel_result")
        if funnel_result:
            for err in funnel_result.get("errors", []):
                st.error(err)

            stages = funnel_result.get("stages", {})
            if stages:
                funnel_df = pd.DataFrame.from_dict(stages, orient="index")
                funnel_df.index.name = "Stage"
                st.bar_chart(funnel_df["cases_reached"])
                st.dataframe(
                    funnel_df.style.format({
                        "pct_of_total": "{:.1f}%", "pct_of_previous_stage": "{:.1f}%", "drop_off_pct": "{:.1f}%"
                    }),
                    width='stretch'
                )
                if funnel_result.get("biggest_drop_off"):
                    st.caption(f"Biggest drop-off: **{funnel_result['biggest_drop_off']}**")
            else:
                st.info("No funnel stages could be computed from the selected steps.")

            funnel_segment_result = st.session_state.get("funnel_segment_result")
            if funnel_segment_result and funnel_segment_result.get("segments"):
                st.divider()
                st.subheader("Drop-off by segment")

                for err in funnel_segment_result.get("errors", []):
                    st.error(err)

                segments = funnel_segment_result["segments"]
                pct_table = {
                    str(seg): {stage: info["pct_of_total"] for stage, info in seg_res["stages"].items()}
                    for seg, seg_res in segments.items()
                }
                pct_df = pd.DataFrame(pct_table)
                pct_df.index.name = "Stage"
                st.caption("% of each segment's cases that reached each stage")
                st.bar_chart(pct_df)

                drop_off_table = {
                    str(seg): {stage: info["drop_off_pct"] for stage, info in seg_res["stages"].items()}
                    for seg, seg_res in segments.items()
                }
                drop_off_df = pd.DataFrame(drop_off_table)
                drop_off_df.index.name = "Stage"
                st.caption("Drop-off % from the previous stage, per segment")
                st.dataframe(drop_off_df.style.format("{:.1f}%", na_rep="-"), width='stretch')
        else:
            st.info("Configure the funnel above and click **Run Funnel Analysis**.")

# ---------------------------------------------------------------------------
# Tab 5: Business Insights
# ---------------------------------------------------------------------------
with tab_biz:
    biz = results.get("repeat_purchase_analysis")
    if biz:
        metrics = biz.get("metrics", {})
        rev = metrics.get("revenue_stats", {})
        cart = metrics.get("cart_abandonment")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Buyers", f"{metrics.get('total_buyers', 0):,}")
        c2.metric("Repeat Rate", f"{metrics.get('repeat_rate', 0):.1f}%")
        c3.metric("Median Days Between", f"{metrics.get('median_days_between', 0):.1f}")
        c4.metric(
            "Value Multiplier",
            f"{rev.get('multiplier', 0):.1f}x" if rev else "N/A",
            help="Average revenue: repeat buyers vs. one-time buyers."
        )

        c5, c6 = st.columns(2)
        c5.metric("Average Order Value", f"{metrics.get('average_order_value', 0):,.2f}")
        c6.metric(
            "Cart Abandonment Rate",
            f"{cart['abandonment_rate']:.1f}%" if cart else "N/A",
            help="Share of add-to-cart sessions that didn't go on to purchase."
        )

        charts = {k: v for k, v in biz.get("charts", {}).items() if v and os.path.exists(v)}
        if charts:
            chart_items = list(charts.items())
            for i in range(0, len(chart_items), 3):
                row = chart_items[i:i + 3]
                for col, (name, path) in zip(st.columns(len(row)), row):
                    with col:
                        st.image(path, caption=name.replace("_", " ").title(), width='stretch')

        with st.expander("Full Report"):
            st.text(format_business_report(biz))
    else:
        st.info(
            "No business insight data. "
            "Ensure the log contains purchase activity labels or a revenue column."
        )

# ---------------------------------------------------------------------------
# Tab: Session Insights
# ---------------------------------------------------------------------------
with tab_sessions:
    session_insights = results.get("session_insights")
    if session_insights:
        sessions_df = session_insights.get("sessions")
        journeys_df = session_insights.get("journeys")

        st.caption(
            "Each session is labelled Browsing, Researching, Cart Abandonment, "
            "or Buying from a priority-ordered rule over its activities "
            "(purchase evidence beats cart evidence beats research-activity "
            "count) - not a machine-learning model, so the label for any "
            "session is always traceable back to what happened in it."
        )

        label_counts = sessions_df["label"].value_counts()
        c1, c2 = st.columns([1, 2])
        with c1:
            for label, count in label_counts.items():
                st.metric(label, f"{count:,}")
        with c2:
            st.bar_chart(label_counts)

        st.subheader("Per-user session journeys")
        st.caption(
            "One row per user, sessions in chronological order - e.g. a user "
            "whose first session only browsed and second session bought "
            "shows as 'Browsing -> Buying'. Most useful with Case grouping "
            "set to 'By user' above, so each row's sessions all belong to "
            "one case."
        )
        if journeys_df is not None and not journeys_df.empty:
            st.dataframe(journeys_df, width='stretch', hide_index=True)
        else:
            st.info("No multi-session user journeys to show.")

        with st.expander("All session labels"):
            st.dataframe(sessions_df, width='stretch', hide_index=True)
    else:
        st.info(
            "No session insight data. Ensure the log has a session ID and a "
            "user ID (both required columns already used elsewhere in PRoX)."
        )

# ---------------------------------------------------------------------------
# Tab: Segment Comparison
# ---------------------------------------------------------------------------
with tab_segments:
    raw_df = st.session_state.get("df")
    saved_config = st.session_state.get("config", {})

    exclude_cols = {"case:concept:name", "concept:name", "time:timestamp", "user_id", "session_id"}
    segment_candidates = [
        c for c in raw_df.columns
        if c not in exclude_cols and 2 <= raw_df[c].nunique(dropna=True) <= 20
    ] if raw_df is not None else []

    if not segment_candidates:
        st.info(
            "No suitable segment columns found. A segment column needs 2-20 unique "
            "values (e.g. device type, traffic source, category)."
        )
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            segment_col = st.selectbox("Segment by column", segment_candidates)
        with c2:
            top_n_segments = st.number_input("Max segments", min_value=2, max_value=10, value=5)
        with c3:
            st.write("")
            st.write("")
            compare_btn = st.button("Compare Segments", width='stretch')

        run_parallel = st.checkbox(
            "Run segments in parallel", value=True,
            help=(
                "Runs one segment's analysis per CPU core at once instead of one "
                "after another. Recommended for local runs (this app is not "
                "intended for shared/hosted deployment). Each parallel segment "
                "run is pinned to a single core internally to avoid "
                "oversubscribing the machine."
            )
        )

        if compare_btn:
            with st.spinner(f"Running analysis per segment of '{segment_col}'..."):
                segment_result = compare_segments(
                    raw_df, segment_col=segment_col, config=saved_config,
                    top_n_segments=int(top_n_segments), parallel=run_parallel
                )
            st.session_state["segment_result"] = segment_result

        segment_result = st.session_state.get("segment_result")
        if segment_result:
            for err in segment_result.get("errors", []):
                st.warning(err)

            comparison_table = segment_result.get("comparison_table", {})
            if comparison_table:
                comp_df = pd.DataFrame.from_dict(comparison_table, orient="index")
                comp_df.index.name = "Segment"
                st.dataframe(
                    comp_df.style.format({
                        "health_score": "{:.0f}", "fitness_score": "{:.1%}",
                        "precision_score": "{:.1%}", "repeat_rate": "{:.1f}%"
                    }),
                    width='stretch'
                )

                st.download_button(
                    "Download Segment Comparison Report",
                    data=generate_segment_comparison_report(segment_result),
                    file_name="prox_segment_comparison_report.html",
                    mime="text/html",
                )

                st.subheader("Happy Path per Segment")
                segments = segment_result.get("segments", {})
                img_cols = st.columns(len(segments)) if segments else []
                for col, (seg_value, seg_results) in zip(img_cols, segments.items()):
                    with col:
                        st.caption(str(seg_value))
                        hp = seg_results.get("visualizations", {}).get("happy_path")
                        if hp and os.path.exists(hp):
                            st.image(hp, width='stretch')
                        else:
                            st.info("Not available.")
            else:
                st.info("Comparison produced no results.")
        else:
            st.info("Choose a column and click **Compare Segments**.")

# ---------------------------------------------------------------------------
# Export: Build a Custom PDF Report
# ---------------------------------------------------------------------------
st.divider()
st.header("6. Build a Custom PDF Report")
render_pdf_builder(results, segment_result=st.session_state.get("segment_result"))
