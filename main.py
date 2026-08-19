import io
import logging
import os
import threading

import pandas as pd
import streamlit as st

from prox import (
    load_and_validate_csv,
    refine_activity_labels,
    optimize_dataframe_memory,
    create_analysis_config,
    run_full_analysis,
    format_business_report,
    generate_html_report,
    generate_segment_comparison_report,
    compare_segments,
    analyze_conversion_funnel,
    generate_mock_csv_bytes,
    filter_event_log,
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
def _cached_load_and_prepare(file_bytes, max_file_size_mb, chunk_threshold_mb, chunk_size):
    """Loads + validates the CSV and applies label refinement/memory optimization.
    Cached on file content and loader params so re-running with the same
    upload (e.g. only sidebar options changed) skips CSV parsing entirely.
    """
    df, messages, has_category = load_and_validate_csv(
        io.BytesIO(file_bytes),
        max_file_size_mb=max_file_size_mb,
        chunk_threshold_mb=chunk_threshold_mb,
        chunk_size=chunk_size,
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
    """
    return run_full_analysis(df_ready, config=config, output_folder=output_folder)

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
    if st.button("Clear Results", use_container_width=True):
        for key in ("results", "df", "load_messages", "segment_result", "funnel_result"):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    if st.session_state.get("confirm_shutdown"):
        st.warning("Shut down PRoX? This stops the local server and closes this tab.")
        cancel_col, confirm_col = st.columns(2)
        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state["confirm_shutdown"] = False
                st.rerun()
        with confirm_col:
            if st.button("Confirm", type="primary", use_container_width=True):
                st.session_state["shutdown_requested"] = True
                st.rerun()
    else:
        if st.button("Shut Down App", use_container_width=True):
            st.session_state["confirm_shutdown"] = True
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Process Excavator")
st.caption("Upload a website event log to discover customer journeys and golden paths.")

st.header("1. Load Data")
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
    if st.button("Generate Mock Data", use_container_width=True):
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
                use_container_width=True,
            )
        with clear_col:
            if st.button("Clear", use_container_width=True):
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

loader_defaults = create_analysis_config()["data_loading"]
with st.spinner("Loading and validating data..."):
    raw_df, df_ready, load_messages, has_category = _cached_load_and_prepare(
        active_file_bytes,
        loader_defaults["max_file_size_mb"],
        loader_defaults["chunk_threshold_mb"],
        loader_defaults["chunk_size"],
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
# Filter events before analysis
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Filter Events")
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

if selected_events:
    filter_mode = "remove_events" if filter_mode_label == "Remove selected events" else "keep_events"
    filter_steps = [{"type": "activity", "activities": selected_events, "mode": filter_mode}]
    preview_df, _ = filter_event_log(raw_df, filter_type="activity", activities=selected_events, mode=filter_mode)
else:
    filter_steps = []
    preview_df = raw_df

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
st.header("3. Sampling")
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
else:
    sample_size = max(post_cases, 1)
    if post_cases > LARGE_CASE_COUNT_THRESHOLD:
        st.warning(
            f"{post_cases:,} cases will be analysed without sampling. Conformance "
            f"checking - especially State Equation A* - can get slow above "
            f"~{LARGE_CASE_COUNT_THRESHOLD:,} cases. Consider enabling sampling above, "
            "or switching to Token Replay in the sidebar."
        )

st.divider()
run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

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
        filter_steps=filter_steps,
    )

    with st.spinner("Running process mining pipeline... This may take a minute."):
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
            use_container_width=True,
        )

st.divider()

tab_map, tab_variants, tab_bottlenecks, tab_conf, tab_funnel, tab_biz, tab_segments = st.tabs([
    "Process Maps",
    "Variants",
    "Bottlenecks",
    "Conformance",
    "Funnel",
    "Business Insights",
    "Segment Comparison",
])

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
            st.image(hp, use_container_width=True)
        else:
            st.info("Happy path image not available. Check that Graphviz is installed.")

    with col2:
        st.subheader("Main Process Flow")
        st.caption("Top-K variants combined, showing common deviations.")
        mf = viz.get("bottlenecks")
        if mf and os.path.exists(mf):
            st.image(mf, use_container_width=True)
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
            st.dataframe(var_df[show_cols], use_container_width=True)
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
            use_container_width=True
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
            use_container_width=True
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
            st.dataframe(pd.DataFrame(dev_rows), use_container_width=True)
        else:
            st.success("All sampled cases follow the model perfectly.")
    else:
        st.info(
            "No per-trace deviation data. "
            "Token Replay mode does not produce trace-level details - "
            "switch to State Equation A* for per-case deviations."
        )

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

        activities = sorted(raw_df["concept:name"].dropna().astype(str).unique().tolist())

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

        run_funnel_btn = st.button("Run Funnel Analysis", use_container_width=True)

        if run_funnel_btn:
            if mode == "Define manually" and not funnel_steps:
                st.warning("Select at least one activity to define a funnel.")
            else:
                with st.spinner("Computing funnel..."):
                    st.session_state["funnel_result"] = analyze_conversion_funnel(
                        raw_df, funnel_steps=funnel_steps
                    )

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
                    use_container_width=True
                )
                if funnel_result.get("biggest_drop_off"):
                    st.caption(f"Biggest drop-off: **{funnel_result['biggest_drop_off']}**")
            else:
                st.info("No funnel stages could be computed from the selected steps.")
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
                        st.image(path, caption=name.replace("_", " ").title(), use_container_width=True)

        with st.expander("Full Report"):
            st.text(format_business_report(biz))
    else:
        st.info(
            "No business insight data. "
            "Ensure the log contains purchase activity labels or a revenue column."
        )

# ---------------------------------------------------------------------------
# Tab: Segment Comparison
# ---------------------------------------------------------------------------
with tab_segments:
    raw_df = st.session_state.get("df")
    saved_config = st.session_state.get("config", {})

    exclude_cols = {"case:concept:name", "concept:name", "time:timestamp", "user_id"}
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
            compare_btn = st.button("Compare Segments", use_container_width=True)

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
                    use_container_width=True
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
                            st.image(hp, use_container_width=True)
                        else:
                            st.info("Not available.")
            else:
                st.info("Comparison produced no results.")
        else:
            st.info("Choose a column and click **Compare Segments**.")
