import logging
import os

import pandas as pd
import streamlit as st

from prox import (
    load_and_validate_csv,
    refine_activity_labels,
    optimize_dataframe_memory,
    create_analysis_config,
    run_full_analysis,
    format_business_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

st.set_page_config(
    page_title="PRoX - Process Excavator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Sidebar - configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("PRoX ⚙️")
    st.caption("Process Excavator")
    st.divider()

    st.header("Data")
    uploaded_file = st.file_uploader("Upload Event Log (CSV)", type=["csv"])

    st.divider()
    st.header("Discovery")
    discovery_algo = st.selectbox(
        "Algorithm",
        ["inductive_miner", "heuristics_miner", "dfg"],
        help=(
            "**Inductive Miner** - guaranteed sound model, best for most datasets.\n\n"
            "**Heuristics Miner** - better for noisy or very large logs.\n\n"
            "**DFG** - Directly-Follows Graph, fastest option, best for a quick first look."
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
        ["token_replay", "state_equation_a_star"],
        help=(
            "**Token Replay** - fast, gives fitness & precision.\n\n"
            "**State Equation A\\*** - slower, gives exact per-trace deviations."
        )
    )
    calculate_precision = st.checkbox("Calculate Precision", value=True)

    st.divider()
    st.header("Sampling")
    sample_size = st.number_input(
        "Sample Size (cases)", min_value=50, max_value=1000, value=250, step=50,
        help="Cases used for conformance. Higher = more accurate but slower."
    )

    st.divider()
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    if st.button("Clear Results", use_container_width=True):
        for key in ("results", "df", "load_messages"):
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Process Excavator")
st.caption("Upload a website event log to discover customer journeys and golden paths.")

if not uploaded_file:
    st.info("Upload a CSV event log in the sidebar to get started.")
    st.stop()

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
    )
    data_loading_cfg = config["data_loading"]

    with st.spinner("Loading and validating data..."):
        df, messages, has_category = load_and_validate_csv(
            uploaded_file,
            max_file_size_mb=data_loading_cfg["max_file_size_mb"],
            chunk_threshold_mb=data_loading_cfg["chunk_threshold_mb"],
            chunk_size=data_loading_cfg["chunk_size"],
        )

    st.session_state["load_messages"] = messages

    if df is None:
        st.error("Failed to load data. See messages below.")
        for msg in messages:
            if "Critical" in msg or "Error" in msg:
                st.error(msg)
            else:
                st.warning(msg)
        st.stop()

    # Pre-processing: convert category columns back to object for label editing
    df_ready = df.copy()
    for col in df_ready.select_dtypes(include=["category"]).columns:
        df_ready[col] = df_ready[col].astype("object")

    if "page_type" in df_ready.columns:
        df_ready = refine_activity_labels(df_ready, target_activity="page_view", context_column="page_type")

    optimize_dataframe_memory(df_ready)

    with st.spinner("Running process mining pipeline... This may take a minute."):
        results = run_full_analysis(df_ready, config=config)

    if results is None:
        st.error("Analysis failed. Check the application logs for details.")
        st.stop()

    st.session_state["results"] = results
    st.session_state["df"] = df
    st.session_state["config"] = config

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if not results:
    st.info("Configure the options in the sidebar and click **Run Analysis**.")
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases", f"{summary.get('Number of Cases', 0):,}")
    c2.metric("Events", f"{summary.get('Number of Events', 0):,}")
    c3.metric("Activities", summary.get("Number of Unique Activities", 0))
    c4.metric("Duration (days)", summary.get("Total Duration (Days)", 0))

st.divider()

tab_map, tab_variants, tab_bottlenecks, tab_conf, tab_biz = st.tabs([
    "Process Maps",
    "Variants",
    "Bottlenecks",
    "Conformance",
    "Business Insights",
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
# Tab 5: Business Insights
# ---------------------------------------------------------------------------
with tab_biz:
    biz = results.get("repeat_purchase_analysis")
    if biz:
        metrics = biz.get("metrics", {})
        rev = metrics.get("revenue_stats", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Buyers", f"{metrics.get('total_buyers', 0):,}")
        c2.metric("Repeat Rate", f"{metrics.get('repeat_rate', 0):.1f}%")
        c3.metric("Median Days Between", f"{metrics.get('median_days_between', 0):.1f}")
        c4.metric(
            "Value Multiplier",
            f"{rev.get('multiplier', 0):.1f}x" if rev else "N/A",
            help="Average revenue: repeat buyers vs. one-time buyers."
        )

        charts = {k: v for k, v in biz.get("charts", {}).items() if v and os.path.exists(v)}
        if charts:
            chart_cols = st.columns(len(charts))
            for col, (name, path) in zip(chart_cols, charts.items()):
                with col:
                    st.image(path, caption=name.replace("_", " ").title(), use_container_width=True)

        with st.expander("Full Report"):
            st.text(format_business_report(biz))
    else:
        st.info(
            "No business insight data. "
            "Ensure the log contains purchase activity labels or a revenue column."
        )
