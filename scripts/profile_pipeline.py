"""
Profiles run_full_analysis() stage-by-stage against synthetic clickstream logs.

Part of Phase 4b (see docs/dev_optimization.md, item #2): the optimization
pass so far was code-reading-based reasoning, not measurement. This generates
a realistically-shaped event log and times each pipeline stage to find out
where wall-clock time actually goes at scale.

Usage:
    python scripts/profile_pipeline.py [--sizes 10000,50000,100000]

Writes timings to stdout as a table; does not touch prox/ or main.py.
"""
import argparse
import io
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prox import load_and_validate_csv, create_analysis_config  # noqa: E402
from prox.data_manager import filter_event_log, FILTER_HANDLERS  # noqa: E402
from prox.discovery import perform_process_discovery  # noqa: E402
from prox.conformance import run_conformance_checking  # noqa: E402
from prox.analytics import get_event_log_summary, analyze_process_performance, analyze_repeat_purchases  # noqa: E402
from prox.visualizer import visualize_focused_insights  # noqa: E402

logging.getLogger("prox").setLevel(logging.WARNING)
logging.getLogger("pm4py").setLevel(logging.ERROR)

# A realistic-ish website journey funnel: most sessions drop off early,
# a minority reach checkout, a minority of those complete a purchase.
FUNNEL = [
    ("session_start", 1.00),
    ("page_view", 0.98),
    ("view_item_list", 0.75),
    ("view_item", 0.55),
    ("add_to_cart", 0.22),
    ("view_cart", 0.18),
    ("begin_checkout", 0.12),
    ("add_shipping_info", 0.10),
    ("add_payment_info", 0.08),
    ("purchase", 0.06),
]
NOISE_EVENTS = ["scroll", "user_engagement", "view_promotion", "select_promotion"]


def generate_synthetic_log(n_events: int, seed: int = 42) -> pd.DataFrame:
    """Builds a synthetic clickstream CSV-shaped DataFrame with ~n_events rows."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    rows = []
    session_counter = 0
    base_time = pd.Timestamp("2026-01-01")

    while len(rows) < n_events:
        session_counter += 1
        session_id = f"s{session_counter}"
        user_id = f"u{session_counter % max(1, session_counter // 3 or 1)}"
        t = base_time + pd.Timedelta(minutes=session_counter * 3)
        reached_purchase = False

        for activity, reach_prob in FUNNEL:
            if rng.random() > reach_prob:
                break
            t += pd.Timedelta(seconds=rng.randint(5, 240))
            price = round(float(np_rng.uniform(10, 250)), 2) if activity == "purchase" else 0.0
            rows.append((session_id, user_id, activity, t, price))
            if activity == "purchase":
                reached_purchase = True

            # Sprinkle a little in-session noise, matching real GA4-style exports.
            if rng.random() < 0.3:
                t += pd.Timedelta(seconds=rng.randint(1, 30))
                rows.append((session_id, user_id, rng.choice(NOISE_EVENTS), t, 0.0))

        if not reached_purchase and rng.random() < 0.05:
            # Occasional repeat buyer session for business-insights coverage.
            t += pd.Timedelta(seconds=30)
            rows.append((session_id, user_id, "purchase", t, round(float(np_rng.uniform(10, 250)), 2)))

    df = pd.DataFrame(rows[:n_events], columns=["session_id", "user_id", "event_name", "timestamp", "event_value"])
    return df


def to_csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


class Timer:
    def __init__(self):
        self.results = []

    def stage(self, name):
        return _StageCtx(self, name)


class _StageCtx:
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        self.timer.results.append((self.name, elapsed))


def profile_size(n_events: int, output_folder: str) -> Timer:
    print(f"\n=== {n_events:,} events ===")
    timer = Timer()

    raw_df = generate_synthetic_log(n_events)
    csv_bytes = to_csv_bytes(raw_df)

    config = create_analysis_config(sample_size=250, enable_sampling=True)

    with timer.stage("1. CSV load + validate"):
        df, messages, _ = load_and_validate_csv(csv_bytes, max_file_size_mb=500, chunk_threshold_mb=50)
    if df is None:
        print("  Load failed:", messages)
        return timer

    with timer.stage("2. Filtering"):
        filter_steps = config.get("filter_steps", [])
        log_df = df.copy()
        for step_config in filter_steps:
            params = step_config.copy()
            f_type = params.pop("type")
            if f_type not in FILTER_HANDLERS:
                continue
            log_df, _ = filter_event_log(log_df, filter_type=f_type, **params)

    with timer.stage("3. Log summary"):
        get_event_log_summary(log_df)

    with timer.stage("4. Discovery (inductive miner)"):
        disc_cfg = config["discovery_params"]
        model_tuple, errors, _ = perform_process_discovery(
            log_df,
            discovery_algo=disc_cfg["algorithm"],
            noise_threshold=disc_cfg["noise_threshold"],
            dependency_threshold=disc_cfg["dependency_threshold"],
            activity_threshold=disc_cfg["activity_threshold"],
        )
    if errors:
        print("  Discovery failed:", errors)
        return timer
    net, im, fm = model_tuple

    with timer.stage("5. Conformance (token_replay, sampled)"):
        speed = config["speed_params"]
        sampling = config["sampling_config"]
        run_conformance_checking(
            log_df, net, im, fm,
            max_align=speed["max_align"],
            max_prec_cases=speed["max_prec_traces"],
            cores=speed["cores"],
            alignment_variant="token_replay",
            enable_detailed_analysis=True,
            perform_sampling=sampling["enabled"],
            strata_col=sampling["strata_col"],
            max_priority_ratio=sampling["max_priority_ratio"],
        )

    with timer.stage("5b. Conformance (state_equation_a_star, sampled)"):
        run_conformance_checking(
            log_df, net, im, fm,
            max_align=speed["max_align"],
            max_prec_cases=speed["max_prec_traces"],
            cores=speed["cores"],
            alignment_variant="state_equation_a_star",
            enable_detailed_analysis=True,
            perform_sampling=sampling["enabled"],
            strata_col=sampling["strata_col"],
            max_priority_ratio=sampling["max_priority_ratio"],
        )

    with timer.stage("6. Performance analysis (bottlenecks/variants)"):
        perf_cfg = config["performance_params"]
        analyze_process_performance(
            log_df,
            time_unit=perf_cfg["time_unit"],
            bottleneck_threshold_percentile=perf_cfg["bottleneck_threshold_percentile"],
            include_variants=True,
        )

    with timer.stage("7. Visualisation (BPMN + bottleneck PNGs)"):
        import pm4py
        log_for_vis = pm4py.convert_to_event_log(log_df)
        visualize_focused_insights(log_for_vis, output_folder=output_folder)

    with timer.stage("8. Business insights (repeat purchase)"):
        biz_cfg = config["business_params"]
        analyze_repeat_purchases(
            log_df,
            output_folder=output_folder,
            user_col=biz_cfg["user_col"],
            purchase_values=biz_cfg["purchase_values"],
            revenue_col=biz_cfg["revenue_col"],
        )

    total = sum(t for _, t in timer.results)
    for name, t in timer.results:
        pct = (t / total * 100) if total else 0
        print(f"  {name:<45} {t:7.3f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<45} {total:7.3f}s")

    return timer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="10000,50000,100000")
    parser.add_argument("--output-folder", default="output/profile_tmp")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    for n in sizes:
        profile_size(n, output_folder=f"{args.output_folder}/{n}")


if __name__ == "__main__":
    main()
