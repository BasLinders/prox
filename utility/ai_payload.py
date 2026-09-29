"""
ai_payload.py

Builds the data payload for the optional "AI conclusion" step (see
utility/ai_client.py): a compact, JSON-safe digest of a finished analysis -
the pipeline results plus whichever follow-up analyses the user ran in the
result tabs (reference-model conformance, funnel by segment, segment
comparison, the propensity model).

Deliberately aggregate-only: no case, user or session IDs, and no
per-resource rows (resource names are often people). What does go out is
what the tabs already show in aggregate - activity names, segment values,
category names and the metrics computed over them. Lists are capped at a
handful of top entries, so the payload stays small enough to keep the model
focused on what matters rather than on a long tail.

Pure Python, no Streamlit: main.py gathers the inputs from session state and
passes them in.

Public entry point: build_ai_payload(results, ...) -> dict.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

PAYLOAD_VERSION = 1

# How many entries to keep from each ranked list (variants, bottlenecks,
# deviations, journeys, ...).
_TOP_N = 5

# Section keys, in payload order, paired with the label the UI uses when it
# lists what's included / not run yet.
SECTION_LABELS = {
    "log_summary": "Log summary",
    "process_performance": "Process performance",
    "variants": "Variants",
    "bottlenecks": "Bottlenecks",
    "conformance_discovered_model": "Conformance (discovered model)",
    "conformance_reference_model": "Conformance vs. reference model",
    "funnel": "Funnel",
    "funnel_by_segment": "Funnel by segment",
    "business": "Business insights",
    "sessions": "Session insights",
    "segment_comparison": "Segment comparison",
    "predictive": "Predictive model",
}


def _clean(value: Any, digits: int = 3) -> Any:
    """Recursively makes `value` JSON-safe and compact: numpy scalars become
    Python ones, NaN/inf become None, and floats are rounded to `digits`
    significant decimals - the model doesn't need 15 of them, and they cost
    tokens."""
    if isinstance(value, dict):
        return {str(k): _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    if isinstance(value, (np.number, np.bool_)):
        value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    return str(value)


def _pct(part: float, whole: float) -> Optional[float]:
    return part / whole * 100 if whole else None


def _top_by(data: Dict[str, Dict[str, Any]], key: str, n: int = _TOP_N) -> List[tuple]:
    return sorted(data.items(), key=lambda kv: kv[1].get(key) or 0, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Section builders - each returns None when its input is missing/empty, so
# build_ai_payload() can list it under "not_run" instead.
# ---------------------------------------------------------------------------

def _context(config: Dict[str, Any], case_grouping: Optional[str]) -> Dict[str, Any]:
    discovery = config.get("discovery_params", {}) or {}
    conformance = config.get("conformance_params", {}) or {}
    sampling = config.get("sampling_config", {}) or {}
    performance = config.get("performance_params", {}) or {}
    loading = config.get("data_loading", {}) or {}
    return {
        "tool": "PRoX (process mining on event logs, typically GA4-style web analytics)",
        "case_grouping": case_grouping,
        "time_unit": performance.get("time_unit"),
        "discovery_algorithm": discovery.get("algorithm"),
        "noise_threshold": discovery.get("noise_threshold"),
        "conformance_method": conformance.get("algorithm"),
        "sampling": {
            "enabled": sampling.get("enabled"),
            "sample_size": sampling.get("total_sample_size") if sampling.get("enabled") else None,
        },
        "duplicate_events_removed": loading.get("remove_duplicates"),
        "filters_applied": config.get("filter_steps") or [],
    }


def _log_summary(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    summary = results.get("log_summary") or {}
    if not summary:
        return None
    return {
        "cases": summary.get("Number of Cases"),
        "events": summary.get("Number of Events"),
        "unique_activities": summary.get("Number of Unique Activities"),
        "avg_events_per_case": summary.get("Average Events per Case"),
        "start": summary.get("Start Timestamp"),
        "end": summary.get("End Timestamp"),
        "duration_days": summary.get("Total Duration (Days)"),
        "activities": summary.get("List of Activities"),
    }


def _process_performance(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    perf = results.get("performance") or {}
    stats = perf.get("summary_statistics") or {}
    case_perf = perf.get("case_performance") or {}
    duration = case_perf.get("duration_stats") or {}
    if not stats and not duration:
        return None
    efficiency = stats.get("efficiency_metrics") or {}
    temporal = perf.get("temporal_patterns") or {}
    return {
        "health_score_0_to_100": stats.get("process_health_score"),
        "lead_time": {k: duration.get(k) for k in ("mean", "median", "q25", "q75", "max", "unit")},
        "duration_variability_pct": efficiency.get("duration_variability_pct"),
        "bottleneck_ratio_pct": efficiency.get("bottleneck_ratio_pct"),
        "case_duration_distribution": case_perf.get("case_duration_distribution"),
        "peak_hours": (temporal.get("hourly_patterns") or {}).get("peak_hours"),
        "busiest_days": (temporal.get("daily_patterns") or {}).get("busiest_days"),
        "rule_based_recommendations": stats.get("recommendations") or [],
    }


def _variants(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    vp = (results.get("performance") or {}).get("variant_performance") or {}
    top = vp.get("top_variants") or {}
    if not top:
        return None
    coverage = vp.get("variant_coverage") or {}
    return {
        "total_variants": vp.get("total_variants"),
        "top_5_coverage_pct": coverage.get("top_5_coverage"),
        "top_10_coverage_pct": coverage.get("top_10_coverage"),
        "top_variants": [
            {
                "path": path,
                "cases": v.get("frequency"),
                "pct_of_cases": v.get("percentage"),
                "median_duration": (v.get("duration") or {}).get("median"),
            }
            for path, v in list(top.items())[:_TOP_N]
        ],
    }


def _bottlenecks(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bn = (results.get("performance") or {}).get("bottlenecks") or {}
    activity_bn = bn.get("activity_bottlenecks") or {}
    transition_bn = bn.get("transition_bottlenecks") or {}
    if not activity_bn and not transition_bn:
        return None

    def rows(data, name_key):
        return [
            {name_key: name, "mean_duration": v.get("mean_duration"),
             "frequency": v.get("frequency"), "severity": v.get("severity")}
            for name, v in _top_by(data, "impact_score")
        ]

    return {
        "note": "Ranked by impact (mean duration x frequency). Durations are waiting time before the step, in context.time_unit.",
        "activities": rows(activity_bn, "activity"),
        "transitions": rows(transition_bn, "transition"),
    }


def _conformance(conf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    overall = (conf or {}).get("overall_summary") or {}
    if not overall:
        return None
    cases = (conf.get("case_analysis") or {}).get("cases") or []
    deviant = [c for c in cases if c.get("fitness", 1.0) < 1.0]
    skipped, unsolicited = Counter(), Counter()
    for c in deviant:
        deviations = c.get("deviations") or {}
        # Counted once per case, so the numbers read as "N cases skipped X".
        skipped.update(set(deviations.get("skipped") or []))
        unsolicited.update(set(deviations.get("unsolicited") or []))

    section = {
        "fitness": overall.get("fitness_score"),
        "precision": overall.get("precision_score"),
        "quality": overall.get("quality_assessment"),
        "pct_perfectly_fitting_traces": (conf.get("fitness") or {}).get("percentage_fit_traces"),
    }
    if cases:
        section.update({
            "cases_checked": len(cases),
            "deviant_cases": len(deviant),
            "most_skipped_activities": [{"activity": a, "cases": n} for a, n in skipped.most_common(_TOP_N)],
            "most_unexpected_activities": [{"activity": a, "cases": n} for a, n in unsolicited.most_common(_TOP_N)],
        })
    timed_out = (conf.get("alignments") or {}).get("timed_out")
    if timed_out:
        section["traces_timed_out"] = timed_out
    return section


def _discovered_conformance(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    section = _conformance(results.get("conformance") or {})
    if section is None:
        return None
    return {
        "note": "Self-consistency: how well a model mined from this same log fits it. Not compliance with an intended process.",
        **section,
    }


def _reference_conformance(reference: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not reference:
        return None
    section = _conformance(reference.get("conformance_result") or {})
    if section is None:
        return None
    coverage = reference.get("coverage_diff") or {}
    return {
        "note": "Real behaviour checked against a reference model the user defined: the process as it is supposed to work.",
        "reference_model": reference.get("description"),
        **section,
        "activities_in_data_not_in_reference": coverage.get("unexpected_in_data", []),
        "activities_in_reference_never_observed": coverage.get("never_observed", []),
    }


def _funnel_stages(funnel: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"stage": step, "cases_reached": s.get("cases_reached"), "pct_of_total": s.get("pct_of_total"),
         "pct_of_previous_stage": s.get("pct_of_previous_stage"), "drop_off_pct": s.get("drop_off_pct")}
        for step, s in (funnel.get("stages") or {}).items()
    ]


def _funnel(funnel: Optional[Dict[str, Any]], auto_detected: bool) -> Optional[Dict[str, Any]]:
    if not funnel or not funnel.get("stages"):
        return None
    return {
        "definition": (
            "auto-detected from each activity's typical position in a case - a rough ordering, not user-defined"
            if auto_detected else "defined by the user"
        ),
        "total_cases": funnel.get("total_cases"),
        "stages": _funnel_stages(funnel),
        "biggest_drop_off_at": funnel.get("biggest_drop_off"),
    }


def _funnel_by_segment(funnel_segment: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    segments = (funnel_segment or {}).get("segments") or {}
    if not segments:
        return None
    return {
        "segment_column": funnel_segment.get("segment_col"),
        "segments": {
            str(value): {
                "cases": seg.get("total_cases"),
                "biggest_drop_off_at": seg.get("biggest_drop_off"),
                "stages": [
                    {"stage": s["stage"], "pct_of_total": s["pct_of_total"], "drop_off_pct": s["drop_off_pct"]}
                    for s in _funnel_stages(seg)
                ],
            }
            for value, seg in segments.items()
        },
    }


def _business(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    biz = results.get("repeat_purchase_analysis")
    if not biz:
        return None
    m = biz.get("metrics") or {}
    rev = m.get("revenue_stats") or {}
    cart = m.get("cart_abandonment") or {}
    categories = m.get("category_breakdown") or {}
    return {
        "total_buyers": m.get("total_buyers"),
        "repeat_rate_pct": m.get("repeat_rate"),
        "median_days_between_purchases": m.get("median_days_between"),
        "average_order_value": m.get("average_order_value"),
        "avg_value_one_time_buyer": rev.get("avg_value_one_time"),
        "avg_value_repeat_buyer": rev.get("avg_value_repeat"),
        "repeat_vs_one_time_value_multiplier": rev.get("multiplier"),
        "cart_abandonment": {
            "cases_added_to_cart": cart.get("cases_added_to_cart"),
            "cases_purchased_after_cart": cart.get("cases_purchased_after_cart"),
            "abandonment_rate_pct": cart.get("abandonment_rate"),
        } if cart else None,
        "top_categories_by_revenue": [
            {"category": name, "revenue": v.get("revenue"), "orders": v.get("orders")}
            for name, v in _top_by(categories, "revenue")
        ],
    }


def _sessions(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    insights = results.get("session_insights") or {}
    sessions_df = insights.get("sessions")
    if sessions_df is None or sessions_df.empty:
        return None
    total = len(sessions_df)
    label_counts = sessions_df["label"].value_counts()
    section = {
        "note": "Labels come from a priority rule over each session's activities (purchase > cart > research count), not an ML model.",
        "total_sessions": total,
        "labels": {
            label: {"sessions": int(n), "pct": _pct(n, total)} for label, n in label_counts.items()
        },
    }
    journeys_df = insights.get("journeys")
    if journeys_df is not None and not journeys_df.empty:
        multi = journeys_df[journeys_df["session_count"] > 1]
        section["users"] = len(journeys_df)
        section["users_with_multiple_sessions"] = len(multi)
        section["most_common_multi_session_journeys"] = [
            {"journey": journey, "users": int(n)}
            for journey, n in multi["journey"].value_counts().head(_TOP_N).items()
        ]
    return section


def _segment_comparison(segment_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    table = (segment_result or {}).get("comparison_table") or {}
    if not table:
        return None
    return {
        "note": "The full analysis run separately per segment value.",
        "segment_column": segment_result.get("segment_col"),
        "segments": [
            {
                "segment": str(value),
                "cases": row.get("cases"),
                "health_score": row.get("health_score"),
                "fitness": row.get("fitness_score"),
                "precision": row.get("precision_score"),
                "repeat_rate_pct": row.get("repeat_rate"),
                "most_common_variant": row.get("top_variant"),
            }
            for value, row in table.items()
        ],
    }


def _predictive(
    model_bundle: Optional[Dict[str, Any]],
    drivers: Optional[List[str]],
    score_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not model_bundle:
        return None
    training = model_bundle.get("training_config") or {}
    outcome = training.get("outcome_activity")
    section: Dict[str, Any] = {
        "note": "A prediction, not a measurement. Drivers are associative, not causal.",
        "model": "logistic regression on trace-prefix features",
        "outcome_activities": outcome if isinstance(outcome, (list, tuple)) else [outcome],
        "case_attributes_used": training.get("case_attribute_cols") or [],
    }
    if model_bundle.get("model") is None:
        section["trained"] = False
        section["reason_not_trained"] = model_bundle.get("errors") or []
        return section

    metrics = model_bundle.get("metrics") or {}
    section.update({
        "trained": True,
        "completed_cases": metrics.get("n_completed"),
        "reached_outcome": metrics.get("n_positive"),
        "did_not_reach_outcome": metrics.get("n_negative"),
        "cross_validated": {
            name: {"mean": metrics.get(f"{name}_mean"), "std": metrics.get(f"{name}_std")}
            for name in ("accuracy", "precision", "recall", "roc_auc")
        },
        "top_drivers": drivers or [],
    })
    if score_summary and score_summary.get("n_scored"):
        section["in_progress_cases"] = {
            "scored": score_summary.get("n_scored"),
            "mean_propensity": score_summary.get("mean_score"),
            "historical_conversion_rate": score_summary.get("historical_positive_rate"),
            "risk_tiers": score_summary.get("risk_tiers"),
        }
    return section


def build_ai_payload(
    results: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    case_grouping: Optional[str] = None,
    reference_conformance: Optional[Dict[str, Any]] = None,
    funnel_result: Optional[Dict[str, Any]] = None,
    funnel_segment_result: Optional[Dict[str, Any]] = None,
    segment_result: Optional[Dict[str, Any]] = None,
    propensity_model: Optional[Dict[str, Any]] = None,
    propensity_drivers: Optional[List[str]] = None,
    propensity_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Digests a finished run into the dict ai_client.generate_conclusion()
    sends to the model.

    `results` is a run_full_analysis() result. The rest are the tabs'
    follow-up analyses, each optional:
      - reference_conformance: main.py's reference_conformance_result
        ({"conformance_result", "coverage_diff", "description", ...})
      - funnel_result: the funnel currently shown in the Funnel tab; falls
        back to the pipeline's auto-detected results["funnel_analysis"]
      - funnel_segment_result: analyze_funnel_by_segment() output, plus a
        "segment_col" key
      - segment_result: compare_segments() output, plus a "segment_col" key
      - propensity_model: a train_propensity_model() bundle, with its
        analyze_propensity_drivers() sentences and
        summarize_propensity_scores() summary

    Every section that couldn't be built (missing input, or not run yet) is
    left out and listed under "not_run" by key, so the model knows not to
    speculate about it.
    """
    results = results or {}
    pipeline_funnel = results.get("funnel_analysis")
    funnel = funnel_result or pipeline_funnel

    sections = {
        "log_summary": _log_summary(results),
        "process_performance": _process_performance(results),
        "variants": _variants(results),
        "bottlenecks": _bottlenecks(results),
        "conformance_discovered_model": _discovered_conformance(results),
        "conformance_reference_model": _reference_conformance(reference_conformance),
        "funnel": _funnel(funnel, auto_detected=funnel is pipeline_funnel),
        "funnel_by_segment": _funnel_by_segment(funnel_segment_result),
        "business": _business(results),
        "sessions": _sessions(results),
        "segment_comparison": _segment_comparison(segment_result),
        "predictive": _predictive(propensity_model, propensity_drivers, propensity_summary),
    }

    payload: Dict[str, Any] = {
        "payload_version": PAYLOAD_VERSION,
        "context": _context(config or {}, case_grouping),
    }
    payload.update({key: section for key, section in sections.items() if section is not None})
    payload["not_run"] = [key for key, section in sections.items() if section is None]
    return _clean(payload)
