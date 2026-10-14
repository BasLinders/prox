import json

import numpy as np
import pandas as pd

from prox.analytics import analyze_funnel_by_segment
from prox.config import create_analysis_config
from prox.pipeline import run_full_analysis
from utility.ai_payload import SECTION_LABELS, build_ai_payload

from conftest import make_event_log, make_simple_variant_log


def test_every_section_key_has_a_label():
    payload = build_ai_payload({})
    assert set(payload["not_run"]) == set(SECTION_LABELS)


def test_empty_results_lists_everything_as_not_run():
    payload = build_ai_payload({}, config={})
    assert payload["payload_version"] == 1
    assert "context" in payload
    assert not set(SECTION_LABELS) & set(payload)


def test_payload_from_a_real_run_is_json_safe_and_has_core_sections(tmp_path):
    df = make_simple_variant_log(n_cases=5)
    config = create_analysis_config()
    results = run_full_analysis(df, config, output_folder=str(tmp_path))

    payload = build_ai_payload(results, config=config, case_grouping="user")

    # allow_nan=False: NaN/inf would make the prompt invalid JSON.
    json.dumps(payload, allow_nan=False)
    for key in ("log_summary", "process_performance", "variants", "conformance_discovered_model"):
        assert key in payload, key
    assert payload["log_summary"]["cases"] == 5
    assert payload["context"]["case_grouping"] == "user"
    # Follow-ups that weren't run are flagged, not silently missing.
    for key in ("conformance_reference_model", "segment_comparison", "predictive", "funnel_by_segment"):
        assert key in payload["not_run"]


def test_payload_never_contains_case_or_user_ids():
    """Aggregate-only: deviation rows and session journeys are counted, their
    case/user IDs are dropped."""
    results = {
        "conformance": {
            "overall_summary": {"fitness_score": 0.8, "precision_score": 0.7, "quality_assessment": "Good"},
            "case_analysis": {"cases": [
                {"case_id": "SECRET-CASE-1", "fitness": 0.5, "deviations": {"skipped": ["pay"], "unsolicited": []}},
                {"case_id": "SECRET-CASE-2", "fitness": 0.6, "deviations": {"skipped": ["pay", "pay"], "unsolicited": ["x"]}},
                {"case_id": "SECRET-CASE-3", "fitness": 1.0, "deviations": {}},
            ]},
        },
        "session_insights": {
            "sessions": pd.DataFrame({"session_id": ["SECRET-S1", "SECRET-S2"], "user_id": ["SECRET-U1"] * 2,
                                      "label": ["Browsing", "Buying"]}),
            "journeys": pd.DataFrame({"user_id": ["SECRET-U1"], "session_count": [2], "journey": ["Browsing -> Buying"]}),
        },
    }
    payload = build_ai_payload(results)

    assert "SECRET" not in json.dumps(payload)
    conf = payload["conformance_discovered_model"]
    assert conf["cases_checked"] == 3
    assert conf["deviant_cases"] == 2
    # Counted once per case, even when a case skipped the same activity twice.
    assert conf["most_skipped_activities"] == [{"activity": "pay", "cases": 2}]
    assert payload["sessions"]["most_common_multi_session_journeys"] == [{"journey": "Browsing -> Buying", "users": 1}]


def test_reference_conformance_and_coverage_diff_are_included():
    reference = {
        "description": "a → b → c",
        "conformance_result": {"overall_summary": {"fitness_score": 0.5, "precision_score": 0.9, "quality_assessment": "Fair"}},
        "coverage_diff": {"unexpected_in_data": ["d"], "never_observed": ["c"]},
    }
    section = build_ai_payload({}, reference_conformance=reference)["conformance_reference_model"]
    assert section["reference_model"] == "a → b → c"
    assert section["fitness"] == 0.5
    assert section["activities_in_data_not_in_reference"] == ["d"]
    assert section["activities_in_reference_never_observed"] == ["c"]


def test_funnel_is_marked_auto_detected_only_when_it_is_the_pipelines_own():
    pipeline_funnel = {"stages": {"a": {"cases_reached": 2, "pct_of_total": 100.0,
                                        "pct_of_previous_stage": 100.0, "drop_off_pct": 0.0}}}
    user_funnel = {"stages": dict(pipeline_funnel["stages"])}
    results = {"funnel_analysis": pipeline_funnel}

    assert build_ai_payload(results)["funnel"]["definition"].startswith("auto-detected")
    assert build_ai_payload(results, funnel_result=pipeline_funnel)["funnel"]["definition"].startswith("auto-detected")
    assert build_ai_payload(results, funnel_result=user_funnel)["funnel"]["definition"] == "defined by the user"


def test_funnel_by_segment_carries_the_segment_column():
    rows = []
    for i, device in enumerate(["mobile", "mobile", "desktop", "desktop"]):
        rows += [(f"c{i}", "view", "2024-01-01 00:00"), (f"c{i}", "buy", "2024-01-01 00:01")]
    df = make_event_log(rows)
    df["device"] = [d for d in ["mobile", "mobile", "desktop", "desktop"] for _ in range(2)]
    combined = analyze_funnel_by_segment(df, segment_col="device", funnel_steps=["view", "buy"])

    section = build_ai_payload({}, funnel_segment_result={**combined, "segment_col": "device"})["funnel_by_segment"]
    assert section["segment_column"] == "device"
    assert set(section["segments"]) == {"mobile", "desktop"}


def test_untrained_propensity_model_reports_why():
    bundle = {"model": None, "errors": ["Too few cases."], "training_config": {"outcome_activity": "buy"}}
    section = build_ai_payload({}, propensity_model=bundle)["predictive"]
    assert section["trained"] is False
    assert section["reason_not_trained"] == ["Too few cases."]
    assert section["outcome_activities"] == ["buy"]


def test_trained_propensity_model_includes_metrics_drivers_and_scores():
    bundle = {
        "model": object(),
        "training_config": {"outcome_activity": ["buy"], "case_attribute_cols": ["device"]},
        "metrics": {"n_completed": 100, "n_positive": 40, "n_negative": 60,
                    "roc_auc_mean": np.float64(0.81234), "roc_auc_std": np.float64(0.05),
                    "accuracy_mean": np.nan, "accuracy_std": np.nan},
    }
    summary = {"n_scored": 7, "mean_score": 0.3, "historical_positive_rate": 0.4,
               "risk_tiers": {"high_risk": 3, "medium_risk": 2, "low_risk": 2}}
    section = build_ai_payload({}, propensity_model=bundle, propensity_drivers=["Driver sentence."],
                               propensity_summary=summary)["predictive"]

    assert section["trained"] is True
    assert section["cross_validated"]["roc_auc"] == {"mean": 0.812, "std": 0.05}
    assert section["cross_validated"]["accuracy"] == {"mean": None, "std": None}
    assert section["top_drivers"] == ["Driver sentence."]
    assert section["in_progress_cases"]["scored"] == 7
    json.dumps(section, allow_nan=False)


def test_segment_comparison_rows():
    segment_result = {
        "segment_col": "device",
        "comparison_table": {"mobile": {"cases": 10, "health_score": 70.0, "fitness_score": 0.9,
                                        "precision_score": 0.8, "repeat_rate": 5.0, "top_variant": "a -> b"}},
    }
    section = build_ai_payload({}, segment_result=segment_result)["segment_comparison"]
    assert section["segment_column"] == "device"
    assert section["segments"][0]["segment"] == "mobile"
    assert section["segments"][0]["most_common_variant"] == "a -> b"


def _variant_outcomes_result():
    return {
        "routes": {
            "a -> b": {"cases": 30, "case_share": 60.0, "conversions": 15, "conversion_rate": 50.0,
                       "revenue": 900.0, "revenue_share": 90.0, "revenue_per_case": 30.0,
                       "revenue_per_conversion": 60.0},
            "a -> c": {"cases": 20, "case_share": 40.0, "conversions": 2, "conversion_rate": 10.0,
                       "revenue": 100.0, "revenue_share": 10.0, "revenue_per_case": 5.0,
                       "revenue_per_conversion": 50.0},
            "a": {"cases": 5, "case_share": 5.0, "conversions": 0, "conversion_rate": 0.0,
                  "revenue": 0.0, "revenue_share": 0.0, "revenue_per_case": 0.0,
                  "revenue_per_conversion": 0.0},
        },
        "summary": {"total_routes": 3, "overall_conversion_rate": 34.0, "total_revenue": 1000.0},
    }


def test_variant_outcomes_section_ranks_routes_and_omits_zero_revenue_ones():
    payload = build_ai_payload({"variant_outcomes": _variant_outcomes_result()})

    section = payload["variant_outcomes"]
    assert "variant_outcomes" not in payload["not_run"]
    assert section["overall_conversion_rate_pct"] == 34.0
    assert [r["route"] for r in section["most_common_routes"]] == ["a -> b", "a -> c", "a"]
    assert section["most_common_routes"][0]["conversion_rate_pct"] == 50.0
    assert section["total_revenue"] == 1000.0
    assert [r["route"] for r in section["top_routes_by_revenue"]] == ["a -> b", "a -> c"]
    json.dumps(payload, allow_nan=False)


def test_variant_outcomes_section_without_revenue_has_conversion_only():
    outcomes = _variant_outcomes_result()
    for stats in outcomes["routes"].values():
        for key in ("revenue", "revenue_share", "revenue_per_case", "revenue_per_conversion"):
            del stats[key]
    del outcomes["summary"]["total_revenue"]

    section = build_ai_payload({"variant_outcomes": outcomes})["variant_outcomes"]

    assert "top_routes_by_revenue" not in section
    assert "total_revenue" not in section
    assert "revenue" not in section["most_common_routes"][0]


def test_variant_outcomes_section_is_capped_to_top_five_routes():
    outcomes = _variant_outcomes_result()
    outcomes["routes"] = {f"route {i}": {**outcomes["routes"]["a -> b"], "cases": 100 - i} for i in range(8)}

    section = build_ai_payload({"variant_outcomes": outcomes})["variant_outcomes"]

    assert len(section["most_common_routes"]) == 5
    assert len(section["top_routes_by_revenue"]) == 5


def test_variant_outcomes_section_is_not_run_without_outcomes():
    assert "variant_outcomes" in build_ai_payload({})["not_run"]
