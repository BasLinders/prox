import json

import pytest

from prox.config import create_analysis_config
from prox.presets import (
    PresetError,
    build_funnel_settings,
    build_preset,
    delete_preset,
    extract_preset_config,
    fit_config_to_log,
    fit_funnel_to_log,
    list_presets,
    load_preset,
    preset_from_json,
    preset_to_json,
    save_preset,
)


def _config():
    return create_analysis_config(
        discovery_algo="heuristics_miner", noise_threshold=0.4, cores=8, chunk_size=123,
        filter_steps=[
            {"type": "activity", "activities": ["scroll", "click"], "mode": "remove_events"},
            {"type": "crop", "activity": ["purchase"]},
        ],
    )


def test_extract_keeps_reusable_settings_only():
    kept = extract_preset_config(_config())

    assert kept["discovery_params"] == {"algorithm": "heuristics_miner", "noise_threshold": 0.4}
    assert kept["data_loading"] == {"remove_duplicates": True, "merge_page_views": True}
    assert "speed_params" not in kept  # cores are machine-specific
    assert kept["filter_steps"][1] == {"type": "crop", "activity": ["purchase"]}


def test_build_funnel_settings_auto_stores_no_steps():
    assert build_funnel_settings("auto", ["a", "b"])["steps"] == []
    assert build_funnel_settings("manual", ["a", "b"], "device")["steps"] == ["a", "b"]
    with pytest.raises(ValueError):
        build_funnel_settings("sideways")


def test_save_load_list_delete_round_trip(tmp_path):
    preset = build_preset("Acme checkout", _config(), build_funnel_settings("manual", ["view", "buy"]))

    preset_id = save_preset(preset, presets_dir=str(tmp_path))

    assert load_preset(preset_id, presets_dir=str(tmp_path)) == preset
    assert [p["name"] for _, p in list_presets(presets_dir=str(tmp_path))] == ["Acme checkout"]
    assert delete_preset(preset_id, presets_dir=str(tmp_path)) is True
    assert list_presets(presets_dir=str(tmp_path)) == []
    assert delete_preset(preset_id, presets_dir=str(tmp_path)) is False


def test_saving_same_name_updates_the_preset(tmp_path):
    save_preset(build_preset("Acme", _config()), presets_dir=str(tmp_path))
    save_preset(build_preset("Acme", create_analysis_config(noise_threshold=0.1)), presets_dir=str(tmp_path))

    listed = list_presets(presets_dir=str(tmp_path))
    assert len(listed) == 1
    assert listed[0][1]["config"]["discovery_params"]["noise_threshold"] == 0.1


def test_json_round_trip():
    preset = build_preset("Acme", _config(), build_funnel_settings("auto"))
    assert preset_from_json(preset_to_json(preset)) == preset
    assert preset_from_json(preset_to_json(preset).encode("utf-8")) == preset


@pytest.mark.parametrize("bad", [
    "not json",
    "[]",
    json.dumps({"version": 99, "name": "x", "config": {}}),
    json.dumps({"version": 1, "name": " ", "config": {}}),
    json.dumps({"version": 1, "name": "x", "config": []}),
    json.dumps({"version": 1, "name": "x", "config": {"filter_steps": {}}}),
    json.dumps({"version": 1, "name": "x", "config": {"filter_steps": [{"no": "type"}]}}),
    json.dumps({"version": 1, "name": "x", "config": {}, "funnel": {"mode": "weird"}}),
])
def test_invalid_presets_are_rejected(bad):
    with pytest.raises(PresetError):
        preset_from_json(bad)


def test_build_preset_requires_a_name():
    with pytest.raises(PresetError):
        build_preset("  ", _config())


def test_unreadable_preset_is_skipped_in_listing(tmp_path):
    save_preset(build_preset("Good", _config()), presets_dir=str(tmp_path))
    (tmp_path / "broken.preset.json").write_text("{nope")

    assert [p["name"] for _, p in list_presets(presets_dir=str(tmp_path))] == ["Good"]
    with pytest.raises(PresetError):
        load_preset("broken", presets_dir=str(tmp_path))


def test_fit_config_drops_what_the_log_lacks():
    config = extract_preset_config(_config())
    config["sampling_config"]["strata_col"] = "has_bought"

    fitted, notes = fit_config_to_log(config, activities=["click", "view"], columns=["concept:name"])

    # 'scroll' is gone from the activity filter, the crop end point isn't in
    # the log so the crop step is removed, and the missing strata column falls
    # back to the "no stratification" sentinel.
    assert fitted["filter_steps"] == [{"type": "activity", "activities": ["click"], "mode": "remove_events"}]
    assert fitted["sampling_config"]["strata_col"] == "case:concept:name"
    assert len(notes) == 3
    # The input isn't modified.
    assert len(config["filter_steps"]) == 2


def test_fit_config_with_nothing_missing_has_no_notes():
    config = extract_preset_config(_config())
    fitted, notes = fit_config_to_log(config, ["scroll", "click", "purchase"], ["purchase"])
    assert notes == []
    assert fitted["filter_steps"] == config["filter_steps"]


def test_directly_follows_and_endpoints_steps_round_trip_through_a_preset():
    config = create_analysis_config(filter_steps=[
        {"type": "directly_follows", "source": "view", "target": "cart", "mode": "contains"},
        {"type": "endpoints", "start_activities": ["view"]},
    ])
    preset = preset_from_json(preset_to_json(build_preset("explorer", config)))
    assert preset["config"]["filter_steps"] == config["filter_steps"]


def test_fit_config_drops_explorer_steps_whose_activities_are_missing():
    config = extract_preset_config(create_analysis_config(filter_steps=[
        {"type": "directly_follows", "source": "view", "target": "cart", "mode": "contains"},
        {"type": "directly_follows", "source": "view", "target": "gone", "mode": "contains"},
        {"type": "endpoints", "end_activities": ["gone"]},
        {"type": "endpoints", "start_activities": ["view"]},
    ]))

    fitted, notes = fit_config_to_log(config, activities=["view", "cart"], columns=["purchase"])

    assert fitted["filter_steps"] == [
        {"type": "directly_follows", "source": "view", "target": "cart", "mode": "contains"},
        {"type": "endpoints", "start_activities": ["view"]},
    ]
    assert len(notes) == 2


def test_fit_funnel_drops_missing_steps_and_segment():
    funnel = build_funnel_settings("manual", ["view", "cart", "buy"], "device")

    fitted, notes = fit_funnel_to_log(funnel, activities=["view", "buy"], segment_candidates=["country"])

    assert fitted == {"mode": "manual", "steps": ["view", "buy"], "segment_col": None}
    assert len(notes) == 2


def test_fit_funnel_none_passes_through():
    assert fit_funnel_to_log(None, ["a"], []) == (None, [])
