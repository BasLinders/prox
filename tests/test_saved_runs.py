import os

from prox.saved_runs import (
    save_run,
    list_saved_runs,
    load_saved_run,
    get_saved_run_manifest,
    load_saved_results,
    delete_saved_run,
    _safe_run_id,
)
from prox.incremental import save_cached_dataset, clear_cached_dataset, merge_incremental

from conftest import make_event_log


# --- save_run / load_saved_run: standalone (no cache link) ---

def test_save_run_standalone_round_trips(tmp_path):
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00'), ('c1', 'b', '2024-01-01 00:01:00')])

    manifest = save_run(df, "Acme Corp - Sept", "user", save_dir=str(tmp_path))

    assert manifest["label"] == "Acme Corp - Sept"
    assert manifest["source_dataset_id"] is None
    assert manifest["n_events"] == 2

    loaded_df, loaded_manifest, error = load_saved_run(manifest["run_id"], save_dir=str(tmp_path))
    assert error is None
    assert len(loaded_df) == 2
    assert loaded_manifest["run_id"] == manifest["run_id"]

    # Standalone saves write their own data file.
    assert os.path.isfile(os.path.join(str(tmp_path), f"{manifest['run_id']}.data.csv.gz"))


def test_save_run_with_same_label_does_not_collide(tmp_path):
    """Unlike incremental.py's dataset_id, a saved-run label is expected to
    repeat across saves (e.g. the same client saved every month) - each save
    must get its own entry rather than overwriting the previous one."""
    september = make_event_log([('c1', 'a', '2024-09-01 00:00:00')])
    october = make_event_log([('c1', 'a', '2024-10-01 00:00:00')])

    first = save_run(september, "Acme Corp", "user", save_dir=str(tmp_path))
    second = save_run(october, "Acme Corp", "user", save_dir=str(tmp_path))

    assert first["run_id"] != second["run_id"]
    manifests = list_saved_runs(save_dir=str(tmp_path))
    assert len(manifests) == 2
    assert {m["label"] for m in manifests} == {"Acme Corp"}


# --- save_run / load_saved_run: linked to an incremental cache ---

def test_save_run_linked_to_cache_does_not_duplicate_data(tmp_path):
    cache_dir = str(tmp_path / "cache")
    save_dir = str(tmp_path / "saved_runs")
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00'), ('c1', 'b', '2024-01-01 00:01:00')])
    save_cached_dataset(df, "GA4 funnel", "user", cache_dir=cache_dir)

    manifest = save_run(df, "Acme Corp - linked", "user", source_dataset_id="GA4 funnel", save_dir=save_dir)

    assert manifest["source_dataset_id"] == "GA4 funnel"
    # No second copy of the data should be written when a cache link exists.
    assert not os.path.isfile(os.path.join(save_dir, f"{manifest['run_id']}.data.csv.gz"))

    loaded_df, _, error = load_saved_run(manifest["run_id"], save_dir=save_dir, cache_dir=cache_dir)
    assert error is None
    assert len(loaded_df) == 2


def test_load_saved_run_surfaces_clear_error_when_linked_cache_is_gone(tmp_path):
    cache_dir = str(tmp_path / "cache")
    save_dir = str(tmp_path / "saved_runs")
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "GA4 funnel", "user", cache_dir=cache_dir)
    manifest = save_run(df, "Acme Corp - linked", "user", source_dataset_id="GA4 funnel", save_dir=save_dir)

    clear_cached_dataset("GA4 funnel", cache_dir=cache_dir)

    loaded_df, loaded_manifest, error = load_saved_run(manifest["run_id"], save_dir=save_dir, cache_dir=cache_dir)
    assert loaded_df is None
    assert loaded_manifest is None
    assert "GA4 funnel" in error
    assert "cleared" in error


# --- list_saved_runs / delete_saved_run ---

def test_list_saved_runs_empty_when_dir_missing(tmp_path):
    assert list_saved_runs(save_dir=str(tmp_path / "does_not_exist")) == []


def test_delete_saved_run_removes_manifest_and_data_then_reports_result(tmp_path):
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    manifest = save_run(df, "Acme Corp", "user", save_dir=str(tmp_path))

    assert delete_saved_run(manifest["run_id"], save_dir=str(tmp_path)) is True
    _, _, error = load_saved_run(manifest["run_id"], save_dir=str(tmp_path))
    assert error is not None
    # Deleting again finds nothing left to remove.
    assert delete_saved_run(manifest["run_id"], save_dir=str(tmp_path)) is False


def test_delete_saved_run_never_touches_the_linked_cache(tmp_path):
    """Deleting a cache-linked saved-run entry must not delete the underlying
    incremental cache - it's a shared resource, managed separately."""
    cache_dir = str(tmp_path / "cache")
    save_dir = str(tmp_path / "saved_runs")
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "GA4 funnel", "user", cache_dir=cache_dir)
    manifest = save_run(df, "Acme Corp - linked", "user", source_dataset_id="GA4 funnel", save_dir=save_dir)

    delete_saved_run(manifest["run_id"], save_dir=save_dir)

    from prox.incremental import load_cached_dataset
    cached_df, _ = load_cached_dataset("GA4 funnel", cache_dir=cache_dir)
    assert cached_df is not None
    assert len(cached_df) == 1


def test_load_saved_run_absent_returns_error(tmp_path):
    df, manifest, error = load_saved_run("nope", save_dir=str(tmp_path))
    assert (df, manifest) == (None, None)
    assert error is not None


# --- regression: labels that sanitize to the same string must not collide ---

def test_run_ids_are_unique_even_for_labels_that_sanitize_identically(tmp_path):
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])

    first = save_run(df, "GA4/report", "user", save_dir=str(tmp_path))
    second = save_run(df, "GA4 report", "user", save_dir=str(tmp_path))

    assert first["run_id"] != second["run_id"]
    assert _safe_run_id("GA4/report") != _safe_run_id("GA4/report")  # timestamp/random suffix differs each call


# --- regression: manifest write is atomic (temp file + rename), no leftover ---

def test_save_run_leaves_no_tmp_file_behind(tmp_path):
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_run(df, "Acme Corp", "user", save_dir=str(tmp_path))

    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == []


# --- duplicate saves ---

def test_resaving_same_label_data_and_config_returns_existing_entry(tmp_path):
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00'), ('c1', 'b', '2024-01-01 00:01:00')])
    config = {"discovery_params": {"algorithm": "inductive_miner"}}

    first = save_run(df, "Acme Corp", "user", save_dir=str(tmp_path), config=config)
    second = save_run(df.copy(), " Acme Corp ", "user", save_dir=str(tmp_path), config=config)

    assert second["already_saved"] is True
    assert second["run_id"] == first["run_id"]
    assert len(list_saved_runs(save_dir=str(tmp_path))) == 1


def test_same_data_with_different_config_is_a_separate_entry(tmp_path):
    """A different config is a different analysis of the same log - worth keeping."""
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])

    save_run(df, "Acme Corp", "user", save_dir=str(tmp_path), config={"noise": 0.2})
    second = save_run(df, "Acme Corp", "user", save_dir=str(tmp_path), config={"noise": 0.5})

    assert "already_saved" not in second
    assert len(list_saved_runs(save_dir=str(tmp_path))) == 2


def test_resaving_a_reloaded_standalone_run_is_recognised_as_duplicate(tmp_path):
    """The load-run-then-save-again path: the reloaded frame must fingerprint
    the same as the one originally saved, including numeric-looking user IDs."""
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00'), ('c2', 'b', '2024-01-01 00:01:00')])
    df['user_id'] = ['12345', 'a1b2c3']
    first = save_run(df, "Acme Corp", "user", save_dir=str(tmp_path), config={"x": 1})

    reloaded, _, _ = load_saved_run(first["run_id"], save_dir=str(tmp_path))
    second = save_run(reloaded, "Acme Corp", "user", save_dir=str(tmp_path), config={"x": 1})

    assert second.get("already_saved") is True


# --- config + results ---

def test_config_is_stored_in_manifest(tmp_path):
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    config = {"filter_steps": [{"type": "crop", "activity": ["purchase"]}], "sampling_config": {"enabled": True}}

    manifest = save_run(df, "Acme Corp", "user", save_dir=str(tmp_path), config=config)

    assert get_saved_run_manifest(manifest["run_id"], save_dir=str(tmp_path))["config"] == config


def test_results_round_trip_with_chart_files_copied_out_of_output_folder(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "happy_path_model.png").write_bytes(b"png")
    (output / "happy_path_model.svg").write_bytes(b"svg")
    png_path = str(output / "happy_path_model.png")
    results = {"log_summary": {"Number of Cases": 1}, "visualizations": {"happy_path": png_path}}
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_dir = str(tmp_path / "saved_runs")

    manifest = save_run(df, "Acme Corp", "user", save_dir=save_dir, results=results)

    # The live results dict (shared with the app's run cache) is left untouched.
    assert results["visualizations"]["happy_path"] == png_path
    # A later run overwrites the output folder - the saved copy must survive that.
    for f in output.iterdir():
        f.unlink()

    loaded, note = load_saved_results(manifest, df, save_dir=save_dir)
    assert note is None
    assert loaded["log_summary"] == {"Number of Cases": 1}
    restored_png = loaded["visualizations"]["happy_path"]
    assert restored_png != png_path
    assert open(restored_png, "rb").read() == b"png"
    assert open(os.path.splitext(restored_png)[0] + ".svg", "rb").read() == b"svg"


def test_load_saved_results_none_when_nothing_was_saved(tmp_path):
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    manifest = save_run(df, "Acme Corp", "user", save_dir=str(tmp_path))

    assert load_saved_results(manifest, df, save_dir=str(tmp_path)) == (None, None)


def test_cache_linked_results_are_stale_once_the_cache_grows(tmp_path):
    """A cache-linked run reads through to the current cache, so after a later
    merge its saved results no longer describe the data - don't show them."""
    cache_dir = str(tmp_path / "cache")
    save_dir = str(tmp_path / "saved_runs")
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    df['user_id'] = 'u1'
    save_cached_dataset(df, "GA4 funnel", "user", cache_dir=cache_dir)
    manifest = save_run(
        df, "Acme Corp", "user", source_dataset_id="GA4 funnel", save_dir=save_dir, results={"ok": True}
    )

    current, _, _ = load_saved_run(manifest["run_id"], save_dir=save_dir, cache_dir=cache_dir)
    assert load_saved_results(manifest, current, save_dir=save_dir) == ({"ok": True}, None)

    more = make_event_log([('c2', 'b', '2024-01-02 00:00:00')])
    more['user_id'] = 'u2'
    merge_incremental(more, "GA4 funnel", "user", cache_dir=cache_dir)
    grown, _, _ = load_saved_run(manifest["run_id"], save_dir=save_dir, cache_dir=cache_dir)

    results, note = load_saved_results(manifest, grown, save_dir=save_dir)
    assert results is None
    assert "out of date" in note


def test_delete_saved_run_removes_results_and_assets(tmp_path):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png")
    save_dir = tmp_path / "saved_runs"
    df = make_event_log([('c1', 'a', '2024-01-01 00:00:00')])
    manifest = save_run(df, "Acme Corp", "user", save_dir=str(save_dir), results={"chart": str(chart)})

    delete_saved_run(manifest["run_id"], save_dir=str(save_dir))

    assert os.listdir(save_dir) == []
