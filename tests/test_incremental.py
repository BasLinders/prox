import os

from prox.incremental import (
    merge_incremental,
    load_cached_dataset,
    save_cached_dataset,
    list_cached_datasets,
    clear_cached_dataset,
    cache_signature,
    _safe_dataset_id,
)

from conftest import make_event_log


def make_log(rows):
    """rows: list of (case, activity, timestamp) -> XES df with a user_id column too,
    since merge_incremental operates on the post-load_and_validate_csv shape."""
    df = make_event_log(rows)
    df['user_id'] = df['case:concept:name']
    return df


# --- merge_incremental: no prior cache ---

def test_merge_incremental_seeds_cache_when_none_exists(tmp_path):
    df = make_log([('c1', 'a', '2024-01-01 00:00:00'), ('c1', 'b', '2024-01-01 00:01:00')])

    merged, stats, messages = merge_incremental(df, "ds1", "session", cache_dir=str(tmp_path))

    assert len(merged) == 2
    assert stats == {
        "cached_events": 0, "new_events": 2, "duplicate_events": 0,
        "total_events": 2, "total_cases": 1, "merged": False,
    }
    assert any("seeding" in m for m in messages)

    cached_df, manifest = load_cached_dataset("ds1", cache_dir=str(tmp_path))
    assert len(cached_df) == 2
    assert manifest["case_grouping"] == "session"
    assert manifest["n_events"] == 2


# --- merge_incremental: growing an existing cache ---

def test_merge_incremental_adds_only_genuinely_new_events(tmp_path):
    first = make_log([('c1', 'a', '2024-01-01 00:00:00'), ('c1', 'b', '2024-01-01 00:01:00')])
    merge_incremental(first, "ds1", "session", cache_dir=str(tmp_path))

    # Second upload re-sends both prior events (as a real recurring export
    # would - same rows plus new ones) and adds one genuinely new event.
    second = make_log([
        ('c1', 'a', '2024-01-01 00:00:00'),
        ('c1', 'b', '2024-01-01 00:01:00'),
        ('c1', 'c', '2024-01-01 00:02:00'),
    ])
    merged, stats, messages = merge_incremental(second, "ds1", "session", cache_dir=str(tmp_path))

    assert len(merged) == 3
    assert stats["cached_events"] == 2
    assert stats["new_events"] == 1
    assert stats["duplicate_events"] == 2
    assert stats["total_events"] == 3
    assert stats["merged"] is True
    assert list(merged['concept:name']) == ['a', 'b', 'c']
    assert any("Merged with cache" in m for m in messages)

    cached_df, manifest = load_cached_dataset("ds1", cache_dir=str(tmp_path))
    assert len(cached_df) == 3
    assert manifest["n_events"] == 3


def test_merge_incremental_is_idempotent_on_rerun(tmp_path):
    """Streamlit reruns the whole script on every widget interaction, not just
    on new uploads - calling merge_incremental again with the same input
    (e.g. because the user tweaked an unrelated sidebar option) must not
    double-count events."""
    df = make_log([('c1', 'a', '2024-01-01 00:00:00'), ('c1', 'b', '2024-01-01 00:01:00')])
    merge_incremental(df, "ds1", "session", cache_dir=str(tmp_path))

    merged, stats, _ = merge_incremental(df, "ds1", "session", cache_dir=str(tmp_path))

    assert len(merged) == 2
    assert stats["new_events"] == 0
    assert stats["duplicate_events"] == 2


def test_merge_incremental_dedup_uses_case_activity_timestamp_key(tmp_path):
    """Same case/activity, different timestamp -> genuinely new event, not a duplicate."""
    first = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    merge_incremental(first, "ds1", "session", cache_dir=str(tmp_path))

    second = make_log([('c1', 'a', '2024-01-02 00:00:00')])
    merged, stats, _ = merge_incremental(second, "ds1", "session", cache_dir=str(tmp_path))

    assert stats["new_events"] == 1
    assert stats["duplicate_events"] == 0
    assert len(merged) == 2


# --- merge_incremental: case-grouping mismatch guard ---

def test_merge_incremental_refuses_merge_on_case_grouping_mismatch(tmp_path):
    first = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    merge_incremental(first, "ds1", "session", cache_dir=str(tmp_path))

    second = make_log([('c2', 'a', '2024-01-01 00:00:00')])
    merged, stats, messages = merge_incremental(second, "ds1", "user", cache_dir=str(tmp_path))

    # Falls back to the new upload alone rather than mixing incompatible case identities.
    assert merged.equals(second)
    assert stats["cached_events"] == 1
    assert stats["merged"] is False
    assert any("different case grouping" in m or "Skipping the merge" in m for m in messages)

    # The stored cache is untouched by the refused merge.
    cached_df, manifest = load_cached_dataset("ds1", cache_dir=str(tmp_path))
    assert len(cached_df) == 1
    assert manifest["case_grouping"] == "session"


# --- cache management helpers ---

def test_list_cached_datasets_reflects_saved_manifests(tmp_path):
    df = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "alpha", "session", cache_dir=str(tmp_path))
    save_cached_dataset(df, "beta", "user", cache_dir=str(tmp_path))

    manifests = list_cached_datasets(cache_dir=str(tmp_path))
    ids = {m["dataset_id"] for m in manifests}
    assert ids == {"alpha", "beta"}


def test_list_cached_datasets_empty_when_dir_missing(tmp_path):
    assert list_cached_datasets(cache_dir=str(tmp_path / "does_not_exist")) == []


def test_clear_cached_dataset_removes_files_and_reports_result(tmp_path):
    df = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "ds1", "session", cache_dir=str(tmp_path))

    assert clear_cached_dataset("ds1", cache_dir=str(tmp_path)) is True
    assert load_cached_dataset("ds1", cache_dir=str(tmp_path)) == (None, None)
    # Clearing again finds nothing left to remove.
    assert clear_cached_dataset("ds1", cache_dir=str(tmp_path)) is False


def test_load_cached_dataset_absent_returns_none_none(tmp_path):
    assert load_cached_dataset("nope", cache_dir=str(tmp_path)) == (None, None)


def test_dataset_id_with_unsafe_characters_is_sanitized_for_the_filesystem(tmp_path):
    df = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "GA4 / checkout funnel!", "session", cache_dir=str(tmp_path))

    cached_df, manifest = load_cached_dataset("GA4 / checkout funnel!", cache_dir=str(tmp_path))
    assert cached_df is not None
    assert manifest["dataset_id"] == "GA4 / checkout funnel!"


# --- regression: dataset IDs that sanitize to the same string must not collide ---

def test_dataset_ids_that_sanitize_identically_get_distinct_caches(tmp_path):
    """'GA4/report' and 'GA4 report' both clean to 'GA4_report' - without an
    injective key, saving one would silently overwrite (or later read as) the
    other's cache."""
    assert _safe_dataset_id("GA4/report") != _safe_dataset_id("GA4 report")

    df_a = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    df_b = make_log([('c2', 'b', '2024-01-01 00:00:00'), ('c2', 'c', '2024-01-01 00:01:00')])
    save_cached_dataset(df_a, "GA4/report", "session", cache_dir=str(tmp_path))
    save_cached_dataset(df_b, "GA4 report", "session", cache_dir=str(tmp_path))

    cached_a, manifest_a = load_cached_dataset("GA4/report", cache_dir=str(tmp_path))
    cached_b, manifest_b = load_cached_dataset("GA4 report", cache_dir=str(tmp_path))

    assert len(cached_a) == 1
    assert len(cached_b) == 2
    assert manifest_a["dataset_id"] == "GA4/report"
    assert manifest_b["dataset_id"] == "GA4 report"


# --- regression: cache round-trip must not mangle NA-like string values ---

def test_cache_round_trip_preserves_ids_that_look_like_na_tokens(tmp_path):
    """pandas' default na_values list (e.g. 'NA', 'null', 'None') would
    otherwise silently turn a case/user ID literally equal to one of those
    tokens into NaN on reload from the cached CSV."""
    df = make_log([('NA', 'a', '2024-01-01 00:00:00'), ('null', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "ds1", "session", cache_dir=str(tmp_path))

    cached_df, _ = load_cached_dataset("ds1", cache_dir=str(tmp_path))

    assert set(cached_df['case:concept:name']) == {'NA', 'null'}
    assert not cached_df['case:concept:name'].isna().any()


def test_merge_incremental_dedup_survives_na_like_case_ids(tmp_path):
    """An 'NA'-valued case ID must still be recognized as already-cached on a
    later merge, not silently re-treated as new every time because it read
    back as NaN and no longer matched the dedup key."""
    first = make_log([('NA', 'a', '2024-01-01 00:00:00')])
    merge_incremental(first, "ds1", "session", cache_dir=str(tmp_path))

    second = make_log([('NA', 'a', '2024-01-01 00:00:00'), ('NA', 'b', '2024-01-01 00:01:00')])
    merged, stats, _ = merge_incremental(second, "ds1", "session", cache_dir=str(tmp_path))

    assert stats["duplicate_events"] == 1
    assert stats["new_events"] == 1
    assert len(merged) == 2


# --- regression: manifest write is atomic (temp file + rename), no leftover ---

def test_save_cached_dataset_leaves_no_tmp_file_behind(tmp_path):
    df = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "ds1", "session", cache_dir=str(tmp_path))

    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == []


# --- cache_signature: cache-busting key for a caller-side memoization layer ---

def test_cache_signature_changes_across_seed_and_clear(tmp_path):
    absent_sig = cache_signature("ds1", cache_dir=str(tmp_path))

    df = make_log([('c1', 'a', '2024-01-01 00:00:00')])
    save_cached_dataset(df, "ds1", "session", cache_dir=str(tmp_path))
    seeded_sig = cache_signature("ds1", cache_dir=str(tmp_path))

    assert seeded_sig != absent_sig

    clear_cached_dataset("ds1", cache_dir=str(tmp_path))
    cleared_sig = cache_signature("ds1", cache_dir=str(tmp_path))

    assert cleared_sig == absent_sig != seeded_sig
