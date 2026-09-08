"""
Incremental analysis: an on-disk cache that lets a recurring event-log export
accumulate across runs instead of being reprocessed from scratch each time.

Scope (see docs/dev_roadmap.md's "Incremental analysis" entry): this is
incremental *data ingestion*, not incremental *algorithmic* discovery/
conformance. pm4py's discovery and conformance-checking algorithms have no
"add one more trace" incremental mode, so run_full_analysis() still runs a
full batch pass every time - what this module avoids is the load/clean/
merge round trip on data already seen before, and it lets a user keep
uploading only the *new* rows of a recurring export (e.g. "this week's GA4
CSV") instead of re-exporting and re-uploading the full history each time.

Deliberately out of scope: the ML/Predictive Insights tab (docs/ML_roadmap.md)
is not wired into this cache. That feature's v1 design retrains synchronously
on whatever log is currently loaded, with no model persistence/versioning -
mixing that with this cache would require solving label churn (a case that
was "in progress" at cache time can resolve to a labelled outcome once new
data arrives) and model versioning, neither of which this module attempts.
If/when the ML layer is built, it should keep reading whatever DataFrame the
rest of the app has assembled (merged-with-cache or not) without this module
needing to know it exists.

Cache layout (one dataset per `dataset_id`, chosen by the user - e.g. "GA4
checkout funnel" - not derived from the uploaded filename, since a recurring
export's filename usually changes every time, e.g. a trailing date):

    <cache_dir>/<dataset_id>-<hash>.manifest.json   - metadata (see _write_manifest)
    <cache_dir>/<dataset_id>-<hash>.cache.csv.gz    - the accumulated event log

The `-<hash>` suffix (see _safe_dataset_id) makes the on-disk key injective:
two different labels that sanitize to the same characters (e.g. "GA4/report"
and "GA4 report" both cleaning to "GA4_report") would otherwise collide on
the same pair of files and silently cross-contaminate two unrelated cached
datasets.

CSV (gzip-compressed), not Parquet: keeps this dependency-free (no pyarrow/
fastparquet beyond what's already pinned), consistent with the project's
"no compiled dependency, runs on a standard laptop" stance in the README.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".prox_cache"

# Same triple check_data_quality() already uses to flag duplicate events -
# reused here as the natural key for "have we already seen this exact event
# before", so incremental merging and duplicate-detection agree on what
# counts as the same event.
_DEDUP_KEY = ['case:concept:name', 'concept:name', 'time:timestamp']

# pd.read_csv's default na_values list (e.g. "NA", "N/A", "null", "None")
# would otherwise silently turn a case/user ID or activity name that happens
# to equal one of those tokens into NaN on cache reload. Genuine missing
# values are still written as an empty field by to_csv's default na_rep='',
# so treating *only* the empty string as NA on read round-trips both
# correctly: real NaNs stay NaN, and a literal "NA" stays the string "NA".
_READ_CSV_NA_KWARGS = {"keep_default_na": False, "na_values": [""]}


def _safe_dataset_id(dataset_id: str) -> str:
    """Collapses a user-provided label to a filesystem-safe, injective cache
    key. The trailing hash (not just the cleaned label) is what makes this
    injective - two labels that clean to the same string (e.g. "GA4/report"
    and "GA4 report") still get different keys."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_id.strip())
    digest = hashlib.sha1(dataset_id.strip().encode("utf-8")).hexdigest()[:8]
    return f"{cleaned or 'default'}-{digest}"


def _manifest_path(dataset_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{_safe_dataset_id(dataset_id)}.manifest.json")


def _data_path(dataset_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{_safe_dataset_id(dataset_id)}.cache.csv.gz")


def cache_signature(dataset_id: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """A cheap (stat-only, no data read), disk-derived value that changes
    whenever the on-disk cache for this dataset actually changes - via a
    fresh merge or a "clear cache" action, from this session or any other.

    Meant as a cache-busting key for a caller-side memoization layer (e.g.
    main.py's st.cache_data wrapper around merge_incremental): keying on
    this instead of re-hashing the merged DataFrame's full content avoids
    that cost on every rerun, and - because it's derived from shared disk
    state rather than any one session's in-memory counter - correctly busts
    a stale memoized result even when a *different* session cleared or
    updated the cache.
    """
    try:
        return str(os.path.getmtime(_manifest_path(dataset_id, cache_dir)))
    except OSError:
        return "absent"


def list_cached_datasets(cache_dir: str = DEFAULT_CACHE_DIR) -> list:
    """Returns manifests for every dataset currently cached, newest first."""
    if not os.path.isdir(cache_dir):
        return []
    manifests = []
    for name in os.listdir(cache_dir):
        if name.endswith(".manifest.json"):
            try:
                with open(os.path.join(cache_dir, name), "r", encoding="utf-8") as f:
                    manifests.append(json.load(f))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping unreadable manifest %s: %s", name, e)
    manifests.sort(key=lambda m: m.get("last_updated", ""), reverse=True)
    return manifests


def load_cached_dataset(
    dataset_id: str, cache_dir: str = DEFAULT_CACHE_DIR
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Returns (cached_df, manifest), or (None, None) if nothing is cached yet
    for this dataset_id."""
    manifest_path = _manifest_path(dataset_id, cache_dir)
    data_path = _data_path(dataset_id, cache_dir)

    if not (os.path.isfile(manifest_path) and os.path.isfile(data_path)):
        return None, None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Defense in depth against a hash collision in _safe_dataset_id (astronomically
        # unlikely, but cheap to check): refuse a cache whose stored label doesn't
        # match what was asked for, rather than silently returning the wrong dataset.
        stored_id = manifest.get("dataset_id")
        if stored_id is not None and stored_id != dataset_id:
            logger.warning(
                "Cache key collision: '%s' and '%s' share an on-disk cache key. "
                "Treating as absent for '%s'.", dataset_id, stored_id, dataset_id
            )
            return None, None

        cached_df = pd.read_csv(data_path, compression="gzip", **_READ_CSV_NA_KWARGS)
        cached_df['time:timestamp'] = pd.to_datetime(cached_df['time:timestamp'])
        cached_df['case:concept:name'] = cached_df['case:concept:name'].astype(str)
        return cached_df, manifest
    except (OSError, json.JSONDecodeError, pd.errors.ParserError, KeyError) as e:
        logger.warning("Cached dataset '%s' unreadable, treating as absent: %s", dataset_id, e)
        return None, None


def _write_manifest(dataset_id: str, cache_dir: str, df: pd.DataFrame, case_grouping: str) -> Dict[str, Any]:
    manifest = {
        "dataset_id": dataset_id,
        "case_grouping": case_grouping,
        "columns": sorted(str(c) for c in df.columns),
        "n_events": int(len(df)),
        "n_cases": int(df['case:concept:name'].nunique()),
        "min_timestamp": df['time:timestamp'].min().isoformat() if not df.empty else None,
        "max_timestamp": df['time:timestamp'].max().isoformat() if not df.empty else None,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = _manifest_path(dataset_id, cache_dir)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)  # atomic on the same filesystem, matches save_cached_dataset
    return manifest


def save_cached_dataset(
    df: pd.DataFrame, dataset_id: str, case_grouping: str, cache_dir: str = DEFAULT_CACHE_DIR
) -> Dict[str, Any]:
    """Writes df as the new cache contents for dataset_id, replacing whatever
    was there before. Callers merge with the existing cache first (see
    merge_incremental) - this always overwrites rather than appending."""
    os.makedirs(cache_dir, exist_ok=True)
    data_path = _data_path(dataset_id, cache_dir)
    tmp_path = data_path + ".tmp"
    df.to_csv(tmp_path, index=False, compression="gzip")
    os.replace(tmp_path, data_path)  # atomic on the same filesystem
    return _write_manifest(dataset_id, cache_dir, df, case_grouping)


def clear_cached_dataset(dataset_id: str, cache_dir: str = DEFAULT_CACHE_DIR) -> bool:
    """Deletes the cache for dataset_id, if any. Returns True if anything was removed."""
    removed = False
    for path in (_manifest_path(dataset_id, cache_dir), _data_path(dataset_id, cache_dir)):
        if os.path.isfile(path):
            os.remove(path)
            removed = True
    return removed


def _build_stats(
    cached_df: Optional[pd.DataFrame], new_events_df: pd.DataFrame, total_df: pd.DataFrame, merged: bool
) -> Dict[str, Any]:
    """Single source of truth for the stats dict merge_incremental returns,
    so its three branches (seed / grouping-mismatch / merge) can't drift out
    of sync with each other on what fields they report.

    `merged` tells the caller whether the returned DataFrame is actually
    different from the freshly-loaded upload it was given (True only for the
    real anti-join-and-concat branch) - False in both the seed branch (cache
    was empty, so the "merged" result is just the upload itself) and the
    grouping-mismatch branch (merge skipped). A caller that already derived
    an analysis-ready frame from that same upload can skip redoing so when
    `merged` is False, instead of discarding and recomputing it against an
    unchanged DataFrame.
    """
    n_cached = int(len(cached_df)) if cached_df is not None else 0
    n_new = int(len(new_events_df))
    return {
        "cached_events": n_cached,
        "new_events": n_new,
        "duplicate_events": 0,
        "total_events": int(len(total_df)),
        "total_cases": int(total_df['case:concept:name'].nunique()) if not total_df.empty else 0,
        "merged": merged,
    }


def merge_incremental(
    new_df: pd.DataFrame,
    dataset_id: str,
    case_grouping: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[pd.DataFrame, Dict[str, int], list]:
    """
    Merges a freshly-loaded event log with whatever's already cached for
    dataset_id, deduplicating on _DEDUP_KEY, then writes the merged result
    back as the new cache (write-through - the next run starts from here).

    Does not mutate new_df. Safe to call every run when incremental mode is
    enabled: with nothing cached yet, this just seeds the cache from new_df.

    Not safe against two concurrent callers merging the same dataset_id at
    once (e.g. two browser tabs both uploading at the same moment) - this
    does a read-modify-write against the cache file with no locking, matching
    the app's single-analyst-on-one-laptop design goal (see README). A
    same-process guard logs a warning if the on-disk cache changed between
    this call's read and its write, so a collision is at least visible
    rather than silently dropping the loser's data with no trace.

    Returns
    -------
    merged_df : pd.DataFrame
        new_df unchanged if no prior cache exists or case_grouping doesn't
        match the cached dataset; otherwise the deduplicated union, sorted
        by case then timestamp.
    stats : dict with keys 'cached_events', 'new_events', 'duplicate_events',
        'total_events', 'total_cases', 'merged' (see _build_stats).
    messages : list of str, human-readable notes for the UI.
    """
    messages = []
    signature_before = cache_signature(dataset_id, cache_dir)
    cached_df, manifest = load_cached_dataset(dataset_id, cache_dir)

    if cached_df is None:
        messages.append(f"No existing cache for '{dataset_id}' - seeding it with this upload.")
        save_cached_dataset(new_df, dataset_id, case_grouping, cache_dir)
        return new_df, _build_stats(None, new_df, new_df, merged=False), messages

    cached_grouping = manifest.get("case_grouping") if manifest else None
    if cached_grouping and cached_grouping != case_grouping:
        messages.append(
            f"Cached data for '{dataset_id}' was built with case grouping "
            f"'{cached_grouping}', but this run uses '{case_grouping}'. Skipping "
            "the merge to avoid mixing incompatible case identities - clear the "
            "cache for this Dataset ID, or switch case grouping back, to merge."
        )
        return new_df, _build_stats(cached_df, new_df, new_df, merged=False), messages

    # Anti-join: keep only incoming rows whose dedup key isn't already cached.
    marker = cached_df[_DEDUP_KEY].drop_duplicates().assign(_cached=True)
    tagged = new_df.merge(marker, on=_DEDUP_KEY, how="left")
    genuinely_new = new_df[tagged["_cached"].isna().to_numpy()]

    n_duplicate = len(new_df) - len(genuinely_new)
    merged_df = (
        pd.concat([cached_df, genuinely_new], ignore_index=True)
        .sort_values(_DEDUP_KEY[:1] + ['time:timestamp'])
        .reset_index(drop=True)
    )

    stats = _build_stats(cached_df, genuinely_new, merged_df, merged=True)
    stats["duplicate_events"] = int(n_duplicate)
    messages.append(
        f"Merged with cache '{dataset_id}': {stats['new_events']:,} new event(s) added to "
        f"{stats['cached_events']:,} cached ({stats['duplicate_events']:,} already-seen event(s) "
        f"skipped) - {stats['total_events']:,} total across {stats['total_cases']:,} cases."
    )

    if cache_signature(dataset_id, cache_dir) != signature_before:
        messages.append(
            f"Note: the cache for '{dataset_id}' changed elsewhere while this merge was "
            "running (e.g. another session). Proceeding, but this write may overwrite "
            "that other update - avoid merging the same Dataset ID from two sessions at once."
        )
    save_cached_dataset(merged_df, dataset_id, case_grouping, cache_dir)
    return merged_df, stats, messages
