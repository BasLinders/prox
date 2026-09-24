"""
Saved runs: a library of event logs the analyst has explicitly chosen to keep
around, labelled by hand (e.g. a client name) so a later session can come
back and pick "which run" to load instead of re-uploading or re-querying.

Distinct from prox/incremental.py's cache: that module keys one recurring
dataset by a *reused* label (the same "GA4 checkout funnel" id always points
at the same accumulating cache). Saved runs intentionally do the opposite -
every save gets its own entry even if the label repeats (e.g. "Acme Corp"
saved once a month, each a separate snapshot to come back to).

Coupling with the incremental cache: if the event log being saved is exactly
what's already sitting in an incremental-cache dataset_id (the run was loaded
via "Load cached dataset", or merged into one via Incremental Analysis), the
saved-run entry records that `source_dataset_id` and does NOT write its own
second copy of the data - it would just be a duplicate of bytes already
durably on disk. Loading such an entry reads through to that cache instead.
A run with no linked cache (a plain CSV/BigQuery pull never merged into an
incremental dataset) gets its own on-disk copy, same as the cache module.

A save also keeps the analysis itself, not just its input: the config the run
was made with (stored in the manifest, so loading it can restore the sidebar/
filter/sampling settings) and the pipeline's results, so a heavy conformance
check doesn't have to be re-run to look at it again. The results reference
chart images by path in the shared output folder, which the next run
overwrites - so those files are copied in alongside and the paths repointed.

Duplicates: repeating a label is expected (monthly snapshots), but saving the
same label with the same event log *and* the same config again - e.g.
clicking Save twice, or re-saving a run that was just loaded - returns the
existing entry instead of adding an identical one.

Layout:

    <save_dir>/<run_id>.manifest.json
    <save_dir>/<run_id>.data.csv.gz      - only present when source_dataset_id is None
    <save_dir>/<run_id>.results.pkl.gz   - only present when results were saved
    <save_dir>/<run_id>.assets/          - chart images the saved results point to

The results file is a pickle - fine for a library this app wrote itself on
the analyst's own machine, but it means a saved-runs folder should never be
loaded from an untrusted source.

`run_id` embeds a timestamp + random suffix (not just the label) precisely
because labels are expected to repeat across saves - unlike incremental.py's
dataset_id, reusing the same label here must not overwrite a prior entry.
"""

import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .incremental import DEFAULT_CACHE_DIR, load_cached_dataset

logger = logging.getLogger(__name__)

DEFAULT_SAVED_RUNS_DIR = ".prox_saved_runs"

# Mirrors incremental.py's _READ_CSV_NA_KWARGS: only an empty field round-trips
# to NaN, so a case/activity value that happens to read "NA"/"null" etc. isn't
# silently corrupted by pandas' default na_values list on reload.
_READ_CSV_NA_KWARGS = {"keep_default_na": False, "na_values": [""]}


def _safe_run_id(label: str) -> str:
    """Builds a unique, filesystem-safe key for one save. Unlike
    incremental._safe_dataset_id, this must never collide on a repeated label -
    saving "Acme Corp" twice in one month has to produce two entries, not
    overwrite the first - so the key includes a timestamp + random suffix
    rather than being derived from the label alone."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip()) or "run"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha1(f"{label}-{ts}-{os.urandom(4).hex()}".encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}-{ts}-{digest}"


def _manifest_path(run_id: str, save_dir: str) -> str:
    return os.path.join(save_dir, f"{run_id}.manifest.json")


def _data_path(run_id: str, save_dir: str) -> str:
    return os.path.join(save_dir, f"{run_id}.data.csv.gz")


def _results_path(run_id: str, save_dir: str) -> str:
    return os.path.join(save_dir, f"{run_id}.results.pkl.gz")


def _assets_dir(run_id: str, save_dir: str) -> str:
    return os.path.join(save_dir, f"{run_id}.assets")


def _data_fingerprint(df: pd.DataFrame) -> Optional[str]:
    """Content hash of the event log, used to recognise a repeat save of the
    same data. None (never matches anything) if the frame can't be hashed."""
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False).values
    except (TypeError, ValueError) as e:
        logger.warning("Could not fingerprint event log, duplicate check skipped: %s", e)
        return None
    digest = hashlib.sha1(row_hashes.tobytes())
    digest.update(",".join(map(str, df.columns)).encode("utf-8"))
    return digest.hexdigest()


def _config_fingerprint(config: Optional[Dict[str, Any]]) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# Chart files a results dict may point to. Only these are copied, so an
# arbitrary string value that happens to name an existing file (an activity
# called "README", say) isn't swept up.
_ASSET_SUFFIXES = (".png", ".svg", ".jpg", ".jpeg", ".csv")


def _relocate_assets(obj: Any, assets_dir: str, copied: Dict[str, str]) -> Any:
    """Returns obj with every chart-file path copied into assets_dir and
    repointed there. Rebuilds dicts/lists/tuples instead of mutating them -
    the results dict passed in is shared with main.py's st.cache_resource
    entry, so editing it in place would corrupt the live run's paths. A PNG's
    .svg sibling (the high-res process-map download) is copied along too."""
    if isinstance(obj, dict):
        return {k: _relocate_assets(v, assets_dir, copied) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_relocate_assets(v, assets_dir, copied) for v in obj)
    if isinstance(obj, str) and obj.lower().endswith(_ASSET_SUFFIXES) and os.path.isfile(obj):
        if obj not in copied:
            os.makedirs(assets_dir, exist_ok=True)
            dest = os.path.join(assets_dir, f"{len(copied)}_{os.path.basename(obj)}")
            shutil.copy2(obj, dest)
            svg_sibling = os.path.splitext(obj)[0] + ".svg"
            if obj.lower().endswith(".png") and os.path.isfile(svg_sibling):
                shutil.copy2(svg_sibling, os.path.splitext(dest)[0] + ".svg")
            copied[obj] = dest
        return copied[obj]
    return obj


def _find_duplicate(save_dir: str, label: str, data_fp: Optional[str], config_fp: str) -> Optional[Dict[str, Any]]:
    if data_fp is None:
        return None
    for m in list_saved_runs(save_dir=save_dir):
        if (m.get("label") == label and m.get("data_fingerprint") == data_fp
                and m.get("config_fingerprint") == config_fp):
            return m
    return None


def list_saved_runs(save_dir: str = DEFAULT_SAVED_RUNS_DIR) -> list:
    """Returns manifests for every saved run, newest first."""
    if not os.path.isdir(save_dir):
        return []
    manifests = []
    for name in os.listdir(save_dir):
        if name.endswith(".manifest.json"):
            try:
                with open(os.path.join(save_dir, name), "r", encoding="utf-8") as f:
                    manifests.append(json.load(f))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping unreadable saved-run manifest %s: %s", name, e)
    manifests.sort(key=lambda m: m.get("saved_at", ""), reverse=True)
    return manifests


def save_run(
    df: pd.DataFrame,
    label: str,
    case_grouping: str,
    source_dataset_id: Optional[str] = None,
    save_dir: str = DEFAULT_SAVED_RUNS_DIR,
    config: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Registers df in the saved-runs library under a fresh run_id.

    If source_dataset_id is given, this event log is already durably stored
    in the incremental cache under that id - the manifest links to it instead
    of writing a second copy. Otherwise df is written to its own gzip CSV,
    same layout as incremental.save_cached_dataset.

    config (the dict passed to run_full_analysis) goes in the manifest;
    results (what it returned) is pickled alongside, with its chart files
    copied in. If an entry with the same label, event log and config already
    exists, nothing is written and that entry's manifest is returned with
    "already_saved": True.
    """
    label = label.strip()
    data_fp = _data_fingerprint(df)
    config_fp = _config_fingerprint(config)
    existing = _find_duplicate(save_dir, label, data_fp, config_fp)
    if existing is not None:
        return {**existing, "already_saved": True}

    os.makedirs(save_dir, exist_ok=True)
    run_id = _safe_run_id(label)

    if source_dataset_id is None:
        data_path = _data_path(run_id, save_dir)
        tmp_path = data_path + ".tmp"
        df.to_csv(tmp_path, index=False, compression="gzip")
        os.replace(tmp_path, data_path)  # atomic on the same filesystem

    if results is not None:
        portable_results = _relocate_assets(results, _assets_dir(run_id, save_dir), {})
        results_path = _results_path(run_id, save_dir)
        tmp_path = results_path + ".tmp"
        with gzip.open(tmp_path, "wb") as f:
            pickle.dump(portable_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, results_path)

    manifest = {
        "run_id": run_id,
        "label": label,
        "source_dataset_id": source_dataset_id,
        "case_grouping": case_grouping,
        "columns": sorted(str(c) for c in df.columns),
        "n_events": int(len(df)),
        "n_cases": int(df['case:concept:name'].nunique()) if 'case:concept:name' in df.columns else None,
        "min_timestamp": df['time:timestamp'].min().isoformat() if 'time:timestamp' in df.columns and not df.empty else None,
        "max_timestamp": df['time:timestamp'].max().isoformat() if 'time:timestamp' in df.columns and not df.empty else None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "has_results": results is not None,
        "data_fingerprint": data_fp,
        "config_fingerprint": config_fp,
    }
    manifest_path = _manifest_path(run_id, save_dir)
    tmp_manifest_path = manifest_path + ".tmp"
    with open(tmp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(tmp_manifest_path, manifest_path)
    return manifest


def load_saved_run(
    run_id: str, save_dir: str = DEFAULT_SAVED_RUNS_DIR, cache_dir: str = DEFAULT_CACHE_DIR
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """Returns (df, manifest, error_message). df/manifest are None and
    error_message is set if the entry (or, for a cache-linked entry, the
    cache it points to) is missing or unreadable."""
    manifest_path = _manifest_path(run_id, save_dir)
    if not os.path.isfile(manifest_path):
        return None, None, f"Saved run '{run_id}' no longer exists."

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, None, f"Saved run '{run_id}' is unreadable: {e}"

    source_dataset_id = manifest.get("source_dataset_id")
    if source_dataset_id is not None:
        df, cache_manifest = load_cached_dataset(source_dataset_id, cache_dir=cache_dir)
        if df is None:
            return None, None, (
                f"'{manifest.get('label', run_id)}' was saved as a link to cached dataset "
                f"'{source_dataset_id}', but that cache has since been cleared. "
                "The saved event log is no longer available."
            )
        return df, manifest, None

    data_path = _data_path(run_id, save_dir)
    if not os.path.isfile(data_path):
        return None, None, f"Saved run '{run_id}' is missing its data file."

    try:
        # Same dtype hint as incremental.load_cached_dataset: without it, a mix of
        # numeric-looking and hashed IDs round-trips as a mixed-type column that
        # later breaks Arrow serialization in st.dataframe.
        df = pd.read_csv(
            data_path, compression="gzip",
            dtype={'case:concept:name': str, 'user_id': str},
            **_READ_CSV_NA_KWARGS,
        )
        if 'time:timestamp' in df.columns:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        return df, manifest, None
    except (OSError, pd.errors.ParserError) as e:
        return None, None, f"Saved run '{run_id}' is unreadable: {e}"


def get_saved_run_manifest(run_id: str, save_dir: str = DEFAULT_SAVED_RUNS_DIR) -> Optional[Dict[str, Any]]:
    """Just the manifest (no data read) - cheap enough to call before the
    data itself is loaded, e.g. to restore a run's settings into widgets
    that render first."""
    try:
        with open(_manifest_path(run_id, save_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_saved_results(
    manifest: Dict[str, Any], df: pd.DataFrame, save_dir: str = DEFAULT_SAVED_RUNS_DIR
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Returns (results, note). results is None when the entry has none (saved
    before results were kept, or without a completed run), when the file is
    unreadable, or when they no longer describe df - a cache-linked entry
    reads through to the *current* cache, which a later incremental merge
    may have grown since. note explains a None result the user would expect
    to have gotten; it's None when there was simply nothing saved."""
    if not manifest.get("has_results"):
        return None, None

    # Cheap staleness check (no full-data hash on every rerun): a merge only
    # ever adds events, so the event count or time range moves with it.
    def _iso(ts):
        return ts.isoformat() if pd.notna(ts) else None
    current = (
        int(len(df)),
        _iso(df['time:timestamp'].min()) if 'time:timestamp' in df.columns and not df.empty else None,
        _iso(df['time:timestamp'].max()) if 'time:timestamp' in df.columns and not df.empty else None,
    )
    saved = (manifest.get("n_events"), manifest.get("min_timestamp"), manifest.get("max_timestamp"))
    if current != saved:
        return None, (
            f"The event log behind '{manifest.get('label')}' has changed since it was saved "
            f"(e.g. new data merged into cache '{manifest.get('source_dataset_id')}'), so its "
            "saved results are out of date. Its settings are restored - click Run Analysis "
            "to redo it on the current data."
        )

    try:
        with gzip.open(_results_path(manifest["run_id"], save_dir), "rb") as f:
            return pickle.load(f), None
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError) as e:
        logger.warning("Saved results for '%s' unreadable: %s", manifest.get("run_id"), e)
        return None, (
            f"The saved results for '{manifest.get('label')}' couldn't be read ({e}). "
            "Its settings are restored - click Run Analysis to redo it."
        )


def delete_saved_run(run_id: str, save_dir: str = DEFAULT_SAVED_RUNS_DIR) -> bool:
    """Deletes a saved-run entry (manifest, plus its own data/results/assets, if any).
    Never touches a linked incremental-cache dataset - that's a shared
    resource managed separately via clear_cached_dataset(). Returns True if
    anything was removed."""
    removed = False
    for path in (_manifest_path(run_id, save_dir), _data_path(run_id, save_dir), _results_path(run_id, save_dir)):
        if os.path.isfile(path):
            os.remove(path)
            removed = True
    assets_dir = _assets_dir(run_id, save_dir)
    if os.path.isdir(assets_dir):
        shutil.rmtree(assets_dir)
        removed = True
    return removed
