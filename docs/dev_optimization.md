# Optimization Methods — Status and Options

*See `dev_roadmap.md` for the current status of all phases at a glance —
that file is the leading document; this one is the detail.*

Companion to `dev_phase2.md`'s Phase 4b section. Written after exposing `cores` in the
UI and shipping segment comparison v1 (`dbf0a5f`), and after confirming CUDA
"ran flat" in practice — consistent with the earlier analysis that PRoX's
workload (Petri net discovery, alignment-based conformance) is combinatorial,
not the dense matrix math GPUs accelerate.

## Already shipped — no longer "available," they're done

- **Vectorization** — `analytics.py` is groupby/agg-based throughout.
- **Clustering-for-speed** — `optimize_variants` in `prox/conformance.py` aligns
  once per unique trace variant, not once per case.
- **Batching** — `calculate_fitness_in_batches` (fitness, batch_size=200 with
  `gc.collect()`) and chunked CSV loading (above 50MB).
- **Multiprocessing** — `cores` is now exposed as a UI control, wired through
  to PM4Py's own internal multiprocessing pool for alignment computation.
  Not GIL-blocked (separate processes, separate interpreters).
  **Correction (2026-09-23, #47):** the "CPU Cores" setting has no effect in
  practice, because PM4Py 2.7's `alignments.apply()` ignores the `cores`
  parameter. The wiring is still in place, but it only helps if a future
  PM4Py version honours it. The only multiprocessing that actually runs today
  is `compare_segments()`, one process per segment (below).
- **CUDA** — considered, and empirically confirmed flat. Not revisitable
  without a fundamentally different algorithm shape (see `dev_phase2.md` for the
  reasoning: many small independent per-trace LP solves, GPU kernel-launch
  overhead dominates; would also require an NVIDIA GPU, contradicting the
  "runs on a standard laptop" goal).

## Shipped this pass — #1 and #2

- **Parallelize `compare_segments()`.** Each segment's `run_full_analysis()`
  call is fully independent (no shared state, no ordering dependency), so
  `prox/segments.py` now runs one segment per worker process via
  `concurrent.futures.ProcessPoolExecutor` (new `parallel` parameter,
  defaults `True`). Each parallel segment run is pinned to a single core
  internally (`speed_params.cores = 1` on a deep-copied per-worker config) —
  otherwise N parallel segments each requesting M alignment cores could
  request N*M cores at once and oversubscribe the machine. Sequential mode
  (`parallel=False`) is kept for cases where nested multiprocessing is
  undesirable, and the UI exposes both via a "Run segments in parallel"
  checkbox on the Segment Comparison tab.

  Explicitly scoped for **local execution only** — this app is run with
  `streamlit run main.py` on a user's own machine, not deployed to
  Streamlit Community Cloud or another shared/hosted environment, so
  spawning worker processes per segment has no multi-tenant resource
  contention to worry about. If PRoX is ever deployed to a shared host,
  this default should be revisited (hosted platforms often cap or forbid
  process-level parallelism).

  Measured on a 4-core machine, 30k-event synthetic log, 4 segments,
  default sampling: **4.92s sequential → 2.41s parallel** (~2x; bound by
  4 cores serving both segment-level and internal alignment parallelism,
  not a clean 4x since each of the 4 concurrent segments still does
  non-trivial single-core work). Tests: `tests/test_segments.py` covers
  both modes and asserts parallel/sequential produce identical
  `comparison_table` output.

- **Profile against a realistically-large synthetic log.** Added
  `scripts/profile_pipeline.py` — generates a synthetic clickstream log
  (session funnel with realistic drop-off + noise events) and times each
  `run_full_analysis()` stage independently at a given size. Results at
  10k / 50k / 100k events (4-core, 15GB dev machine):

  | Stage | 10k events | 50k events | 100k events |
  |---|---|---|---|
  | CSV load + validate | 0.04s | 0.11s | 0.15s |
  | Filtering | 0.00s | 0.00s | 0.01s |
  | Discovery (inductive miner) | 0.18s | 1.00s | 2.12s |
  | Conformance, token replay (sampled) | 0.50s | 0.90s | 1.66s |
  | Conformance, state equation A* (sampled) | 1.27s | 1.80s | 2.73s |
  | Performance analysis | 0.17s | 0.46s | 0.87s |
  | Visualisation (BPMN + bottleneck PNGs) | 0.27s | 0.93s | 1.98s |
  | Business insights | 0.48s | 0.47s | 0.54s |
  | **Total** | **2.90s** | **5.68s** | **10.06s** |

  **Finding that changes the picture assumed going into this pass:**
  conformance checking is capped by stratified sampling
  (`speed_params.max_align`, default 250 traces) regardless of log size, so
  it does *not* scale with the full log — it stays roughly flat while
  **discovery and visualisation scale linearly with total events and
  dominate at scale** (a combined ~41% of wall-clock at 100k events, vs.
  conformance's ~44%). This means:
  - `cores`/multiprocessing (already shipped) genuinely helps only the
    alignment-based conformance stage, which is exactly the one stage that's
    *already* bounded by sampling — its ceiling is capped by design, so
    there's a natural limit to what more cores buy there.
  - Segment-comparison parallelization (#1, above) is a better win than
    originally scoped: because it parallelizes the *entire* per-segment
    pipeline — discovery and visualisation included, not just alignment —
    it gets a proportionally bigger speedup than alignment-only
    parallelism would, which the 2x measurement above reflects.

  **Since superseded in part:** these numbers assume sampling is always on,
  which was the default at the time. Sampling became opt-in on 2026-08-19
  (#20). Since 2026-09-22 (#43), when sampling is on it applies once, right
  after filtering, and every later stage reuses that sample, not just
  conformance. With sampling on, discovery and visualisation no longer scale
  with the full log either. With it off, conformance scales with the full log
  too.

## Shipped — Streamlit caching

`main.py` now caches CSV loading/prep (`st.cache_data`, keyed on file content)
and the full `run_full_analysis()` call (`st.cache_resource`, keyed on the
prepared DataFrame + config). Re-running with unchanged inputs is now near-
instant instead of redoing filtering, discovery, conformance, etc. from
scratch. Verified end-to-end: identical config on a repeat run went from
~2.1s to ~0.01s, while a changed config still correctly recomputes.

This pass also surfaced and fixed a real correctness bug it depended on
being able to test against: `optimize_dataframe_memory()` was converting
`case:concept:name`/`concept:name` to `category` dtype, which
`pm4py.convert_to_event_log()` rejects — silently breaking discovery on
real event logs. Both columns are now excluded from category downcasting.

**Updated since:** every loader (`_cached_load_and_prepare`, the
incremental merge, "Load cached dataset" and "Load saved run") now uses
`st.cache_resource` with `max_entries` set, not `st.cache_data`. For logs of
tens of millions of rows, `cache_data` pickles and unpickles a fresh copy on
every rerun, which caused an `ArrowMemoryError` while configuring a run from a
large cache (#49). `_cached_run_full_analysis` keeps at most 2 entries (#47).

## Shipped — Memory bounding for large logs (2026-09-23, #47)

Large runs could exhaust memory and kill the Streamlit server. Four fixes,
the first being the main one:

- **ETConformance precision budget.** PM4Py's precision replays every
  unique prefix of every trace, so its cost grows with the square of trace
  length. With user-level cases, one heavy user can have thousands of
  events. `cap_traces_for_precision()` keeps the prefix log within
  `MAX_PRECISION_PREFIX_EVENTS` (250k, about 190 MB worst case) by cutting
  traces to the longest common length that fits. When the cap applies, the
  Conformance tab says so. The score barely moves (0.077 → 0.075 across
  200/400/800-event caps on a long-trace log).
- **DataFrames to PM4Py, not an `EventLog`.** `convert_to_event_log()`
  built a Python object per event, about 18x the DataFrame's memory.
  `to_pm4py_frame()` now passes a slim, time-sorted frame to discovery and
  the process maps instead. At 300k events, peak memory dropped from 388 MB
  to 98 MB. Side effect: discovery now uses timestamp order rather than file
  row order, so models on real data can differ from earlier runs.
- **Capped Streamlit caches** (`max_entries` on the loaders and the
  analysis run).
- **30 s per-trace limit for State Equation A\***
  (`MAX_ALIGN_SECONDS_PER_TRACE`). Traces that time out are counted and shown
  as a warning, instead of silently dropping out of fitness.

Related: the 500 MB upload limit was removed (`d5ed8ed`), incremental merges
find new rows with a key lookup instead of a merge that copied the whole
upload (`5afdddc`), and the browser is pinged
every 30 s so long runs don't drop the session (`49b18ce`).

## Shipped — Fewer recomputations per rerun (2026-09-22 to 2026-09-29)

Streamlit reruns the whole script on every widget change, and the results
fragment reruns on every tab switch. Several passes cut what each rerun
redoes:

- **Apply Configuration form** (#45, later scoped to Filter Events and
  Sampling in `11c1be0`). Filter and sampling changes only recompute when
  the button is clicked. The reference-model conformance config got the same
  treatment (`7dc4749`).
- **Nothing loads at startup** (#52, #53). The data source, cached dataset
  and saved run pickers start empty, so no large cache is read into memory
  before the user picks one.
- **Identity-keyed caching of log-derived data** (#55). Deduplication, the
  data quality check, winsorizing, the incremental-merge `df_ready`, filter
  steps and per-column `nunique` scans are cached with `st.cache_resource`,
  keyed on DataFrame identity (`hash_funcs={pd.DataFrame: id}`). "Download
  Event Log" and "Download Full Report" build their file only when clicked.
  `_cached_run_full_analysis` is also keyed on `df_ready` identity, because
  Streamlit's content hash only samples 10k rows. That had returned stale,
  uncapped results when winsorizing capped only a few outliers.

## Remaining item — deprioritized, not a ready next step

**Precision / discovery variant-dedup or downsampling for visualisation.**
Visualisation's linear scaling comes from PM4Py/Graphviz rendering a Petri
net sized to the full (post-filter) log, not from anything PRoX controls
directly. Discovery already runs once per call, not once per trace, so
"variant-dedup" doesn't apply the way it did for alignments — the lever
here would be pre-discovery downsampling for very large logs, which needs
a concrete pain report before it's worth the accuracy trade-off. Precision
variant-dedup specifically (the original phrasing of this item) is
deprioritized: precision uses ETConformance token replay, already fast in
the profiling data above.

## Suggested next step

None ready. The cheap, clearly-justified optimization work (vectorization,
variant-clustering, batching, parallel segment comparison, Streamlit
caching, memory bounding, rerun caching) is done. Incremental analysis and
the BigQuery data source, once listed here, have both shipped (see
`dev_roadmap.md`, Phases 5 and 7).

Known performance issues, neither scheduled:
- **Token Replay on very long traces is slow** (about 4 minutes for 3 cases ×
  about 33k events). Only runtime is affected; memory stays bounded.
- **"CPU Cores" has no effect** (see the Multiprocessing correction above).
  Either remove the control or wait for a PM4Py version that honours
  `cores`.

Pre-discovery downsampling still needs a concrete pain report before it's
worth the accuracy trade-off.
