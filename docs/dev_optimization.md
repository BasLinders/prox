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
variant-clustering, batching, multi-core alignment, parallel segment
comparison, Streamlit caching) is done. What's left — pre-discovery
downsampling, incremental analysis (`dev_phase2.md` Phase 5), a BigQuery
data source (`dev_roadmap.md`) — each needs a real usage signal to justify
before it's worth building. Revisit once one of those actually hurts.
