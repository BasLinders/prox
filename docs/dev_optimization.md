# Optimization Methods — Status and Options

Companion to `phase2.md`'s Phase 4b section. Written after exposing `cores` in the
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
  without a fundamentally different algorithm shape (see `phase2.md` for the
  reasoning: many small independent per-trace LP solves, GPU kernel-launch
  overhead dominates; would also require an NVIDIA GPU, contradicting the
  "runs on a standard laptop" goal).

## Still genuinely available, ranked by expected payoff

1. **Parallelize `compare_segments()` itself.**
   It currently loops over segments sequentially, but each segment's
   `run_full_analysis()` call is fully independent — no shared state, no
   ordering dependency. This is an embarrassingly-parallel case for
   `concurrent.futures.ProcessPoolExecutor` or `multiprocessing.Pool`, using
   the same GIL-irrelevant argument as `cores`. With N segments this could
   mean close to an N-fold wall-clock improvement on segment comparison
   specifically. Ranked highest because it's a concrete, already-identified
   gap rather than something that needs discovery first.

2. **Profile against a realistically-large synthetic log.**
   Everything in Phase 4b so far has been code-reading-based reasoning, not
   measurement. Generate a large synthetic log (~50k-100k events) and time
   each pipeline stage to see where time is actually going at scale — it's
   entirely possible discovery or repeat-purchase chart generation dominates,
   not alignment, in which case cores/parallelization wouldn't be the next
   lever to pull. Do this alongside #1 since it's cheap and de-risks #4.

3. **Streamlit caching** (`st.cache_data` / `st.cache_resource`).
   App-layer, not engine-layer. Every "Run Analysis" click currently
   reprocesses everything from scratch even if only one sidebar control
   changed (e.g. toggling precision on/off re-runs filtering and discovery
   unnecessarily). Caching the CSV-load and discovery stages keyed on their
   inputs would make iterating on settings noticeably snappier without
   touching the engine.

4. **Precision variant-dedup.**
   Flagged as uncertain in `phase2.md`: precision uses ETConformance
   token-based replay, which may already be fast enough that variant-deduping
   wouldn't matter. Needs #2's profiling data to justify before touching.

## Suggested next step

#1 and #2 together: #1 has an obvious, low-risk win; #2 tells us whether #3/#4
are worth pursuing at all rather than guessing further.
