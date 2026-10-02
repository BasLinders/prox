# PRoX Development Roadmap

**This is the leading document for PRoX development status.** Every phase —
completed, in progress, or roadmapped — is tracked here first. Where a
phase has enough detail to warrant its own file, this document carries an
accurate summary and links out to it: `dev_phase2.md` for the phase-by-phase
execution log (tests/CI/extensibility/capability work), `dev_optimization.md`
for performance work, `ML_roadmap.md` for predictive/statistical ML ideas,
`AI_summary_roadmap.md` for generative-AI ideas. Keep the summaries here
current even when the detail lives elsewhere — this is the one page meant
to answer "where does PRoX development actually stand?" without opening
five files.

Last assessed 2026-09-30, against `main` — 239 tests passing, `pyflakes`
clean.

---

## Status at a glance

| Phase | Status | Detail |
|---|---|---|
| Phase 1 — Core correctness | Complete | `dev_phase2.md` |
| Phase 2 — Safety net (tests, CI, housekeeping) | Complete | `dev_phase2.md` |
| Phase 3 — Extensibility (registries) | Complete | `dev_phase2.md` |
| Phase 4 — Expand capability | Complete* | `dev_phase2.md` |
| Phase 4b — Optimization | Complete | `dev_optimization.md` |
| Phase 5 — BigQuery live data source (via `foe.data`) | Complete | below |
| Phase 6 — Session insight, reporting & data controls | Complete | below |
| Phase 6b — Full-pipeline correctness pass | Complete | below |
| Phase 7 — Incremental analysis (data-level caching) | Complete | below |
| Phase 7b — High-res process map export & saved-run library | Complete | below |
| Phase 8 — Memory, rerun cost & analysis consistency | Complete | below, `dev_optimization.md` |
| ML layer (conversion propensity + drivers) | Complete (engine + Predictive Insights tab) | below, `ML_roadmap.md` |
| AI Conclusion (optional, Gemini) | Complete | below, `AI_summary_roadmap.md` |
| Follow-ups on shipped features (ML, performance, BigQuery) | Roadmapped | below |
| Process mining capability gaps (5 items, by effort) | Roadmapped, not scoped (resource perspective partly shipped) | below |
| Product development suggestions | Roadmapped | below |

\* One sub-item — segment comparison v2 (automated golden-path diffing) —
is deliberately deferred; see "Medium bets" below. Everything else under
Phase 4 is shipped.

---

## Completed phases (summary)

### Phase 1 — Core correctness
Token-based replay genuinely implemented (`pm4py.fitness_token_based_replay`),
`create_analysis_config()` fully parameterized, DFG discovery exposed in the
UI alongside Inductive and Heuristics Miner. Caught and fixed a real pm4py
API-drift bug in the DFG-to-Petri-net conversion along the way. Full detail
in `dev_phase2.md`.

### Phase 2 — Safety net
Test suite (`tests/`, 74 tests at the time, 239 as of 2026-09-30), CI
(`.github/workflows/ci.yml` — pyflakes then pytest on every PR/push to
`main`), `.gitignore`, and pinned dependency upper bounds. Caught and fixed
a real silent bug in `optimize_dataframe_memory()` while writing its test
(pandas 2.x+/3.x's `str` dtype wasn't matched by the old `== 'object'`
check). Full detail in `dev_phase2.md`.

### Phase 3 — Extensibility
Discovery, conformance, and filter dispatch replaced with registries
(`DISCOVERY_ALGORITHMS`, `CONFORMANCE_METHODS`, `FILTER_HANDLERS`); the UI
now derives its selectbox options and help text from the same registries
instead of a second hardcoded list, so adding a new algorithm or filter type
means one registry entry, not an "edit two files" seam. Full detail in
`dev_phase2.md`.

### Phase 4 — Expand capability
- **HTML report export**, later overhauled with a plain-language Executive
  Summary (health verdict, translated fitness/precision, most common
  journey, biggest bottleneck, business/funnel highlights), embedded
  business-insight charts, and a click-to-zoom lightbox for the
  process-map diagrams. `generate_segment_comparison_report()` brings the
  same treatment to segment comparison, with its own download button.
- **Segment comparison v1** — compare health score, fitness, precision,
  repeat rate, and happy path across the top-N values of any column.
  Parallelized in Phase 4b.
- **Business insights** — fixed three correctness bugs in
  `analyze_repeat_purchases()` (revenue/price values alone were treated as
  purchase evidence, causing cart-abandoners to be counted as buyers;
  order value used `max(price)` across a whole case instead of the actual
  purchase event; three different, inconsistent `purchase_values` defaults
  existed across the codebase). Added cart abandonment rate, average
  order value, category-level revenue breakdown, and a revenue-over-time
  trend.
- **Funnel analysis** — `analyze_conversion_funnel()` plus a dedicated
  **Funnel** tab, letting the user define their own funnel from any
  activities in the log, in any order (industry-agnostic by design — not
  just e-commerce), with auto-derivation from the data as a fallback
  starting point.
- **Deferred**: segment comparison v2 (automated golden-path diffing) —
  see "Medium bets" below.

Full detail in `dev_phase2.md`.

### Phase 4b — Optimization
`compare_segments()` parallelization (~2x wall-clock on a 4-core machine),
pipeline profiling against realistic synthetic logs (found discovery and
visualisation dominate at scale, not conformance — conformance is capped by
sampling), and Streamlit-layer caching (~200x on a repeat run with
unchanged inputs). Found and fixed a real correctness bug along the way:
`optimize_dataframe_memory()` was converting `case:concept:name`/
`concept:name` to `category` dtype, which `pm4py.convert_to_event_log()`
rejects — silently breaking discovery on real event logs. Full detail in
`dev_optimization.md`. Later memory and rerun work is under Phase 8.

### Phase 5 — BigQuery live data source (via `first-order-engine`'s `foe.data`)

**Shipped 2026-08-20** (PR #28, `utility/bigquery_source.py`) — kept in full below
as the design record; see the "Resolved" note under Open Questions at the
end for how those were actually settled.

**Idea**: instead of requiring the user to export, clean, and upload a CSV,
let PRoX connect directly to BigQuery and query GA4 event data live.
Positioned as a second, separate data-source workflow alongside the
existing CSV upload — not a replacement.

**Update 2026-08-20**: most of what this section originally scoped from
scratch (OAuth flow, BigQuery client/session handling, dry-run cost
estimation, GA4 SQL builders) already exists, built and tested, in a
sibling project: [`first-order-engine`](https://github.com/BasLinders/first-order-engine)'s
`foe.data` package. PRoX should consume it rather than reimplement it —
this turns the item below from "build a BigQuery integration" into "wire
an existing engine into the Streamlit UI."

#### Why

Today's flow (`main.py`) gates the entire app behind a single upload widget:
`uploaded_file = st.file_uploader(...)` in the sidebar, then `if not
uploaded_file: st.stop()` before anything else renders. For a user whose
event data already lives in BigQuery (a common case for GA4/web-analytics
exports), that means an export-clean-upload round trip every time they want
to look at a different date range or dataset. A live connection removes
that round trip entirely.

#### What `foe.data.DataEngine` already provides

- **Framework-free OAuth** (`build_auth_url` / `exchange_code`) — the PKCE
  verifier and any caller state are encoded into the OAuth `state` param
  itself, so no server-side session storage is needed to survive the
  redirect. This is a good match for Streamlit specifically, whose reruns
  don't give you a durable server-side session the way a typical web
  framework does. `refresh_if_expired`, `credentials_to_dict` /
  `credentials_from_dict` round out token lifecycle handling.
- **Discovery**: `list_projects()`, `list_datasets(project)` for a
  project/dataset picker — no free-form SQL editor needed for v1.
- **Cost safety**: `dry_run(sql)` estimates bytes scanned before executing;
  `monthly_usage(dataset)` tracks the 1TB/month free tier.
- **`extract_event_log(EventLogExtractionParams, limit=0)`** — this is the
  key piece: it already returns exactly the shape PRoX's engine expects,
  one row per `(case_id, activity, timestamp)`, sourced from a GA4
  `events_*` export. `case_id_col` defaults to `user_pseudo_id`,
  `activity_col` to `event_name`; both are overridable to any top-level or
  dotted struct column. Optional `event_names` restricts which events are
  pulled, `attribute_params` unnests extra `event_params` keys as columns
  (e.g. `page_location`), and an optional user-scoping filter
  (`UserFilterType.CONTAINS` / `REGEX` / `EVENT`) narrows to a subset of
  users. The returned DataFrame can go straight into PRoX's existing
  validation path.
- Gated behind the `foe[bigquery]` extra — importing `foe.data` itself
  never requires `google-cloud-bigquery`, only instantiating `DataEngine`
  does. Same "don't force Google auth deps on CSV-only users" property
  this section originally asked for, already built in.

#### Proposed shape

- **A data-source choice, shown first** — before the existing sidebar
  controls become interactive, similar to a landing step: "Upload CSV" or
  "Connect to BigQuery." This matches the existing gating pattern (`st.stop()`
  until data is ready) but adds a fork before it, rather than replacing it.
- **BigQuery path**:
  1. "Sign in with Google" — `DataEngine.build_auth_url(client_id,
     client_secret, redirect_uri)` using OAuth client credentials read from
     `st.secrets["bigquery"]` (see `.streamlit/secrets.toml`, prepared
     below). Store the returned `verifier` in `st.session_state` as a
     belt-and-suspenders measure (it's also recoverable from `state`
     alone, per the module's own design). On the callback, call
     `DataEngine.exchange_code(...)` and stash the resulting `Credentials`
     via `credentials_to_dict()` in `st.session_state` — never on disk.
  2. Build a `DataEngine` via `DataEngine.from_credentials(...)`, then
     `list_projects()` / `list_datasets(project)` for a picker rather than
     free-form SQL, for the same "no accidentally-expensive or destructive
     query" reasons the original scoping called for.
  3. Construct `EventLogExtractionParams` (connection, date range,
     case/activity column choice, optional event-name/attribute filters),
     call `dry_run()` on the generated SQL first and show the estimated
     bytes/free-tier percentage before running anything.
  4. Call `extract_event_log(params)`, get a DataFrame back, and feed it
     into the same validation/cleaning path the CSV upload already uses.
- **Shared validation logic**: `load_and_validate_csv()` currently mixes
  CSV-specific concerns (file-size checks, chunked reading) with genuinely
  reusable logic (column auto-mapping against `COLUMN_MAPPINGS`, composite
  case-ID creation, timestamp parsing, critical-column validation). This is
  a real refactor opportunity: split it into `_load_csv_source(...)` (CSV-only)
  and a shared `validate_and_clean_dataframe(df, ...)` that both the CSV path
  and the new BigQuery path call. Avoids duplicating the column-mapping and
  cleaning logic in two places. `extract_event_log`'s output already uses
  `case_id`/`activity`/`timestamp` as column names, which simplifies the
  mapping step considerably versus a raw GA4 table.

#### Dependencies and scope

- New dependency: `first-order-engine[bigquery]` (which itself pulls in
  `google-cloud-bigquery`, `google-auth-oauthlib`, and
  `google-cloud-resourcemanager`). Kept as an **optional extra** in
  `requirements.txt`/`setup.py` — most users running the CSV-only workflow
  shouldn't need to install or configure Google auth libraries at all.
  Consistent with the "runs on a standard laptop" design goal already
  documented in the README.
- OAuth `client_id` / `client_secret` / `redirect_uri` live in
  `.streamlit/secrets.toml` (gitignored, per-deployment) — see below.
  Exchanged `Credentials` live only in `st.session_state` for the session;
  never persisted to disk.
- **Out of scope for v1**: writing back to BigQuery (not needed — PRoX is
  read-only by design, and `DataEngine` itself never issues DDL/DML for
  this recipe), scheduled/incremental refresh (this is the same territory
  as Phase 5's incremental-analysis idea above and should stay deferred
  alongside it), multi-account switching, and query-cost governance beyond
  the basic dry-run estimate.

#### Secrets scaffold (prepared)

`.streamlit/secrets.toml` (gitignored) and a checked-in
`.streamlit/secrets.toml.example` now exist with the `[bigquery]` keys
`DataEngine`'s OAuth flow needs (`client_id`, `client_secret`,
`redirect_uri`) plus the default `BQConnectionConfig` fields (`project`,
`dataset`, optional `location`) so a picker has sane defaults before the
user has authenticated. Values are placeholders — fill in from the GCP
OAuth client used for this deployment.

#### Utility assessment — resolved (2026-08-20)

`foe.data.sql.event_log.build_event_log()` was reassessed against PRoX's
actual consumption path (`load_and_validate_csv()`'s `CRITICAL_COLS`,
`COLUMN_MAPPINGS`, and `analytics.py`'s revenue/user-column resolution).
Three real gaps were found and have since been fixed upstream in
`first-order-engine`:

- **No separate `user_id` column** (needed for PRoX's composite key) —
  fixed via `include_user_id` (default `True`), which emits
  `user_pseudo_id AS user_id` alongside `case_id`.
- **No session-level case granularity** (only user-for-the-whole-range) —
  fixed via `session_id_param`: pass `'ga_session_id'` and the query pulls
  the nested int-valued event_params key via a correlated subquery, cast
  to STRING, instead of a flat `case_id_col`. Paired with `include_user_id`,
  this produces exactly the (user_id, session_id) pair PRoX's composite key
  expects — mutually exclusive with a non-default `case_id_col` (enforced
  by a model validator).
- **No numeric revenue** (`attribute_params` only reads `string_value`,
  numeric params come back NULL) — fixed via `include_purchase_revenue`
  (adds `ecommerce.purchase_revenue AS revenue`, a flat typed `FLOAT64`
  column) and the more general `numeric_attribute_params` for other numeric
  event params.

**Recommended `EventLogExtractionParams` defaults for the PRoX
integration**: `session_id_param="ga_session_id"`, `include_user_id=True`,
`include_purchase_revenue=True`, `activity_col="event_name"` (default).
With these, `extract_event_log()`'s output needs zero PRoX-side column
mapping changes — `case_id`/`activity`/`timestamp`/`user_id`/`revenue` all
land on existing `COLUMN_MAPPINGS` entries.

**New consideration surfaced during this check**: `foe`'s `pyproject.toml`
lists `prophet`, `statsmodels`, `pingouin`, and `patsy` as unconditional
base dependencies (not gated behind the `bigquery` extra), since
`foe.data` is a subpackage of the whole `foe` library — installing
`foe[bigquery]` for just `DataEngine` also pulls in Prophet, which
typically needs a compiled Stan backend on first install. Worth stating
plainly in install docs as the known cost of this optional path, since it
cuts against PRoX's "no compilation step, runs on a standard laptop"
positioning. Also: `foe` requires Python ≥3.10 vs. PRoX's README-stated
3.9+ base requirement — noted directly in `requirements.txt`'s `[bigquery]`
comment as shipped, so this doesn't need a separate doc update.

#### Segment/revenue wiring — resolved (2026-09-22)

Revenue and standard GA4 segment dimensions (device, traffic source,
category) were reported missing from live BigQuery extractions. Root
cause for revenue: not a bug — `ecommerce.purchase_revenue` is only ever
populated on `purchase` events, so it reads NULL elsewhere by design.
`first-order-engine` (PR #14) added a validator rejecting
`include_purchase_revenue=True` combined with a restricted `event_names`
list that omits `'purchase'`, since that combination silently zeroes out
every row's revenue. Device/traffic source/category had no extraction
path at all (they live outside `event_params`, so `attribute_params`
couldn't reach them) — fixed via three new `EventLogExtractionParams`
flags: `include_device`, `include_traffic_source` (session-scoped
last-click, COALESCEd to the event-level struct for older export
schemas), `include_item_category`. A capped preview
(`DataEngine.preview_columns()`, ≤50 rows, narrowed date window) was
added alongside these so a caller can verify column data is populated
before running a full extraction.

`utility/bigquery_source.py` now sets all three new flags (matching the
existing always-on `include_user_id`/`include_purchase_revenue` pattern —
no UI toggle, since PRoX's downstream features want these whenever
they're available), wraps `EventLogExtractionParams` construction in a
try/except for the new validator, and adds a "Verify columns before
running" expander that calls `preview_columns()` and shows per-column
non-null counts. No `prox/config.py` changes were needed: `device_category`/
`traffic_source`/`traffic_medium` aren't required concepts, they pass
through as ordinary columns and are picked up automatically by the
Segment Comparison tab's generic low-cardinality column scan. The static
SQL template in `main.py` (for CSV-export users) was updated to match, so
both of PRoX's GA4 data paths agree on column names.

`first-order-engine` later added a fourth flag, `include_geo` (adds
`geo.country AS geo_country`), as a more reliable fallback segment
dimension than `category` — GA4's automatic IP geolocation populates it
regardless of the property's ecommerce/GTM setup, unlike `item_category`
which depends on that setup actually being in place. `utility/
bigquery_source.py` sets `include_geo=True` alongside the other always-on
flags, and the `main.py` SQL template gained a matching `geo.country AS
geo_country` line. Same reasoning as above applies: no `prox/config.py`
change needed, `geo_country` is picked up automatically by the Segment
Comparison tab's generic column scan.

#### Open questions to resolve before implementation

- Where does OAuth client registration (GCP project, redirect URI) live —
  is this a per-deployment config the person running PRoX sets up once
  (the `.streamlit/secrets.toml` approach above assumes this), or something
  each end user configures themselves? The secrets-file approach only
  covers the former; a multi-tenant deployment would need a different
  answer.
- `event_names`/`attribute_params`/`numeric_attribute_params` need a UI
  decision: expose as advanced/overridable fields, or hardcode sane
  defaults for v1 and revisit if a real dataset needs otherwise.

**Resolved (2026-08-20, as shipped):** OAuth client registration is a
per-deployment `.streamlit/secrets.toml` config the person running PRoX
sets up once - the multi-tenant case was out of scope. `event_names` is
exposed as an advanced, comma-separated text field ("restrict to specific
events"); `attribute_params`/`numeric_attribute_params` were left
hardcoded to sane v1 defaults rather than exposed, since no real dataset
has needed otherwise yet.

### Phase 6 — Session insight, reporting & data controls

- **Case grouping default switched to user-level** - `case:concept:name`
  now defaults to `user_id` (previously always a `user_id + session_id`
  composite), so a case can span a user's whole session history instead of
  just one session. A per-user-scoped `session_id` column is always kept
  regardless, and a UI toggle switches back to the previous per-session
  grouping.
- **Session-level intent classification** - `classify_sessions()` /
  `summarize_user_journeys()` in `prox/analytics.py`: a transparent,
  priority-ordered rule set (Buying > Cart Abandonment > Researching >
  Browsing) labels every session from its activities, then rolls a user's
  session labels into a chronological journey string (e.g. "Browsing ->
  Researching -> Buying"). Surfaced in a new **Session Insights** tab.
- **Configurable process end point** - a "Process end point" selector in
  the Filter Events step anchors analysis to a chosen activity (defaulting
  to `purchase` when present) via the existing `crop` filter, instead of
  that only being settable in code.
- **Modular, opt-in PDF report builder** (`utility/pdf_builder.py`) - a separate
  tool from `prox/report.py`'s all-in-one `generate_html_report()`: checks
  per available results tab (Process Maps, Variants, Bottlenecks,
  Conformance, Funnel, Business Insights, Session Insights, Segment
  Comparison), builds a PDF containing only what's checked, via `reportlab`
  (pure Python, no system rendering dependency - no wkhtmltopdf binary, no
  Cairo/Pango).
- **Smoothly ticking progress percentage** - the analysis progress bar
  previously only updated at 6 coarse pipeline-stage boundaries, sitting
  frozen for long stretches (especially during State Equation A*
  conformance). `run_full_analysis` now executes on a background thread
  while the main thread drives the bar from elapsed time via an asymptotic
  curve (capped at 95% until actually done), independent of the real
  per-stage callback, which still supplies the stage label text.
- **Sampling stratification exposed in the UI** - `strata_col` /
  `max_priority_ratio` (stratified sampling that reserves part of the
  conformance sample for cases where a chosen column = 1, e.g. purchases,
  so they aren't sampled away) existed in the pipeline but were hardcoded
  to `'purchase'` and never surfaced. Now a "Prioritise a column when
  sampling" selector (binary 0/1-style columns only) plus a max-priority-
  share slider in the Sampling step.
- **Opt-in revenue/price winsorization** - `prox.winsorize_series()` (same
  technique as first-order-engine's
  `ContinuousMetricEngine.winsorize_series`: cap at mean +/- N std devs, or
  a percentile band) applied to the revenue/price column right after data
  is loaded/cached, before any filtering/sampling/analysis reads it - caps
  outlier values instead of dropping the rows, so Average Order
  Value/revenue trend/category breakdown aren't diluted by a handful of
  extreme orders.

### Phase 6b — Full-pipeline correctness pass

A deliberate line-by-line review across `prox/discovery.py`,
`conformance.py`, `analytics.py`, and `data_manager.py`, prompted by the
Phase 6 work above. Each finding was verified empirically or by direct
reproduction before fixing, with a regression test added per fix:

- Inductive Miner discovery called `inductive_miner.apply()` without
  `variant=Variants.IMf`, so the Noise Threshold slider had been a
  complete no-op - noise filtering only exists under `IMf`, not the
  default `IM` variant.
- `_fitness_state_equation_alignments()` accepted `initial_marking`/
  `final_marking` but never used them, instead guessing markings from net
  topology - usually harmless for a discovered net, but capable of
  silently corrupting fitness/alignments for reference-model topologies
  where the real start/end doesn't coincide with sourceless/sinkless
  places. Now uses the real markings, matching the sibling
  `_fitness_token_replay()`.
- Purchase/cart/research-keyword matching built a regex via
  `'|'.join(values)` with no guard for an empty list - `'|'.join([])` is
  `''`, and `str.contains('')` matches every row, silently misclassifying
  everything as a match instead of nothing. Fixed at all 5 call sites via
  a shared `_contains_any()` helper.
- `refine_activity_labels()` decided whether to apply URL-cleaning based
  only on the first matched row's value, applying that one decision to the
  whole column - a column mixing plain and URL-like values leaked raw
  query strings/slashes into activity names for every row that didn't
  match the first row's style. Cleaning is now applied per row.

### Phase 7 — Incremental analysis (data-level caching)

**Shipped 2026-09-08** (`prox/incremental.py`, wired into `main.py`'s new
"2. Incremental Analysis" step).

**Scope, deliberately narrower than the name suggests**: this is incremental
*data ingestion*, not incremental *algorithmic* discovery/conformance. pm4py
has no "add one more trace" incremental mode for either, so
`run_full_analysis()` still runs a full batch pass every time - what this
avoids is the load/clean/merge round trip on data already seen before, and
it lets a user upload only the *new* rows of a recurring export (e.g. "this
week's GA4 CSV") instead of re-exporting and re-uploading the full history
each time.

**How it works**: an opt-in "Merge with cached data for a recurring dataset"
checkbox, keyed on a user-chosen **Dataset ID** (not the uploaded filename,
which usually changes every export - e.g. a trailing date). On merge, the
freshly-loaded upload is anti-joined against the cached copy on
`(case:concept:name, concept:name, time:timestamp)` - the same triple
`check_data_quality()` already uses to flag exact duplicates - so only
genuinely new events are added; the merged result is written back as the
new cache (write-through) and fed into the rest of the page exactly as if a
bigger file had been uploaded directly. A "Clear cache for this Dataset ID"
button resets it. Cache lives on disk under `.prox_cache/` (gitignored) as
one gzipped CSV + a small JSON manifest per Dataset ID - no new dependency
(no pyarrow/fastparquet), consistent with the "no compiled dependency, runs
on a standard laptop" stance in the README.

**Guardrail**: a cache built under one case-grouping setting (`user` vs.
`session`) refuses to merge with a run using the other - case identities
aren't comparable across the two, so mixing them would silently corrupt
case boundaries. The merge is skipped with a warning instead; the upload
still runs (unmerged), it just doesn't get cached under that mismatched
identity.

**Deliberately out of scope, revisit only if the ML layer's own design
changes**: the ML/Predictive Insights tab (`ML_roadmap.md`) is not wired
into this cache. That feature's v1 design retrains synchronously on
whatever log is currently loaded, with no model persistence/versioning -
mixing it with this cache would mean solving label churn (a case "in
progress" at cache time can resolve to a labelled outcome once new data
arrives) and model versioning, neither of which this module attempts. See
`prox/incremental.py`'s module docstring for the full reasoning.

### Phase 7b — High-res process map export & saved-run library

**Shipped 2026-09-23** (`prox/saved_runs.py`, `prox/visualizer.py`, wired
into `main.py`).

**High-res process map export**: `visualize_focused_insights()` and
`render_petri_net()` now render an SVG alongside every process-map PNG
(happy path, main flow, per-segment happy paths, and the discovered/
reference-model images in the Conformance tab), reusing the same rendered
graphviz object so it costs one extra `dot` invocation rather than a
re-discovery. SVG is vector, so activity names stay legible at any zoom or
print size - the on-screen PNG was too low-resolution to read once
exported. A "Download High-Res (SVG)" button sits next to each process map
image.

**Saved-run library**: a "Save Event Log to Library" action (next to the
results header's existing report download) lets an analyst label and
persist the event log behind a completed run, then pick it back up later
from Step 1 → "Load saved run" - without re-running conformance checking,
sampling, or any other configurable, which can be too slow to redo on
demand. Unlike `prox/incremental.py`'s Dataset ID (a reused key for one
recurring cache), a saved-run label is expected to repeat across saves
(e.g. the same client saved every month), so each save gets its own
`run_id` keyed by label + timestamp, never overwriting a prior entry.

A save keeps the analysis, not just its input: the run's config goes in the
manifest, and loading the entry restores it into the sidebar, filter, and
sampling widgets; the pipeline results are pickled alongside (with the chart
images copied out of the shared `output/` folder, which the next run
overwrites), so the results tabs come back without clicking Run Analysis.
Saving the same label with the same event log and config again returns the
existing entry instead of adding a duplicate. For a cache-linked entry whose
cache has since grown via an incremental merge, the saved results no longer
describe the data, so only the settings are restored and the user is told
to re-run.

**Coupling with the incremental cache**: if the event log being saved is
already sitting in an incremental-cache dataset (loaded via "Load cached
dataset", or merged into one via Incremental Analysis), the saved-run
manifest records that `source_dataset_id` and does not write a second copy
of the data - loading it reads through to that cache instead. If the
linked cache is later cleared, loading the saved run surfaces a clear
error rather than a crash or stale data. Deleting a saved-run entry never
touches the cache it links to, since that's a shared resource managed
separately via "Clear cache for this Dataset ID".

### ML layer — conversion propensity + drivers

**Engine shipped 2026-09-21** (#40, `prox/predictive.py`,
`tests/test_predictive.py`) - the "conversion propensity + driver analysis"
idea scoped in `ML_roadmap.md`: `train_propensity_model()`,
`analyze_propensity_drivers()`, `summarize_propensity_scores()`, and the
completed/in-progress case split (`split_completed_in_progress()`) it all
sits on. Optional dependency (`pip install -e ".[ml]"`, scikit-learn), same
pattern as the BigQuery extra above.

**UI shipped 2026-09-21** (#41): a **Predictive Insights** tab with an
outcome picker, optional attribute/revenue columns, validation metrics,
plain-language drivers and an aggregate in-progress summary, captioned as a
prediction rather than a measurement. Since `bc208e5` (2026-09-29) the model
trains on the raw log with only event-level filter steps and no sampling. A
crop at the outcome would remove every negative case, and the stratified
sample over-represents purchases. This makes it the one tab whose case set
can differ from the rest. Detail in `ML_roadmap.md`.

Deviates from `ML_roadmap.md`'s original framing in three ways, decided
during implementation: plain logistic regression instead of
HistGradientBoostingClassifier (explainability, and the driver analysis is
model-agnostic permutation importance either way); stratified k-fold CV
instead of a single train/test split (more honest given case counts will
often sit close to the minimum-data guard); and no public per-case scoring
output at all (not even a "highest risk sessions" table) - PRoX runs on a
manually exported, reviewed-later log, so a named in-progress case would
very likely have already resolved by the time a stakeholder reads it,
risking a stale-looking prediction casting doubt on the feature's other,
aggregate numbers. `ML_roadmap.md`'s open questions are resolved inline
there.

### AI Conclusion (optional, Gemini)

**Shipped 2026-09-29** (#51, `utility/ai_payload.py`, `utility/ai_client.py`).
An "AI Conclusion" expander under the results tabs sends an aggregate-only
digest of the run to Gemini: no case, user or session IDs, and no
per-resource rows. The digest includes any follow-up analyses run in the tabs
(reference-model conformance, funnel by segment, segment comparison, the
propensity model). Gemini returns a summary, key findings and next steps in
Dutch or English. The conclusion is kept separate from the deterministic
Executive Summary. It can be added to the custom PDF report and is dropped
from the PDF once the results change. Needs `pip install -e ".[ai]"` and
`GEMINI_API_KEY` in `.streamlit/secrets.toml`. Its scope is wider than the
recommendations-only idea it was designed as. Follow-ups shipped
2026-09-30: a preview of the exact JSON payload before anything is sent, a
note that the user's own key and Google's pricing apply, and the conclusion
frozen into the HTML report as plain text. As-built detail is in
`AI_summary_roadmap.md`.

### Phase 8 — Memory, rerun cost & analysis consistency

**Shipped 2026-09-22 to 2026-09-29.** Hardening driven by real, large GA4
logs rather than new capability. Performance detail is in
`dev_optimization.md`.

- **Memory** (#47): a budget on ETConformance precision (the main cause of
  the OOM kills), DataFrames passed to PM4Py instead of `EventLog` objects
  (about 4x less peak memory), capped Streamlit caches, and a 30 s per-trace
  limit for State Equation A\*. Loaders moved to `st.cache_resource` (#49),
  and the 500 MB upload limit was removed.
- **Rerun cost**: an Apply Configuration form for filters and sampling (#45),
  nothing loaded until a data source is picked (#52, #53), and
  identity-keyed caching of log-derived data with downloads built on click
  (#55).
- **Sampling applies to the whole pipeline** (#43): when enabled, it is taken
  once after filtering and reused by every stage, so business insights and
  conformance describe the same cases.
- **Every tab reads the filtered log** (#54, `e5068a8`): Reference Model,
  Funnel and the pipeline all use the configured `filter_steps`, so events
  removed as noise no longer count as deviations. The one deliberate
  exception is Predictive Insights (see the ML layer above).
- **Duplicate events removed by default** (#50): see "Data-quality
  pre-check" under Product development suggestions.
- **`purchase`/`add_to_cart` flags derived from activity names** (#42), so
  stratified sampling works on real data, not just the mock data.

---

## In progress

Nothing currently in progress.

---

## Roadmapped (not yet scheduled)

Nothing below is committed, scoped or sequenced. Rechecked against `main`
on 2026-09-30. The ML layer and the AI Conclusion used to be listed here.
Both have shipped (see Completed phases above), so only their follow-ups
remain. The AI Conclusion's follow-ups have shipped too.

### Follow-ups on shipped features

- **Predictive Insights** (`ML_roadmap.md`): left out of v1 on purpose and
  still open:
  - multi-class outcomes
  - saving or exporting trained models
  - automatic hyperparameter tuning
  - using the incremental cache, which first needs a fix for label churn
- **Performance** (`dev_optimization.md`):
  - The "CPU Cores" control has no effect, because PM4Py 2.7 ignores
    `cores`. Either remove the control or wait for a PM4Py version that
    honours it.
  - Token Replay is slow on very long traces (runtime only; memory stays
    bounded).
  - Pre-discovery downsampling is waiting on a concrete pain report.
- **BigQuery source**: `attribute_params`/`numeric_attribute_params` are
  still hardcoded, and switching between accounts isn't supported. Neither
  has been needed on a real dataset yet.

### Process mining capability gaps

Surfaced from a sophistication assessment (2026-08-21). PRoX is a real
discovery/conformance engine (Inductive Miner, Heuristics Miner, DFG, Token
Replay, State Equation A\* alignments, reference-model conformance). On top
of that it has an e-commerce/CRO analytics layer and, since 2026-09-21, a
predictive layer. That puts it ahead of a hobby pm4py script and behind an
enterprise platform like Celonis or Disco. Most of the gap is in five
capabilities those platforms have, listed smallest effort first:

1. **Organizational/resource-perspective mining (partly shipped).** The
   Bottlenecks tab's Resource Performance table (#25) shows events, cases
   touched and mean/median processing time for each resource, when the log
   has a resource column. Still missing: a handover-of-work network (who
   hands cases to whom, and how often) and workload over time. Both extend
   the same `analyze_process_performance()` code path and need no new data.

2. **Interactive process explorer.** Process maps are static Graphviz
   images: a PNG on screen, plus an SVG download since Phase 7b. There is no
   clickable, filterable or animated flow view yet. This is real work but
   bounded: an interactive graph component (or a vis.js/d3 embed) in place of
   the images, with click-to-filter wired into the existing filter config.

3. **Decision-point (data-aware) mining.** This explains *why* a case took
   one branch at a choice point, e.g. "cases with `device=mobile` skip the
   comparison step 80% of the time". The standard approach is to find the
   XOR choice points in the discovered model, then fit a decision tree per
   choice point on the attributes seen before it. PRoX has no code for this
   yet. The optional `[ml]` extra (scikit-learn) already provides the
   decision tree.

4. **Time-perspective prediction (remaining-time/SLA forecasting).**
   Timing analysis is still descriptive: bottleneck durations, lead time,
   duration by hour and day. It can't answer "this case is at step X, when
   will it finish, and will it breach an SLA?" The most natural home is an
   extension of the shipped propensity model in `prox/predictive.py`. That
   model already has prefix-based features, the completed/in-progress split
   and cross-validation. What it would need is a regression target and a
   validation approach suited to it.

5. **Multi-tenant / hosted deployment layer.** Authentication, session
   isolation, per-user storage and audit logging. Analyses now persist to
   local disk (`.prox_cache/` for the incremental cache, `.prox_saved_runs/`
   for saved runs), but that storage is shared, single-user and has no
   access control. Moving from one local Streamlit process on an analyst's
   laptop (the README's stated design goal) to a shared, hosted service is
   by far the largest item. It's a different deployment model for the whole
   app, not a new analytical capability. Out of scope unless that
   positioning changes.

### Product development options

Where the product could go next, given what has shipped: discovery,
conformance (including against a reference model), bottlenecks, variants,
funnel and business insights, session insights, segment comparison,
predictive insights, the AI Conclusion, HTML/PDF reporting, the BigQuery
source, incremental analysis and saved runs. Roughly ranked by effort vs.
payoff.

#### Quick wins (small, builds on what already exists)

**Shipped:**
- **Funnel x Segment cross-analysis — done.** The Funnel tab has an optional
  "Split by segment" selector (same 2-20-unique-value column candidates as
  Segment Comparison). `analyze_funnel_by_segment()` in `prox/analytics.py`
  reuses the overall funnel's stage order for every segment, so results are
  directly comparable - "does mobile drop off earlier than desktop?" is now
  a chart and a table, not a manual cross-reference between two tabs.
- **Data-quality pre-check — done.** A "Data Quality Check" step runs right
  after loading, before filtering/analysis. `check_data_quality()` in
  `prox/data_manager.py` flags exact duplicate events, single-event cases
  (no transitions to analyse), and events logged out of chronological order
  within a case - the log-shape problems `load_and_validate_csv()`'s own
  null/timestamp checks don't catch. Flagged duplicates are then removed
  before analysis by `drop_duplicate_events()` (keeping the first of each),
  controlled by the on-by-default **Remove Duplicate Events** sidebar
  option.
- **Page-view merging — done.** Templated pages fire `page_view` plus their
  own event (`view_item`, `view_item_list`, ...) on one page load, which
  showed the page twice in the journey. `merge_page_views_into_page_events()`
  in `prox/data_manager.py` drops a `page_view` when a page-specific event
  fires with it (same case, same time or up to 5s later, same URL if a
  `page_location` column exists), leaving `page_view` for CMS pages. It needs
  no `page_type` column and runs before `refine_activity_labels`; it applies
  to the analysis frame only, not the raw log the Funnel/Segments tabs read.
  Controlled by the on-by-default **Merge page_view into page-specific
  events** sidebar option. Single-page-app caveat: it assumes those events
  mark a page load, so one that fires without a page change (a
  `view_item_list` carousel on the homepage) absorbs that page's `page_view`.

- **Config presets — done.** A preset is a small JSON file
  (`prox/presets.py`, stored in `.prox_presets/`) holding the sidebar,
  filter, sampling and funnel settings, with no event data, so it applies to
  any new upload. It is saved from the **Save settings as preset** expander
  above Run Analysis, and applied, imported (JSON file) or deleted from the
  sidebar's **Configuration preset**. Machine-specific tuning (CPU cores,
  chunk sizes) is left out. Activities, columns and funnel steps the new log
  doesn't have are dropped with a warning instead of failing.

**Still open:**
- **Funnel in saved runs.** Deliberately not part of the presets change, but
  prepared for it: the funnel definition has its own shape
  (`build_funnel_settings`), the funnel widgets already default from a
  `restored_funnel`, and `fit_funnel_to_log` is not preset-specific. Storing
  a `"funnel"` key in the saved-run manifest (written in the Save to Library
  handler) is all that is left; `main.py` already reads it back.
- **Predictive, AI and segment-comparison settings in presets.** Left out of
  v1 on purpose.

#### Medium bets (real feature work, clear value)

- **Segment comparison v2 — automated golden-path diffing.** Scoped and
  deliberately deferred in `dev_phase2.md`'s Phase 4 entry ("segment A
  visits checkout, segment B doesn't"). Worth revisiting now that v1 has
  real usage (parallel execution, its own report export, GA4 device, traffic
  source and country segments from BigQuery).
- **Cohort/retention view.** Current loyalty metrics (repeat rate,
  days between purchases) are transaction-level. A cohort retention curve
  (% of users from cohort week N still active in weeks N+1, N+2, ...) is a
  complementary lens that product/growth stakeholders look for. The input
  is already there: `user_id` is always kept and cases default to user
  level, and an incremental cache gives the longer history a cohort needs.
- **Two-log comparison.** Compare this week's export with last week's, or
  this month's with last month's: a diff over time, unlike segment
  comparison's split of a single upload. The inputs now exist, because
  saved runs keep each save of the same label (e.g. a client, every month)
  as a separate entry with its results. What's missing is the comparison
  itself: loading two runs side by side and diffing their metrics, variants
  and process maps.

#### Longer-term

The one large item still open is
the multi-tenant/hosted deployment layer, under Process mining capability
gaps above.
