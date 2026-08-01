# Roadmap

Longer-horizon feature ideas, distinct from `phase2.md`'s execution plan
(tests/CI/hardening) and `dev_optimization.md`'s performance work. Entries
here are scoped for discussion, not committed to a phase number yet.

## Product development suggestions

A working list of where the product could go next, given everything shipped
so far (discovery, conformance, bottlenecks, variants, business insights,
funnel analysis, segment comparison, HTML reporting). Roughly ranked by
effort vs. payoff — none of these are committed, scoped, or sequenced yet.

### Quick wins (small, builds on what already exists)

- **Funnel x Segment cross-analysis.** The Funnel tab and Segment Comparison
  are currently independent features that were built separately but combine
  naturally: "show the funnel drop-off for mobile vs. desktop." Wiring the
  Funnel tab to optionally split by the same segment column would surface
  exactly the kind of insight non-technical stakeholders actually ask for.
- **Config presets.** Save/load the sidebar configuration (discovery algo,
  sample size, filters, funnel definition) as a small JSON file. Removes the
  "reconfigure everything every session" friction for a repeat analyst —
  directly serves the "runs on a laptop, used repeatedly" use case.
- **Data-quality pre-check.** Before running the full pipeline, a lightweight
  summary of missing values, duplicate case IDs, and timestamp gaps/
  out-of-order events. Non-technical users currently only find out their
  data is messy after a confusing downstream result.

### Medium bets (real feature work, clear value)

- **Segment comparison v2 — automated golden-path diffing.** Already scoped
  and deliberately deferred (`phase2.md`'s Phase 4 segment-comparison entry:
  "segment A visits checkout, segment B doesn't"). Worth revisiting now that
  v1 has real usage patterns (parallel execution, its own report export).
- **Cohort/retention view.** Current loyalty metrics (repeat rate,
  days-between-purchases) are transaction-level. A cohort retention curve
  (% of users from cohort week N still active in week N+1, N+2, ...) is a
  different, complementary lens that product/growth stakeholders
  specifically look for and PRoX doesn't have yet.
- **Two-log comparison.** Compare this week's export vs. last week's, or
  this cohort's export vs. last month's — a temporal/version diff, distinct
  from segment comparison's categorical split of a single upload.

### Longer-term (bigger, already flagged elsewhere in these docs)

- **Phase 5 — incremental analysis** for recurring large logs
  (`phase2.md`). Still explicitly "no pain signal yet" — worth doing once
  someone is actually re-running PRoX on a growing dataset regularly, not
  before.
- **BigQuery live data source** (below). Removes the export-clean-upload
  cycle for GA4-in-BigQuery users. Biggest lift of anything on this list —
  only worth it if CSV upload is a genuine recurring friction point.

## BigQuery live data source (via Google OAuth)

**Idea**: instead of requiring the user to export, clean, and upload a CSV,
let PRoX connect directly to BigQuery and query event data live. Positioned
as a second, separate data-source workflow alongside the existing CSV
upload — not a replacement.

### Why

Today's flow (`main.py`) gates the entire app behind a single upload widget:
`uploaded_file = st.file_uploader(...)` in the sidebar, then `if not
uploaded_file: st.stop()` before anything else renders. For a user whose
event data already lives in BigQuery (a common case for GA4/web-analytics
exports), that means an export-clean-upload round trip every time they want
to look at a different date range or dataset. A live connection removes
that round trip entirely.

### Proposed shape

- **A data-source choice, shown first** — before the existing sidebar
  controls become interactive, similar to a landing step: "Upload CSV" or
  "Connect to BigQuery." This matches the existing gating pattern (`st.stop()`
  until data is ready) but adds a fork before it, rather than replacing it.
- **BigQuery path**:
  1. "Sign in with Google" — OAuth 2.0, scoped to `bigquery.readonly` only
     (PRoX is analysis-only; no write access should ever be requested).
  2. Once authenticated: either (a) a project/dataset/table picker that
     auto-generates a query, or (b) a free-form SQL editor for full
     flexibility. Recommend starting with (a) for v1 — safer, no risk of
     an accidentally-expensive or destructive query — and offering (b) later
     once there's a real need for arbitrary queries.
  3. Run a `dry_run` query first to estimate bytes scanned and warn the user
     before executing anything that could be costly, given BigQuery's
     pay-per-byte-scanned pricing.
  4. Execute, get a DataFrame back (via `google-cloud-bigquery`'s
     `to_dataframe()`), and feed it into the same validation/cleaning path
     the CSV upload already uses.
- **Shared validation logic**: `load_and_validate_csv()` currently mixes
  CSV-specific concerns (file-size checks, chunked reading) with genuinely
  reusable logic (column auto-mapping against `COLUMN_MAPPINGS`, composite
  case-ID creation, timestamp parsing, critical-column validation). This is
  a real refactor opportunity: split it into `_load_csv_source(...)` (CSV-only)
  and a shared `validate_and_clean_dataframe(df, ...)` that both the CSV path
  and a new BigQuery path call. Avoids duplicating the column-mapping and
  cleaning logic in two places.

### Dependencies and scope

- New dependency: `google-cloud-bigquery` plus an OAuth flow library
  (e.g. `google-auth-oauthlib`). This should be an **optional extra**
  (e.g. `pip install prox[bigquery]`), not a hard requirement — most users
  running the CSV-only workflow shouldn't need to install or configure
  Google auth libraries at all. Consistent with the "runs on a standard
  laptop" design goal already documented in the README.
- Credentials live only in `st.session_state` for the session; never
  persisted to disk.
- **Out of scope for v1**: writing back to BigQuery (not needed — PRoX is
  read-only by design), scheduled/incremental refresh (this is the same
  territory as `phase2.md`'s Phase 5 incremental-analysis idea and should
  stay deferred alongside it), multi-account switching, and query-cost
  governance beyond the basic dry-run estimate.

### Open questions to resolve before implementation

- Does the target BigQuery table already look like an event log (one row
  per event, with case/activity/timestamp-ish columns `COLUMN_MAPPINGS` can
  match), or would the query need to reshape data before it reaches PRoX?
  Likely varies per user/dataset — may need a lightweight "preview + confirm
  column mapping" step for the BigQuery path specifically, since there's no
  file to eyeball beforehand the way there is with a CSV.
- Where does OAuth client registration (GCP project, redirect URI) live —
  is this a per-deployment config the person running PRoX sets up once, or
  something each end user configures themselves? Affects how "opt-in" this
  really is in practice.
