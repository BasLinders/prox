# ML Roadmap

*See `dev_roadmap.md` for the current status of all phases at a glance —
that file is the leading document; this one is the detail.*

Machine-learning feature ideas for PRoX, kept separate from `dev_roadmap.md`
(data-source/product ideas) since they carry a different kind of risk: unlike
every other metric in PRoX, ML output is probabilistic, needs validation, and
can be wrong in ways that are harder to explain to a non-technical
stakeholder than "here's the bottleneck." Entries here are scoped for
discussion, not committed to a phase number yet.

## Conversion propensity + root-cause driver analysis

**Idea**: an opt-in "Predictive Insights" capability with two parts:
1. A binary classifier that predicts whether an in-progress case will reach
   a defined success outcome (e.g. `purchase`, or any funnel end-stage),
   trained on completed cases and scored against currently-incomplete ones.
2. A driver/root-cause analysis layer built from the same model, explaining
   *why* cases succeed or fail — not just reporting the rate.

### Why

PRoX already computes cart abandonment rate and funnel drop-off, but only
as aggregate, after-the-fact numbers. It doesn't say why specific cases
abandon, or flag which currently in-progress cases are trending toward
abandonment. A propensity model turns an aggregate stat into case-level,
actionable signal; a driver analysis explains root causes in the same
plain-language style the Executive Summary already uses ("mobile sessions
at 8pm are 3x more likely to abandon at checkout" vs. just "78% drop-off at
checkout"). Since PRoX runs after-the-fact on exported logs (batch, not
real-time), "in progress" prediction is inherently retrospective: train on
completed cases labelled by their actual outcome, validate on held-out
cases, then apply the trained model to whichever cases in the current log
haven't reached an outcome yet.

### Proposed shape

- **Outcome definition, user-picked.** Reuse the Funnel tab's pattern:
  let the user choose which activity counts as "success" from the
  activities already in their log (e.g. `purchase`, or any funnel
  end-stage), rather than hardcoding an e-commerce concept — keeps this
  industry-agnostic like the rest of PRoX.
- **Feature engineering from trace prefixes.** For each case, build
  features only from events *before* the outcome (or a cutoff point):
  activities visited so far, event count so far, elapsed time since case
  start, available case attributes (device/category/segment column,
  revenue-so-far), time-of-day/day-of-week of the first event. This is
  standard "predictive process monitoring" prefix-based feature
  engineering — no need to invent a new approach.
- **Model.** Start with one interpretable model family — scikit-learn's
  `HistGradientBoostingClassifier` (ships with scikit-learn, no extra
  heavy dependency like xgboost/lightgbm) or plain logistic regression as
  a simpler, more transparent baseline. No deep learning — consistent
  with the CUDA-rejection precedent in `dev_optimization.md` and the
  "runs on a standard laptop" design goal.
- **Validation, reported honestly.** Train/test split or k-fold CV on
  completed cases only, reporting standard classification metrics
  (accuracy, precision/recall, ROC-AUC) alongside the prediction — PRoX's
  other metrics are deterministic and fully auditable, so a probabilistic
  model needs to visibly show its own reliability rather than presenting
  predictions as fact.
- **Driver analysis.** Extract top-N important features from the trained
  model and translate them into plain-language sentences, reusing
  `report.py`'s narrative-generation pattern (e.g. "Sessions that reach
  'add_to_cart' within 2 minutes are 3x more likely to convert").
  Prefer **permutation importance** (built into scikit-learn, no new
  dependency) over raw impurity-based feature importances, which are
  known to bias toward high-cardinality features and would produce
  misleading "drivers."
- **Scoring in-progress cases.** Apply the trained model to
  currently-incomplete cases to produce a per-case abandonment-risk score,
  shown as a sortable "highest risk sessions" table — the actionable
  "who's about to abandon" output.
- **UI integration.** New tab (or an extension of the Funnel tab, since it
  shares the outcome-picker concept): outcome picker, "Train Model"
  button, validation metrics, a driver/feature-importance chart, and the
  scored case table. Visually distinguished from the deterministic metrics
  elsewhere (e.g. a clear "Predicted" label) to preserve the trust
  boundary between measured numbers and model output.

### Dependencies and scope

- New dependency: `scikit-learn` (pure Python + numpy/scipy, no GPU).
  Should be an **optional extra** (e.g. `pip install prox[ml]`), matching
  the pattern proposed for the BigQuery data source below — most users
  running the core discovery/conformance workflow shouldn't need to
  install it.
- Training happens synchronously within the Streamlit session, batch not
  online-learning — no model persistence/versioning needed for v1;
  retrains each session on the currently loaded log.
- **Out of scope for v1**: real-time/streaming scoring, deep learning
  models, automatic hyperparameter tuning (use sensible defaults),
  multi-class outcomes (v1 is binary success/failure only), saving or
  exporting trained models. Also out of scope: wiring into
  `prox/incremental.py`'s cache (`dev_roadmap.md`'s Phase 7) - that cache
  merges *data* across runs, but this feature's v1 retrains synchronously
  on whatever log is currently loaded with no model persistence/versioning,
  and mixing the two would mean solving label churn (a case "in progress"
  at cache time can resolve to a labelled outcome once new data arrives) -
  deliberately deferred, not attempted here.

### Open questions — resolved (2026-09-18)

- **Minimum data volume — resolved.** Hard-refuse (not warn), below 30
  completed cases in either class — a deliberate departure from the rest
  of the codebase's "always warn, never block" convention, since below
  that count a held-out CV fold leaves single-digit class counts, making
  every metric sampling-noise-dominated rather than actually informative.
  Implemented in `train_propensity_model()`'s `min_cases_per_class` guard
  (`prox/predictive.py`).
- **Leakage risk — resolved.** A single shared prefix-truncation function
  (`_extract_case_features()`, private) is the only place raw case events
  become a feature row, called identically by both the training-set and
  scoring-set builders — the two paths can't structurally diverge in what
  they include, only in which cutoff timestamp they pass. Covered by
  explicit regression tests in `tests/test_predictive.py`
  (`test_train_propensity_model_excludes_post_outcome_events_from_features`
  and others).
- **Where this fits relative to Funnel analysis — resolved.** A genuinely
  separate future "Predictive Insights" tab, not a Funnel tab extension —
  keeps deterministic Funnel metrics and probabilistic model output
  visually and architecturally separate, matching the trust-boundary
  concern above. The outcome-picker is its own independent parameter, not
  shared state with `analyze_conversion_funnel()`.

Three further decisions were made during implementation, not anticipated
above: **logistic regression** instead of HistGradientBoostingClassifier
(driver analysis is model-agnostic permutation importance either way, so
this costs little, and buys simpler, more transparent output); **stratified
k-fold CV** instead of a single train/test split (more honest validation
when case counts sit close to the minimum-data guard); and **no public
per-case scoring output** at all, not even a "highest risk sessions" table
— PRoX runs on a manually exported, reviewed-later log, so a named
in-progress case would very likely have already resolved to a real outcome
by the time a stakeholder reads a report, risking a stale-looking
prediction that casts doubt on the feature's other, aggregate numbers.
`summarize_propensity_scores()` is the only public scoring output:
aggregate distribution/risk-tier stats and plain-language narrative, never
a named case.
