"""
prox/predictive.py — Conversion propensity engine (predictive process monitoring).

Trains a binary classifier that predicts whether a case will reach a
user-chosen "success" activity, using only features derivable from the
events of a case *before* that outcome (or from all its events so far, for a
case still in progress) — standard prefix-based feature engineering for
predictive process monitoring, applied to PM4Py-standard event logs
(columns: case:concept:name, concept:name, time:timestamp, plus whatever
optional case attribute/revenue columns the log has).

Optional dependency: requires scikit-learn (`pip install prox[ml]`).
split_completed_in_progress() works with pandas/numpy alone, so a caller can
inspect "how many completed/in-progress cases do I have" without the [ml]
extra installed. train_propensity_model() (and, transitively,
analyze_propensity_drivers()/summarize_propensity_scores(), which both
consume its output) require it, and degrade to a message in the returned
`errors` list rather than raising ImportError - see _SKLEARN_AVAILABLE.

Central ambiguity, inherited from PRoX being a batch/offline tool (see
incremental.py's module docstring for the same framing applied to a
different problem): a static event log has no wall-clock "now", only its
own max timestamp. "Has this case had its full opportunity to convert and
failed to?" is therefore a heuristic, not a fact - see
split_completed_in_progress()'s docstring for exactly how it's decided.

Deliberately no public per-case scoring output. PRoX runs on a manually
exported, batch log reviewed after the fact, not live monitoring - by the
time an analysis reaches a stakeholder, any specific "in-progress" case has
very likely already resolved to a real outcome, making a named case-level
prediction look stale or simply wrong through no fault of the model, and
casting doubt on the feature's other, aggregate numbers alongside it. None
of PRoX's other metrics name individual cases (funnel drop-off %, cart
abandonment rate are aggregate snapshots) - this module follows that
precedent. Per-case scoring exists only as the private
_score_in_progress_cases() helper, consumed exclusively by
summarize_propensity_scores(), which is the only public scoring output.

Deliberately out of scope for v1 (see docs/ML_roadmap.md): UI, real-time/
streaming scoring, model persistence/export, multi-class outcomes, automatic
hyperparameter tuning, and wiring into incremental.py's cache (that cache
merges *data* across runs; this feature retrains synchronously on whatever
log is currently loaded, with no model persistence/versioning - mixing the
two would mean solving label churn, which isn't attempted here).

Public API
----------
split_completed_in_progress  Split cases into completed (positive/negative)
                              and in-progress buckets
train_propensity_model       Train + cross-validate a conversion propensity
                              classifier (requires prox[ml])
analyze_propensity_drivers   Plain-language top-driver sentences from a
                              trained model bundle
summarize_propensity_scores  Aggregate, non-case-level summary of in-progress
                              case scoring
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .analytics import _contains_any

logger = logging.getLogger(__name__)

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

_SKLEARN_INSTALL_MSG = (
    "scikit-learn is required for predictive modelling but isn't installed. "
    "Install it with: pip install prox[ml]"
)

_TOP_K_ACTIVITIES = 20
_MIN_CASES_PER_CLASS = 30
_MIN_POSITIVES_FOR_CENSOR_PERCENTILE = 5


def _coerce_to_list(values) -> list:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def split_completed_in_progress(
    df: pd.DataFrame,
    outcome_activity,
    case_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    timestamp_col: str = 'time:timestamp',
    censor_percentile: float = 0.90,
) -> Dict[str, Any]:
    """
    Splits cases into three buckets used throughout this module:

      - positive:    reached outcome_activity at least once -> completed,
                      label=1. Feature prefixes must stop strictly before the
                      *first* occurrence's timestamp.
      - negative:     never reached outcome_activity, AND is judged to have
                      had its full opportunity to -> completed, label=0. All
                      of the case's events are usable as features.
      - in_progress:  never reached outcome_activity, and is still within its
                      plausible conversion window -> excluded from training,
                      used only as scoring input.

    Heuristic ("had its full opportunity"), since a static batch log has no
    wall-clock "now" other than its own max timestamp:

      1. T = the `censor_percentile`-th percentile of (outcome_timestamp -
         case_start_timestamp) across POSITIVE cases only - "how long does a
         case that *does* convert typically take, with margin for the slow
         tail". Requires >=5 positive cases to estimate T stably; below that,
         falls back to the same percentile of ALL cases' (last_event -
         first_event) duration, and a note is added to 'errors' (not fatal).
      2. A non-outcome-reaching case is a confirmed NEGATIVE if
         (log_max_timestamp - case_last_event_timestamp) >= T - enough time
         has elapsed since this case's last event, relative to the log's own
         end, that a normally-converting case would already have converted
         by now. Otherwise it's IN_PROGRESS.

    This is a right-censoring heuristic (survival-analysis framing): only
    cases that have "aged out" of the plausible conversion window count as
    confirmed negatives; anything more recent is treated as unresolved
    rather than guessed at. This is deliberately conservative - it costs
    training negatives near the end of the log's coverage, but avoids
    injecting label noise from cases that were simply about to convert when
    the export was taken.

    Known limitation (documented, not solved here): a genuinely abandoned
    case that happens to sit just inside the window is excluded from
    training (fewer negatives, not wrong ones); a genuinely slow-but-still-
    active case beyond the window is mislabelled negative (label noise). No
    log-only heuristic can fully distinguish these - a real "now" timestamp,
    not present in a static export, would remove the ambiguity.

    Returns
    -------
    dict with keys: 'positive_case_ids', 'negative_case_ids',
    'in_progress_case_ids' (all list), 'outcome_timestamps'
    (dict[case_id, pd.Timestamp], positive cases only - first occurrence of
    outcome_activity), 'censor_cutoff' (pd.Timedelta), 'errors' (list of str)
    """
    errors: List[str] = []
    outcome_values = _coerce_to_list(outcome_activity)

    empty = {
        'positive_case_ids': [], 'negative_case_ids': [], 'in_progress_case_ids': [],
        'outcome_timestamps': {}, 'censor_cutoff': None, 'errors': errors,
    }

    if case_col not in df.columns or activity_col not in df.columns or timestamp_col not in df.columns:
        errors.append("Missing required columns for completed/in-progress split.")
        return empty
    if df.empty:
        errors.append("Cannot split an empty event log.")
        return empty
    if not outcome_values:
        errors.append("No outcome activity given.")
        return empty

    outcome_mask = _contains_any(df[activity_col], outcome_values)
    first_outcome_ts = df[outcome_mask].groupby(case_col)[timestamp_col].min()
    case_start = df.groupby(case_col)[timestamp_col].min()
    case_last = df.groupby(case_col)[timestamp_col].max()
    log_max = df[timestamp_col].max()

    positive_ids = first_outcome_ts.index.tolist()

    if len(positive_ids) >= _MIN_POSITIVES_FOR_CENSOR_PERCENTILE:
        positive_durations = first_outcome_ts - case_start.loc[positive_ids]
        censor_cutoff = positive_durations.quantile(censor_percentile)
    else:
        errors.append(
            f"Only {len(positive_ids)} positive case(s) found - too few to estimate a "
            "reliable conversion-time window. Falling back to the overall case-duration "
            "percentile, which is a coarser proxy."
        )
        all_durations = case_last - case_start
        censor_cutoff = all_durations.quantile(censor_percentile)

    non_outcome_ids = [c for c in df[case_col].unique() if c not in set(positive_ids)]
    age = log_max - case_last.loc[non_outcome_ids]
    negative_ids = age[age >= censor_cutoff].index.tolist()
    in_progress_ids = age[age < censor_cutoff].index.tolist()

    return {
        'positive_case_ids': positive_ids,
        'negative_case_ids': negative_ids,
        'in_progress_case_ids': in_progress_ids,
        'outcome_timestamps': first_outcome_ts.to_dict(),
        'censor_cutoff': censor_cutoff,
        'errors': errors,
    }


def _extract_case_features(
    case_events: pd.DataFrame,
    cutoff_timestamp,
    activity_vocab: list,
    case_attribute_cols: list,
    revenue_col,
    activity_col: str,
    timestamp_col: str,
) -> Dict[str, Any]:
    """
    Builds one feature row from a case's events, truncated to strictly
    before `cutoff_timestamp` (or all events, if cutoff_timestamp is None).

    This is the ONLY place raw case events become a feature row - called
    identically by the training-set builder (cutoff = first outcome
    timestamp for positives, None for negatives) and the scoring builder
    (cutoff = None, since an in-progress case's own events already ARE the
    prefix - there is nothing after it in the log to leak). Structurally,
    there is no second implementation that could quietly diverge and admit
    post-outcome information - only the cutoff argument differs per caller.

    The reference time for every "time since" feature is the prefix's own
    last event, never the log's overall max timestamp - using the log's max
    would leak future information into a positive case's features.
    """
    prefix = case_events if cutoff_timestamp is None \
        else case_events[case_events[timestamp_col] < cutoff_timestamp]
    prefix = prefix.sort_values(timestamp_col)

    features: Dict[str, Any] = {}
    event_count = len(prefix)
    features['event_count_so_far'] = event_count
    features['unique_activities_so_far'] = prefix[activity_col].nunique() if event_count else 0

    visited = set(prefix[activity_col]) if event_count else set()
    for act in activity_vocab:
        features[f'visited_{act}'] = int(act in visited)

    features['last_activity'] = prefix.iloc[-1][activity_col] if event_count else '<none>'

    if event_count:
        reference_time = prefix[timestamp_col].max()
        start_time = prefix[timestamp_col].min()
        features['elapsed_seconds_since_case_start'] = (reference_time - start_time).total_seconds()
        features['first_event_hour'] = start_time.hour
        features['first_event_dayofweek'] = start_time.dayofweek
    else:
        features['elapsed_seconds_since_case_start'] = 0.0
        features['first_event_hour'] = np.nan
        features['first_event_dayofweek'] = np.nan

    if event_count > 1:
        diffs = prefix[timestamp_col].diff().dt.total_seconds()
        features['mean_seconds_between_events'] = diffs.mean()
    else:
        features['mean_seconds_between_events'] = np.nan

    if revenue_col and revenue_col in prefix.columns:
        features['revenue_so_far'] = pd.to_numeric(prefix[revenue_col], errors='coerce').fillna(0).sum()

    for col in case_attribute_cols:
        if col in prefix.columns:
            non_null = prefix[col].dropna()
            features[col] = non_null.iloc[0] if not non_null.empty else '<unknown>'
        else:
            features[col] = '<unknown>'

    return features


def _build_propensity_features(
    df: pd.DataFrame,
    outcome_activity,
    case_col: str,
    activity_col: str,
    timestamp_col: str,
    case_attribute_cols: list,
    revenue_col,
    top_k_activities: int = _TOP_K_ACTIVITIES,
    censor_percentile: float = 0.90,
) -> Dict[str, Any]:
    """
    Orchestrates split_completed_in_progress() + _extract_case_features()
    into two feature tables. The activity_vocab used for `visited_<act>`
    columns is derived from COMPLETED cases only (never in-progress cases),
    so the feature space at scoring time is always a subset of what training
    saw and can't introduce columns the model was never fit on.
    """
    split = split_completed_in_progress(
        df, outcome_activity, case_col=case_col, activity_col=activity_col,
        timestamp_col=timestamp_col, censor_percentile=censor_percentile,
    )
    errors = list(split['errors'])

    completed_ids = split['positive_case_ids'] + split['negative_case_ids']
    completed_df = df[df[case_col].isin(completed_ids)]
    activity_vocab = (
        completed_df[activity_col].value_counts().head(top_k_activities).index.tolist()
        if not completed_df.empty else []
    )

    by_case = {cid: events for cid, events in df.groupby(case_col)}

    def build_rows(case_ids, cutoff_map):
        rows = []
        for cid in case_ids:
            events = by_case.get(cid)
            if events is None:
                continue
            row = _extract_case_features(
                events, cutoff_map.get(cid), activity_vocab, case_attribute_cols,
                revenue_col, activity_col, timestamp_col,
            )
            row[case_col] = cid
            rows.append(row)
        return pd.DataFrame(rows)

    positive_rows = build_rows(split['positive_case_ids'], split['outcome_timestamps'])
    negative_rows = build_rows(split['negative_case_ids'], {})
    in_progress_rows = build_rows(split['in_progress_case_ids'], {})

    if not positive_rows.empty:
        positive_rows['label'] = 1
    if not negative_rows.empty:
        negative_rows['label'] = 0

    training_features = pd.concat([positive_rows, negative_rows], ignore_index=True) \
        if not (positive_rows.empty and negative_rows.empty) else pd.DataFrame()

    return {
        'training_features': training_features,
        'scoring_features': in_progress_rows,
        'activity_vocab': activity_vocab,
        'categorical_columns': ['last_activity'] + list(case_attribute_cols),
        'split': split,
        'errors': errors,
    }


def train_propensity_model(
    df: pd.DataFrame,
    outcome_activity,
    case_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    timestamp_col: str = 'time:timestamp',
    case_attribute_cols: list = None,
    revenue_col: str = None,
    cv_folds: int = 5,
    min_cases_per_class: int = _MIN_CASES_PER_CLASS,
    random_state: int = 42,
    logreg_kwargs: dict = None,
) -> Dict[str, Any]:
    """
    Trains a logistic regression classifier to predict whether a case will
    reach `outcome_activity`, on trace-prefix features built from completed
    cases only (see split_completed_in_progress()). Chosen over a tree
    ensemble for explainability; since driver analysis
    (analyze_propensity_drivers()) uses model-agnostic permutation
    importance rather than reading model internals, this costs little.
    Unlike a tree ensemble, logistic regression has no native categorical/
    missing-value handling, so categorical columns are one-hot encoded
    (unseen categories at scoring time are ignored, not an error) and
    numeric columns are median-imputed then standardised, via one
    sklearn Pipeline/ColumnTransformer.

    Hard data-adequacy guard (a deliberate departure from the rest of
    PRoX's "always warn, never block" convention - see docs/ML_roadmap.md):
    refuses to train, returning model=None and a populated 'errors' list,
    unless there are at least `min_cases_per_class` (default 30) completed
    cases in BOTH the positive and negative classes. Below that, a held-out
    CV fold leaves single-digit counts per class, making every reported
    metric dominated by sampling noise rather than signal - not just "less
    accurate" but actively misleading if presented as a number. 30/class is
    a documented, overridable judgment call (rule-of-thumb floor for a
    stable evaluation), not a derived constant.

    Validation: stratified `cv_folds`-fold cross-validation (default 5) over
    all completed cases - more honest than a single train/test split when
    case counts are close to the minimum threshold, where one split's
    metrics would be high-variance. Permutation importance
    (sklearn.inspection.permutation_importance, scored on ROC-AUC) is
    computed per-fold on that fold's held-out portion and averaged, so
    driver analysis is also out-of-sample. The model returned for scoring
    is refit once on ALL completed cases afterward (more training data,
    better production model - standard practice) - documented trade-off:
    the reported metrics describe the cross-validation process's typical
    performance, not literally this exact refit model.

    Returns
    -------
    dict with keys:
        'model'                : fitted sklearn Pipeline, or None if
                                  refused/unavailable
        'feature_columns'      : list[str]
        'categorical_columns'  : list[str]
        'activity_vocab'       : list[str]
        'metrics'               : {'n_completed', 'n_positive', 'n_negative',
                                   'accuracy_mean', 'accuracy_std',
                                   'precision_mean', 'precision_std',
                                   'recall_mean', 'recall_std',
                                   'roc_auc_mean', 'roc_auc_std'} - {} if refused
        'permutation_importance': {feature: mean_importance across folds} -
                                   {} if refused
        'training_config'      : echo of outcome_activity/columns/
                                  censor_percentile actually used, so
                                  summarize_propensity_scores() can reproduce
                                  an identical split/feature space later
        'errors'                : list[str] - non-empty means refused-to-train
                                  or a hard failure; callers must check this
                                  BEFORE treating 'model'/'metrics' as present
    """
    case_attribute_cols = case_attribute_cols or []
    logreg_kwargs = logreg_kwargs or {}
    errors: List[str] = []

    training_config = {
        'outcome_activity': outcome_activity, 'case_col': case_col,
        'activity_col': activity_col, 'timestamp_col': timestamp_col,
        'case_attribute_cols': case_attribute_cols, 'revenue_col': revenue_col,
    }
    empty = {
        'model': None, 'feature_columns': [], 'categorical_columns': [],
        'activity_vocab': [], 'metrics': {}, 'permutation_importance': {},
        'training_config': training_config, 'errors': errors,
    }

    if not _SKLEARN_AVAILABLE:
        errors.append(_SKLEARN_INSTALL_MSG)
        return empty

    if case_col not in df.columns or activity_col not in df.columns or timestamp_col not in df.columns:
        errors.append("Missing required columns for propensity model training.")
        return empty
    if df.empty:
        errors.append("Cannot train a propensity model on an empty event log.")
        return empty

    built = _build_propensity_features(
        df, outcome_activity, case_col, activity_col, timestamp_col,
        case_attribute_cols, revenue_col,
    )
    errors.extend(built['errors'])
    training_features = built['training_features']

    n_positive = int((training_features['label'] == 1).sum()) if not training_features.empty else 0
    n_negative = int((training_features['label'] == 0).sum()) if not training_features.empty else 0

    if n_positive < min_cases_per_class or n_negative < min_cases_per_class:
        outcome_label = ' or '.join(_coerce_to_list(outcome_activity))
        errors.append(
            f"Not enough completed cases to train a trustworthy model: {n_positive} reached "
            f"'{outcome_label}', {n_negative} confirmed they didn't "
            f"(need at least {min_cases_per_class} of each). Refusing to train."
        )
        return empty

    feature_columns = [
        c for c in training_features.columns if c not in (case_col, 'label')
    ]
    categorical_columns = [c for c in built['categorical_columns'] if c in feature_columns]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns]

    X = training_features[feature_columns].copy()
    for c in categorical_columns:
        X[c] = X[c].astype(str)
    y = training_features['label'].astype(int)

    def make_pipeline():
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
            ('num', Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]), numeric_columns),
        ])
        clf_kwargs = {'class_weight': 'balanced', 'max_iter': 1000, 'random_state': random_state}
        clf_kwargs.update(logreg_kwargs)
        return Pipeline([('preprocess', preprocessor), ('clf', LogisticRegression(**clf_kwargs))])

    n_splits = max(2, min(cv_folds, n_positive, n_negative))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': []}
    importances_per_fold = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = make_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics['roc_auc'].append(
            roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else np.nan
        )

        try:
            perm = permutation_importance(
                pipeline, X_test, y_test, scoring='roc_auc', n_repeats=10,
                random_state=random_state,
            )
            importances_per_fold.append(dict(zip(feature_columns, perm.importances_mean)))
        except ValueError:
            pass

    metrics = {
        'n_completed': n_positive + n_negative,
        'n_positive': n_positive,
        'n_negative': n_negative,
    }
    for key, values in fold_metrics.items():
        clean = [v for v in values if not np.isnan(v)]
        metrics[f'{key}_mean'] = float(np.mean(clean)) if clean else np.nan
        metrics[f'{key}_std'] = float(np.std(clean)) if clean else np.nan

    permutation_importances = {}
    if importances_per_fold:
        for feat in feature_columns:
            vals = [fold.get(feat, 0.0) for fold in importances_per_fold]
            permutation_importances[feat] = float(np.mean(vals))

    final_model = make_pipeline()
    final_model.fit(X, y)

    return {
        'model': final_model,
        'feature_columns': feature_columns,
        'categorical_columns': categorical_columns,
        'activity_vocab': built['activity_vocab'],
        'metrics': metrics,
        'permutation_importance': permutation_importances,
        'training_config': training_config,
        'errors': errors,
        # Private (not part of the documented return shape): the raw,
        # un-encoded training feature table, kept only so
        # analyze_propensity_drivers() can compute descriptive
        # direction/magnitude per feature without re-deriving features from
        # the original log a second time.
        '_training_features_for_narrative': training_features,
    }


def analyze_propensity_drivers(
    model_bundle: Dict[str, Any],
    top_n: int = 5,
) -> List[str]:
    """
    Translates the top `top_n` features by mean permutation importance
    (already averaged across cross-validation folds in
    train_propensity_model() - no separate held-out set needed here) into
    plain-language sentences, in the same f-string/threshold style as
    analytics._generate_performance_recommendations().

    Direction and magnitude come from a simple descriptive comparison on the
    model's own training data: the positive rate for cases where the
    feature is present (binary features) or above its median (numeric
    features) vs. absent/below it. This is deliberately an associative, not
    causal, statement - it is NOT derived from the permutation importance
    score itself, which only ranks *how much* a feature matters, not *which
    direction*.

    Returns [] if model_bundle['model'] is None (refused/untrained) or there
    are no importances to report. Callers should check
    model_bundle['errors'] first to distinguish "nothing to say" from
    "training was refused".
    """
    if model_bundle.get('model') is None:
        return []

    importances = model_bundle.get('permutation_importance', {})
    if not importances:
        return []

    top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    sentences: List[str] = []

    for feature, _importance in top_features:
        sentences.append(_describe_driver(feature, model_bundle))

    return [s for s in sentences if s]


def _compare_rates(rate_a: float, rate_b: float):
    """
    Compares two positive-rates and returns (comparison_word, multiplier)
    for the sentence "<group A> reached the outcome <multiplier>x
    <comparison_word> often than <group B>" - i.e. always phrased from
    group A's perspective, flipping to "less" (and inverting the multiplier)
    when A's rate is actually lower, so a below-average feature never gets
    described as "more often" with a misleadingly small multiplier. Returns
    None when the lower rate is 0 (multiplier undefined/infinite).
    """
    if rate_a >= rate_b:
        if rate_b <= 0:
            return None
        return 'more', rate_a / rate_b
    if rate_a <= 0:
        return None
    return 'less', rate_b / rate_a


def _describe_driver(feature: str, model_bundle: Dict[str, Any]) -> str:
    training_features = model_bundle.get('_training_features_for_narrative')
    if training_features is None or training_features.empty or feature not in training_features.columns:
        return ""

    label = training_features['label']
    values = training_features[feature]

    if feature.startswith('visited_') or pd.api.types.is_bool_dtype(values):
        present_mask = values.astype(bool)
        activity_name = feature[len('visited_'):] if feature.startswith('visited_') else feature
        with_rate = label[present_mask].mean() if present_mask.any() else 0.0
        without_rate = label[~present_mask].mean() if (~present_mask).any() else 0.0
        comparison = _compare_rates(with_rate, without_rate)
        if comparison is None:
            return ""
        word, multiplier = comparison
        return (
            f"Cases that visited '{activity_name}' reached the outcome "
            f"{multiplier:.1f}x {word} often ({with_rate * 100:.0f}% vs {without_rate * 100:.0f}%) "
            "than cases that didn't."
        )

    if pd.api.types.is_numeric_dtype(values):
        median = values.median()
        above_mask = values > median
        above_rate = label[above_mask].mean() if above_mask.any() else 0.0
        below_rate = label[~above_mask].mean() if (~above_mask).any() else 0.0
        comparison = _compare_rates(above_rate, below_rate)
        if comparison is None:
            return ""
        word, multiplier = comparison
        return (
            f"Cases with an above-median '{feature}' (> {median:.1f}) reached the outcome "
            f"{multiplier:.1f}x {word} often ({above_rate * 100:.0f}% vs {below_rate * 100:.0f}%) "
            "than cases below it."
        )

    top_value = values.value_counts().idxmax() if not values.empty else None
    if top_value is None:
        return ""
    match_mask = values == top_value
    match_rate = label[match_mask].mean() if match_mask.any() else 0.0
    other_rate = label[~match_mask].mean() if (~match_mask).any() else 0.0
    comparison = _compare_rates(match_rate, other_rate)
    if comparison is None:
        return ""
    word, multiplier = comparison
    return (
        f"Cases where '{feature}' = '{top_value}' reached the outcome "
        f"{multiplier:.1f}x {word} often ({match_rate * 100:.0f}% vs {other_rate * 100:.0f}%) "
        "than other cases."
    )


def _score_in_progress_cases(df: pd.DataFrame, model_bundle: Dict[str, Any]) -> pd.DataFrame:
    """
    Not exported - internal plumbing only, consumed exclusively by
    summarize_propensity_scores(). See the module docstring for why no
    public per-case scoring output exists.

    Reproduces the split/features for in-progress cases via
    model_bundle['training_config'], scores with predict_proba().
    'propensity_score' is P(the case eventually reaches the outcome) - high
    = likely to convert, not "at risk" - documented explicitly to avoid a
    sign-flip bug internally.

    Returns an empty DataFrame (correct columns, zero rows) if
    model_bundle['model'] is None or there are no in-progress cases.
    """
    columns = ['case_id', 'propensity_score', 'last_activity',
               'event_count_so_far', 'elapsed_seconds_since_case_start']
    model = model_bundle.get('model')
    if model is None:
        return pd.DataFrame(columns=columns)

    config = model_bundle['training_config']
    built = _build_propensity_features(
        df, config['outcome_activity'], config['case_col'], config['activity_col'],
        config['timestamp_col'], config['case_attribute_cols'], config['revenue_col'],
    )
    scoring_features = built['scoring_features']
    if scoring_features.empty:
        return pd.DataFrame(columns=columns)

    feature_columns = model_bundle['feature_columns']
    categorical_columns = model_bundle['categorical_columns']
    X = scoring_features.reindex(columns=feature_columns)
    for c in categorical_columns:
        if c in X.columns:
            X[c] = X[c].astype(str)

    scores = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        'case_id': scoring_features[config['case_col']],
        'propensity_score': scores,
        'last_activity': scoring_features['last_activity'],
        'event_count_so_far': scoring_features['event_count_so_far'],
        'elapsed_seconds_since_case_start': scoring_features['elapsed_seconds_since_case_start'],
    })
    return result


_RISK_TIERS = ('high_risk', 'medium_risk', 'low_risk')


def summarize_propensity_scores(df: pd.DataFrame, model_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    The only public scoring output (see module docstring for why there's no
    per-case equivalent). Scores every currently in-progress case
    internally via _score_in_progress_cases(), then aggregates - no
    per-case data is returned.

    Returns dict with keys:
        'n_scored'                : int
        'mean_score', 'median_score', 'q1_score', 'q3_score' : float
        'risk_tiers'              : {'high_risk', 'medium_risk', 'low_risk': int}
                                     - tertile-based bands (lowest-scoring
                                     third = high_risk), not fixed absolute
                                     thresholds, since propensity
                                     distributions vary by dataset
        'historical_positive_rate': float - share of completed cases that
                                     were positive, for comparison against
                                     mean_score
        'narrative'                : List[str] plain-language sentences

    Returns a zeroed/empty-shaped dict (no narrative) if there's no model or
    no in-progress cases to score.
    """
    empty = {
        'n_scored': 0, 'mean_score': None, 'median_score': None,
        'q1_score': None, 'q3_score': None,
        'risk_tiers': {tier: 0 for tier in _RISK_TIERS},
        'historical_positive_rate': None, 'narrative': [],
    }

    if model_bundle.get('model') is None:
        return empty

    scored = _score_in_progress_cases(df, model_bundle)
    if scored.empty:
        return empty

    scores = scored['propensity_score']
    metrics = model_bundle.get('metrics', {})
    n_positive = metrics.get('n_positive', 0)
    n_negative = metrics.get('n_negative', 0)
    n_completed = n_positive + n_negative
    historical_rate = (n_positive / n_completed) if n_completed else None

    q1, median, q3 = scores.quantile([0.25, 0.5, 0.75])
    tier_bounds = scores.quantile([1 / 3, 2 / 3])
    low_cut, high_cut = tier_bounds.iloc[0], tier_bounds.iloc[1]
    risk_tiers = {
        'high_risk': int((scores <= low_cut).sum()),
        'medium_risk': int(((scores > low_cut) & (scores <= high_cut)).sum()),
        'low_risk': int((scores > high_cut).sum()),
    }

    narrative: List[str] = []
    n_scored = len(scored)
    high_risk_pct = risk_tiers['high_risk'] / n_scored * 100 if n_scored else 0.0
    narrative.append(
        f"{risk_tiers['high_risk']} of {n_scored} in-progress cases ({high_risk_pct:.0f}%) "
        "are in the bottom (highest-risk) propensity tier."
    )
    if historical_rate is not None:
        mean_score = float(scores.mean())
        direction = "below" if mean_score < historical_rate else "above"
        narrative.append(
            f"In-progress cases are scoring {direction} the historical conversion rate on "
            f"average ({mean_score * 100:.0f}% average propensity vs a {historical_rate * 100:.0f}% "
            "historical conversion rate)."
        )

    return {
        'n_scored': n_scored,
        'mean_score': float(scores.mean()),
        'median_score': float(median),
        'q1_score': float(q1),
        'q3_score': float(q3),
        'risk_tiers': risk_tiers,
        'historical_positive_rate': historical_rate,
        'narrative': narrative,
    }
