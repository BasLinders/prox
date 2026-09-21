import pandas as pd
import pytest

pytest.importorskip("sklearn")

from prox.predictive import (
    split_completed_in_progress,
    train_propensity_model,
    analyze_propensity_drivers,
    summarize_propensity_scores,
    _extract_case_features,
    _build_propensity_features,
    _score_in_progress_cases,
)

from conftest import make_event_log

BASE = pd.Timestamp('2024-01-01 00:00:00')


def _rows_for_case(case_id, activities, start, gap=pd.Timedelta(minutes=1)):
    rows = []
    ts = start
    for act in activities:
        rows.append((case_id, act, ts))
        ts += gap
    return rows


def make_conversion_log(n_positive=40, n_negative=40, n_in_progress=5,
                         predictive_activity='browse_deal', outcome='purchase'):
    """
    n_negative cases come first chronologically, n_positive cases after -
    so even with n_in_progress=0, the log's own last event is always a
    converting case's outcome, and every negative case is unambiguously
    "aged out" (there's no case-with-nothing-after-it ambiguity, which
    would otherwise make the single most recent negative case look
    in-progress rather than confirmed-negative - a real property of the
    split heuristic, not a bug, so the test log has to account for it).

    n_positive cases reach `outcome`; 90% of them visit predictive_activity
    first (signal), 10% don't (noise). n_negative cases never reach it; 90%
    never visit predictive_activity either, 10% visit it but still abandon
    (noise). n_in_progress cases are single-event, placed right at the
    log's own end - inside the conversion window, never reaching the
    outcome.
    """
    rows = []
    for i in range(n_negative):
        start = BASE + pd.Timedelta(hours=i)
        acts = ['view', predictive_activity] if i % 10 == 0 else ['view']
        rows.extend(_rows_for_case(f'neg{i}', acts, start))
    for i in range(n_positive):
        start = BASE + pd.Timedelta(hours=n_negative + i)
        acts = ['view', predictive_activity, outcome] if i % 10 != 0 else ['view', outcome]
        rows.extend(_rows_for_case(f'pos{i}', acts, start))
    log_end = BASE + pd.Timedelta(hours=n_negative + n_positive) + pd.Timedelta(days=60)
    for i in range(n_in_progress):
        start = log_end - pd.Timedelta(minutes=1, seconds=i)
        rows.extend(_rows_for_case(f'prog{i}', ['view'], start))
    return make_event_log(rows)


def make_leakage_log(n_positive=6, n_negative=6):
    """Positive cases visit an activity ('thank_you_page') that only ever
    occurs AFTER the outcome ('purchase') - the feature-building code must
    never see it."""
    rows = []
    for i in range(n_positive):
        start = BASE + pd.Timedelta(hours=i)
        rows.extend(_rows_for_case(f'pos{i}', ['view', 'purchase', 'thank_you_page'], start))
    for i in range(n_negative):
        start = BASE + pd.Timedelta(hours=n_positive + i)
        rows.extend(_rows_for_case(f'neg{i}', ['view'], start))
    return make_event_log(rows)


def make_split_test_log():
    rows = []
    for i in range(5):
        start = BASE + pd.Timedelta(hours=i)
        rows.extend(_rows_for_case(f'pos{i}', ['view', 'purchase'], start))
    rows.extend(_rows_for_case('neg_old', ['view'], BASE + pd.Timedelta(hours=5)))
    log_end = BASE + pd.Timedelta(days=90)
    rows.extend(_rows_for_case('recent', ['view'], log_end - pd.Timedelta(minutes=1)))
    return make_event_log(rows)


# --- split_completed_in_progress ---

def test_split_completed_in_progress_labels_outcome_reaching_cases_positive():
    df = make_split_test_log()
    result = split_completed_in_progress(df, 'purchase')
    assert set(result['positive_case_ids']) == {f'pos{i}' for i in range(5)}


def test_split_completed_in_progress_labels_aged_out_non_outcome_cases_negative():
    df = make_split_test_log()
    result = split_completed_in_progress(df, 'purchase')
    assert 'neg_old' in result['negative_case_ids']


def test_split_completed_in_progress_labels_recent_non_outcome_cases_in_progress():
    df = make_split_test_log()
    result = split_completed_in_progress(df, 'purchase')
    assert 'recent' in result['in_progress_case_ids']
    assert 'recent' not in result['negative_case_ids']


def test_split_completed_in_progress_falls_back_when_too_few_positives_for_percentile():
    rows = []
    for i in range(2):
        start = BASE + pd.Timedelta(hours=i)
        rows.extend(_rows_for_case(f'pos{i}', ['view', 'purchase'], start))
    rows.extend(_rows_for_case('neg0', ['view'], BASE + pd.Timedelta(hours=10)))
    df = make_event_log(rows)
    result = split_completed_in_progress(df, 'purchase')
    assert any('too few' in e.lower() for e in result['errors'])


# --- leakage regression (the most important tests in this file) ---

def test_extract_case_features_excludes_events_at_or_after_cutoff_timestamp():
    case_events = pd.DataFrame({
        'concept:name': ['view', 'purchase', 'thank_you_page'],
        'time:timestamp': [BASE, BASE + pd.Timedelta(minutes=1), BASE + pd.Timedelta(minutes=2)],
    })
    features = _extract_case_features(
        case_events, cutoff_timestamp=BASE + pd.Timedelta(minutes=1),
        activity_vocab=['view', 'purchase', 'thank_you_page'], case_attribute_cols=[],
        revenue_col=None, activity_col='concept:name', timestamp_col='time:timestamp',
    )
    assert features['event_count_so_far'] == 1
    assert features['last_activity'] == 'view'
    assert features['visited_purchase'] == 0
    assert features['visited_thank_you_page'] == 0


def test_train_propensity_model_excludes_post_outcome_events_from_features():
    df = make_leakage_log()
    built = _build_propensity_features(
        df, 'purchase', 'case:concept:name', 'concept:name', 'time:timestamp', [], None,
    )
    training_features = built['training_features']
    assert 'visited_thank_you_page' in training_features.columns
    assert (training_features['visited_thank_you_page'] == 0).all()
    positive_rows = training_features[training_features['label'] == 1]
    assert (positive_rows['event_count_so_far'] == 1).all()


def test_score_in_progress_cases_uses_full_available_prefix_no_future_leakage():
    df = make_conversion_log(n_positive=6, n_negative=6, n_in_progress=1)
    built = _build_propensity_features(
        df, 'purchase', 'case:concept:name', 'concept:name', 'time:timestamp', [], None,
    )
    scoring_row = built['scoring_features'].iloc[0]
    assert scoring_row['event_count_so_far'] == 1  # the single 'view' event, in full


# --- min-sample hard-refuse guard ---

def test_train_propensity_model_refuses_below_min_positive_cases():
    df = make_conversion_log(n_positive=3, n_negative=10, n_in_progress=0)
    result = train_propensity_model(df, 'purchase', min_cases_per_class=5)
    assert result['model'] is None
    assert any('not enough' in e.lower() for e in result['errors'])


def test_train_propensity_model_refuses_below_min_negative_cases():
    df = make_conversion_log(n_positive=10, n_negative=3, n_in_progress=0)
    result = train_propensity_model(df, 'purchase', min_cases_per_class=5)
    assert result['model'] is None
    assert any('not enough' in e.lower() for e in result['errors'])


def test_train_propensity_model_refusal_returns_errors_and_no_model():
    df = make_conversion_log(n_positive=3, n_negative=3, n_in_progress=0)
    result = train_propensity_model(df, 'purchase', min_cases_per_class=5)
    assert result['model'] is None
    assert result['metrics'] == {}
    assert result['permutation_importance'] == {}
    assert result['errors']


def test_train_propensity_model_trains_when_thresholds_exactly_met():
    df = make_conversion_log(n_positive=5, n_negative=5, n_in_progress=0)
    result = train_propensity_model(df, 'purchase', min_cases_per_class=5, cv_folds=5)
    assert result['model'] is not None
    assert result['errors'] == []


# --- sklearn-missing degrade path ---

def test_train_propensity_model_missing_sklearn_returns_clear_error(monkeypatch):
    import prox.predictive as predictive_module
    monkeypatch.setattr(predictive_module, '_SKLEARN_AVAILABLE', False)
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = predictive_module.train_propensity_model(df, 'purchase')
    assert result['model'] is None
    assert any('scikit-learn' in e.lower() for e in result['errors'])


def test_analyze_propensity_drivers_empty_model_bundle_returns_empty_list():
    assert analyze_propensity_drivers({'model': None}) == []


def test_summarize_propensity_scores_empty_model_bundle_returns_empty_shape():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=5)
    summary = summarize_propensity_scores(df, {'model': None})
    assert summary['n_scored'] == 0
    assert summary['narrative'] == []


# --- happy path: CV metrics, refit, scoring, drivers, summary ---

def test_train_propensity_model_happy_path_returns_populated_cv_metrics():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = train_propensity_model(df, 'purchase')
    assert result['errors'] == []
    metrics = result['metrics']
    assert metrics['n_completed'] == 80
    assert metrics['n_positive'] == 40
    assert metrics['n_negative'] == 40
    for key in ('accuracy_mean', 'precision_mean', 'recall_mean', 'roc_auc_mean'):
        assert metrics[key] == metrics[key]  # not NaN


def test_train_propensity_model_auc_meaningfully_above_chance_on_separable_log():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = train_propensity_model(df, 'purchase')
    assert result['metrics']['roc_auc_mean'] > 0.75


def test_train_propensity_model_final_model_fit_on_all_completed_cases():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = train_propensity_model(df, 'purchase')
    model = result['model']
    assert list(model.named_steps['clf'].classes_) == [0, 1]
    training_features = result['_training_features_for_narrative']
    X = training_features[result['feature_columns']].copy()
    for c in result['categorical_columns']:
        X[c] = X[c].astype(str)
    probabilities = model.predict_proba(X)[:, 1]
    assert len(probabilities) == len(training_features)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_analyze_propensity_drivers_surfaces_the_known_predictive_activity():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = train_propensity_model(df, 'purchase')
    sentences = analyze_propensity_drivers(result, top_n=5)
    assert sentences
    assert any('browse_deal' in s for s in sentences)


def test_score_in_progress_cases_returns_expected_columns_and_sort_order():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=5)
    result = train_propensity_model(df, 'purchase')
    scored = _score_in_progress_cases(df, result)
    assert list(scored.columns) == [
        'case_id', 'propensity_score', 'last_activity',
        'event_count_so_far', 'elapsed_seconds_since_case_start',
    ]
    assert len(scored) == 5


def test_score_in_progress_cases_propensity_scores_are_between_0_and_1():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=5)
    result = train_propensity_model(df, 'purchase')
    scored = _score_in_progress_cases(df, result)
    assert ((scored['propensity_score'] >= 0) & (scored['propensity_score'] <= 1)).all()


def test_summarize_propensity_scores_reports_risk_tiers_and_historical_comparison():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=5)
    result = train_propensity_model(df, 'purchase')
    summary = summarize_propensity_scores(df, result)
    assert summary['n_scored'] == 5
    assert sum(summary['risk_tiers'].values()) == 5
    assert summary['historical_positive_rate'] == pytest.approx(0.5)
    assert summary['narrative']


def test_summarize_propensity_scores_narrative_never_names_individual_cases():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=5)
    result = train_propensity_model(df, 'purchase')
    summary = summarize_propensity_scores(df, result)
    case_ids = df['case:concept:name'].unique().tolist()
    for sentence in summary['narrative']:
        assert not any(cid in sentence for cid in case_ids)


def test_summarize_propensity_scores_no_in_progress_cases_returns_empty_shape():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = train_propensity_model(df, 'purchase')
    summary = summarize_propensity_scores(df, result)
    assert summary['n_scored'] == 0
    assert summary['narrative'] == []


# --- categorical/missing-value handling ---

def test_train_propensity_model_handles_unseen_activity_at_scoring_time():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = train_propensity_model(df, 'purchase')

    extra_rows = _rows_for_case(
        'unseen_activity_case', ['never_before_seen_activity'],
        BASE + pd.Timedelta(hours=80) + pd.Timedelta(days=60) - pd.Timedelta(minutes=1),
    )
    df_with_unseen = pd.concat([df, make_event_log(extra_rows)], ignore_index=True)

    summary = summarize_propensity_scores(df_with_unseen, result)  # should not raise
    assert summary['n_scored'] >= 1


def test_train_propensity_model_case_attribute_columns_included_when_present():
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    df['category'] = 'electronics'
    result = train_propensity_model(df, 'purchase', case_attribute_cols=['category'])
    assert 'category' in result['categorical_columns']
    assert result['model'] is not None
    assert result['errors'] == []


def test_train_propensity_model_imputes_missing_mean_seconds_between_events():
    # make_conversion_log's negative "noise" cases are single-event ('view'
    # only) -> mean_seconds_between_events is NaN for them; training must
    # not crash on that.
    df = make_conversion_log(n_positive=40, n_negative=40, n_in_progress=0)
    result = train_propensity_model(df, 'purchase')
    assert result['model'] is not None
    assert result['errors'] == []
