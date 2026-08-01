from prox.config import create_analysis_config
from prox.pipeline import run_full_analysis

from conftest import make_simple_variant_log


def test_run_full_analysis_end_to_end():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)

    assert results is not None
    assert results['log_summary']['Number of Cases'] == 3
    assert results['conformance']['errors'] == []


def test_run_full_analysis_rejects_unknown_filter_type_upfront():
    """Unknown filter type should fail fast before running any filter steps,
    not surface confusingly later as 'produced an empty dataset'."""
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(
        filter_steps=[{'type': 'not_a_real_filter'}],
        sample_size=10
    )
    results = run_full_analysis(df, config=config)
    assert results is None


def test_run_full_analysis_rejects_filter_step_missing_type():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(
        filter_steps=[{'activities': ['a']}],
        sample_size=10
    )
    results = run_full_analysis(df, config=config)
    assert results is None


def test_run_full_analysis_includes_funnel_analysis():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)

    assert results is not None
    assert 'funnel_analysis' in results
    assert results['funnel_analysis']['total_cases'] == 3
    assert results['funnel_analysis']['stages']
