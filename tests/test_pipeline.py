import pandas as pd

from prox.config import create_analysis_config
from prox.pipeline import run_full_analysis, _PROGRESS_STAGES

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


def test_run_full_analysis_reports_progress_for_every_stage_in_order():
    """Drives the Streamlit progress bar - stage numbers, total, and labels
    must exactly match _PROGRESS_STAGES, in order, one call per stage."""
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)

    calls = []
    results = run_full_analysis(df, config=config, progress_callback=lambda *args: calls.append(args))

    assert results is not None
    assert calls == [(i + 1, len(_PROGRESS_STAGES), label) for i, label in enumerate(_PROGRESS_STAGES)]


def test_run_full_analysis_none_progress_callback_is_a_no_op():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)

    results = run_full_analysis(df, config=config, progress_callback=None)

    assert results is not None


def test_run_full_analysis_includes_variant_outcomes_when_log_has_purchases():
    from conftest import make_event_log
    base = '2024-01-01 00:00:00'
    rows = []
    for i in range(4):
        t = pd.Timestamp(base) + pd.Timedelta(hours=i)
        rows += [(f'c{i}', 'view_item', t), (f'c{i}', 'add_to_cart', t + pd.Timedelta(minutes=1))]
        if i % 2 == 0:
            rows.append((f'c{i}', 'purchase', t + pd.Timedelta(minutes=2)))
    df = make_event_log(rows)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)

    assert results is not None
    outcomes = results['variant_outcomes']
    assert outcomes['routes']['view_item -> add_to_cart']['conversion_rate'] == 50


def test_run_full_analysis_sampling_keeps_whole_users_for_session_cases():
    """Session-level cases with sampling on: whole users are drawn, so a user
    who bought in two sessions is still a repeat buyer in the sample."""
    rows = []
    for u in range(6):
        for s in range(2):
            t = pd.Timestamp('2024-01-01') + pd.Timedelta(days=u + 3 * s)
            case = f'u{u}_s{s}'
            rows += [
                dict(user_id=f'u{u}', **{'case:concept:name': case, 'concept:name': 'view_item', 'time:timestamp': t}),
                dict(user_id=f'u{u}', **{'case:concept:name': case, 'concept:name': 'purchase',
                                         'time:timestamp': t + pd.Timedelta(minutes=1)}),
            ]
    df = pd.DataFrame(rows)
    config = create_analysis_config(filter_steps=[], sample_size=4, enable_sampling=True)
    results = run_full_analysis(df, config=config)

    assert results is not None
    sampled = results['processed_log']
    assert (sampled.groupby('user_id')['case:concept:name'].nunique() == 2).all()
    assert results['repeat_purchase_analysis']['metrics']['repeat_rate'] == 100


def test_run_full_analysis_omits_variant_outcomes_without_purchases():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)

    assert results is not None
    assert 'variant_outcomes' not in results


def test_run_full_analysis_includes_process_graph():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)

    graph = results['process_graph']
    assert graph['n_cases'] == 3
    assert graph['happy_path'] == ['a', 'b', 'c']


def test_run_full_analysis_includes_loop_analysis_with_configured_groups():
    from conftest import make_event_log
    base = pd.Timestamp('2024-01-01')
    rows = []
    for i in range(3):
        for j, act in enumerate(['list', 'item', 'list', 'search']):
            rows.append((f'c{i}', act, base + pd.Timedelta(hours=i, minutes=j)))
    df = make_event_log(rows)
    config = create_analysis_config(
        filter_steps=[], sample_size=10,
        loop_params={'activity_groups': {'Browse': ['list', 'item'], 'Search': ['search']}, 'group_col': None},
    )
    results = run_full_analysis(df, config=config)

    loops = results['loop_analysis']
    assert loops['summary']['back_nav_case_share'] == 100
    assert list(loops['group_switches']['transitions']) == ['Browse -> Search']


def test_run_full_analysis_loop_analysis_without_groups():
    df = make_simple_variant_log(n_cases=3)
    config = create_analysis_config(filter_steps=[], sample_size=10)
    results = run_full_analysis(df, config=config)

    assert results['loop_analysis']['summary']['back_nav_events'] == 0
    assert results['loop_analysis']['group_switches'] is None
