import pandas as pd
import pytest

from prox.loops import EXIT_LABEL, OTHER_GROUP, analyze_loops, fit_loop_params_to_log

from conftest import make_event_log


def _log(traces):
    """traces: {case_id: [activity, ...]} -> event log, one minute per step."""
    base = pd.Timestamp('2024-01-01')
    return make_event_log([
        (case_id, act, base + pd.Timedelta(minutes=i))
        for case_id, acts in traces.items()
        for i, act in enumerate(acts)
    ])


LIST_ITEM_LOG = {
    '1': ['list', 'item', 'list', 'item', 'purchase'],
    '2': ['list', 'item', 'item'],
    '3': ['search', 'list', 'search'],
    '4': ['list', 'item'],
}


def test_back_navigation_counts_returns_to_the_previous_activity():
    result = analyze_loops(_log(LIST_ITEM_LOG))

    assert result['errors'] == []
    summary = result['summary']
    assert summary['total_cases'] == 4
    # case 1 returns twice (list, then item), case 3 once
    assert summary['back_nav_events'] == 3
    assert summary['back_nav_cases'] == 2
    assert summary['back_nav_case_share'] == 50

    patterns = result['back_navigation']
    assert set(patterns) == {
        'list -> item -> list', 'item -> list -> item', 'search -> list -> search',
    }
    assert patterns['search -> list -> search'] == {'occurrences': 1, 'cases': 1, 'case_share': 25}


def test_immediate_repeat_is_a_repeat_visit_but_not_back_navigation():
    result = analyze_loops(_log({'1': ['a', 'b', 'b', 'c']}))

    summary = result['summary']
    assert summary['back_nav_events'] == 0
    assert summary['self_loop_events'] == 1
    assert summary['repeat_events'] == 1
    assert result['repeats']['b']['self_loop_events'] == 1


def test_repeats_per_activity():
    repeats = analyze_loops(_log(LIST_ITEM_LOG))['repeats']

    # 'list' is in all 4 cases but only repeats in case 1
    assert repeats['list']['cases_with_activity'] == 4
    assert repeats['list']['repeat_cases'] == 1
    assert repeats['list']['repeat_case_share'] == 25
    assert repeats['list']['visits_per_case'] == pytest.approx(5 / 4)
    # 'item' repeats in cases 1 (via back-navigation) and 2 (immediately)
    assert repeats['item']['repeat_events'] == 2
    assert repeats['item']['self_loop_events'] == 1
    assert repeats['item']['back_nav_returns'] == 1
    # activities that never repeat are left out
    assert 'purchase' not in repeats


def test_after_loop_reports_next_activity_exit_and_position():
    after = analyze_loops(_log(LIST_ITEM_LOG))['after_loop']['back_navigation']

    assert after['loops'] == 3
    # case 1: list (step 3) -> item, item (step 4) -> purchase; case 3: search (step 3) ends the case
    assert after['next_activity'] == {
        'item': {'count': 1, 'share': pytest.approx(100 / 3)},
        'purchase': {'count': 1, 'share': pytest.approx(100 / 3)},
        EXIT_LABEL: {'count': 1, 'share': pytest.approx(100 / 3)},
    }
    assert after['exit_share'] == pytest.approx(100 / 3)
    # relative positions 2/4 (middle), 3/4 and 2/2 (late)
    assert after['position'] == {
        'early': 0, 'middle': pytest.approx(100 / 3), 'late': pytest.approx(200 / 3),
    }
    assert after['mean_step'] == pytest.approx((3 + 4 + 3) / 3)


def test_outcomes_compare_conversion_with_and_without_loops():
    outcomes = analyze_loops(_log(LIST_ITEM_LOG))['outcomes']

    assert outcomes['back_navigation']['with'] == {'cases': 2, 'conversions': 1, 'conversion_rate': 50}
    assert outcomes['back_navigation']['without'] == {'cases': 2, 'conversions': 0, 'conversion_rate': 0}


def test_outcomes_are_none_without_purchases():
    assert analyze_loops(_log({'1': ['a', 'b', 'a']}))['outcomes'] is None


def test_case_order_follows_timestamps_not_row_order():
    df = _log(LIST_ITEM_LOG)
    shuffled = df.sample(frac=1, random_state=3).reset_index(drop=True)

    assert analyze_loops(shuffled)['summary'] == analyze_loops(df)['summary']


def test_no_group_switches_without_groups():
    assert analyze_loops(_log(LIST_ITEM_LOG))['group_switches'] is None


def test_group_switches_skip_ungrouped_events_and_collapse_same_group_runs():
    df = _log({
        # nav, (ungrouped), nav -> one visit; then search, back to nav
        '1': ['menu', 'page', 'list', 'search', 'menu'],
        '2': ['menu', 'list'],
        '3': ['search', 'search'],
    })
    groups = {'Navigation': ['menu', 'list'], 'Search': ['search']}
    switches = analyze_loops(df, activity_groups=groups)['group_switches']

    assert switches['source'] == 'activities'
    assert switches['groups'] == groups
    assert switches['transitions'] == {
        'Navigation -> Search': {'from': 'Navigation', 'to': 'Search', 'occurrences': 1, 'cases': 1,
                                 'case_share': pytest.approx(100 / 3)},
        'Search -> Navigation': {'from': 'Search', 'to': 'Navigation', 'occurrences': 1, 'cases': 1,
                                 'case_share': pytest.approx(100 / 3)},
    }
    summary = switches['summary']
    assert summary['ungrouped_events'] == 1
    assert summary['switch_events'] == 2
    assert summary['switch_cases'] == 1
    assert summary['return_events'] == 1  # Navigation -> Search -> Navigation
    assert switches['group_stats'] == {
        'Navigation': {'events': 5, 'cases': 2}, 'Search': {'events': 3, 'cases': 2},
    }


def test_activity_in_two_groups_counts_for_the_first():
    df = _log({'1': ['a', 'b']})
    switches = analyze_loops(df, activity_groups={'G1': ['a', 'b'], 'G2': ['b']})['group_switches']

    assert switches['transitions'] == {}
    assert switches['group_stats'] == {'G1': {'events': 2, 'cases': 1}}


def test_group_switches_from_a_column():
    df = _log({'1': ['view', 'view', 'view'], '2': ['view', 'view']})
    df['category'] = ['A', 'B', None, 'A', 'A']

    switches = analyze_loops(df, group_col='category')['group_switches']

    assert switches['source'] == 'column'
    assert switches['group_col'] == 'category'
    assert list(switches['transitions']) == ['A -> B']
    assert switches['summary']['ungrouped_events'] == 1
    assert switches['summary']['switch_case_share'] == 50


def test_fewer_than_two_groups_is_an_error():
    result = analyze_loops(_log(LIST_ITEM_LOG), activity_groups={'Only': ['list'], 'Empty': []})

    assert result['group_switches'] is None
    assert any('at least two' in e for e in result['errors'])
    # the rest of the analysis still runs
    assert result['summary']['back_nav_events'] == 3


def test_missing_group_column_is_reported_and_skipped():
    result = analyze_loops(_log(LIST_ITEM_LOG), group_col='nope')

    assert result['group_switches'] is None
    assert any("'nope' not found" in e for e in result['errors'])


def test_handles_empty_and_missing_columns():
    assert analyze_loops(pd.DataFrame())['summary'] == {}
    result = analyze_loops(pd.DataFrame({'concept:name': ['a']}))
    assert result['summary'] == {}
    assert any('missing' in e for e in result['errors'])


def test_include_ungrouped_counts_switches_through_other_activity():
    df = _log({'1': ['menu', 'page', 'menu'], '2': ['menu', 'search']})
    groups = {'Navigation': ['menu'], 'Search': ['search']}

    skipped = analyze_loops(df, activity_groups=groups)['group_switches']
    assert list(skipped['transitions']) == ['Navigation -> Search']
    assert skipped['summary']['return_events'] == 0
    assert skipped['include_ungrouped'] is False

    other = analyze_loops(df, activity_groups=groups, include_ungrouped=True)['group_switches']
    assert set(other['transitions']) == {
        f'Navigation -> {OTHER_GROUP}', f'{OTHER_GROUP} -> Navigation', 'Navigation -> Search',
    }
    # Navigation -> (other) -> Navigation is a switch back
    assert other['summary']['return_events'] == 1
    assert other['summary']['ungrouped_events'] == 1
    assert other['summary']['grouped_events'] == 4
    assert other['group_stats'][OTHER_GROUP] == {'events': 1, 'cases': 1}


def test_one_group_is_enough_with_include_ungrouped():
    df = _log({'1': ['a', 'search', 'b']})

    assert analyze_loops(df, activity_groups={'Search': ['search']})['group_switches'] is None
    switches = analyze_loops(df, activity_groups={'Search': ['search']}, include_ungrouped=True)['group_switches']
    assert switches['summary']['switch_events'] == 2


def test_fit_loop_params_drops_what_the_log_lacks():
    params = {'activity_groups': {'Nav': ['menu', 'gone'], 'Old': ['gone']}, 'group_col': 'nope',
              'include_ungrouped': True}

    fitted, notes = fit_loop_params_to_log(params, ['menu'], ['category'])

    assert fitted == {'activity_groups': {'Nav': ['menu']}, 'group_col': None, 'include_ungrouped': True}
    assert len(notes) == 3
    assert fit_loop_params_to_log(None, ['menu'], []) == (None, [])
