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


# --- Page URLs ---------------------------------------------------------------

from prox.loops import (  # noqa: E402
    SECTION_COL, URL_LOOP_KINDS, detect_url_column, format_sections_text, format_url_rules_text,
    normalize_url_paths, page_sections, parse_sections_text, parse_url_rules_text, resolve_loop_params,
)


def _url_log(traces, sessions=None):
    """traces: {case_id: [(activity, url), ...]} -> event log with page_location."""
    base = pd.Timestamp('2024-01-01')
    rows, urls, session_ids = [], [], []
    for case_id, steps in traces.items():
        for i, (act, url) in enumerate(steps):
            rows.append((case_id, act, base + pd.Timedelta(minutes=i)))
            urls.append(url)
            session_ids.append((sessions or {}).get(case_id, [case_id] * len(steps))[i])
    df = make_event_log(rows)
    df['page_location'] = urls
    df['session_id'] = session_ids
    return df


SHOP = 'https://www.shop.nl'
URL_LOG = {
    # list, product 1, back to the same list, then product 2: a real return, then comparing
    '1': [('view_item_list', f'{SHOP}/c/shoes'), ('view_item', f'{SHOP}/p/100001'),
          ('view_item_list', f'{SHOP}/c/shoes?page=1'), ('view_item', f'{SHOP}/p/100002'),
          ('purchase', f'{SHOP}/checkout/thanks')],
    # an add_to_cart on the product page between two view_items: no page change
    '2': [('view_item', f'{SHOP}/p/100001'), ('add_to_cart', f'{SHOP}/p/100001'),
          ('view_item', f'{SHOP}/p/100001#reviews')],
    # content pages that are all page_view: a page loop event names can't show
    '3': [('page_view', f'{SHOP}/faq'), ('page_view', f'{SHOP}/shipping'), ('page_view', f'{SHOP}/FAQ/')],
}


def test_normalize_url_paths_keeps_only_a_clean_path():
    urls = pd.Series([
        'https://www.shop.nl/P/123?gclid=x#top', 'http://shop.nl//c/shoes/', 'www.shop.nl/c/shoes',
        '/search?q=me@example.com', 'https://shop.nl', '', None,
        'https://shop.nl/p/red-shoe-123456', 'https://shop.nl/o/0f8fad5b-d9cb-469f-a165-70867728950e',
        'https://shop.nl/p/air-max-90',
    ])
    assert normalize_url_paths(urls).tolist() == [
        '/p/{id}', '/c/shoes', '/c/shoes', '/search', '/', None, None,
        '/p/{id}', '/o/{id}', '/p/air-max-90',
    ]
    assert normalize_url_paths(urls, collapse_ids=False)[0] == '/p/123'


def test_normalize_url_paths_handles_categorical_columns():
    urls = pd.Series(['https://a.nl/x?y=1', None, 'https://a.nl/x'], dtype='category')
    assert normalize_url_paths(urls).tolist() == ['/x', None, '/x']


def test_url_rules_apply_in_order_before_id_collapse():
    urls = pd.Series(['https://a.nl/nl/p/red-shoe', 'https://a.nl/en', 'https://a.nl/nl/c/42'])
    rules = [['^/(nl|en)(/|$)', '/'], ['^/p/[^/]+$', '/p/{slug}']]
    assert normalize_url_paths(urls, rules).tolist() == ['/p/{slug}', '/', '/c/{id}']


def test_invalid_url_rules_are_skipped_and_reported():
    result = analyze_loops(_url_log(URL_LOG), url_col='page_location', url_rules=[['(', 'x'], ['^/faq$', '/help']])
    assert any("not a valid regex" in e for e in result['errors'])
    assert 'faq' not in str(result['page_loops']['back_navigation'])
    assert '/help -> /shipping -> /help' in result['page_loops']['back_navigation']


def test_page_sections_use_rules_then_first_segment():
    paths = pd.Series(['/p/{id}', '/c/shoes', '/', '/checkout/thanks', None])
    sections = {'Product': ['^/p/'], 'Checkout': ['^/(cart|checkout)']}
    assert page_sections(paths, sections).tolist() == ['Product', 'c', 'home', 'Checkout', None]


def test_back_navigation_is_split_by_url():
    result = analyze_loops(_url_log(URL_LOG), url_col='page_location')

    assert result['errors'] == []
    summary = result['summary']
    # case 1: list -> item -> list on /c/shoes twice (query string ignored), then
    # item -> list -> item on another product; case 2: all on one page
    assert summary['back_nav_events'] == 3
    assert summary['back_nav_same_url_events'] == 1
    assert summary['back_nav_other_url_events'] == 1
    assert summary['back_nav_in_page_events'] == 1
    assert summary['back_nav_unknown_url_events'] == 0
    assert summary['back_nav_same_url_case_share'] == pytest.approx(100 / 3)

    patterns = result['back_navigation']
    assert patterns['view_item_list -> view_item -> view_item_list']['same_url'] == 1
    assert patterns['view_item -> view_item_list -> view_item']['other_url'] == 1
    assert patterns['view_item -> add_to_cart -> view_item']['in_page'] == 1


def test_repeats_on_the_same_url_and_distinct_urls():
    repeats = analyze_loops(_url_log(URL_LOG), url_col='page_location')['repeats']

    # view_item repeats in case 1 (another product) and case 2 (same product)
    assert repeats['view_item']['repeat_events'] == 2
    assert repeats['view_item']['same_url_repeat_events'] == 1
    # case 1: 2 products, case 2: 1
    assert repeats['view_item']['distinct_urls_per_case'] == pytest.approx(1.5)


def test_url_loop_kinds_get_after_loop_and_outcomes():
    result = analyze_loops(_url_log(URL_LOG), url_col='page_location')

    assert set(URL_LOOP_KINDS) <= set(result['after_loop'])
    assert result['after_loop']['back_navigation_same_url']['loops'] == 1
    assert result['after_loop']['back_navigation_same_url']['next_activity'] == {
        'view_item': {'count': 1, 'share': 100.0},
    }
    assert result['outcomes']['back_navigation_same_url']['with'] == {
        'cases': 1, 'conversions': 1, 'conversion_rate': 100.0,
    }


def test_page_loops_collapse_events_on_one_page_and_find_page_loops():
    page = analyze_loops(_url_log(URL_LOG), url_col='page_location')['page_loops']

    assert page['url_col'] == 'page_location'
    summary = page['summary']
    # case 1: 5 visits, case 2: one visit (3 events on /p/{id}), case 3: 3 visits
    assert summary['page_visits'] == 9
    assert summary['events_without_url'] == 0
    assert summary['self_loop_events'] == 0
    # IDs collapse, so case 1 is list -> product -> list -> product -> checkout
    assert set(page['back_navigation']) == {
        '/c/shoes -> /p/{id} -> /c/shoes', '/p/{id} -> /c/shoes -> /p/{id}', '/faq -> /shipping -> /faq',
    }
    assert page['repeats']['/faq']['repeat_events'] == 1
    assert page['outcomes']['back_navigation']['with']['conversion_rate'] == 50


def test_page_visits_do_not_run_on_across_sessions():
    df = _url_log({'u1': [('page_view', '/a'), ('page_view', '/a')]}, sessions={'u1': ['s1', 's2']})

    page = analyze_loops(df, url_col='page_location')['page_loops']
    assert page['summary']['page_visits'] == 2
    assert page['summary']['self_loop_events'] == 1


def test_events_without_url_are_left_out_of_page_loops():
    df = _url_log({'1': [('page_view', '/a'), ('purchase', None), ('page_view', '/b'), ('page_view', '/a')]})

    result = analyze_loops(df, url_col='page_location')
    assert result['page_loops']['summary']['events_without_url'] == 1
    assert '/a -> /b -> /a' in result['page_loops']['back_navigation']


def test_no_url_analysis_without_url_col():
    result = analyze_loops(_url_log(URL_LOG))
    assert result['page_loops'] is None
    assert 'back_nav_same_url_events' not in result['summary']
    assert set(result['after_loop']) == {'back_navigation', 'repeat'}


def test_missing_or_empty_url_col_is_reported():
    df = _log(LIST_ITEM_LOG)
    assert any("URL column 'page_location' not found" in e
               for e in analyze_loops(df, url_col='page_location')['errors'])
    df['page_location'] = None
    result = analyze_loops(df, url_col='page_location')
    assert any("has no values" in e for e in result['errors'])
    assert result['page_loops'] is None


def test_group_switches_by_site_section():
    switches = analyze_loops(
        _url_log(URL_LOG), url_col='page_location', group_by_section=True,
        sections={'Product': ['^/p/'], 'Category': ['^/c/']},
    )['group_switches']

    assert switches['source'] == 'section'
    assert switches['group_col'] == SECTION_COL
    assert switches['transitions']['Category -> Product']['occurrences'] == 2
    # case 1: Category -> Product -> Category and back again; case 3: faq -> shipping -> faq
    assert switches['summary']['return_events'] == 3
    assert set(switches['group_stats']) == {'Product', 'Category', 'checkout', 'faq', 'shipping'}


def test_group_by_section_needs_a_url_column():
    result = analyze_loops(_log(LIST_ITEM_LOG), group_by_section=True)
    assert result['group_switches'] is None
    assert any("Site sections need a URL column" in e for e in result['errors'])


def test_resolve_loop_params_detects_or_disables_the_url_column():
    columns = ['case:concept:name', 'page_url', 'page_location']
    assert detect_url_column(columns) == 'page_location'
    assert resolve_loop_params(None, columns)['url_col'] == 'page_location'
    assert resolve_loop_params({'url_col': None}, columns)['url_col'] is None
    assert resolve_loop_params({'url_col': 'page_url'}, columns)['url_col'] == 'page_url'
    assert resolve_loop_params(None, ['case:concept:name'])['url_col'] is None


def test_fit_loop_params_drops_unknown_url_column_and_invalid_patterns():
    params = {
        'activity_groups': {}, 'url_col': 'page_url', 'url_rules': [['(', ''], ['^/nl', '']],
        'sections': {'Bad': ['['], 'Product': ['^/p/']},
    }
    fitted, notes = fit_loop_params_to_log(params, [], ['page_location'])

    assert 'url_col' not in fitted
    assert fitted['url_rules'] == [['^/nl', '']]
    assert fitted['sections'] == {'Product': ['^/p/']}
    assert any("using 'page_location' instead" in n for n in notes)
    assert len(notes) == 3


def test_rule_and_section_text_round_trip():
    rules, errors = parse_url_rules_text("^/(nl|en)(/|$) => /\n\n^/p/[^/]+$ => /p/{slug}\n^/x =>")
    assert errors == []
    assert rules == [['^/(nl|en)(/|$)', '/'], ['^/p/[^/]+$', '/p/{slug}'], ['^/x', '']]
    assert parse_url_rules_text(format_url_rules_text(rules))[0] == rules

    sections, errors = parse_sections_text("Product = ^/p/\nProduct = ^/item/\nLook = (?=x)")
    assert errors == []
    assert sections == {'Product': ['^/p/', '^/item/'], 'Look': ['(?=x)']}
    assert parse_sections_text(format_sections_text(sections))[0] == sections


def test_rule_and_section_text_report_bad_lines():
    assert parse_url_rules_text("no arrow")[1] == ["Rule line 1 has no '=>': no arrow"]
    assert "not a valid regex" in parse_url_rules_text("( => x")[1][0]
    assert parse_sections_text("= ^/p/")[1] == ["Section line 1 is not 'name = pattern': = ^/p/"]
