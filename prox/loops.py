"""
Loop, back-navigation and switch-pattern analysis.

Web journeys are dominated by repeat and switch behaviour - a visitor bounces
between a product list and product pages, or between search and the menu -
and the process map only shows those loops as arrows. This module measures
them, per case and per event:

- Back-navigation: a return to the previous activity (A -> B -> A, A != B).
  A browser back isn't an event in a clickstream, so it can only be inferred
  from this pattern; clicking a link back to A looks exactly the same.
- Repeat visits: an activity seen more than once within a case, including
  immediate self-loops (A -> A).
- Group switches: moves between user-defined groups of activities (e.g. menu
  navigation -> search), or between the values of a column (e.g. category A
  -> category B), since activity naming differs per site.
- What follows a loop (next activity or exit), where in the journey it
  happens, and - when the log has purchases - how cases with loops convert
  compared to cases without.
- With a page URL column (e.g. GA4's page_location):
  - each back-navigation split into a return to the same URL, a move to
    another URL of the same kind of page (product A -> list -> product B),
    or an event within one page (all three events on one URL);
  - repeat visits to the same URL, and distinct URLs per case per activity;
  - the same loop analysis at page level, on page visits (consecutive events
    on one page collapse into one visit) - which finds the loops event names
    can't, e.g. between CMS pages that all share the name page_view;
  - site sections from the URL paths, to use as the groups for switches.
  URLs are reduced to their path (no host, query string or fragment - those
  can carry personal data), optionally rewritten by user-defined regex rules
  and with ID-like path segments collapsed to {id}.

Builds on the same per-case directly-follows ordering as
analytics.analyze_process_performance()'s transition data, and is run by the
pipeline so compare_segments() picks it up per segment.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .analytics import _normalise_values, _purchase_row_mask

EXIT_LABEL = '(exit)'

# The group events in no user-defined group (or with no value in the group
# column) fall into when include_ungrouped is set.
OTHER_GROUP = '(other)'

# The limitation every view of these results should state.
LOOP_INFERENCE_NOTE = (
    "A browser back isn't recorded as an event, so back-navigation is inferred from an "
    "A -> B -> A pattern; following a link back to A looks exactly the same."
)

# A loop's relative position within its case (event index / last index):
# below the first bound is early, below the second is middle, else late.
POSITION_BOUNDS = (1 / 3, 2 / 3)

LOOP_KINDS = ('back_navigation', 'repeat')

# The back-navigation kinds added when the log has page URLs (see analyze_loops).
URL_LOOP_KINDS = ('back_navigation_same_url', 'back_navigation_other_url')

LOOP_KIND_LABELS = {
    'back_navigation': 'Back-navigation',
    'repeat': 'Repeat visit',
    'back_navigation_same_url': 'Back-navigation to the same URL',
    'back_navigation_other_url': 'Back-navigation to another URL',
}

# Columns holding the URL an event fired on, in order of preference, for when
# loop_params doesn't name one.
URL_COLUMN_CANDIDATES = ('page_location', 'page_url', 'page_path')

# The loop_params keys for the URL analysis (see resolve_loop_params).
URL_PARAM_KEYS = ('url_col', 'url_rules', 'collapse_ids', 'sections', 'group_by_section')

# The group column analyze_loops derives when group_by_section is set.
SECTION_COL = '_page_section'
HOME_SECTION = 'home'
ID_PLACEHOLDER = '{id}'

_SCHEME_HOST = re.compile(r'^(?:[a-z][a-z0-9+.\-]*:)?//[^/?#]*', re.IGNORECASE)
# A bare host without a scheme, e.g. 'www.shop.nl/p/1'.
_BARE_HOST = re.compile(r'^[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.[a-z]{2,}(?::\d+)?(?=[/?#]|$)', re.IGNORECASE)
# Path segments that are IDs rather than page names: all digits, a UUID, a
# long hex string, or containing a run of 5+ digits (e.g. 'red-shoe-123456').
_ID_SEGMENT = re.compile(
    r'^(?:\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{16,}|.*\d{5,}.*)$',
    re.IGNORECASE,
)


def _share(part: float, whole: float) -> float:
    return float(part / whole * 100) if whole else 0.0


def _clean_groups(activity_groups: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Drops unnamed or empty groups; values become lists of str."""
    cleaned: Dict[str, List[str]] = {}
    for name, activities in (activity_groups or {}).items():
        name = str(name).strip()
        if isinstance(activities, str):
            activities = [activities]
        activities = [str(a) for a in activities or [] if str(a).strip()]
        if name and activities:
            cleaned[name] = activities
    return cleaned


def _after_loop(work: pd.DataFrame, loop_mask: pd.Series, top_n: int) -> Dict[str, Any]:
    """What comes right after the loop events in loop_mask, and where in their
    case they occur."""
    loops = work.loc[loop_mask]
    total = len(loops)
    if not total:
        return {'loops': 0, 'exit_share': 0.0, 'next_activity': {}, 'position': {}, 'mean_step': None}

    next_counts = loops['_next'].fillna(EXIT_LABEL).value_counts()
    next_activity = {
        str(act): {'count': int(n), 'share': _share(n, total)}
        for act, n in next_counts.head(top_n).items()
    }

    last_index = (loops['_case_len'] - 1).clip(lower=1)
    relative = loops['_step'] / last_index
    early, middle = POSITION_BOUNDS
    position = {
        'early': _share((relative < early).sum(), total),
        'middle': _share(((relative >= early) & (relative < middle)).sum(), total),
        'late': _share((relative >= middle).sum(), total),
    }

    return {
        'loops': int(total),
        'exit_share': _share(loops['_next'].isna().sum(), total),
        'next_activity': next_activity,
        'position': position,
        'mean_step': float((loops['_step'] + 1).mean()),
    }


def detect_url_column(columns: Iterable[str]) -> Optional[str]:
    """The first of URL_COLUMN_CANDIDATES in columns, or None."""
    columns = set(columns)
    return next((c for c in URL_COLUMN_CANDIDATES if c in columns), None)


def compile_url_rules(url_rules) -> Tuple[List[Tuple[re.Pattern, str]], List[str]]:
    """[[pattern, replacement], ...] -> (compiled rules, errors). A rule that
    isn't a pair or doesn't compile is left out, with an error naming it."""
    compiled, errors = [], []
    for rule in url_rules or []:
        if not isinstance(rule, (list, tuple)) or len(rule) != 2:
            errors.append(f"URL rule {rule!r} is not a [pattern, replacement] pair; skipped.")
            continue
        pattern, replacement = str(rule[0]), str(rule[1])
        try:
            compiled.append((re.compile(pattern, re.IGNORECASE), replacement))
        except re.error as e:
            errors.append(f"URL rule pattern '{pattern}' is not a valid regex ({e}); skipped.")
    return compiled, errors


def compile_sections(sections) -> Tuple[List[Tuple[str, re.Pattern]], List[str]]:
    """{section name: [regex, ...]} -> ([(name, compiled), ...], errors)."""
    compiled, errors = [], []
    for name, patterns in (sections or {}).items():
        name = str(name).strip()
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns or []:
            pattern = str(pattern)
            if not name or not pattern.strip():
                continue
            try:
                compiled.append((name, re.compile(pattern, re.IGNORECASE)))
            except re.error as e:
                errors.append(f"Section '{name}' pattern '{pattern}' is not a valid regex ({e}); skipped.")
    return compiled, errors


def _base_path(url: str) -> Optional[str]:
    """A URL's path: no scheme, host, query string or fragment, lower case,
    no repeated or trailing slashes. None for an empty value."""
    url = url.strip()
    if not url:
        return None
    url = _SCHEME_HOST.sub('', url)
    if not url.startswith('/'):
        url = _BARE_HOST.sub('', url)
    path = re.split(r'[?#]', url, maxsplit=1)[0].lower()
    path = '/' + re.sub(r'/{2,}', '/', path).strip('/')
    return path


def _template_path(path: str, rules: List[Tuple[re.Pattern, str]], collapse_ids: bool) -> str:
    for pattern, replacement in rules:
        path = pattern.sub(replacement, path)
    if collapse_ids:
        path = '/'.join(ID_PLACEHOLDER if _ID_SEGMENT.match(seg) else seg for seg in path.split('/'))
    return path or '/'


def _map_unique(values: pd.Series, fn) -> pd.Series:
    """fn applied once per distinct value (URL columns repeat a lot); missing
    values stay missing."""
    codes, uniques = pd.factorize(values)
    mapped = np.array([fn(u) for u in uniques] + [None], dtype=object)
    return pd.Series(mapped[codes], index=values.index, dtype=object)  # code -1 picks the None


def normalize_url_paths(urls: pd.Series, url_rules=None, collapse_ids: bool = True) -> pd.Series:
    """URLs -> page paths: _base_path, then the url_rules ([[regex,
    replacement], ...], applied in order with re.sub, case-insensitive),
    then - with collapse_ids - ID-like path segments replaced by {id}.
    Invalid rules are skipped (compile_url_rules reports them)."""
    rules = compile_url_rules(url_rules)[0]

    def normalise(url):
        path = _base_path(str(url))
        return None if path is None else _template_path(path, rules, collapse_ids)

    return _map_unique(urls, normalise)


def page_sections(paths: pd.Series, sections=None) -> pd.Series:
    """Page paths -> site section: the first section whose regex matches the
    path (re.search), else the path's first segment ('home' for '/')."""
    compiled = compile_sections(sections)[0]

    def section(path):
        for name, pattern in compiled:
            if pattern.search(path):
                return name
        return path.strip('/').split('/')[0] or HOME_SECTION

    return _map_unique(paths, section)


def parse_url_rules_text(text: str) -> Tuple[List[List[str]], List[str]]:
    """'pattern => replacement' per line -> ([[pattern, replacement], ...],
    errors). Blank lines are ignored; the replacement may be empty."""
    rules, errors = [], []
    for n, line in enumerate((text or '').splitlines(), 1):
        if not line.strip():
            continue
        if '=>' not in line:
            errors.append(f"Rule line {n} has no '=>': {line.strip()}")
            continue
        pattern, replacement = (part.strip() for part in line.split('=>', 1))
        rules.append([pattern, replacement])
    return rules, errors + compile_url_rules(rules)[1]


def format_url_rules_text(url_rules) -> str:
    return '\n'.join(f"{r[0]} => {r[1]}" for r in url_rules or [] if isinstance(r, (list, tuple)) and len(r) == 2)


def parse_sections_text(text: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """'section name = regex' per line -> ({name: [regex, ...]}, errors). A
    name on several lines collects all its patterns, in order."""
    sections: Dict[str, List[str]] = {}
    errors = []
    for n, line in enumerate((text or '').splitlines(), 1):
        if not line.strip():
            continue
        name, sep, pattern = line.partition('=')
        if not sep or not name.strip() or not pattern.strip():
            errors.append(f"Section line {n} is not 'name = pattern': {line.strip()}")
            continue
        sections.setdefault(name.strip(), []).append(pattern.strip())
    return sections, errors + compile_sections(sections)[1]


def format_sections_text(sections) -> str:
    return '\n'.join(
        f"{name} = {pattern}"
        for name, patterns in (sections or {}).items()
        for pattern in ([patterns] if isinstance(patterns, str) else patterns or [])
    )


def resolve_loop_params(loop_params: Optional[Dict[str, Any]], columns: Iterable[str]) -> Dict[str, Any]:
    """loop_params -> keyword arguments for analyze_loops. Without a 'url_col'
    key the URL column is detected (detect_url_column); a url_col of None
    turns the URL analysis off."""
    loop_params = dict(loop_params or {})
    kwargs = {
        'activity_groups': loop_params.get('activity_groups'),
        'group_col': loop_params.get('group_col'),
        'include_ungrouped': bool(loop_params.get('include_ungrouped', False)),
        'url_col': loop_params['url_col'] if 'url_col' in loop_params else detect_url_column(columns),
        'url_rules': loop_params.get('url_rules'),
        'collapse_ids': bool(loop_params.get('collapse_ids', True)),
        'sections': loop_params.get('sections'),
        'group_by_section': bool(loop_params.get('group_by_section', False)),
    }
    return kwargs


def _outcomes(case_ids: pd.Index, loop_cases: Dict[str, set], converted_cases: set) -> Dict[str, Any]:
    """Conversion of cases with vs. without each loop kind."""
    converted = case_ids.isin(list(converted_cases))
    out = {}
    for kind, cases in loop_cases.items():
        has_loop = case_ids.isin(list(cases))
        groups = {}
        for label, mask in (('with', has_loop), ('without', ~has_loop)):
            n = int(mask.sum())
            conversions = int((converted & mask).sum())
            groups[label] = {
                'cases': n,
                'conversions': conversions,
                'conversion_rate': _share(conversions, n),
            }
        out[kind] = groups
    return out


def _group_switches(
    work: pd.DataFrame,
    case_col: str,
    activity_col: str,
    activity_groups: Dict[str, List[str]],
    group_col: Optional[str],
    include_ungrouped: bool,
    total_cases: int,
    top_n: int,
) -> Dict[str, Any]:
    if group_col:
        groups = work[group_col].astype(str).where(work[group_col].notna())
        source = 'column'
    else:
        lookup: Dict[str, str] = {}
        for name, activities in activity_groups.items():
            for act in activities:
                lookup.setdefault(act, name)  # first group wins on overlap
        groups = work[activity_col].map(lookup)
        source = 'activities'
    ungrouped_events = int(groups.isna().sum())
    if include_ungrouped:
        groups = groups.fillna(OTHER_GROUP)

    grouped = work.assign(_group=groups).loc[groups.notna(), [case_col, '_group']]
    g_stats = grouped.groupby('_group', sort=False).agg(
        events=(case_col, 'size'), cases=(case_col, 'nunique')
    ).sort_values('events', ascending=False)
    group_stats = {
        str(g): {'events': int(row['events']), 'cases': int(row['cases'])}
        for g, row in g_stats.iterrows()
    }

    # Ungrouped events are skipped (unless include_ungrouped puts them in
    # OTHER_GROUP) and consecutive events in the same group collapse into one
    # visit, so a switch is a move to a different group however many skipped
    # events sit in between.
    prev = grouped.groupby(case_col, sort=False)['_group'].shift(1)
    visits = grouped.loc[prev.isna() | (grouped['_group'] != prev)].copy()
    visits['_prev'] = visits.groupby(case_col, sort=False)['_group'].shift(1)
    visits['_prev2'] = visits.groupby(case_col, sort=False)['_group'].shift(2)
    switches = visits.loc[visits['_prev'].notna()]

    transitions = {}
    if not switches.empty:
        t_stats = switches.groupby(['_prev', '_group'], sort=False).agg(
            occurrences=(case_col, 'size'), cases=(case_col, 'nunique')
        ).sort_values(['occurrences', 'cases'], ascending=False).head(top_n)
        for (source_group, target_group), row in t_stats.iterrows():
            transitions[f"{source_group} -> {target_group}"] = {
                'from': str(source_group),
                'to': str(target_group),
                'occurrences': int(row['occurrences']),
                'cases': int(row['cases']),
                'case_share': _share(row['cases'], total_cases),
            }

    returns = switches.loc[switches['_prev2'] == switches['_group']]
    switch_cases = switches[case_col].nunique()
    return {
        'source': source,
        'group_col': group_col,
        'groups': activity_groups if source == 'activities' else {},
        'include_ungrouped': include_ungrouped,
        'group_stats': group_stats,
        'transitions': transitions,
        'summary': {
            'grouped_events': int(len(work) - ungrouped_events),
            'ungrouped_events': ungrouped_events,
            'cases_in_groups': int(grouped[case_col].nunique()),
            'switch_events': int(len(switches)),
            'switch_cases': int(switch_cases),
            'switch_case_share': _share(switch_cases, total_cases),
            'return_events': int(len(returns)),
            'return_cases': int(returns[case_col].nunique()),
            'return_case_share': _share(returns[case_col].nunique(), total_cases),
        },
    }


def _sequence_loops(
    work: pd.DataFrame, case_col: str, activity_col: str, top_n: int,
) -> Tuple[Dict[str, Any], Dict[str, pd.Series]]:
    """Back-navigation and repeats over work, already in case/time order with
    a clean index. Adds _next, _step and _case_len to work. Returns (results
    with summary, back_navigation and repeats; the loop masks by kind, plus
    'self_loop')."""
    by_case = work.groupby(case_col, sort=False)[activity_col]
    prev = by_case.shift(1)
    prev2 = by_case.shift(2)
    work['_next'] = by_case.shift(-1)
    work['_step'] = work.groupby(case_col, sort=False).cumcount()
    work['_case_len'] = work.groupby(case_col, sort=False)[case_col].transform('size')

    acts = work[activity_col]
    back_nav = (acts == prev2) & (acts != prev) & prev2.notna()
    repeat = work.groupby([case_col, activity_col], sort=False).cumcount() > 0
    self_loop = acts == prev

    total_cases = work[case_col].nunique()
    back_nav_cases = work.loc[back_nav, case_col].nunique()
    repeat_cases = work.loc[repeat, case_col].nunique()
    self_loop_cases = work.loc[self_loop, case_col].nunique()

    result: Dict[str, Any] = {
        'summary': {
            'total_cases': int(total_cases),
            'total_events': int(len(work)),
            'back_nav_events': int(back_nav.sum()),
            'back_nav_cases': int(back_nav_cases),
            'back_nav_case_share': _share(back_nav_cases, total_cases),
            'repeat_events': int(repeat.sum()),
            'repeat_cases': int(repeat_cases),
            'repeat_case_share': _share(repeat_cases, total_cases),
            'self_loop_events': int(self_loop.sum()),
            'self_loop_cases': int(self_loop_cases),
            'self_loop_case_share': _share(self_loop_cases, total_cases),
        },
        'back_navigation': {},
        'repeats': {},
    }

    if back_nav.any():
        patterns = (prev2 + ' -> ' + prev + ' -> ' + acts)[back_nav]
        p_stats = pd.DataFrame({'pattern': patterns, 'case': work.loc[back_nav, case_col]}).groupby(
            'pattern', sort=False
        ).agg(occurrences=('case', 'size'), cases=('case', 'nunique')).sort_values(
            ['occurrences', 'cases'], ascending=False
        ).head(top_n)
        result['back_navigation'] = {
            pattern: {
                'occurrences': int(row['occurrences']),
                'cases': int(row['cases']),
                'case_share': _share(row['cases'], total_cases),
            }
            for pattern, row in p_stats.iterrows()
        }

    flags = pd.DataFrame({
        'activity': acts, 'case': work[case_col],
        'repeat': repeat, 'self_loop': self_loop, 'back_nav': back_nav,
    })
    a_stats = flags.groupby('activity', sort=False).agg(
        events=('case', 'size'), cases_with_activity=('case', 'nunique'),
        repeat_events=('repeat', 'sum'), self_loop_events=('self_loop', 'sum'),
        back_nav_returns=('back_nav', 'sum'),
    )
    a_stats['repeat_cases'] = (
        flags.loc[flags['repeat']].groupby('activity')['case'].nunique()
        .reindex(a_stats.index, fill_value=0)
    )
    a_stats = a_stats.loc[a_stats['repeat_events'] > 0].sort_values(
        ['repeat_events', 'repeat_cases'], ascending=False
    ).head(top_n)
    result['repeats'] = {
        activity: {
            'cases_with_activity': int(row['cases_with_activity']),
            'repeat_cases': int(row['repeat_cases']),
            'repeat_case_share': _share(row['repeat_cases'], row['cases_with_activity']),
            'repeat_events': int(row['repeat_events']),
            'self_loop_events': int(row['self_loop_events']),
            'back_nav_returns': int(row['back_nav_returns']),
            'visits_per_case': float(row['events'] / row['cases_with_activity']),
        }
        for activity, row in a_stats.iterrows()
    }

    masks = {'back_navigation': back_nav, 'repeat': repeat, 'self_loop': self_loop}
    return result, masks


def _kind_details(
    work: pd.DataFrame, masks: Dict[str, pd.Series], kinds: Iterable[str], case_col: str,
    converted_cases: Optional[set], top_n: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """(after_loop, outcomes) for each loop kind; outcomes is None without
    purchases."""
    after = {kind: _after_loop(work, masks[kind], top_n) for kind in kinds}
    if not converted_cases:
        return after, None
    loop_cases = {kind: set(work.loc[masks[kind], case_col]) for kind in kinds}
    return after, _outcomes(pd.Index(work[case_col].unique()), loop_cases, converted_cases)


def _url_split(
    work: pd.DataFrame, paths: pd.Series, masks: Dict[str, pd.Series], case_col: str,
    activity_col: str, result: Dict[str, Any],
) -> None:
    """Splits event-level back-navigation and repeats by URL (paths: each
    event's exact page path, or None), into result and masks in place."""
    back_nav, repeat = masks['back_navigation'], masks['repeat']
    by_case = paths.groupby(work[case_col], sort=False)
    p1, p2 = by_case.shift(1), by_case.shift(2)

    # A -> B -> A on URLs uA -> uB -> uA'. One page throughout is an in-page
    # event, not navigation; a known uA' == uA is a return to the same URL;
    # different URLs are the same kind of page, another one (product A ->
    # list -> product B). B without a URL (a server-side event) doesn't
    # decide anything.
    known = back_nav & paths.notna() & p2.notna()
    same_end = known & (paths == p2)
    in_page = same_end & (p1 == paths)
    same_url = same_end & ~in_page
    other_url = known & (paths != p2)
    masks['back_navigation_same_url'] = same_url
    masks['back_navigation_other_url'] = other_url

    total_cases = result['summary']['total_cases']
    summary = result['summary']
    for key, mask in (('same_url', same_url), ('other_url', other_url), ('in_page', in_page)):
        cases = work.loc[mask, case_col].nunique()
        summary[f'back_nav_{key}_events'] = int(mask.sum())
        summary[f'back_nav_{key}_cases'] = int(cases)
        summary[f'back_nav_{key}_case_share'] = _share(cases, total_cases)
    summary['back_nav_unknown_url_events'] = int((back_nav & ~known).sum())

    # A repeat of the activity on a URL it already had in this case.
    keyed = work[[case_col, activity_col]].assign(_path=paths.fillna(''))
    repeat_same_url = repeat & paths.notna() & (
        keyed.groupby([case_col, activity_col, '_path'], sort=False).cumcount() > 0
    )
    cases = work.loc[repeat_same_url, case_col].nunique()
    summary['repeat_same_url_events'] = int(repeat_same_url.sum())
    summary['repeat_same_url_cases'] = int(cases)
    summary['repeat_same_url_case_share'] = _share(cases, total_cases)

    if result['back_navigation']:
        patterns = (
            work.groupby(case_col, sort=False)[activity_col].shift(2) + ' -> '
            + work.groupby(case_col, sort=False)[activity_col].shift(1) + ' -> ' + work[activity_col]
        )
        split = pd.DataFrame({
            'pattern': patterns[back_nav], 'same': same_url[back_nav],
            'other': other_url[back_nav], 'in_page': in_page[back_nav],
        }).groupby('pattern', sort=False).sum()
        for pattern, row in result['back_navigation'].items():
            row['same_url'] = int(split.at[pattern, 'same'])
            row['other_url'] = int(split.at[pattern, 'other'])
            row['in_page'] = int(split.at[pattern, 'in_page'])

    if result['repeats']:
        has_url = paths.notna()
        frame = pd.DataFrame({
            'activity': work.loc[has_url, activity_col], 'case': work.loc[has_url, case_col],
            'path': paths[has_url], 'same': repeat_same_url[has_url],
        })
        distinct = frame.groupby(['activity', 'case'], sort=False)['path'].nunique().groupby('activity').mean()
        same = frame.groupby('activity', sort=False)['same'].sum()
        for activity, row in result['repeats'].items():
            row['same_url_repeat_events'] = int(same.get(activity, 0))
            row['distinct_urls_per_case'] = (
                float(distinct[activity]) if activity in distinct.index else None
            )


def _page_loops(
    work: pd.DataFrame, page_paths: pd.Series, case_col: str, session_col: Optional[str],
    converted_cases: Optional[set], top_n: int,
) -> Dict[str, Any]:
    """The loop analysis on page visits: events without a URL are left out,
    and consecutive events on one page (within one session) are one visit."""
    has_url = page_paths.notna()
    cols = [case_col] + ([session_col] if session_col else [])
    events = work.loc[has_url, cols].assign(_page=page_paths[has_url])
    by_case = events.groupby(case_col, sort=False)
    starts = by_case['_page'].shift(1).isna() | (events['_page'] != by_case['_page'].shift(1))
    if session_col:
        starts |= events[session_col] != by_case[session_col].shift(1)
    visits = events.loc[starts, [case_col, '_page']].reset_index(drop=True)

    if visits.empty:
        return {}
    result, masks = _sequence_loops(visits, case_col, '_page', top_n)
    result['after_loop'], result['outcomes'] = _kind_details(
        visits, masks, LOOP_KINDS, case_col, converted_cases, top_n
    )
    result['summary']['page_visits'] = int(len(visits))
    result['summary']['events_without_url'] = int((~has_url).sum())
    result['summary']['distinct_pages'] = int(visits['_page'].nunique())
    return result


def analyze_loops(
    df: pd.DataFrame,
    activity_groups: Optional[Dict[str, List[str]]] = None,
    group_col: Optional[str] = None,
    include_ungrouped: bool = False,
    purchase_values=None,
    url_col: Optional[str] = None,
    url_rules=None,
    collapse_ids: bool = True,
    sections: Optional[Dict[str, List[str]]] = None,
    group_by_section: bool = False,
    case_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    timestamp_col: str = 'time:timestamp',
    session_col: str = 'session_id',
    top_n: int = 15,
) -> Dict[str, Any]:
    """
    Measures back-navigation, repeat visits and group switches in an event log.

    Parameters
    ----------
    activity_groups : dict, optional
        {group name: [activities]} for switch analysis, e.g.
        {'Navigation': ['view_item_list', 'menu_click'], 'Search': ['search']}.
        Activities in no group are skipped; an activity in several groups
        counts for the first. Ignored when group_col is given.
    group_col : str, optional
        Column whose values are the groups instead (e.g. 'category'). Events
        with no value are skipped.
    include_ungrouped : bool
        Put the events that would be skipped in a group of their own,
        OTHER_GROUP, instead. A switch is then also a move into or out of
        ungrouped activity, and G1 -> (other) -> G1 counts as a switch back -
        what to use to see the loops that run through activities outside
        the groups. With it, one activity group is enough.
    purchase_values : list, optional
        Purchase detection, shared with the Business Insights analysis. When
        the log has purchases, conversion is compared for cases with vs.
        without each loop kind.
    url_col : str, optional
        Column with the URL each event fired on (e.g. 'page_location'). Adds
        the URL split of back-navigation and repeats, and 'page_loops'.
        resolve_loop_params() detects it when loop_params doesn't name one.
    url_rules : list, optional
        [[regex, replacement], ...] applied in order to each page path for
        the page-level analysis and sections (e.g. [['^/p/[^/]+$', '/p/{slug}']]
        to treat all product pages as one, or [['^/(nl|en)(/|$)', '/']] to
        drop a language prefix). Not used for the URL split, which compares
        exact paths.
    collapse_ids : bool
        Replace ID-like path segments (digits, UUIDs, long hex, runs of 5+
        digits) with {id} in the page-level analysis and sections.
    sections : dict, optional
        {section name: [regex, ...]} matched against the page path (after
        url_rules); a page matching none gets its path's first segment.
    group_by_section : bool
        Use the site sections as the groups for switches, instead of
        activity_groups / group_col.
    session_col : str
        Where a case spans sessions (cases per user), a page visit doesn't
        run on across a session boundary. Ignored when the column is missing.

    Returns
    -------
    dict with keys:
        'summary'         : {total_cases, total_events, back_nav_events,
                             back_nav_cases, back_nav_case_share, repeat_events,
                             repeat_cases, repeat_case_share, self_loop_events,
                             self_loop_cases, self_loop_case_share}; with a URL
                            column also back_nav_{same_url, other_url,
                            in_page}_{events, cases, case_share},
                            back_nav_unknown_url_events and
                            repeat_same_url_{events, cases, case_share}
        'back_navigation' : {"A -> B -> A": {occurrences, cases, case_share}},
                            top_n by occurrences; with a URL column also
                            same_url, other_url and in_page counts
        'repeats'         : {activity: {cases_with_activity, repeat_cases,
                             repeat_case_share, repeat_events, self_loop_events,
                             back_nav_returns, visits_per_case}}, top_n by
                            repeat_events; repeat_case_share is of the cases
                            that have the activity at all; with a URL column
                            also same_url_repeat_events and
                            distinct_urls_per_case
        'after_loop'      : {kind: {loops, exit_share,
                             next_activity: {activity | '(exit)': {count, share}},
                             position: {early, middle, late}, mean_step}}, for
                            LOOP_KINDS plus, with a URL column, URL_LOOP_KINDS
        'outcomes'        : {kind: {'with' | 'without': {cases, conversions,
                             conversion_rate}}} for the same kinds, or None
                            without purchases
        'group_switches'  : {source, group_col, groups, include_ungrouped, group_stats,
                             transitions: {"G1 -> G2": {from, to, occurrences,
                             cases, case_share}}, summary}, or None when no
                            groups are defined; source is 'activities',
                            'column' or 'section'
        'page_loops'      : None without a URL column, else {url_col,
                             url_rules, collapse_ids, summary, back_navigation,
                             repeats, after_loop, outcomes} - the same analysis
                            on page visits (activities are page paths, so
                            self_loop counts are 0 within a session); summary
                            adds page_visits, events_without_url and
                            distinct_pages
        'errors'          : list of str
    Shares and rates are percentages of all cases unless noted.
    """
    errors: List[str] = []
    result: Dict[str, Any] = {
        'summary': {}, 'back_navigation': {}, 'repeats': {}, 'after_loop': {},
        'outcomes': None, 'group_switches': None, 'page_loops': None, 'errors': errors,
    }

    if df is None or df.empty:
        errors.append("Event log is empty.")
        return result
    required = [case_col, activity_col, timestamp_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Loop analysis needs columns {required}; missing: {missing}.")
        return result
    if group_col and group_col not in df.columns:
        errors.append(f"Group column '{group_col}' not found; group switches skipped.")
        group_col = None
    if url_col and url_col not in df.columns:
        errors.append(f"URL column '{url_col}' not found; URL analysis skipped.")
        url_col = None
    if group_by_section and not url_col:
        errors.append("Site sections need a URL column; group switches by section skipped.")
        group_by_section = False
    if session_col not in df.columns or session_col == case_col:
        session_col = None

    work = df.loc[df[activity_col].notna()].sort_values([case_col, timestamp_col], kind='stable')
    work = work.reset_index(drop=True)
    work[activity_col] = work[activity_col].astype(str)

    core, masks = _sequence_loops(work, case_col, activity_col, top_n)
    result.update(core)

    purchase_mask = _purchase_row_mask(
        work, activity_col, _normalise_values(purchase_values, ['purchase', 'has_purchase'])
    )
    converted_cases = set(work.loc[purchase_mask, case_col]) if purchase_mask.any() else None

    kinds = list(LOOP_KINDS)
    page_paths = None
    if url_col:
        _, rule_errors = compile_url_rules(url_rules)
        _, section_errors = compile_sections(sections)
        errors.extend(rule_errors + section_errors)
        exact_paths = normalize_url_paths(work[url_col], collapse_ids=False)
        if exact_paths.notna().any():
            _url_split(work, exact_paths, masks, case_col, activity_col, result)
            kinds += list(URL_LOOP_KINDS)
            page_paths = normalize_url_paths(work[url_col], url_rules, collapse_ids)
            page = _page_loops(work, page_paths, case_col, session_col, converted_cases, top_n)
            if page:
                result['page_loops'] = {
                    'url_col': url_col, 'url_rules': url_rules or [], 'collapse_ids': collapse_ids, **page,
                }
        else:
            errors.append(f"URL column '{url_col}' has no values; URL analysis skipped.")
            group_by_section = False

    result['after_loop'], result['outcomes'] = _kind_details(
        work, masks, kinds, case_col, converted_cases, top_n
    )

    groups = _clean_groups(activity_groups)
    total_cases = result['summary']['total_cases']
    if group_by_section:
        work[SECTION_COL] = page_sections(page_paths, sections)
        switches = _group_switches(
            work, case_col, activity_col, {}, SECTION_COL, include_ungrouped, total_cases, top_n
        )
        switches['source'] = 'section'
        switches['sections'] = sections or {}
        result['group_switches'] = switches
    elif group_col or groups:
        if not group_col and len(groups) + include_ungrouped < 2:
            errors.append(
                "Define at least two activity groups, or one plus the ungrouped activities, "
                "to measure switches between them."
            )
        else:
            result['group_switches'] = _group_switches(
                work, case_col, activity_col, groups, group_col, include_ungrouped, total_cases, top_n
            )

    return result


def fit_loop_params_to_log(
    loop_params: Optional[Dict[str, Any]], activities: List[str], columns: List[str]
) -> tuple:
    """Returns (loop_params, notes): loop_params without the grouped
    activities, empty groups, group column and URL column the log doesn't
    have, and without URL rules or section patterns that aren't valid regex
    (an unknown URL column is left to be detected again). notes is
    one readable line per thing dropped. For settings carried over from
    another log (a preset or a saved run)."""
    if not loop_params:
        return loop_params, []
    known_acts = set(activities)
    notes: List[str] = []
    groups = {}
    for name, acts in _clean_groups(loop_params.get('activity_groups')).items():
        missing = [a for a in acts if a not in known_acts]
        kept = [a for a in acts if a in known_acts]
        if not kept:
            notes.append(f"Loop group '{name}' has no activities in this log, so it was dropped.")
            continue
        if missing:
            notes.append(f"Loop group '{name}': activities not in this log were dropped: {', '.join(missing)}.")
        groups[name] = kept
    group_col = loop_params.get('group_col')
    if group_col and group_col not in set(columns):
        notes.append(f"Loop groups: column '{group_col}' is not in this log - no group switches measured.")
        group_col = None
    fitted = {**loop_params, 'activity_groups': groups, 'group_col': group_col}

    url_col = fitted.get('url_col')
    if url_col and url_col not in set(columns):
        detected = detect_url_column(columns)
        notes.append(
            f"Loop URL column '{url_col}' is not in this log - "
            + (f"using '{detected}' instead." if detected else "no URL analysis.")
        )
        fitted.pop('url_col')  # detected again at run time
    rules, rule_errors = compile_url_rules(fitted.get('url_rules'))
    if rule_errors:
        notes.extend(rule_errors)
        fitted['url_rules'] = [[p.pattern, r] for p, r in rules]
    sections, section_errors = compile_sections(fitted.get('sections'))
    if section_errors:
        notes.extend(section_errors)
        kept: Dict[str, List[str]] = {}
        for name, pattern in sections:
            kept.setdefault(name, []).append(pattern.pattern)
        fitted['sections'] = kept
    return fitted, notes
