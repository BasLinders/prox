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

Builds on the same per-case directly-follows ordering as
analytics.analyze_process_performance()'s transition data, and is run by the
pipeline so compare_segments() picks it up per segment.
"""

from typing import Any, Dict, List, Optional

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


def _outcomes(work: pd.DataFrame, loop_cases: Dict[str, set], purchase_mask: pd.Series, case_col: str) -> Dict[str, Any]:
    """Conversion of cases with vs. without each loop kind."""
    case_ids = pd.Index(work[case_col].unique())
    converted = case_ids.isin(work.loc[purchase_mask, case_col].unique())
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


def analyze_loops(
    df: pd.DataFrame,
    activity_groups: Optional[Dict[str, List[str]]] = None,
    group_col: Optional[str] = None,
    include_ungrouped: bool = False,
    purchase_values=None,
    case_col: str = 'case:concept:name',
    activity_col: str = 'concept:name',
    timestamp_col: str = 'time:timestamp',
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

    Returns
    -------
    dict with keys:
        'summary'         : {total_cases, total_events, back_nav_events,
                             back_nav_cases, back_nav_case_share, repeat_events,
                             repeat_cases, repeat_case_share, self_loop_events,
                             self_loop_cases, self_loop_case_share}
        'back_navigation' : {"A -> B -> A": {occurrences, cases, case_share}},
                            top_n by occurrences
        'repeats'         : {activity: {cases_with_activity, repeat_cases,
                             repeat_case_share, repeat_events, self_loop_events,
                             back_nav_returns, visits_per_case}}, top_n by
                            repeat_events; repeat_case_share is of the cases
                            that have the activity at all
        'after_loop'      : {'back_navigation' | 'repeat': {loops, exit_share,
                             next_activity: {activity | '(exit)': {count, share}},
                             position: {early, middle, late}, mean_step}}
        'outcomes'        : {'back_navigation' | 'repeat': {'with' | 'without':
                             {cases, conversions, conversion_rate}}}, or None
                            without purchases
        'group_switches'  : {source, group_col, groups, include_ungrouped, group_stats,
                             transitions: {"G1 -> G2": {from, to, occurrences,
                             cases, case_share}}, summary}, or None when no
                            groups are defined
        'errors'          : list of str
    Shares and rates are percentages of all cases unless noted.
    """
    errors: List[str] = []
    result: Dict[str, Any] = {
        'summary': {}, 'back_navigation': {}, 'repeats': {}, 'after_loop': {},
        'outcomes': None, 'group_switches': None, 'errors': errors,
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

    work = df.loc[df[activity_col].notna()].sort_values([case_col, timestamp_col], kind='stable')
    work = work.reset_index(drop=True)
    work[activity_col] = work[activity_col].astype(str)

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
    loop_cases = {
        'back_navigation': set(work.loc[back_nav, case_col]),
        'repeat': set(work.loc[repeat, case_col]),
    }
    self_loop_cases = work.loc[self_loop, case_col].nunique()

    result['summary'] = {
        'total_cases': int(total_cases),
        'total_events': int(len(work)),
        'back_nav_events': int(back_nav.sum()),
        'back_nav_cases': len(loop_cases['back_navigation']),
        'back_nav_case_share': _share(len(loop_cases['back_navigation']), total_cases),
        'repeat_events': int(repeat.sum()),
        'repeat_cases': len(loop_cases['repeat']),
        'repeat_case_share': _share(len(loop_cases['repeat']), total_cases),
        'self_loop_events': int(self_loop.sum()),
        'self_loop_cases': int(self_loop_cases),
        'self_loop_case_share': _share(self_loop_cases, total_cases),
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

    result['after_loop'] = {
        'back_navigation': _after_loop(work, back_nav, top_n),
        'repeat': _after_loop(work, repeat, top_n),
    }

    purchase_mask = _purchase_row_mask(
        work, activity_col, _normalise_values(purchase_values, ['purchase', 'has_purchase'])
    )
    if purchase_mask.any():
        result['outcomes'] = _outcomes(work, loop_cases, purchase_mask, case_col)

    groups = _clean_groups(activity_groups)
    if group_col or groups:
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
    activities, empty groups and group column the log doesn't have. notes is
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
    return {**loop_params, 'activity_groups': groups, 'group_col': group_col}, notes
