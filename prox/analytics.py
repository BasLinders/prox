import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


def get_event_log_summary(event_log_df: pd.DataFrame) -> Tuple[Dict[str, Any] | None, list]:
    """
    Computes high-level statistics for an event log.

    Returns
    -------
    summary : dict or None
    errors  : list of str
    """
    errors = []
    SECONDS_PER_DAY = 86400

    if event_log_df is None or event_log_df.empty:
        errors.append("Critical Error: Event log is empty or None.")
        return None, errors

    required = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing = [c for c in required if c not in event_log_df.columns]
    if missing:
        errors.append(f"Critical Error: Missing required columns: {missing}.")
        return None, errors

    try:
        if not pd.api.types.is_datetime64_any_dtype(event_log_df['time:timestamp']):
            df = event_log_df.copy()
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
        else:
            df = event_log_df

        invalid = df['time:timestamp'].isnull().sum()
        if invalid > 0:
            errors.append(f"Warning: {invalid} invalid timestamps removed.")
            df = df.dropna(subset=['time:timestamp'])

        if df.empty:
            errors.append("Critical Error: No valid data after timestamp handling.")
            return None, errors

        num_events = len(df)
        num_cases = df['case:concept:name'].nunique()
        start_time = df['time:timestamp'].min()
        end_time = df['time:timestamp'].max()
        unique_activities = df['concept:name'].nunique()
        duration_days = (end_time - start_time).total_seconds() / SECONDS_PER_DAY

        summary = {
            'Number of Cases': num_cases,
            'Number of Events': num_events,
            'Start Timestamp': str(start_time),
            'End Timestamp': str(end_time),
            'Total Duration (Days)': round(duration_days, 2),
            'Number of Unique Activities': unique_activities,
            'Average Events per Case': round(num_events / num_cases, 2) if num_cases > 0 else 0,
            'List of Activities': sorted(df['concept:name'].unique().tolist())
        }
        return summary, errors

    except Exception as e:
        errors.append(f"Unexpected error: {e}")
        return None, errors


def analyze_process_performance(
    event_log_df: pd.DataFrame,
    aggregation_level: str = 'mean',
    bottleneck_threshold_percentile: float = 75,
    include_variants: bool = True,
    time_unit: str = 'hours'
) -> Dict[str, Any]:
    """
    Computes case durations, activity performance, bottlenecks, variant stats,
    temporal patterns, and an overall process health score.

    Returns a nested results dict; all errors are collected in results['errors'].
    """
    results = {
        'case_performance': {},
        'activity_performance': {},
        'transition_performance': {},
        'bottlenecks': {},
        'variant_performance': {},
        'temporal_patterns': {},
        'resource_performance': {},
        'summary_statistics': {},
        'errors': []
    }

    divisors = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}
    if time_unit not in divisors:
        results['errors'].append(f"Unknown time unit '{time_unit}'. Defaulting to hours.")
        time_unit = 'hours'
    time_divisor = divisors[time_unit]

    try:
        required = ['case:concept:name', 'concept:name', 'time:timestamp']
        missing = [c for c in required if c not in event_log_df.columns]
        if missing:
            results['errors'].append(f"Critical Error: Missing columns: {missing}")
            return results

        if not pd.api.types.is_datetime64_any_dtype(event_log_df['time:timestamp']):
            event_log_df = event_log_df.copy()
            event_log_df['time:timestamp'] = pd.to_datetime(event_log_df['time:timestamp'], errors='coerce')
            event_log_df.dropna(subset=['time:timestamp'], inplace=True)

        df = event_log_df.copy()
        df.sort_values(['case:concept:name', 'time:timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- Case performance ---
        case_groups = df.groupby('case:concept:name')
        case_durations = case_groups['time:timestamp'].agg(['min', 'max'])
        case_durations['duration'] = (
            case_durations['max'] - case_durations['min']
        ).dt.total_seconds() / time_divisor
        case_durations['num_events'] = case_groups.size()
        case_durations['throughput_rate'] = (
            case_durations['num_events'] / case_durations['duration'].replace(0, np.nan)
        )

        q25 = case_durations['duration'].quantile(0.25)
        q75 = case_durations['duration'].quantile(0.75)

        results['case_performance'] = {
            'total_cases': len(case_durations),
            'duration_stats': {
                'mean': float(case_durations['duration'].mean()),
                'median': float(case_durations['duration'].median()),
                'std': float(case_durations['duration'].std()),
                'min': float(case_durations['duration'].min()),
                'max': float(case_durations['duration'].max()),
                'q25': float(q25),
                'q75': float(q75),
                'unit': time_unit
            },
            'throughput_stats': {
                'mean_events_per_case': float(case_durations['num_events'].mean()),
                'mean_throughput_rate': float(case_durations['throughput_rate'].mean()),
                'median_throughput_rate': float(case_durations['throughput_rate'].median()),
            },
            'case_duration_distribution': {
                'short_cases': int((case_durations['duration'] <= q25).sum()),
                'medium_cases': int(((case_durations['duration'] > q25) & (case_durations['duration'] <= q75)).sum()),
                'long_cases': int((case_durations['duration'] > q75).sum())
            }
        }

        # --- Activity performance ---
        df['prev_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(1)
        df['time_since_prev'] = (
            df['time:timestamp'] - df['prev_timestamp']
        ).dt.total_seconds() / time_divisor

        activity_stats = df.groupby('concept:name').agg({
            'case:concept:name': 'count',
            'time_since_prev': ['mean', 'median', 'std', 'min', 'max']
        }).round(3)
        activity_stats.columns = ['frequency', 'mean_duration', 'median_duration',
                                   'std_duration', 'min_duration', 'max_duration']

        df['case_start'] = df.groupby('case:concept:name')['time:timestamp'].transform('min')
        df['waiting_time'] = (df['time:timestamp'] - df['case_start']).dt.total_seconds() / time_divisor
        waiting_stats = df.groupby('concept:name')['waiting_time'].agg(['mean', 'median', 'std']).round(3)

        activity_performance = {}
        for activity, stats in activity_stats.iterrows():
            activity_performance[activity] = {
                'frequency': int(stats['frequency']),
                'frequency_percentage': float(stats['frequency'] / len(df) * 100),
                'duration': {k: float(stats[f'{k}_duration']) if pd.notna(stats[f'{k}_duration']) else 0
                             for k in ('mean', 'median', 'std', 'min', 'max')},
                'waiting_time': {
                    'mean': float(waiting_stats.loc[activity, 'mean']),
                    'median': float(waiting_stats.loc[activity, 'median']),
                    'std': float(waiting_stats.loc[activity, 'std'])
                }
            }
        results['activity_performance'] = activity_performance

        # --- Transition performance ---
        df['next_activity'] = df.groupby('case:concept:name')['concept:name'].shift(-1)
        df['next_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(-1)
        transitions = df[df['next_activity'].notna()].copy()
        transitions['transition'] = transitions['concept:name'] + ' -> ' + transitions['next_activity']
        transitions['transition_time'] = (
            transitions['next_timestamp'] - transitions['time:timestamp']
        ).dt.total_seconds() / time_divisor

        t_stats = transitions.groupby('transition').agg({
            'transition_time': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'case:concept:name': 'nunique'
        }).round(3)
        t_stats.columns = ['frequency', 'mean_time', 'median_time',
                           'std_time', 'min_time', 'max_time', 'num_cases']

        transition_performance = {}
        for t in t_stats.index:
            source, target = t.split(' -> ')
            transition_performance[t] = {
                'source': source, 'target': target,
                'frequency': int(t_stats.loc[t, 'frequency']),
                'num_cases': int(t_stats.loc[t, 'num_cases']),
                'duration': {k: float(t_stats.loc[t, f'{k}_time'])
                             for k in ('mean', 'median', 'std', 'min', 'max')}
            }
        results['transition_performance'] = transition_performance

        # --- Bottlenecks ---
        act_means = [s['duration']['mean'] for s in activity_performance.values() if s['duration']['mean'] > 0]
        act_threshold = np.percentile(act_means, bottleneck_threshold_percentile) if act_means else 0

        activity_bottlenecks = {
            act: {
                'mean_duration': s['duration']['mean'],
                'frequency': s['frequency'],
                'impact_score': s['duration']['mean'] * s['frequency'],
                'severity': 'high' if s['duration']['mean'] > act_threshold * 1.5 else 'medium'
            }
            for act, s in activity_performance.items()
            if s['duration']['mean'] > act_threshold
        }
        activity_bottlenecks = dict(
            sorted(activity_bottlenecks.items(), key=lambda x: x[1]['impact_score'], reverse=True)
        )

        t_times = [s['duration']['mean'] for s in transition_performance.values()]
        transition_bottlenecks = {}
        if t_times:
            t_threshold = np.percentile(t_times, bottleneck_threshold_percentile)
            transition_bottlenecks = {
                t: {
                    'mean_duration': s['duration']['mean'],
                    'frequency': s['frequency'],
                    'impact_score': s['duration']['mean'] * s['frequency'],
                    'severity': 'high' if s['duration']['mean'] > t_threshold * 1.5 else 'medium'
                }
                for t, s in transition_performance.items()
                if s['duration']['mean'] > t_threshold
            }
            transition_bottlenecks = dict(
                sorted(transition_bottlenecks.items(), key=lambda x: x[1]['impact_score'], reverse=True)
            )

        results['bottlenecks'] = {
            'activity_bottlenecks': activity_bottlenecks,
            'transition_bottlenecks': transition_bottlenecks,
            'threshold_percentile': bottleneck_threshold_percentile,
            'summary': {
                'num_activity_bottlenecks': len(activity_bottlenecks),
                'num_transition_bottlenecks': len(transition_bottlenecks),
                'top_activity_bottleneck': next(iter(activity_bottlenecks), None),
                'top_transition_bottleneck': next(iter(transition_bottlenecks), None),
            }
        }

        # --- Variant performance ---
        if include_variants:
            variants = df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x))
            variant_counts = variants.value_counts()
            top_variants = variant_counts.head(20)

            variant_performance = {}
            for variant, count in top_variants.items():
                v_cases = variants[variants == variant].index
                v_dur = case_durations.loc[v_cases, 'duration']
                variant_performance[variant] = {
                    'frequency': int(count),
                    'percentage': float(count / len(variants) * 100),
                    'duration': {
                        'mean': float(v_dur.mean()), 'median': float(v_dur.median()),
                        'std': float(v_dur.std()), 'min': float(v_dur.min()), 'max': float(v_dur.max())
                    },
                    'num_activities': len(variant.split(' -> '))
                }

            results['variant_performance'] = {
                'total_variants': len(variant_counts),
                'top_variants': variant_performance,
                'variant_coverage': {
                    'top_5_coverage': float(variant_counts.head(5).sum() / len(variants) * 100),
                    'top_10_coverage': float(variant_counts.head(10).sum() / len(variants) * 100),
                    'top_20_coverage': float(variant_counts.head(20).sum() / len(variants) * 100)
                }
            }

        # --- Temporal patterns ---
        df['hour'] = df['time:timestamp'].dt.hour
        df['day_name'] = df['time:timestamp'].dt.day_name()
        hourly = df.groupby('hour').size()
        daily = df.groupby('day_name').size()

        case_temporal = df.groupby('case:concept:name')['time:timestamp'].min().to_frame()
        case_temporal['duration'] = case_durations['duration']
        case_temporal['hour'] = case_temporal['time:timestamp'].dt.hour
        case_temporal['day_name'] = case_temporal['time:timestamp'].dt.day_name()

        results['temporal_patterns'] = {
            'hourly_patterns': {
                'peak_hours': hourly.nlargest(3).index.tolist(),
                'hourly_distribution': hourly.to_dict()
            },
            'daily_patterns': {
                'busiest_days': daily.nlargest(3).index.tolist(),
                'daily_distribution': daily.to_dict()
            },
            'performance_by_hour': case_temporal.groupby('hour')['duration'].mean().to_dict(),
            'performance_by_day': case_temporal.groupby('day_name')['duration'].mean().to_dict()
        }

        # --- Resource performance ---
        resource_col = next(
            (c for c in ['org:resource', 'resource', 'user', 'operator'] if c in df.columns), None
        )
        if resource_col:
            res_stats = df.groupby(resource_col).agg({
                'case:concept:name': ['count', 'nunique'],
                'time_since_prev': ['mean', 'median']
            }).round(3)
            res_stats.columns = ['total_events', 'unique_cases', 'mean_proc_time', 'median_proc_time']
            results['resource_performance'] = {
                'num_resources': len(res_stats),
                'resource_metrics': res_stats.to_dict('index')
            }
        else:
            results['resource_performance']['note'] = 'No resource column found'

        # --- Summary / health score ---
        avg_dur = case_durations['duration'].mean()
        variability = case_durations['duration'].std() / avg_dur if avg_dur > 0 else 1
        bn_ratio = len(activity_bottlenecks) / len(activity_performance) if activity_performance else 0
        health_score = max(0, min(100, 100 * (1 - variability * 0.3) * (1 - bn_ratio * 0.5)))

        recommendations = _generate_performance_recommendations(results)
        results['summary_statistics'] = {
            'process_health_score': round(health_score, 2),
            'efficiency_metrics': {
                'average_case_duration': round(avg_dur, 2),
                'duration_variability_pct': round(variability * 100, 2),
                'bottleneck_ratio_pct': round(bn_ratio * 100, 2)
            },
            'recommendations': recommendations
        }

    except Exception as e:
        results['errors'].append(f"Unexpected error: {e}")
        results['errors'].append(traceback.format_exc())

    return results


def _generate_performance_recommendations(results: Dict[str, Any]) -> List[str]:
    recommendations = []
    try:
        case_stats = results.get('case_performance', {}).get('duration_stats', {})
        if case_stats:
            mean = case_stats.get('mean', 1)
            std = case_stats.get('std', 0)
            cv = std / mean if mean > 0 else 0
            if cv > 0.5:
                recommendations.append(
                    f"High process variability (CV: {cv * 100:.1f}%). "
                    "Consider standardising process paths or investigating outlier cases."
                )

        bn_summary = results.get('bottlenecks', {}).get('summary', {})
        top_bn = bn_summary.get('top_activity_bottleneck')
        if top_bn:
            recommendations.append(
                f"Critical bottleneck at '{top_bn}'. Prioritise optimisation here."
            )

        variant_info = results.get('variant_performance', {})
        if variant_info:
            total_v = variant_info.get('total_variants', 0)
            top5 = variant_info.get('variant_coverage', {}).get('top_5_coverage', 0)
            if total_v > 50 and top5 < 50:
                recommendations.append(
                    f"High complexity: {total_v} variants with low concentration. "
                    "Consider process standardisation."
                )

        peak_hours = results.get('temporal_patterns', {}).get('hourly_patterns', {}).get('peak_hours', [])
        if peak_hours:
            recommendations.append(
                f"Peak activity hours: {peak_hours}. Consider resource scaling during these periods."
            )

        health = results.get('summary_statistics', {}).get('process_health_score', 0)
        if health < 50:
            recommendations.append(f"Low health score ({health:.0f}/100). Process redesign may be needed.")
        elif health > 80:
            recommendations.append(f"Good health score ({health:.0f}/100). Focus on continuous monitoring.")

        if not recommendations:
            recommendations.append("Process within normal parameters. Continue monitoring.")

    except Exception as e:
        recommendations.append(f"Unable to generate recommendations: {e}")

    return recommendations


def analyze_repeat_purchases(
    df: pd.DataFrame,
    output_folder: str = "output",
    user_col: str = "user_id",
    activity_col: str = "concept:name",
    timestamp_col: str = "time:timestamp",
    revenue_col: str = "event_value",
    case_col: str = "case:concept:name",
    purchase_values=None
) -> Dict[str, Any]:
    """
    Identifies repeat buyers, computes loyalty metrics, inter-purchase timing,
    and revenue lift from repeat customers. Saves up to three charts to output_folder.

    Returns dict with 'metrics' and 'charts' keys.
    """
    logger.info("Analyzing repeat purchases.")

    empty = {
        "metrics": {"total_buyers": 0, "repeat_rate": 0, "avg_days_between": 0,
                    "median_days_between": 0, "revenue_stats": {}},
        "charts": {}
    }

    if purchase_values is None:
        purchase_values = ['purchase', 'order', 'has_purchase', 'payment', 'transaction']
    elif isinstance(purchase_values, str):
        purchase_values = [purchase_values]

    os.makedirs(output_folder, exist_ok=True)
    cols_map = {c.lower(): c for c in df.columns}

    # --- Column detection ---
    real_user_col = cols_map.get(user_col.lower())
    if not real_user_col:
        for c in ['customer_id', 'user', 'user_id', 'case_id']:
            if c in cols_map:
                real_user_col = cols_map[c]
                break

    real_case_col = cols_map.get(case_col.lower())
    if not real_case_col:
        for c in ['case_id', 'case', 'session_id', 'session']:
            if c in cols_map:
                real_case_col = cols_map[c]
                break

    real_time_col = cols_map.get(timestamp_col.lower())
    real_activity_col = cols_map.get(activity_col.lower(), activity_col)

    # Smart revenue column detection: pick first candidate that has values > 0
    real_rev_col = None
    candidates = list(dict.fromkeys([revenue_col, 'event_value', 'revenue', 'total_amount', 'value', 'price', 'amount']))
    for cand in candidates:
        if cand.lower() in cols_map:
            col_name = cols_map[cand.lower()]
            try:
                if pd.to_numeric(df[col_name], errors='coerce').sum() > 0:
                    real_rev_col = col_name
                    logger.info("Revenue column selected: '%s'", real_rev_col)
                    break
            except Exception:
                continue
    if not real_rev_col:
        for cand in candidates:
            if cand.lower() in cols_map:
                real_rev_col = cols_map[cand.lower()]
                logger.warning("No revenue column with >0 values found. Defaulting to '%s'.", real_rev_col)
                break

    if not real_user_col or not real_time_col or not real_case_col or real_activity_col not in df.columns:
        logger.warning("Missing required columns for repeat purchase analysis.")
        return empty

    df = df.copy()
    df[real_time_col] = pd.to_datetime(df[real_time_col], utc=True)

    # --- Purchase detection (three strategies, combined) ---
    pattern = '|'.join([p.lower() for p in purchase_values])
    mask_text = df[real_activity_col].astype(str).str.lower().str.contains(pattern, na=False)

    mask_col = pd.Series(False, index=df.index)
    for flag_col in [c for c in df.columns if 'purchase' in c.lower() or 'conversion' in c.lower()]:
        if flag_col == real_activity_col:
            continue
        try:
            hits = pd.to_numeric(df[flag_col], errors='coerce').fillna(0) > 0
            if hits.sum() > 0:
                mask_col = mask_col | hits
        except Exception:
            pass

    mask_rev = pd.Series(False, index=df.index)
    if real_rev_col:
        try:
            mask_rev = pd.to_numeric(df[real_rev_col], errors='coerce').fillna(0) > 0
        except Exception:
            pass

    purchase_case_ids = df[mask_text | mask_col | mask_rev][real_case_col].unique()
    if len(purchase_case_ids) == 0:
        logger.info("No purchase cases identified.")
        return empty

    logger.info("Found %d unique purchase sessions.", len(purchase_case_ids))

    purchase_traces = df[df[real_case_col].isin(purchase_case_ids)].copy()
    agg_rules = {real_time_col: 'min', real_user_col: 'first'}
    if real_rev_col:
        purchase_traces[real_rev_col] = pd.to_numeric(purchase_traces[real_rev_col], errors='coerce').fillna(0)
        agg_rules[real_rev_col] = 'max'

    purchase_df = purchase_traces.groupby(real_case_col).agg(agg_rules).reset_index()

    user_counts = purchase_df.groupby(real_user_col).size().reset_index(name='purchase_count')
    repeat_buyers = user_counts[user_counts['purchase_count'] > 1]
    repeat_rate = (len(repeat_buyers) / len(user_counts)) * 100 if len(user_counts) > 0 else 0

    # --- Chart 1: Purchase distribution ---
    chart_dist = None
    try:
        chart_dist = os.path.join(output_folder, "repeat_purchases_dist.png")
        viz = user_counts.copy()
        viz['bucket'] = viz['purchase_count'].apply(lambda x: str(x) if x < 5 else "5+")
        plt.figure(figsize=(8, 5))
        sns.countplot(data=viz, x='bucket', order=['1', '2', '3', '4', '5+'],
                      palette="viridis", hue='bucket', legend=False)
        plt.title(f"Orders per Customer (Repeat Rate: {repeat_rate:.1f}%)")
        plt.ylabel("Customer Count")
        plt.tight_layout()
        plt.savefig(chart_dist)
        plt.close()
    except Exception as e:
        logger.warning("Distribution chart failed: %s", e)
        chart_dist = None

    # --- Chart 2: Time between purchases ---
    avg_days = 0.0
    med_days = 0.0
    chart_time = None

    purchase_df = purchase_df.sort_values([real_user_col, real_time_col])
    purchase_df['prev_time'] = purchase_df.groupby(real_user_col)[real_time_col].shift(1)
    purchase_df['days_diff'] = (
        purchase_df[real_time_col] - purchase_df['prev_time']
    ).dt.total_seconds() / (3600 * 24)
    repeat_data = purchase_df.dropna(subset=['days_diff']).copy()

    if not repeat_data.empty:
        avg_days = repeat_data['days_diff'].mean()
        med_days = repeat_data['days_diff'].median()

        repeat_data = repeat_data.merge(
            user_counts[[real_user_col, 'purchase_count']], on=real_user_col, how='left'
        )
        repeat_data['bucket'] = repeat_data['purchase_count'].apply(lambda x: str(x) if x < 5 else "5+")

        if len(repeat_data) > 5:
            chart_time = os.path.join(output_folder, "time_between_purchases.png")
            try:
                plt.figure(figsize=(8, 5))
                max_d = repeat_data['days_diff'].max()
                if pd.isna(max_d):
                    max_d = 0
                bin_w = None if max_d <= 1 else (1 if max_d < 60 else 7)
                use_kde = repeat_data['days_diff'].std() != 0

                ax = sns.histplot(
                    data=repeat_data, x='days_diff', hue='bucket',
                    hue_order=['2', '3', '4', '5+'], kde=use_kde,
                    binwidth=bin_w, palette="viridis", multiple="stack"
                )
                plt.axvline(med_days, color='r', linestyle='--')
                plt.text(
                    med_days + (max_d * 0.02), ax.get_ylim()[1] * 0.95,
                    f'Median: {med_days:.1f} days', color='r', fontweight='bold'
                )
                plt.title("Time Between Purchases")
                plt.xlabel("Days")
                if max_d > 0:
                    plt.xlim(0, max_d * 1.05)
                plt.tight_layout()
                plt.savefig(chart_time)
                plt.close()
            except Exception as e:
                logger.warning("Timing chart failed: %s", e)
                chart_time = None

    # --- Chart 3: Revenue comparison ---
    rev_stats = {}
    chart_rev = None

    if real_rev_col:
        purchase_df = purchase_df.merge(
            user_counts[[real_user_col, 'purchase_count']], on=real_user_col, how='left'
        )
        purchase_df['type'] = purchase_df['purchase_count'].apply(
            lambda x: 'Repeat Buyer' if x > 1 else 'One-time Buyer'
        )
        clv_df = purchase_df.groupby([real_user_col, 'type'])[real_rev_col].sum().reset_index()

        avg_rep = clv_df[clv_df['type'] == 'Repeat Buyer'][real_rev_col].mean()
        avg_once = clv_df[clv_df['type'] == 'One-time Buyer'][real_rev_col].mean()
        avg_rep = 0.0 if pd.isna(avg_rep) else float(avg_rep)
        avg_once = 0.0 if pd.isna(avg_once) else float(avg_once)
        multiplier = avg_rep / avg_once if avg_once > 0 else 0.0

        rev_stats = {
            "avg_value_one_time": avg_once,
            "avg_value_repeat": avg_rep,
            "multiplier": multiplier
        }

        if avg_rep > 0 or avg_once > 0:
            chart_rev = os.path.join(output_folder, "value_comparison.png")
            try:
                plt.figure(figsize=(6, 5))
                plot_data = pd.DataFrame({
                    'Type': ['One-time', 'Repeater'],
                    'Value': [avg_once, avg_rep]
                })
                sns.barplot(data=plot_data, x='Type', y='Value', hue='Type',
                            legend=False, palette="rocket")
                plt.title(f"Avg Lifetime Value (Repeaters: {multiplier:.1f}x)")
                plt.ylabel(f"Revenue ({real_rev_col})")
                plt.tight_layout()
                plt.savefig(chart_rev)
                plt.close()
            except Exception as e:
                logger.warning("Revenue chart failed: %s", e)
                chart_rev = None

    return {
        "metrics": {
            "total_buyers": len(user_counts),
            "repeat_rate": repeat_rate,
            "avg_days_between": float(avg_days),
            "median_days_between": float(med_days),
            "revenue_stats": rev_stats
        },
        "charts": {
            "distribution": chart_dist,
            "timing": chart_time,
            "revenue": chart_rev
        }
    }


def format_business_report(results: Dict[str, Any]) -> str:
    """Returns a formatted summary string of the repeat purchase analysis."""
    if not results:
        return "No business insights available."

    m = results.get('metrics', {})
    lines = ["=" * 40, "REPEAT PURCHASE INSIGHTS", "=" * 40]

    rate = m.get('repeat_rate', 0)
    buyers = m.get('total_buyers', 0)
    lines.append(f"LOYALTY: {rate:.2f}% of {buyers} buyers purchased more than once.")
    if rate > 40:
        lines.append("   STATUS: Excellent. High retention.")
    elif rate < 10:
        lines.append("   STATUS: Low. Focus on post-purchase engagement.")
    else:
        lines.append("   STATUS: Moderate.")

    med_days = m.get('median_days_between', 0)
    avg_days = m.get('avg_days_between', 0)
    if med_days > 0 or avg_days > 0:
        lines.append(f"\nTIMING: Customers return after ~{med_days:.1f} days (median).")
        if med_days < 1 and avg_days < 1:
            lines.append("   NOTE: Purchases cluster within the same session.")
    else:
        lines.append("\nTIMING: No return timing data available.")

    rev = m.get('revenue_stats', {})
    if rev:
        mult = rev.get('multiplier', 0)
        rep_val = rev.get('avg_value_repeat', 0)
        if rep_val > 0:
            lines.append(f"\nVALUE: Repeaters are {mult:.1f}x more valuable than one-timers.")
        else:
            lines.append(f"\nVALUE: Revenue data present but values appear to be 0 (multiplier: {mult:.1f}x).")
    else:
        lines.append("\nVALUE: No revenue data detected.")

    lines.append("=" * 40)
    return "\n".join(lines)
