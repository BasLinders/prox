import pandas as pd
from typing import Dict, Any

def get_event_log_summary(event_log_df: pd.DataFrame) -> Tuple[Dict[str, Any] | None, list]:
    """
    Purpose:
        Provides quick statistics about the loaded event log, such as the number of cases,
        events, start/end timestamps, and unique activities.

    Args:
        event_log_df (pd.DataFrame): The input event log DataFrame. It must contain
                                     'case:concept:name', 'concept:name', and
                                     'time:timestamp' columns.

    Returns:
        A tuple containing:
        - A dictionary of summary statistics.
        - A list of error messages. Returns (None, errors) if a critical error occurs.
    """
    errors = []
    SECONDS_PER_DAY = 86400

    # --- 1. Input Validation ---
    if event_log_df is None or event_log_df.empty:
        errors.append("Critical Error: The event log DataFrame is empty or None.")
        return None, errors

    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_cols = [col for col in required_cols if col not in event_log_df.columns]
    if missing_cols:
        errors.append(f"Critical Error: The DataFrame is missing required columns: {missing_cols}.")
        return None, errors

    try:
        # Ensure timestamp is in datetime format for calculations
        if not pd.api.types.is_datetime64_any_dtype(event_log_df['time:timestamp']):
            df = event_log_df.copy()
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
        
        if df['time:timestamp'].isnull().any():
            errors.append("Warning: Some timestamps were invalid and could not be converted.")
            df.dropna(subset=['time:timestamp'], inplace=True)
        else:
            df = event_log_df
        
        if df.empty:
            errors.append("Critical Error: No valid data remains after handling timestamps.")
            return None, errors

        # --- 2. Calculate Statistics ---
        num_events = len(df)
        num_cases = df['case:concept:name'].nunique()
        start_time = df['time:timestamp'].min()
        end_time = df['time:timestamp'].max()
        unique_activities = df['concept:name'].nunique()
        
        # Calculate duration
        duration = end_time - start_time
        
        summary = {
            'Number of Cases': num_cases,
            'Number of Events': num_events,
            'Start Timestamp': str(start_time),
            'End Timestamp': str(end_time),
            'Total Duration (Days)': round(duration.total_seconds() / SECONDS_PER_DAY, 2),
            'Number of Unique Activities': unique_activities,
            'Average Events per Case': round(num_events / num_cases, 2) if num_cases > 0 else 0,
            'List of Activities': sorted(df['concept:name'].unique().tolist())
        }
        
        return summary, errors

    except Exception as e:
        errors.append(f"An unexpected error occurred: {e}")
        return None, errors

def analyze_process_performance(
    event_log_df: pd.DataFrame,
    aggregation_level: str = 'mean',
    bottleneck_threshold_percentile: float = 75,
    include_variants: bool = True,
    time_unit: str = 'hours'
) -> Dict[str, Any]:
    """
    Purpose: Extracts comprehensive performance metrics from the event log including activity durations,
             case durations, waiting times, bottleneck identification, and resource utilization patterns.

    Args:
        event_log_df (pd.DataFrame): PM4Py-formatted event log with columns:
                                    ['case:concept:name', 'concept:name', 'time:timestamp']
        aggregation_level (str): Statistical aggregation for metrics ('mean', 'median', 'both')
        bottleneck_threshold_percentile (float): Percentile threshold for bottleneck identification (0-100)
        include_variants (bool): Whether to include variant-specific performance analysis
        time_unit (str): Time unit for duration reporting ('seconds', 'minutes', 'hours', 'days')

    Returns:
        Dict[str, Any]: Comprehensive performance statistics.
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

    # Time unit conversion factors
    time_conversions = {
        'seconds': 1,
        'minutes': 60,
        'hours': 3600,
        'days': 86400
    }

    if time_unit not in time_conversions:
        results['errors'].append(f"Warning: unknown time unit: {time_unit}. Using hours.")
        time_unit = 'hours'

    time_divisor = time_conversions[time_unit]

    try:
        # Validate required columns
        required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
        missing_cols = [col for col in required_cols if col not in event_log_df.columns]
        if missing_cols:
            results['errors'].append(f"Critical Error: Missing required columns: {missing_cols}")
            return results

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(event_log_df['time:timestamp']):
            event_log_df['time:timestamp'] = pd.to_datetime(event_log_df['time:timestamp'], errors='coerce')
            event_log_df.dropna(subset=['time:timestamp'], inplace=True)

        # Create working copy and sort by case and timestamp
        df = event_log_df.copy()
        df.sort_values(by=['case:concept:name', 'time:timestamp'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- 1. Case-level performance metrics ---
        case_groups = df.groupby('case:concept:name')

        # Calculate case durations
        case_durations = case_groups['time:timestamp'].agg(['min', 'max'])
        case_durations['duration'] = (case_durations['max'] - case_durations['min']).dt.total_seconds() / time_divisor
        case_durations['num_events'] = case_groups.size()

        # Calculate throughput rate (events per time unit)
        case_durations['throughput_rate'] = case_durations['num_events'] / case_durations['duration']
        case_durations['throughput_rate'] = case_durations['throughput_rate'].replace([np.inf, -np.inf], np.nan)

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
                'unit': f"events per {time_unit[:-1]}" if time_unit != 'seconds' else "events per second"
            },
            'case_duration_distribution': {
                'short_cases': int((case_durations['duration'] <= q25).sum()),
                'medium_cases': int(((case_durations['duration'] > q25) & (case_durations['duration'] <= q75)).sum()),
                'long_cases': int((case_durations['duration'] > q75).sum())
            }
        }

        # --- 2. Activity level performance metrics ---
        # Calculate time between consecutive events within each case
        df['prev_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(1)
        df['time_since_prev'] = (df['time:timestamp'] - df['prev_timestamp']).dt.total_seconds() / time_divisor

        # Activity frequency and duration statistics
        activity_stats = df.groupby('concept:name').agg({
            'case:concept:name': 'count',
            'time_since_prev': ['mean', 'median', 'std', 'min', 'max']
        }).round(3)

        activity_stats.columns = ['frequency', 'mean_duration', 'median_duration',
                                  'std_duration', 'min_duration', 'max_duration']

        # Calculate waiting time (time until this activity starts after case starts)
        df['case_start'] = df.groupby('case:concept:name')['time:timestamp'].transform('min')
        df['waiting_time'] = (df['time:timestamp'] - df['case_start']).dt.total_seconds() / time_divisor

        waiting_stats = df.groupby('concept:name')['waiting_time'].agg(['mean', 'median', 'std']).round(3)

        activity_performance = {}
        for activity, stats in activity_stats.iterrows():
            activity_performance[activity] = {
                'frequency': int(stats['frequency']),
                'frequency_percentage': float(stats['frequency'] / len(df) * 100),
                'duration': {
                    'mean': float(stats['mean_duration']) if pd.notna(stats['mean_duration']) else 0,
                    'median': float(stats['median_duration']) if pd.notna(stats['median_duration']) else 0,
                    'std': float(stats['std_duration']) if pd.notna(stats['std_duration']) else 0,
                    'min': float(stats['min_duration']) if pd.notna(stats['min_duration']) else 0,
                    'max': float(stats['max_duration']) if pd.notna(stats['max_duration']) else 0
                },
                'waiting_time': {
                    'mean': float(waiting_stats.loc[activity, 'mean']),
                    'median': float(waiting_stats.loc[activity, 'median']),
                    'std': float(waiting_stats.loc[activity, 'std'])
                }
            }
        results['activity_performance'] = activity_performance

        # --- 3. Transition performance (direct-follows relations) ---
        df['next_activity'] = df.groupby('case:concept:name')['concept:name'].shift(-1)
        transitions = df[df['next_activity'].notna()].copy()
        transitions['transition'] = transitions['concept:name'] + ' -> ' + transitions['next_activity']

        transitions['next_timestamp'] = df.groupby('case:concept:name')['time:timestamp'].shift(-1)
        transitions['transition_time'] = (transitions['next_timestamp'] - transitions['time:timestamp']).dt.total_seconds() / time_divisor

        transition_stats = transitions.groupby('transition').agg({
            'transition_time': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'case:concept:name': 'nunique'
        }).round(3)

        transition_stats.columns = ['frequency', 'mean_time', 'median_time',
                                    'std_time', 'min_time', 'max_time', 'num_cases']

        transition_performance = {}
        for transition in transition_stats.index:
            source, target = transition.split(' -> ')
            transition_performance[transition] = {
                'source': source,
                'target': target,
                'frequency': int(transition_stats.loc[transition, 'frequency']),
                'num_cases': int(transition_stats.loc[transition, 'num_cases']),
                'duration': {
                    'mean': float(transition_stats.loc[transition, 'mean_time']),
                    'median': float(transition_stats.loc[transition, 'median_time']),
                    'std': float(transition_stats.loc[transition, 'std_time']),
                    'min': float(transition_stats.loc[transition, 'min_time']),
                    'max': float(transition_stats.loc[transition, 'max_time'])
                }
            }
        results['transition_performance'] = transition_performance

        # --- 4. Bottleneck identification ---
        activity_mean_durations = [stats['duration']['mean'] for stats in activity_performance.values() if stats['duration']['mean'] > 0]
        act_threshold = np.percentile(activity_mean_durations, bottleneck_threshold_percentile) if activity_mean_durations else 0

        activity_bottlenecks = {
            activity: {
                'mean_duration': stats['duration']['mean'],
                'frequency': stats['frequency'],
                'impact_score': stats['duration']['mean'] * stats['frequency'],
                'severity': 'high' if stats['duration']['mean'] > act_threshold * 1.5 else 'medium'
            }
            for activity, stats in activity_performance.items()
            if stats['duration']['mean'] > act_threshold
        }

        transition_times = [stats['duration']['mean'] for stats in transition_performance.values()]
        transition_bottlenecks = {}
        if transition_times:
            trans_threshold = np.percentile(transition_times, bottleneck_threshold_percentile)
            transition_bottlenecks = {
                trans: {
                    'mean_duration': stats['duration']['mean'],
                    'frequency': stats['frequency'],
                    'impact_score': stats['duration']['mean'] * stats['frequency'],
                    'severity': 'high' if stats['duration']['mean'] > trans_threshold * 1.5 else 'medium'
                }
                for trans, stats in transition_performance.items()
                if stats['duration']['mean'] > trans_threshold
            }

        # Sort bottlenecks by Impact Score
        activity_bottlenecks = dict(sorted(activity_bottlenecks.items(), key=lambda x: x[1]['impact_score'], reverse=True))
        transition_bottlenecks = dict(sorted(transition_bottlenecks.items(), key=lambda x: x[1]['impact_score'], reverse=True))

        results['bottlenecks'] = {
            'activity_bottlenecks': activity_bottlenecks,
            'transition_bottlenecks': transition_bottlenecks,
            'threshold_percentile': bottleneck_threshold_percentile,
            'summary': {
                'num_activity_bottlenecks': len(activity_bottlenecks),
                'num_transition_bottlenecks': len(transition_bottlenecks),
                'top_activity_bottleneck': list(activity_bottlenecks.keys())[0] if activity_bottlenecks else None,
                'top_transition_bottleneck': list(transition_bottlenecks.keys())[0] if transition_bottlenecks else None
            }
        }

        # --- 5. Variant performance analysis ---
        if include_variants:
            variants = df.groupby('case:concept:name')['concept:name'].apply(lambda x: ' -> '.join(x))
            variant_counts = variants.value_counts()
            top_variants = variant_counts.head(20)

            variant_performance = {}
            for variant, count in top_variants.items():
                variant_cases = variants[variants == variant].index
                v_durations = case_durations.loc[variant_cases, 'duration']

                variant_performance[variant] = {
                    'frequency': int(count),
                    'percentage': float(count / len(variants) * 100),
                    'duration': {
                        'mean': float(v_durations.mean()),
                        'median': float(v_durations.median()),
                        'std': float(v_durations.std()),
                        'min': float(v_durations.min()),
                        'max': float(v_durations.max())
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

        # --- 6. Temporal patterns analysis ---
        df['hour'] = df['time:timestamp'].dt.hour
        df['day_name'] = df['time:timestamp'].dt.day_name()

        hourly_activity = df.groupby('hour').size()
        daily_activity = df.groupby('day_name').size()

        case_temporal = df.groupby('case:concept:name')['time:timestamp'].min().to_frame()
        case_temporal['duration'] = case_durations['duration']
        case_temporal['hour'] = case_temporal['time:timestamp'].dt.hour
        case_temporal['day_name'] = case_temporal['time:timestamp'].dt.day_name()

        h_duration = case_temporal.groupby('hour')['duration'].mean()
        d_duration = case_temporal.groupby('day_name')['duration'].mean()

        results['temporal_patterns'] = {
            'hourly_patterns': {
                'peak_hours': hourly_activity.nlargest(3).index.tolist(),
                'hourly_distribution': hourly_activity.to_dict()
            },
            'daily_patterns': {
                'busiest_days': daily_activity.nlargest(3).index.tolist(),
                'daily_distribution': daily_activity.to_dict()
            },
            'performance_by_hour': h_duration.to_dict(),
            'performance_by_day': d_duration.to_dict()
        }

        # --- 7. Resource performance ---
        resource_cols = ['org:resource', 'resource', 'user', 'operator']
        resource_col = next((c for c in resource_cols if c in df.columns), None)

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

        # --- 8. Summary statistics ---
        avg_case_duration = case_durations['duration'].mean()
        duration_variability = case_durations['duration'].std() / avg_case_duration if avg_case_duration > 0 else 1
        bottleneck_severity = len(activity_bottlenecks) / len(activity_performance) if len(activity_performance) > 0 else 0

        health_score = max(0, min(100, 100 * (1 - duration_variability * 0.3) * (1 - bottleneck_severity * 0.5)))
        
        results['summary_statistics'] = {
            'process_health_score': round(health_score, 2),
            'efficiency_metrics': {
                'average_case_duration': round(avg_case_duration, 2),
                'duration_variability_pct': round(duration_variability * 100, 2),
                'bottleneck_ratio_pct': round(bottleneck_severity * 100, 2)
            }
        }

    except Exception as e:
        import traceback
        results['errors'].append(f'Unexpected error: {str(e)}')
        results['errors'].append(f'Traceback: {traceback.format_exc()}')

    return results

def _generate_performance_recommendations(results: Dict[str, Any]) -> List[str]:
    """
    Analyzes processed results to provide actionable business intelligence.
    """
    recommendations = []

    try:
        # Check for high duration variability
        case_stats = results.get('case_performance', {}).get('duration_stats', {})
        if case_stats:
            mean = case_stats.get('mean', 1)
            std = case_stats.get('std', 0)
            cv = std / mean if mean > 0 else 0
            
            if cv > 0.5:
                recommendations.append(
                    f'High process variability detected (Coefficient of Variation: {round(cv * 100, 2)}%). '
                    'Consider standardizing process paths or investigating outlier cases.'
                )

        # Check for bottlenecks
        bottleneck_summary = results.get('bottlenecks', {}).get('summary', {})
        num_bottlenecks = bottleneck_summary.get('num_activity_bottlenecks', 0)
        
        if num_bottlenecks > 0:
            top_bottleneck = bottleneck_summary.get('top_activity_bottleneck')
            if top_bottleneck:
                recommendations.append(
                    f"Critical bottleneck identified at '{top_bottleneck}'. "
                    'Prioritize optimization efforts on this activity.'
                )

        # Check variant complexity
        variant_info = results.get('variant_performance', {})
        if variant_info:
            total_variants = variant_info.get('total_variants', 0)
            top_5_coverage = variant_info.get('variant_coverage', {}).get('top_5_coverage', 0)

            if total_variants > 50 and top_5_coverage < 50:
                recommendations.append(
                    f'High process complexity: {total_variants} variants with low concentration. '
                    'Consider process standardization to reduce complexity.'
                )

        # Check temporal patterns
        temporal = results.get('temporal_patterns', {})
        if temporal:
            peak_hours = temporal.get('hourly_patterns', {}).get('peak_hours', [])
            if peak_hours:
                recommendations.append(
                    f'Peak activity hours identified: {peak_hours}. '
                    'Consider resource scaling during these periods.'
                )

        # Resource recommendations
        resource_perf = results.get('resource_performance', {})
        if resource_perf and 'resource_metrics' in resource_perf:
            recommendations.append(
                'Resource performance data available. '
                'Review individual resource metrics for training or workload balancing opportunities.'
            )

        # Overall health score
        health_score = results.get('summary_statistics', {}).get('process_health_score', 0)
        if health_score < 50:
            recommendations.append(
                f'Low process health score: {health_score:.1f}/100. '
                'Comprehensive process redesign may be beneficial.'
            )
        elif health_score > 80:
            recommendations.append(
                f'Good process health score: {health_score:.1f}/100. '
                'Focus on maintaining performance and continuous monitoring.'
            )

        if not recommendations:
            recommendations.append('Process within normal parameters. Continue monitoring for changes.')

    except Exception as e:
        recommendations.append(f'Unable to generate specific recommendations due to analysis errors: {e}')

    return recommendations

def analyze_repeat_purchases(df, output_folder="output", user_col="user_id", activity_col="concept:name", purchase_values=None):
    """
    Analyzes repeat purchases by grouping data by User ID.
    
    Args:
        df: The Pandas DataFrame containing the event log.
        output_folder: Where to save the chart.
        user_col: The column name representing the unique User (NOT the Session/Case ID).
        activity_col: The column containing activity names.
        purchase_values: List of strings that represent a purchase (e.g. ['purchase', 'order_confirmation']).
                         If None, defaults to ['purchase'].
    
    Returns:
        Dictionary containing KPIs (Repeat Rate, etc.) and path to the chart.
    """
    print("--- Business Logic: Analyzing Repeat Purchases ---")
    
    if purchase_values is None:
        purchase_values = ['purchase']
        
    # 1. Validation
    # Check case-insensitive matching for columns if exact match fails
    available_cols = [c.lower() for c in df.columns]
    
    real_user_col = None
    if user_col in df.columns: 
        real_user_col = user_col
    elif user_col.lower() in available_cols:
        real_user_col = df.columns[available_cols.index(user_col.lower())]
    
    if not real_user_col:
        print(f"   -> Warning: User column '{user_col}' not found. Cannot track repeat buyers.")
        print(f"      Available columns: {list(df.columns)}")
        return None

    # 2. Filter only Purchase events
    # Normalize to string and lowercase to be robust
    purchase_df = df[df[activity_col].astype(str).str.lower().isin([p.lower() for p in purchase_values])]
    
    if purchase_df.empty:
        print(f"   -> Warning: No activities found matching {purchase_values}.")
        return None

    # 3. Group by User and Count
    user_purchase_counts = purchase_df.groupby(real_user_col).size().reset_index(name='purchase_count')
    
    total_buyers = len(user_purchase_counts)
    one_time_buyers = len(user_purchase_counts[user_purchase_counts['purchase_count'] == 1])
    repeat_buyers = len(user_purchase_counts[user_purchase_counts['purchase_count'] > 1])
    
    repeat_rate = (repeat_buyers / total_buyers) * 100 if total_buyers > 0 else 0
    
    print(f"   -> Found {total_buyers} unique buyers.")
    print(f"   -> Repeat Buyers: {repeat_buyers} ({repeat_rate:.2f}%)")

    # 4. Generate Visualization (Distribution)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    chart_path = os.path.join(output_folder, "repeat_purchases_dist.png")
    
    plt.figure(figsize=(10, 6))
    
    # Cap the visual at "5+" to keep the chart readable
    viz_data = user_purchase_counts.copy()
    viz_data['purchase_bucket'] = viz_data['purchase_count'].apply(lambda x: str(x) if x < 5 else "5+")
    
    # Order: 1, 2, 3, 4, 5+
    order = ['1', '2', '3', '4', '5+']
    
    ax = sns.countplot(data=viz_data, x='purchase_bucket', order=order, palette="viridis", hue='purchase_bucket', legend=False)
    plt.title(f"Distribution of Purchases per User (Repeat Rate: {repeat_rate:.1f}%)")
    plt.xlabel("Number of Purchases")
    plt.ylabel("Number of Unique Users")
    
    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline')
    
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.show()
    plt.close()
    
    print(f"   -> Chart saved: {chart_path}")

    return {
        "total_buyers": total_buyers,
        "one_time_buyers": one_time_buyers,
        "repeat_buyers": repeat_buyers,
        "repeat_rate_percent": repeat_rate,
        "chart_path": chart_path
    }
