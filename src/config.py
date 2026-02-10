from typing import Dict, Any

# --- CONFIGURATION ---
COLUMN_MAPPINGS = {
    # Case Identifier
    'case:concept:name': frozenset([
        'session id', 'session_id', 'sessie id', 'ga_session_id',
        'case_id', 'case', 'trace_id', 'transaction_id', 'id', 
        'session_unique_id'
    ]),

    # Activity Name
    'concept:name': frozenset([
        'event', 'event_name', 'event_type', 'gebeurtenis',
        'naam_gebeurtenis', 'activity', 'activity_name', 'product', 'action'
    ]),

    # Timestamp
    'time:timestamp': frozenset([
        'tijd', 'timestamp', 'tijdstempel', 'event_timestamp',
        'date', 'time', 'created_at', 'started_at', 'datetime'
    ]),

    # User / Resource
    'user_id': frozenset([
        'user id', 'user_id', 'gebruikers id', 'user_pseudo_id',
        'client_ref', 'customer_id', 'resource', 'org:resource', 
        'client_id'
    ]),

    # E-Commerce Specifics
    'price': frozenset(['prijs', 'price', 'order total', 'value', 'revenue', 'amount']),
    'category': frozenset(['categorie', 'category', 'product_category']),
    'purchase': frozenset(['purchase', 'transaction', 'conversion', 'order_placed']),
    'add_to_cart': frozenset(['add_to_cart', 'add_to_basket']),
    'page_type': frozenset(['page_type', 'pagetype', 'screen_class'])
}

# === FUNCTION FOR COLUMN MAPPINGS IN STREAMLIT APP ===
def get_column_mappings():
    """
    Returns a comprehensive dictionary of aliases for standardizing event log columns.

    This function provides the reference sets used to automatically map various 
    source column names (including Dutch translations and e-commerce specific 
    identifiers) to a standardized internal schema. This is a critical step 
    for cross-compatibility between different SQL exports and the PM4Py engine.

    Returns
    -------
    mappings : dict of frozenset
        A dictionary where each key represents a standard process mining field 
        and each value is a `frozenset` of common aliases found in raw CSV data.

    Notes
    -----
    The mappings are categorized into four core process pillars and 
    supplementary e-commerce attributes:

    **Core Pillars:**
    * `case:concept:name`: Identifies the unique process instance (Session, Trace, Case).
    * `concept:name`: The label for the activity or event occurring.
    * `time:timestamp`: The temporal marker for the event.
    * `user_id`: The resource or actor responsible for the event.

    **E-Commerce Attributes:**
    * `price`, `category`, `purchase`, `add_to_cart`, and `page_type`.
    
    Example
    -------
    >>> mappings = get_column_mappings()
    >>> 'ga_session_id' in mappings['case:concept:name']
    True
    """
    return {
        # Case Identifier
        'case:concept:name': frozenset([
            'session id', 'session_id', 'sessie id', 'ga_session_id',
            'case_id', 'case', 'trace_id', 'transaction_id', 'id', 
            'session_unique_id'
        ]),

        # Activity Name
        'concept:name': frozenset([
            'event', 'event_name', 'event_type', 'gebeurtenis',
            'naam_gebeurtenis', 'activity', 'activity_name', 'product', 'action'
        ]),

        # Timestamp
        'time:timestamp': frozenset([
            'tijd', 'timestamp', 'tijdstempel', 'event_timestamp',
            'date', 'time', 'created_at', 'started_at', 'datetime'
        ]),

        # User / Resource
        'user_id': frozenset([
            'user id', 'user_id', 'gebruikers id', 'user_pseudo_id',
            'client_ref', 'customer_id', 'resource', 'org:resource', 
            'client_id'
        ]),

        # E-Commerce Specifics
        'price': frozenset([
            'prijs', 'price', 'order total', 'value', 'revenue', 'amount'
        ]),
        'category': frozenset([
            'categorie', 'category', 'product_category'
        ]),
        'purchase': frozenset([
            'purchase', 'transaction', 'conversion', 'order_placed'
        ]),
        'add_to_cart': frozenset([
            'add_to_cart', 'add_to_basket'
        ]),
        'page_type': frozenset([
            'page_type', 'pagetype', 'screen_class'
        ])
    }

# --- 2. BASE CONFIGURATION (Balanced) ---
CONFIG = {
    "app_name": "Process Miner",

    "active_case_id": "session_unique_id",
    
    # Data Loading Limits
    "data_loading": {
        "max_file_size_mb": 500,
        "chunk_threshold_mb": 50,
        "chunk_size": 50000,
    },

    # Parameters for capping the event log after sampling to increase computing speed
    "speed_params": {
        "max_align": 250, # Choose a number lower than the sample below to increase performance, or set it as equal
        "cores": 1, # set to 0 for max core utilization
        "max_prec_traces": 250 # Choose a number lower than the sample to increase performance, or set it as equal
    },

    # 1. Discovery (The Visual Model)
    "discovery_params": {
        "algorithm": "inductive_miner", # inductive_minder, heuristics_miner, alpha_miner
        "noise_threshold": 0.2,       # 0.4 removes 40% of rarest edges (cleaner map)
        "dependency_threshold": 0.9   # Used only for Heuristics Miner
    },

    # 2. Conformance (The Deviations)
    "conformance_params": {
        # 'state_equation_a_star' gives trace deviations.
        # 'token_replay' is faster but gives fewer details.
        # 'dijkstra' is very slow and with a lot of potentially useless noise.
        "algorithm": "token_replay",
        
        # Calculate alignments per variant instead of per case.
        # True = Calculate once per unique path (Factor 10-100x faster for repetitive logs)
        # False = Calculate every case individually (Slow, use only if data-content differs per case)
        "optimize_variants": True,

        # Calculate Precision? (Slowest part of the pipeline)
        "calculate_precision": True,
        "calculate_fitness": False # Preliminary fitness; irrelevant and expensive, EXCEPT for TBR (required).
    },

    # 3. Sampling (Speed Optimization)
    "sampling_config": {
        "enabled": True,
        "total_sample_size": 250,
        "max_priority_ratio": 0.5, # Max 50% priority cases (e.g. purchases)

        # Auto-detect priority column (Purchase > Transaction > Error)
        # Set to explicit string (e.g., "purchase") to force it.
        "strata_col": 'purchase'
    },

    # 4. Performance (Bottlenecks)
    "performance_params": {
        "time_unit": "minutes",        # Choose from days, hours, minutes, seconds
        "bottleneck_threshold_percentile": 75, # Top 25% slowest are bottlenecks
        "include_variants": True
    },

    # 5. Filters (Preprocessing)
    # List of dicts. Applied in order.
    "filter_steps": [
        {
            "type": "activity",
            "activities": [
                "experience_impression", 
                "view_cookie_bar", 
                "javascript_error", 
                "scroll", 
                "view_item_list_empty",
                "user_engagement", 
                "page_timestamp",
                "session_start",
                "first_visit"
            ],
            "mode": "remove_events" # Filters out EVENTS
        },
        {
            "type": "crop",
            "activity": ["purchase", "event_value", "has_purchase"],
            "top_n": 10
        }
    ],

    # Visualization params
    "visualisation_params": {
        "bottleneck_top_k": 50,
        "max_bottleneck_edges": 2
    },

    # Business insights
    "business_params": {
        "user_col": "user_id",       # Or 'email', 'user'
        "revenue_col": "event_value",   # Or 'price', 'purchase_revenue'
        "purchase_values": ["purchase", "has_purchase"]
    }
}

# === CONFIG FUNCTION FOR STREAMLIT ===
def create_analysis_config(
    discovery_algo: str = "inductive_miner",
    active_case_id: str = "session_unique_id",
    cores: int = 1,
    noise_threshold: float = 0.6,
    dependency_threshold: float = 0.9,
    conformance_algo: str = "token_replay",
    optimize_variants: bool = True,
    calculate_precision: bool = True,
    calculate_fitness: bool = False,
    sample_size: int = 15,
    time_unit: str = "minutes",
    strata_col: str = "purchase",
    enable_sampling: bool = True
):
    """
    Generates a comprehensive configuration profile for the process analysis pipeline.

    This function synchronizes UI-level inputs with the multi-stage backend logic. 
    It defines the behavior for data ingestion, discovery, conformance checking, 
    and performance profiling, ensuring that computational limits (like CPU cores 
    and sample sizes) are respected to maintain application stability.

    Parameters
    ----------
    discovery_algo : {'inductive_miner', 'heuristics_miner', 'alpha_miner'}, optional
        The algorithm used to discover the process model from the event log 
        (default is "inductive_miner").
    active_case_id : str, optional
        The column name representing the unique process instance identifier 
        (e.g., 'session_id' or 'trace_id') (default is "session_unique_id").
    cores : int, optional
        The number of CPU cores to utilize for parallel processing. Set to 1 
        for serial execution or 0 for maximum core utilization (default is 1).
    noise_threshold : float, optional
        Filtering sensitivity for the Inductive Miner (0.0 to 1.0). 
        A higher value (e.g., 0.6) aggressively removes infrequent paths 
        to produce a clearer, more abstract process map (default is 0.6).
    dependency_threshold : float, optional
        Threshold used exclusively by the Heuristics Miner to determine 
        if an edge between activities represents a strong causal 
        dependency (default is 0.9).
    conformance_algo : {'token_replay', 'dijkstra'}, optional
        The mathematical approach for alignment. 'token_replay' is significantly 
        faster but less precise, while 'dijkstra' provides optimal alignments 
        at a higher computational cost (default is "token_replay").
    optimize_variants : bool, optional
        If True, executes conformance checking once per unique trace variant 
        rather than per individual case. This typically results in a 
        10x-100x performance increase (default is True).
    calculate_precision : bool, optional
        Whether to calculate the precision score, which measures the degree 
        to which the model prohibits behavior not seen in the log. This is 
        often the most computationally intensive step (default is True).
    calculate_fitness : bool, optional
        Whether to calculate the fitness score, measuring how much of the 
        log behavior can be replayed by the model. Required for 
        Token-Based Replay (default is False).
    sample_size : int, optional
        The total number of cases to be retained in the analysis sample. 
        Higher values improve accuracy but increase processing time (default is 15).
    time_unit : {'days', 'hours', 'minutes', 'seconds'}, optional
        The unit of measurement for all duration-based performance metrics 
        and bottleneck analyses (default is "minutes").
    strata_col : str, optional
        The column used for stratified sampling to ensure rare but critical 
        events (like conversions) are represented in the sample (default is "purchase").
    enable_sampling : bool, optional
        Toggles the sampling engine. When True, the pipeline operates on 
        the specified `sample_size` to ensure fast feedback loops (default is True).

    Returns
    -------
    dict
        A nested configuration dictionary ready for the `run_full_analysis` function.

    Notes
    -----
    The relationship between fitness and precision is a fundamental trade-off 
    in process discovery. Adjusting the `noise_threshold` directly impacts 
    these metrics by simplifying the discovered model.

    See Also
    --------
    run_full_analysis : The main orchestrator that consumes this configuration.
    sample_log_stratified : The engine controlled by the sampling parameters.
    """
    return {
        "app_name": "Process Miner Pro",
        "active_case_id": active_case_id,
        
        "data_loading": {
            "max_file_size_mb": 500,
            "chunk_threshold_mb": 50,
            "chunk_size": 50000,
        },

        "speed_params": {
            "max_align": sample_size,
            "cores": cores,
            "max_prec_traces": sample_size
        },

        "discovery_params": {
            "algorithm": discovery_algo,
            "noise_threshold": noise_threshold,
            "dependency_threshold": dependency_threshold 
        },

        "conformance_params": {
            "algorithm": conformance_algo,
            "optimize_variants": optimize_variants,
            "calculate_precision": calculate_precision,
            "calculate_fitness": calculate_fitness 
        },

        "sampling_config": {
            "enabled": enable_sampling,
            "total_sample_size": sample_size,
            "max_priority_ratio": 0.5,
            "strata_col": strata_col
        },

        "performance_params": {
            "time_unit": time_unit,
            "bottleneck_threshold_percentile": 75,
            "include_variants": True
        },

        "filter_steps": [
            {
                "type": "activity",
                "activities": ["experience_impression"],
                "mode": "not_contains"
            }
        ],

        "visualisation_params": {
            "bottleneck_top_k": 50,
            "max_bottleneck_edges": 2
        }
    }
