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

# --- 2. BASE CONFIGURATION (Balanced) ---
# This is the default setup: Good balance between speed and detail.
CONFIG = {
    "app_name": "Process Miner Pro",

    "active_case_id": "session_unique_id",
    
    # Data Loading Limits
    "data_loading": {
        "max_file_size_mb": 500,
        "chunk_threshold_mb": 50,
        "chunk_size": 50000,
    },

    # Parameters for capping the event log after sampling to increase computing speed
    "speed_params": {
        "max_align": 15, # Choose a number lower than the sample below to increase performance, or set it as equal
        "cores": 1, # set to 0 for max core utilization
        "max_prec_traces": 15 # Choose a number lower than the sample to increase performance, or set it as equal
    },

    # 1. Discovery (The Visual Model)
    "discovery_params": {
        "algorithm": "inductive_miner", # inductive_minder, heuristics_miner, alpha_miner
        "noise_threshold": 0.6,       # 0.4 removes 40% of rarest edges (cleaner map)
        "dependency_threshold": 0.9   # Used only for Heuristics Miner
    },

    # 2. Conformance (The Deviations)
    "conformance_params": {
        # 'state_equation_a_star' gives trace deviations.
        # 'token_replay' is faster but gives fewer details.
        # 'dijkstra' is very slow and with a lot of potentially useless noise.
        "algorithm": "token_replay", # dijkstra, token_replay
        
        # Calculate alignments per variant instead of per case.
        # True = Calculate once per unique path (Factor 10-100x faster for repetitive logs)
        # False = Calculate every case individually (Slow, use only if data-content differs per case)
        "optimize_variants": True,

        # Calculate Precision? (Slowest part of the pipeline)
        "calculate_precision": True,
        "calculate_fitness": True # Preliminary fitness; irrelevant and expensive, EXCEPT for TBR (required).
    },

    # 3. Sampling (Speed Optimization)
    "sampling_config": {
        "enabled": True,
        "total_sample_size": 15,
        "max_priority_ratio": 0.5, # Max 50% priority cases (e.g. purchases)

        # Auto-detect priority column (Purchase > Transaction > Error)
        # Set to explicit string (e.g., "purchase") to force it.
        "strata_col": 'has_purchase'
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
            "activities": ["experience_impression"],
            "mode": "not_contains" # Filters out CASES
        }
        #{
        #    "type": "activity",
        #    "activities": ["add_to_cart"],
        #    "mode": "contains" # includes EVENTS
        #}
    ],

    # Visualization params
    "visualisation_params": {
        "bottleneck_top_k": 50,
        "max_bottleneck_edges": 2
    }
}
