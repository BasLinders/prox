from typing import Dict, Any

COLUMN_MAPPINGS = {
    'case:concept:name': frozenset([
        'session id', 'session_id', 'sessie id', 'ga_session_id',
        'case_id', 'case', 'trace_id', 'transaction_id', 'id',
        'session_unique_id'
    ]),
    'concept:name': frozenset([
        'event', 'event_name', 'event_type', 'gebeurtenis',
        'naam_gebeurtenis', 'activity', 'activity_name', 'product', 'action'
    ]),
    'time:timestamp': frozenset([
        'tijd', 'timestamp', 'tijdstempel', 'event_timestamp',
        'date', 'time', 'created_at', 'started_at', 'datetime'
    ]),
    'user_id': frozenset([
        'user id', 'user_id', 'gebruikers id', 'user_pseudo_id',
        'client_ref', 'customer_id', 'client_id'
    ]),
    'price': frozenset(['prijs', 'price', 'order total', 'value', 'revenue', 'amount']),
    'category': frozenset(['categorie', 'category', 'product_category']),
    'purchase': frozenset(['purchase', 'transaction', 'conversion', 'order_placed']),
    'add_to_cart': frozenset(['add_to_cart', 'add_to_basket']),
    'page_type': frozenset(['page_type', 'pagetype', 'screen_class'])
}


def get_column_mappings() -> dict:
    return COLUMN_MAPPINGS


CONFIG = {
    "app_name": "PRoX",
    "active_case_id": "session_unique_id",

    "data_loading": {
        "max_file_size_mb": 500,
        "chunk_threshold_mb": 50,
        "chunk_size": 50000,
    },

    "speed_params": {
        "max_align": 250,
        "cores": 1,
        "max_prec_traces": 250
    },

    "discovery_params": {
        "algorithm": "inductive_miner",
        "noise_threshold": 0.2,
        "dependency_threshold": 0.9,
        "activity_threshold": 0
    },

    "conformance_params": {
        "algorithm": "token_replay",
        "optimize_variants": True,
        "calculate_precision": True,
        "calculate_fitness": False
    },

    "sampling_config": {
        "enabled": True,
        "total_sample_size": 250,
        "max_priority_ratio": 0.5,
        "strata_col": "purchase"
    },

    "performance_params": {
        "time_unit": "minutes",
        "bottleneck_threshold_percentile": 75,
        "include_variants": True
    },

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
            "mode": "remove_events"
        },
        {
            "type": "crop",
            "activity": ["purchase", "event_value", "has_purchase"],
            "top_n": 10
        }
    ],

    "visualisation_params": {
        "bottleneck_top_k": 50,
        "max_bottleneck_edges": 2
    },

    "business_params": {
        "user_col": "user_id",
        "revenue_col": "event_value",
        "purchase_values": ["purchase", "has_purchase"],
        "research_keywords": ["search", "filter", "view_item", "product"],
        "research_min_events": 3
    }
}


def create_analysis_config(
    discovery_algo: str = "inductive_miner",
    active_case_id: str = "session_unique_id",
    cores: int = 1,
    noise_threshold: float = 0.2,
    dependency_threshold: float = 0.9,
    activity_threshold: int = 0,
    conformance_algo: str = "token_replay",
    optimize_variants: bool = True,
    calculate_precision: bool = True,
    calculate_fitness: bool = False,
    sample_size: int = 250,
    time_unit: str = "minutes",
    strata_col: str = "purchase",
    max_priority_ratio: float = 0.5,
    enable_sampling: bool = True,
    bottleneck_threshold_percentile: float = 75,
    filter_steps: list = None,
    bottleneck_top_k: int = 50,
    max_bottleneck_edges: int = 2,
    business_params: dict = None,
    max_file_size_mb: int = 500,
    chunk_threshold_mb: int = 50,
    chunk_size: int = 50000,
) -> Dict[str, Any]:
    return {
        "app_name": "PRoX",
        "active_case_id": active_case_id,

        "data_loading": {
            "max_file_size_mb": max_file_size_mb,
            "chunk_threshold_mb": chunk_threshold_mb,
            "chunk_size": chunk_size,
        },

        "speed_params": {
            "max_align": sample_size,
            "cores": cores,
            "max_prec_traces": sample_size
        },

        "discovery_params": {
            "algorithm": discovery_algo,
            "noise_threshold": noise_threshold,
            "dependency_threshold": dependency_threshold,
            "activity_threshold": activity_threshold
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
            "max_priority_ratio": max_priority_ratio,
            "strata_col": strata_col
        },

        "performance_params": {
            "time_unit": time_unit,
            "bottleneck_threshold_percentile": bottleneck_threshold_percentile,
            "include_variants": True
        },

        "filter_steps": filter_steps if filter_steps is not None else [
            {
                "type": "activity",
                "activities": ["experience_impression"],
                "mode": "remove_events"
            }
        ],

        "visualisation_params": {
            "bottleneck_top_k": bottleneck_top_k,
            "max_bottleneck_edges": max_bottleneck_edges
        },

        "business_params": business_params if business_params is not None else {
            "user_col": "user_id",
            "revenue_col": "event_value",
            "purchase_values": ["purchase", "has_purchase"],
            "research_keywords": ["search", "filter", "view_item", "product"],
            "research_min_events": 3
        }
    }
