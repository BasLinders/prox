"""
Generates a mock GA4-style e-commerce event log so users can try PRoX's full
workflow - process discovery, conformance, funnel, business insights, and
segment comparison - without needing their own data.
"""
import random

import numpy as np
import pandas as pd

CATEGORIES = ["Electronics", "Apparel", "Home & Garden", "Sports", "Beauty"]
DEVICES = ["desktop", "mobile", "tablet"]
DEVICE_WEIGHTS = [0.45, 0.45, 0.10]
TRAFFIC_SOURCES = ["organic", "paid_search", "email", "social", "direct"]
TRAFFIC_WEIGHTS = [0.35, 0.25, 0.10, 0.15, 0.15]

# Funnel stages with per-stage continuation probability (drop-off baked in).
FUNNEL = [
    ("session_start", 1.00),
    ("view_item_list", 0.88),
    ("view_item", 0.75),
    ("add_to_cart", 0.45),
    ("begin_checkout", 0.55),
    ("purchase", 0.70),
]

# Extra noise events that can be sprinkled in without breaking the funnel order.
NOISE_EVENTS = ["scroll", "search", "view_promotion", "apply_coupon_attempt"]

PRICE_RANGES = {
    "Electronics": (49, 899),
    "Apparel": (15, 180),
    "Home & Garden": (12, 350),
    "Sports": (18, 260),
    "Beauty": (8, 120),
}


def _gen_user_pool(n_users):
    return [f"u{idx:05d}" for idx in range(1, n_users + 1)]


def _build_session(user_id, session_num, start_time, rng):
    rows = []
    session_id = f"s{session_num:06d}"
    device = rng.choices(DEVICES, weights=DEVICE_WEIGHTS, k=1)[0]
    traffic_source = rng.choices(TRAFFIC_SOURCES, weights=TRAFFIC_WEIGHTS, k=1)[0]
    category = rng.choice(CATEGORIES)
    lo, hi = PRICE_RANGES[category]
    price = round(rng.uniform(lo, hi), 2)

    t = start_time

    for stage, continue_prob in FUNNEL:
        if stage != "session_start" and rng.random() > continue_prob:
            break

        t += pd.Timedelta(seconds=rng.randint(8, 90))
        event_value = price if stage in ("add_to_cart", "begin_checkout", "purchase") else np.nan

        rows.append({
            "user_pseudo_id": user_id,
            "ga_session_id": session_id,
            "event_name": stage,
            "event_timestamp": t,
            "event_value": event_value,
            "category": category,
            "device": device,
            "traffic_source": traffic_source,
            "page_type": "product" if stage in ("view_item", "add_to_cart") else "listing",
            "purchase": stage == "purchase",
            "add_to_cart": stage == "add_to_cart",
        })

        # Occasionally inject a noise event between funnel steps (process variants).
        if stage in ("view_item_list", "view_item") and rng.random() < 0.25:
            t += pd.Timedelta(seconds=rng.randint(5, 40))
            rows.append({
                "user_pseudo_id": user_id,
                "ga_session_id": session_id,
                "event_name": rng.choice(NOISE_EVENTS),
                "event_timestamp": t,
                "event_value": np.nan,
                "category": category,
                "device": device,
                "traffic_source": traffic_source,
                "page_type": "listing",
                "purchase": False,
                "add_to_cart": False,
            })

    return rows


def generate_mock_event_log(n_sessions: int = 400, seed: int = 42, repeat_buyer_ratio: float = 0.30) -> pd.DataFrame:
    """
    Builds a synthetic e-commerce clickstream: session_start -> view_item_list
    -> view_item -> add_to_cart -> begin_checkout -> purchase, with realistic
    drop-off at each stage, occasional noise events (process variants), and
    columns for repeat-buyer, category, revenue, and segment analysis.

    Returns a DataFrame with raw (pre-mapping) column names, ready to be
    written to CSV and loaded via `load_and_validate_csv` exactly like a
    user-provided file.
    """
    rng = random.Random(seed)

    # Fewer users than sessions so some users return (repeat-purchase signal).
    n_users = max(1, int(n_sessions * (1 - repeat_buyer_ratio * 0.5)))
    users = _gen_user_pool(n_users)

    base_time = pd.Timestamp("2026-06-01 08:00:00")
    all_rows = []
    session_num = 0

    for user_id in users:
        # Most users get 1 session; a subset (repeat buyers/browsers) get 2-4.
        n_user_sessions = 1 if rng.random() > repeat_buyer_ratio else rng.randint(2, 4)
        day_offset = rng.randint(0, 59)
        for k in range(n_user_sessions):
            session_num += 1
            if session_num > n_sessions:
                break
            start_time = (
                base_time
                + pd.Timedelta(days=day_offset + k * rng.randint(1, 7))
                + pd.Timedelta(minutes=rng.randint(0, 12 * 60))
            )
            all_rows.extend(_build_session(user_id, session_num, start_time, rng))
        if session_num >= n_sessions:
            break

    df = pd.DataFrame(all_rows)
    df.sort_values(["user_pseudo_id", "ga_session_id", "event_timestamp"], inplace=True)
    return df.reset_index(drop=True)


def generate_mock_csv_bytes(n_sessions: int = 400, seed: int = 42) -> bytes:
    """Convenience wrapper: generates the mock log and encodes it as CSV bytes."""
    df = generate_mock_event_log(n_sessions=n_sessions, seed=seed)
    return df.to_csv(index=False).encode("utf-8")
