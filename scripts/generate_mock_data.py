"""
CLI wrapper around `prox.mock_data.generate_mock_event_log` for generating a
mock GA4-style e-commerce event log from the command line. The same
generator also powers the "Generate Mock Data" button in the Streamlit UI.

Usage:
    python scripts/generate_mock_data.py [output_path] [--sessions N] [--seed N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prox.mock_data import generate_mock_event_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="mock_event_log.csv")
    parser.add_argument("--sessions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_mock_event_log(n_sessions=args.sessions, seed=args.seed)
    out_path = Path(args.output)
    df.to_csv(out_path, index=False)

    n_purchases = (df["event_name"] == "purchase").sum()
    print(f"Wrote {len(df)} events, {df['ga_session_id'].nunique()} sessions, "
          f"{df['user_pseudo_id'].nunique()} users, {n_purchases} purchases -> {out_path}")


if __name__ == "__main__":
    main()
