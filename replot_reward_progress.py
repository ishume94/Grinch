import argparse
import pickle
from pathlib import Path

from gflow_qoc.result_analysis import plot_rewards


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the reward-progress plot from a saved rewards file."
    )
    parser.add_argument(
        "rewards_file",
        type=Path,
        help="Pickle file written by driver.py (for example, 32_GHZ_rewards.p).",
    )
    args = parser.parse_args()

    if not args.rewards_file.is_file():
        parser.error(f"Rewards file does not exist: {args.rewards_file}")

    with args.rewards_file.open("rb") as handle:
        rewards = pickle.load(handle)

    if rewards is None or len(rewards) == 0:
        parser.error(f"Rewards file is empty: {args.rewards_file}")

    plot_rewards(rewards)
    print(f"Loaded {len(rewards)} rewards from {args.rewards_file}")
    print("Saved rewards_progress.svg and rewards_progress.png")


if __name__ == "__main__":
    main()
