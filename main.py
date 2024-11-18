import argparse

from mdp import MDP
from maps import MAPS


def main():
    parser = argparse.ArgumentParser(description="Run an MDP simulation.")
    parser.add_argument(
        "--map",
        type=str,
        choices=MAPS.keys(),
        default="large",
        help="Choose a predefined map, e.g., 'large'.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps for iterative policy evaluation (default: 100).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run learned policy (default: 1).",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="Folder to save video recordings. Set to None to disable recording.",
    )

    args = parser.parse_args()
    m = MDP(MAPS[args.map])
    m.run_iterative_policy_evaluation(steps=args.steps)
    m.run(episodes=args.episodes, video_folder=args.video_folder)


if __name__ == "__main__":
    main()
