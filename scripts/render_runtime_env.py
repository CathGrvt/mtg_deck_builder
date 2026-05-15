from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtg_shared.runtime_env import build_gcloud_env_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render runtime env vars as a gcloud --set-env-vars payload.")
    parser.add_argument(
        "--target",
        default="backend",
        choices=["backend", "agent-engine", "all"],
        help="Runtime target to render env vars for.",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Only include env vars explicitly set in the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        build_gcloud_env_arg(
            target=args.target,
            include_defaults=not bool(args.no_defaults),
        )
    )


if __name__ == "__main__":
    main()
