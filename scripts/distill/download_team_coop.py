"""Download the CooperBench/team-coop dataset from HuggingFace to data/team-coop/.

Usage:
    uv run python scripts/distill/download_team_coop.py
    uv run python scripts/distill/download_team_coop.py --run cmp-full-team
    uv run python scripts/distill/download_team_coop.py --dest data/my-dir

Auth:
    Public dataset — no HF_TOKEN required. Set HF_TOKEN env var for higher
    rate limits or if the repo is made private later.

Output layout (mirrors the HF repo structure):
    data/team-coop/
        cmp-full-team/
            coop/<repo>/<task_id>/<f_pair>/
                agent1_traj.json  agent2_traj.json
                conversation.json eval.json metadata.json result.json
                agent1.patch      agent2.patch
        cmp-full-team-noproto/  ...
        coop/                   ...  (Qwen baseline runs)
        summary.json
        config.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "CooperBench/team-coop"
DEFAULT_DEST = Path(__file__).resolve().parent.parent.parent / "data" / "team-coop"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Local directory to download into (default: {DEFAULT_DEST})",
    )
    p.add_argument(
        "--run",
        default=None,
        metavar="RUN_NAME",
        help="Download only a single top-level run directory, e.g. 'cmp-full-team'.",
    )
    p.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*.patch"],
        metavar="PATTERN",
        help="Glob patterns to skip (default: ['*.patch'] — saves ~half the disk space). Pass '' to download everything.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")

    allow_patterns = [f"{args.run}/**"] if args.run else None
    ignore_patterns = [p for p in (args.ignore_patterns or []) if p] or None

    print(f"repo:            {REPO_ID}")
    print(f"dest:            {dest}")
    print(f"allow_patterns:  {allow_patterns or '(all)'}")
    print(f"ignore_patterns: {ignore_patterns or '(none)'}")
    print()

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        token=token,
    )
    print(f"\nDownloaded to: {local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
