"""Sync the local `dataset/` directory to the HuggingFace dataset repo.

Mirrors `dataset/` 1:1 into `CodeConflict/cooperbench-dataset`. The local
README.md becomes the HF repo's dataset card.

Usage:
    python scripts/upload_dataset_to_hf.py
    python scripts/upload_dataset_to_hf.py --commit-message "Add task 9999"
    python scripts/upload_dataset_to_hf.py --create-pr

Auth:
    Requires HF_TOKEN env var (or a prior `huggingface-cli login`) with write
    access to the target repo.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "CodeConflict/cooperbench-dataset"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "dataset"
IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/.DS_Store",
    "**/*.pyc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--commit-message",
        default="Update dataset",
        help="Commit message for the HF revision.",
    )
    p.add_argument(
        "--create-pr",
        action="store_true",
        help="Open a PR on the dataset repo instead of committing to main.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not LOCAL_DIR.is_dir():
        print(f"error: {LOCAL_DIR} is not a directory", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN")
    print(f"local dir:     {LOCAL_DIR}")
    print(f"repo:          {REPO_ID} (dataset)")

    create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=token)

    result = HfApi(token=token).upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(LOCAL_DIR),
        commit_message=args.commit_message,
        create_pr=args.create_pr,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
