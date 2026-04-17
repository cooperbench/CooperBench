"""Prepare the CooperBench dataset locally from the HuggingFace mirror.

Downloads every file of `CodeConflict/cooperbench-dataset` into `./dataset`,
reproducing the layout expected by `cooperbench run` / `cooperbench eval`.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "CodeConflict/cooperbench-dataset"


def prepare_dataset(force: bool = False) -> Path:
    """Download the CooperBench dataset into `./dataset`."""
    dest = Path.cwd() / "dataset"

    if dest.exists() and force:
        print(f"removing existing {dest}")
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    print(f"repo:          {REPO_ID}")
    print(f"destination:   {dest}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        token=os.environ.get("HF_TOKEN"),
    )

    # snapshot_download creates `.cache/huggingface/` inside local_dir for bookkeeping; exclude it from the summary.
    files = [p for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts]
    total = sum(p.stat().st_size for p in files)
    print(f"done: {len(files)} files, {total / 1e6:.2f} MB at {dest}")
    return dest


def _prepare_command(args) -> None:
    try:
        prepare_dataset(force=args.force)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
