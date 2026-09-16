"""Download the final LoRA adapter from the Modal volume and push to HuggingFace.

Uploads only the adapter files (not checkpoint subdirs or optimizer states):
  adapter_model.safetensors
  adapter_config.json
  tokenizer.json / tokenizer_config.json / chat_template.jinja
  training_args.bin

Target repo: CooperBench/qwen3.5-9b-tool-use-sft

Usage:
    uv run python scripts/distill/upload_adapter.py
    uv run python scripts/distill/upload_adapter.py --skip-download   # if already local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VOLUME_NAME = "qwen35-tool-use-sft-v1-volume"
REMOTE_ROOT = "/qwen35-tool-use-sft-v1"
LOCAL_DIR   = Path("data/checkpoints/qwen35-tool-use-sft-v1")
HF_REPO_ID  = "CooperBench/qwen3.5-9b-tool-use-sft"

# Files to upload — root-level only, skip checkpoint-* subdirs and optimizer states
UPLOAD_PATTERNS = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "training_args.bin",
    "README.md",
}

MODEL_CARD = """\
---
base_model: Qwen/Qwen3.5-9B
library_name: peft
tags:
  - lora
  - qwen3
  - tool-use
  - multi-agent
  - cooperbench
license: apache-2.0
---

# Qwen3.5-9B Tool-Use SFT (CooperBench)

LoRA adapter fine-tuned on top of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
to improve coordination tool usage in multi-agent coding settings.

## Training data

1,230 successful multi-agent trajectories from the
[CooperBench/team-coop](https://huggingface.co/datasets/CooperBench/team-coop) dataset:

- **1,153** converted teacher trajectories (gpt-5.5-hao via Codex, converted to
  mini_swe_agent_v2 format)
- **77** native Qwen3.5-9B successful trajectories

Tools taught: `coop-task-create`, `coop-task-claim`, `coop-task-update`,
`coop-task-list`, `coop-task-request`, `coop-task-respond`, `coop-task-pending`.

## LoRA config

| Parameter | Value |
|-----------|-------|
| r | 64 |
| alpha | 128 |
| dropout | 0.05 |
| target modules | q/k/v/o/gate/up/down proj |
| epochs | 3 |
| lr | 2e-4 |
| seq len | 2048 |

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B")
model = PeftModel.from_pretrained(model, "CooperBench/qwen3.5-9b-tool-use-sft")
tokenizer = AutoTokenizer.from_pretrained("CooperBench/qwen3.5-9b-tool-use-sft")
```
"""


def download_from_volume(local_dir: Path) -> None:
    try:
        import modal
    except ImportError:
        print("modal not installed — skipping download", file=sys.stderr)
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(VOLUME_NAME)
    downloaded = 0
    for entry in vol.iterdir(REMOTE_ROOT, recursive=False):
        fname = Path(entry.path).name
        if fname not in UPLOAD_PATTERNS:
            continue
        dest = local_dir / fname
        data = b"".join(vol.read_file(entry.path))
        dest.write_bytes(data)
        print(f"  ← {fname} ({len(data) / 1e6:.1f} MB)")
        downloaded += 1
    print(f"Downloaded {downloaded} files to {local_dir}")


def upload_to_hf(local_dir: Path) -> None:
    import os
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Creating/verifying repo {HF_REPO_ID} …")
    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, token=token)

    # Write model card
    card_path = local_dir / "README.md"
    if not card_path.exists():
        card_path.write_text(MODEL_CARD)

    print(f"Uploading to {HF_REPO_ID} …")
    for fpath in sorted(local_dir.iterdir()):
        if fpath.name not in UPLOAD_PATTERNS:
            continue
        print(f"  → {fpath.name} ({fpath.stat().st_size / 1e6:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fpath.name,
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=token,
        )

    print(f"\nDone: https://huggingface.co/{HF_REPO_ID}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-download", action="store_true",
                   help="Skip Modal download, use files already in data/checkpoints/")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.skip_download:
        print(f"Downloading adapter from Modal volume {VOLUME_NAME} …")
        download_from_volume(LOCAL_DIR)

    # Check we have the required files
    missing = [f for f in ("adapter_model.safetensors", "adapter_config.json")
               if not (LOCAL_DIR / f).exists()]
    if missing:
        print(f"error: missing required files: {missing}", file=sys.stderr)
        print(f"Check {LOCAL_DIR} or run without --skip-download", file=sys.stderr)
        return 1

    print(f"\nUploading adapter to HuggingFace …")
    upload_to_hf(LOCAL_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
