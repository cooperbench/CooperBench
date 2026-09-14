"""vLLM server on Modal A100 serving the fine-tuned Qwen3.5-9B LoRA adapter.

Exposes an OpenAI-compatible /v1/chat/completions endpoint so CooperBench
can evaluate the fine-tuned model.

Usage:
    # Start the server (keeps running until Ctrl+C)
    uv run modal serve scripts/distill/serve_vllm_modal.py

    # Then run CooperBench pointing at the printed URL, e.g.:
    uv run cooperbench run \\
      --base-url https://<your-modal-url> \\
      --auth-token dummy \\
      -m qwen35-sft \\
      -a claude_code \\
      --setting team \\
      -s flash_10 \\
      -c 2

    # Deploy persistently (survives terminal close, billed while up)
    uv run modal deploy scripts/distill/serve_vllm_modal.py
    uv run modal app stop serve-qwen35-sft   # to shut it down
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL           = "Qwen/Qwen3.5-9B"
LORA_REPO            = "CooperBench/qwen3.5-9b-tool-use-sft"
MODEL_ALIAS          = "qwen35-sft"
APP_NAME             = "serve-qwen35-sft"
GPU                  = "A100-80GB"
MAX_MODEL_LEN        = 32768
VLLM_PORT            = 8000
VLLM_VERSION         = "0.19.0"
TRANSFORMERS_VERSION = "5.5.4"
MINUTES              = 60

# ---------------------------------------------------------------------------
# Volumes — shared HF / vLLM weight cache (reused across runs)
# ---------------------------------------------------------------------------

_hf_cache_vol   = modal.Volume.from_name("huggingface-cache",  create_if_missing=True)
_vllm_cache_vol = modal.Volume.from_name("vllm-cache",         create_if_missing=True)

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .uv_pip_install(
        f"transformers=={TRANSFORMERS_VERSION}",
        "huggingface-hub[hf_xet]>=0.36.0",
        "peft>=0.14.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App(name=APP_NAME)


@app.function(
    image=_image,
    gpu=GPU,
    timeout=4 * MINUTES * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": _hf_cache_vol,
        "/root/.cache/vllm":        _vllm_cache_vol,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--host",                   "0.0.0.0",
        "--port",                   str(VLLM_PORT),
        "--max-model-len",          str(MAX_MODEL_LEN),
        "--dtype",                  "bfloat16",
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser",       "qwen3_coder",
        "--enable-lora",
        "--max-lora-rank",          "64",
        "--lora-modules",           f"{MODEL_ALIAS}={LORA_REPO}",
        "--trust-remote-code",
        "--enforce-eager",
    ]

    print(f"Starting vLLM: {' '.join(cmd)}")
    subprocess.Popen(cmd)
