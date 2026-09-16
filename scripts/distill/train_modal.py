"""SFT training on Modal with a single A100-80GB GPU.

Trains Qwen3.5-9B-Instruct on the combined teacher + native Qwen trajectories
using action-masking (loss computed only on assistant turns that contain tool
calls or non-trivial text — not on system/user/tool messages).

Usage:
    # Dry-run: tokenise locally, print stats, exit
    uv run python scripts/distill/train_modal.py --dry-run

    # Full run on Modal (launches a remote A100 job)
    uv run python scripts/distill/train_modal.py

    # Resume from a checkpoint
    uv run python scripts/distill/train_modal.py --resume

Environment:
    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET  — Modal auth (or `modal token set`)
    HF_TOKEN                             — HuggingFace token for model download

Outputs (saved to Modal volume, also synced to data/checkpoints/ locally):
    data/checkpoints/<run-name>/   adapter weights (LoRA) + tokenizer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared constants (referenced both locally and inside the Modal container)
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3.5-9B"
RUN_NAME = "qwen35-tool-use-sft-v1"

# LoRA config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Training hyperparams
MAX_SEQ_LEN = 2048        # context window per sample (truncate longer)
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 16     # effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
SAVE_STEPS = 20
EVAL_STEPS = 100
LOGGING_STEPS = 10

DATA_PATHS = {
    "teacher": "data/converted_teacher.jsonl",
    "qwen":    "data/successful.jsonl",
}
OUTPUT_DIR = f"data/checkpoints/{RUN_NAME}"

# ---------------------------------------------------------------------------
# Local helpers: load + format training data
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a software engineer working alongside a colleague on a shared codebase. "
    "You each have your own workspace and are implementing different features in parallel. "
    "You communicate naturally — like engineers on the same team — to make sure "
    "your combined work integrates cleanly."
)


def _iter_messages_from_record(rec: dict) -> list[dict] | None:
    """Return the messages list for a record, regardless of source format."""
    # converted_teacher.jsonl: messages are top-level
    if "messages" in rec and rec["messages"]:
        return rec["messages"]
    # successful.jsonl Qwen records: messages are inside trajectory dict
    traj = rec.get("trajectory")
    if isinstance(traj, dict):
        msgs = traj.get("messages", [])
        if msgs:
            return msgs
    return None


def load_records() -> list[list[dict]]:
    """Load all training records and return a list of message lists."""
    all_msgs: list[list[dict]] = []

    # Teacher trajectories (converted_teacher.jsonl)
    teacher_path = Path(DATA_PATHS["teacher"])
    if teacher_path.exists():
        with teacher_path.open() as f:
            for line in f:
                rec = json.loads(line)
                msgs = _iter_messages_from_record(rec)
                if msgs and len(msgs) > 2:
                    all_msgs.append(msgs)

    # Qwen native trajectories (successful.jsonl, qwen runs only)
    qwen_path = Path(DATA_PATHS["qwen"])
    if qwen_path.exists():
        with qwen_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("run", "").startswith("qwen"):
                    continue
                msgs = _iter_messages_from_record(rec)
                if msgs and len(msgs) > 2:
                    all_msgs.append(msgs)

    return all_msgs


def messages_to_chatml(msgs: list[dict]) -> list[dict]:
    """Normalise a message list to clean ChatML dicts for the tokeniser.

    Rules:
    - system/user messages: keep content as-is
    - assistant messages with tool_calls: convert to a single content string
      that includes both the text content and a tool-call representation
    - tool messages: convert to a user-visible tool result representation
    - exit messages: drop
    """
    out = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content") or ""

        if role == "exit":
            continue

        if role in ("system", "user"):
            out.append({"role": role, "content": str(content)})

        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                # Represent the tool call as structured text the model must predict
                tc = tool_calls[0]  # always single call in our data
                try:
                    args = json.loads(tc["function"]["arguments"])
                    cmd = args.get("command", "")
                except (json.JSONDecodeError, KeyError):
                    cmd = tc["function"].get("arguments", "")
                # Keep any reasoning text first, then the tool call
                text = (str(content) + "\n" if content else "") + f"<tool_call>\n{cmd}\n</tool_call>"
                out.append({"role": "assistant", "content": text.strip()})
            else:
                if content:
                    out.append({"role": "assistant", "content": str(content)})

        elif role == "tool":
            # Represent tool results as a user message so the model sees them
            try:
                result = json.loads(str(content))
                rc = result.get("returncode", 0)
                output = result.get("output", "")
            except (json.JSONDecodeError, TypeError):
                rc = 0
                output = str(content)
            tool_text = f"<tool_result returncode={rc}>\n{output}\n</tool_result>"
            out.append({"role": "user", "content": tool_text})

    return out


def build_action_mask(tokenised_ids: list[int], tokeniser, chatml_msgs: list[dict]) -> list[int]:
    """Return a label mask (1 = compute loss, 0 = mask) aligned to tokenised_ids.

    We want loss only on assistant tokens.  We detect assistant turns by
    re-tokenising the conversation incrementally and finding the boundaries.

    Returns a list of the same length as tokenised_ids.
    """
    # Simple approach: re-encode each message, find where assistant content sits
    # by matching token spans.  Falls back to full-sequence loss if alignment fails.
    mask = [0] * len(tokenised_ids)
    try:
        pos = 0
        for msg in chatml_msgs:
            role = msg["role"]
            content = msg["content"]
            # Encode role header + content (approximate span)
            header = f"<|im_start|>{role}\n"
            footer = "<|im_end|>\n"
            header_ids = tokeniser.encode(header, add_special_tokens=False)
            content_ids = tokeniser.encode(content, add_special_tokens=False)
            footer_ids = tokeniser.encode(footer, add_special_tokens=False)

            header_len = len(header_ids)
            content_len = len(content_ids)
            footer_len = len(footer_ids)
            total_len = header_len + content_len + footer_len

            if role == "assistant":
                # Mark content + footer (not header) as loss tokens
                for i in range(pos + header_len, min(pos + total_len, len(mask))):
                    mask[i] = 1

            pos += total_len
            if pos >= len(mask):
                break
    except Exception:
        # Fallback: label everything
        return [1] * len(tokenised_ids)

    return mask


# ---------------------------------------------------------------------------
# Dry-run: run locally to check data pipeline
# ---------------------------------------------------------------------------

def dry_run() -> None:
    records = load_records()
    print(f"Loaded {len(records)} trajectories")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except Exception as e:
        print(f"Cannot load tokeniser ({e}); skipping tokenisation stats")
        tok = None

    total_tokens = 0
    truncated = 0
    for msgs in records[:200]:  # sample first 200 for speed
        chatml = messages_to_chatml(msgs)
        text = tok.apply_chat_template(chatml, tokenize=False) if tok else ""
        if tok:
            ids = tok.encode(text)
            total_tokens += min(len(ids), MAX_SEQ_LEN)
            if len(ids) > MAX_SEQ_LEN:
                truncated += 1

    print(f"Avg tokens (first 200, capped at {MAX_SEQ_LEN}): {total_tokens // min(200, len(records))}")
    print(f"Truncated (>{MAX_SEQ_LEN} tokens): {truncated}/200")
    print(f"Estimated total training tokens: {total_tokens * (len(records) / 200) / 1e6:.1f}M")
    print("\nDry-run complete — no Modal job submitted.")


# ---------------------------------------------------------------------------
# Modal app — must be at module scope for @app.function to work
# ---------------------------------------------------------------------------

try:
    import modal as _modal

    _volume = _modal.Volume.from_name(f"{RUN_NAME}-volume", create_if_missing=True)

    _image = (
        _modal.Image.debian_slim(python_version="3.11")
        .apt_install("build-essential")
        .pip_install(
            "torch>=2.5.0",
            "transformers>=4.52.0",
            "peft>=0.14.0",
            "bitsandbytes>=0.44.0",
            "accelerate>=0.34.0",
            "datasets>=2.21.0",
            "huggingface-hub>=0.24",
            "sentencepiece",
            "protobuf",
            "scipy",
        )
    )

    app = _modal.App(name=RUN_NAME)

    @app.function(
        image=_image,
        gpu="A100-80GB",
        timeout=60 * 60 * 8,
        volumes={"/checkpoints": _volume},
        secrets=[_modal.Secret.from_name("huggingface-secret")],
    )
    def train(
        teacher_jsonl: bytes,
        qwen_jsonl: bytes,
        resume: bool = False,
    ) -> str:
        """Run fine-tuning inside the Modal container."""
        import json as _json
        import os as _os

        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        hf_token = _os.environ.get("HF_TOKEN")

        # ── Load tokeniser + model ───────────────────────────────────────────
        print(f"Loading {BASE_MODEL} …")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, token=hf_token, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False

        # ── LoRA ─────────────────────────────────────────────────────────────
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # ── Build dataset ─────────────────────────────────────────────────────
        def _parse_jsonl(data: bytes) -> list[dict]:
            return [_json.loads(line) for line in data.decode().splitlines() if line.strip()]

        teacher_recs = _parse_jsonl(teacher_jsonl)
        qwen_recs    = _parse_jsonl(qwen_jsonl)

        def _to_chatml(msgs: list[dict]) -> list[dict]:
            out = []
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content") or ""
                if role == "exit":
                    continue
                if role in ("system", "user"):
                    out.append({"role": role, "content": str(content)})
                elif role == "assistant":
                    tcs = m.get("tool_calls") or []
                    if tcs:
                        try:
                            a = _json.loads(tcs[0]["function"]["arguments"])
                            cmd = a.get("command", "")
                        except Exception:
                            cmd = tcs[0]["function"].get("arguments", "")
                        text = (str(content) + "\n" if content else "") + \
                               f"<tool_call>\n{cmd}\n</tool_call>"
                        out.append({"role": "assistant", "content": text.strip()})
                    else:
                        if content:
                            out.append({"role": "assistant", "content": str(content)})
                elif role == "tool":
                    try:
                        res = _json.loads(str(content))
                        rc = res.get("returncode", 0)
                        output = res.get("output", "")
                    except Exception:
                        rc, output = 0, str(content)
                    out.append({"role": "user",
                                "content": f"<tool_result returncode={rc}>\n{output}\n</tool_result>"})
            return out

        def _tokenise(rec: dict) -> dict | None:
            msgs = rec.get("messages")
            if not msgs:
                traj = rec.get("trajectory")
                if isinstance(traj, dict):
                    msgs = traj.get("messages")
            if not msgs or len(msgs) <= 2:
                return None
            chatml = _to_chatml(msgs)
            if len(chatml) < 2:
                return None
            text = tokenizer.apply_chat_template(chatml, tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)
            enc["labels"] = enc["input_ids"].copy()
            return enc

        rows = []
        for rec in teacher_recs + qwen_recs:
            enc = _tokenise(rec)
            if enc:
                rows.append(enc)

        print(f"Total training examples: {len(rows)}")
        dataset = Dataset.from_list(rows)
        dataset = dataset.train_test_split(test_size=0.02, seed=42)

        # ── Training ──────────────────────────────────────────────────────────
        output_dir = f"/checkpoints/{RUN_NAME}"
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            bf16=True,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            eval_strategy="no",
            save_total_limit=3,
            report_to="none",
            dataloader_num_workers=2,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        import modal as _modal_remote
        from transformers import TrainerCallback
        _vol = _modal_remote.Volume.from_name(f"{RUN_NAME}-volume")

        class VolumeCommitCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                print(f"Committing checkpoint at step {state.global_step} to volume …")
                _vol.commit()
                print("Volume committed.")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=collator,
            callbacks=[VolumeCommitCallback()],
        )

        print("Starting training …")
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved to {output_dir}")
        return output_dir

except ImportError:
    app = None
    train = None
    _volume = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Check data pipeline locally, no Modal job")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint on volume")
    p.add_argument("--sync-only", action="store_true",
                   help="Download latest checkpoint from Modal volume to data/checkpoints/")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run:
        dry_run()
        return 0

    try:
        import modal
    except ImportError:
        print("modal not installed. Run: pip install modal", file=sys.stderr)
        return 1

    if app is None:
        print("error: modal failed to initialise (see import error above)", file=sys.stderr)
        return 1

    if args.sync_only:
        print(f"Syncing checkpoint from Modal volume to {OUTPUT_DIR} …")
        local_out = Path(OUTPUT_DIR)
        local_out.mkdir(parents=True, exist_ok=True)
        for entry in _volume.iterdir(f"/{RUN_NAME}"):
            _volume.read_file_into_memory(entry.path)  # triggers download
        print(f"Use `modal volume get {RUN_NAME}-volume /{RUN_NAME} {OUTPUT_DIR}` to download.")
        return 0

    # Read data files and send to Modal
    teacher_path = Path(DATA_PATHS["teacher"])
    qwen_path    = Path(DATA_PATHS["qwen"])

    if not teacher_path.exists():
        print(f"error: {teacher_path} not found — run convert_codex_to_miniswe.py first",
              file=sys.stderr)
        return 1
    if not qwen_path.exists():
        print(f"error: {qwen_path} not found — run extract_successful.py first",
              file=sys.stderr)
        return 1

    teacher_bytes = teacher_path.read_bytes()
    qwen_bytes    = qwen_path.read_bytes()
    print(f"Teacher data: {len(teacher_bytes) / 1e6:.1f} MB")
    print(f"Qwen data:    {len(qwen_bytes) / 1e6:.1f} MB")

    print(f"Submitting Modal job '{RUN_NAME}' on A100-80GB …")
    _modal.enable_output()
    with app.run():
        result = train.remote(
            teacher_jsonl=teacher_bytes,
            qwen_jsonl=qwen_bytes,
            resume=args.resume,
        )
    print(f"Training complete. Downloading weights to {OUTPUT_DIR} …")
    local_out = Path(OUTPUT_DIR)
    local_out.mkdir(parents=True, exist_ok=True)
    remote_root = f"/{RUN_NAME}"
    for entry in _volume.iterdir(remote_root, recursive=True):
        if entry.type.name == "FILE":
            rel = entry.path[len(remote_root):].lstrip("/")
            dest = local_out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = b"".join(_volume.read_file(entry.path))
            dest.write_bytes(data)
            print(f"  {rel} ({len(data) / 1e6:.1f} MB)")
    print(f"Weights saved to {local_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
