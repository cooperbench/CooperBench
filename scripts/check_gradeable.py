"""Check every feature is gradeable: its tests must FAIL on the base commit and PASS with gold.

This is the property the whole dataset depends on, and neither half is safe to assume:

    tests alone  ->  must FAIL     otherwise the tests do not measure the feature, and it scores
                                   for any submission at all (pallets_jinja/1621 f5 passed 6/6 on
                                   an untouched tree, so it scored 5/5 across archived runs
                                   regardless of what the agent wrote)
    tests + gold ->  must PASS     otherwise the reference contradicts its own tests and no
                                   correct implementation can score

Both runs go through `runner.sh`, which is the real grading path — language-specific invocation,
filters and exit-code handling included — rather than a hand-rolled pytest call that would miss
exactly the runner defects this is meant to catch.

One sandbox per feature, so a runner's `git clean -fdx` cleanup trap cannot leak into the next
measurement. Runs on Modal; nothing is built locally.

    python scripts/check_gradeable.py                        # all 199
    python scripts/check_gradeable.py pillow_task/task290     # one task
    python scripts/check_gradeable.py --workers 16
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooperbench.eval.backends import get_backend  # noqa: E402
from cooperbench.utils import get_image_name  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "dataset"
REPORT = DATASET / "gradeable_report.json"
REPORT_LOCAL = DATASET / "gradeable_report_localrunner.json"


def _amd64_ref(image: str) -> str:
    """Pin the amd64 manifest by digest.

    These images are multi-arch; Modal runs amd64 only, and when it resolves the tag itself it can
    pick the arm64 manifest and fail the image build with "image architecture arm64 not supported"
    — then CACHE that failure, so every later run of that task fails instantly. Handing it a
    digest removes the choice. Falls back to the plain tag if the registry cannot be reached.
    """
    try:
        raw = subprocess.run(["docker", "buildx", "imagetools", "inspect", image, "--raw"],
                             capture_output=True, text=True, timeout=60)
        if raw.returncode != 0:
            return image
        for m in json.loads(raw.stdout).get("manifests", []):
            p = m.get("platform", {})
            if p.get("architecture") == "amd64" and p.get("os") == "linux":
                return f"{image}@{m['digest']}"
    except Exception:
        pass
    return image


def _write(sb, path: str, content: str) -> None:
    enc = base64.b64encode(content.encode()).decode()
    sb.exec("bash", "-c", f"echo '{enc}' | base64 -d > {path}")


def check_feature(repo: str, task: str, feature: str, local_runner: bool = False) -> dict:
    fd = DATASET / repo / task / feature
    tests, gold = fd / "tests.patch", fd / "feature.patch"
    out = {"task": f"{repo}/{task}", "feature": feature}
    if not (tests.is_file() and gold.is_file()):
        return {**out, "verdict": "SKIP", "note": "missing tests.patch or feature.patch"}

    image = _amd64_ref(get_image_name(repo, int(task.replace("task", ""))))
    started = time.time()
    sb = get_backend("modal").create_sandbox(image, timeout=3600)
    try:
        sb.exec("bash", "-c", "mkdir -p /patches")
        if local_runner:
            # runner.sh is COPY'd into the image at build time, so edits to it in this repo are
            # invisible until the image is rebuilt and pushed. Overwrite it in the sandbox to
            # measure the fixed runner rather than the stale baked one.
            _write(sb, "/usr/local/bin/runner.sh", (DATASET / repo / task / "runner.sh").read_text())
            sb.exec("bash", "-c", "chmod +x /usr/local/bin/runner.sh")
        _write(sb, "/patches/tests.patch", tests.read_text())
        _write(sb, "/patches/feature.patch", gold.read_text())

        # NOT `runner.sh ... | tail`: a pipeline exits with the status of its LAST command, so
        # piping into tail reports tail's 0 and every feature looks like it passes. Redirect to a
        # file, keep the runner's own status, then read the tail back.
        def run(args: str):
            # Grep the signal out rather than tail blindly: these runners print a long
            # `git clean` inventory on exit, which pushes the actual error off the end.
            r = sb.exec("bash", "-c",
                        f"bash /usr/local/bin/runner.sh {args} > /tmp/out.log 2>&1; echo RC=$?; "
                        "grep -iE 'does not apply|failed to apply|error:|FAILED|assert|"
                        "[0-9]+ (passed|failed)|no tests|collected|panic|cannot' /tmp/out.log "
                        "| grep -viE 'Removing |Repository (cleaned|restored)' | tail -25")
            body = r.stdout_read() + r.stderr_read()
            rc = next((int(ln[3:]) for ln in body.splitlines() if ln.startswith("RC=")), -1)
            return rc, body

        base_rc, base_body = run("tests.patch")
        gold_rc, gold_body = run("tests.patch feature.patch")

        if gold_rc != 0:
            verdict = "GOLD_FAILS"          # reference contradicts its own tests
        elif base_rc == 0:
            verdict = "PASSES_ON_BASE"      # tests do not measure the feature
        else:
            verdict = "OK"
        return {
            **out,
            "verdict": verdict,
            "base_rc": base_rc,
            "gold_rc": gold_rc,
            "seconds": round(time.time() - started),
            "base_tail": base_body.strip()[-400:],
            "gold_tail": gold_body.strip()[-400:],
        }
    except Exception as exc:
        return {**out, "verdict": "ERROR", "note": f"{type(exc).__name__}: {exc}"}
    finally:
        sb.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="repo/taskN, default all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--local-runner", action="store_true",
                    help="overwrite the image's baked-in runner.sh with the one in dataset/, so "
                         "runner fixes are exercised without rebuilding and pushing images")
    args = ap.parse_args()

    feats = [
        (p.parent.parent.name, p.parent.name, p.name)
        for p in sorted(DATASET.glob("*/task*/feature*"))
        if p.is_dir() and (not args.task or f"{p.parent.parent.name}/{p.parent.name}" == args.task)
    ]
    if not feats:
        raise SystemExit(f"no features matched {args.task!r}")
    print(f"checking {len(feats)} features, {args.workers} sandboxes at a time\n", flush=True)

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_feature, *f, args.local_runner): f for f in feats}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if r["verdict"] != "OK":
                print(f"  [{done}/{len(feats)}] {r['verdict']:15s} {r['task']} {r['feature']}"
                      f" {r.get('note','')}", flush=True)
            elif done % 20 == 0:
                print(f"  [{done}/{len(feats)}] ...", flush=True)

    (REPORT_LOCAL if args.local_runner else REPORT).write_text(json.dumps(sorted(results, key=lambda r: (r["task"], r["feature"])), indent=1))
    tally: dict[str, int] = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("\n" + " · ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    print(f"report -> {REPORT_LOCAL if args.local_runner else REPORT}")


if __name__ == "__main__":
    main()
