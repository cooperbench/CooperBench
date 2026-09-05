"""Check every task's combined.patch passes every one of its features' tests.

Third leg of the objective dataset check (the other two are in check_gradeable.py):

    tests alone            -> must FAIL
    tests + feature.patch  -> must PASS
    tests + combined.patch -> must PASS      <- this script

combined.patch is the full PR, i.e. what the "all features in one tree" reference looks like.
If a feature's tests fail against it, either the combined patch does not contain that feature
or its tests are incompatible with a sibling feature landing in the same tree — both mean the
merged-eval path can never score that pair.

Same sandbox path as check_gradeable.py: the image's own `runner.sh`, one sandbox per feature.
`--backend modal` is linux/amd64; `--backend docker` is the local daemon (arm64 on Apple
Silicon), and the published images are multi-arch, so running both covers both architectures.

    python scripts/check_combined.py                        # all 199 on modal
    python scripts/check_combined.py --backend docker       # all 199 locally
    python scripts/check_combined.py pillow_task/task290    # one task
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from check_gradeable import _amd64_ref  # noqa: E402

from cooperbench.eval.backends import get_backend  # noqa: E402
from cooperbench.utils import get_image_name  # noqa: E402

DATASET = Path(__file__).resolve().parents[1] / "dataset"


def report_path(backend: str) -> Path:
    return DATASET / f"combined_report{'' if backend == 'modal' else '_' + backend}.json"


def _write(sb, path: str, content: str) -> None:
    # Chunked: the whole command line rides in the exec's argv, and Modal caps that at 64 KiB
    # (ARG_MAX). pallets_jinja/1465's combined.patch alone is >100 KB.
    enc = base64.b64encode(content.encode()).decode()
    sb.exec("bash", "-c", f": > {path}")
    for i in range(0, len(enc), 32_000):
        sb.exec("bash", "-c", f"echo '{enc[i : i + 32_000]}' | base64 -d >> {path}")


def check_feature(repo: str, task: str, feature: str, backend: str) -> dict:
    td = DATASET / repo / task
    tests, combined = td / feature / "tests.patch", td / "combined.patch"
    out = {"task": f"{repo}/{task}", "feature": feature}
    if not (tests.is_file() and combined.is_file()):
        return {**out, "verdict": "SKIP", "note": "missing tests.patch or combined.patch"}

    image = get_image_name(repo, int(task.replace("task", "")))
    if backend == "modal":
        image = _amd64_ref(image)
    started = time.time()
    sb = get_backend(backend).create_sandbox(image, timeout=3600)
    try:
        sb.exec("bash", "-c", "mkdir -p /patches")
        _write(sb, "/patches/tests.patch", tests.read_text())
        _write(sb, "/patches/combined.patch", combined.read_text())
        # Same shape as check_gradeable.run: keep the runner's own exit status (never pipe it),
        # then grep the signal lines back out of the log.
        r = sb.exec(
            "bash",
            "-c",
            "bash /usr/local/bin/runner.sh tests.patch combined.patch > /tmp/out.log 2>&1; echo RC=$?; "
            "grep -iE 'does not apply|failed to apply|error:|FAILED|assert|"
            "[0-9]+ (passed|failed)|no tests|collected|panic|cannot|Tests:|test result:|^--- FAIL' "
            "/tmp/out.log | grep -viE 'Removing |Repository (cleaned|restored)' | tail -25",
        )
        body = r.stdout_read() + r.stderr_read()
        rc = next((int(ln[3:]) for ln in body.splitlines() if ln.startswith("RC=")), -1)
        return {
            **out,
            "verdict": "OK" if rc == 0 else "COMBINED_FAILS",
            "rc": rc,
            "seconds": round(time.time() - started),
            "tail": body.strip()[-600:],
        }
    except Exception as exc:
        return {**out, "verdict": "ERROR", "note": f"{type(exc).__name__}: {exc}"}
    finally:
        sb.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="repo/taskN, default all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--backend", choices=["modal", "docker"], default="modal")
    args = ap.parse_args()

    feats = [
        (p.parent.parent.name, p.parent.name, p.name)
        for p in sorted(DATASET.glob("*/task*/feature*"))
        if p.is_dir() and (not args.task or f"{p.parent.parent.name}/{p.parent.name}" == args.task)
    ]
    if not feats:
        raise SystemExit(f"no features matched {args.task!r}")
    report = report_path(args.backend)
    print(f"checking {len(feats)} features on {args.backend}, {args.workers} sandboxes at a time\n", flush=True)

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_feature, *f, args.backend): f for f in feats}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if r["verdict"] != "OK":
                print(
                    f"  [{done}/{len(feats)}] {r['verdict']:15s} {r['task']} {r['feature']} {r.get('note', '')}",
                    flush=True,
                )
            elif done % 20 == 0:
                print(f"  [{done}/{len(feats)}] ...", flush=True)

    report.write_text(json.dumps(sorted(results, key=lambda r: (r["task"], r["feature"])), indent=1))
    tally: dict[str, int] = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("\n" + " · ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
