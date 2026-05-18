# Core-subset horizontal comparison

Four agent frameworks evaluated on the 10-pair `core` subset
(`dataset/subsets/core.json`) in `team` setting.  Each framework was
paired with its natural model (`claude_code` → `claude-sonnet-4-5`;
`codex`, `mini_swe_agent_v2`, `openhands_sdk` → `gpt-5.5`).  Backend
is Docker (concurrency=3) except `openhands_sdk`, which runs its
agent-server in a Modal sandbox.

| Agent framework | Pass | Cost (USD) | Wall time | Run name |
|---|---|---|---|---|
| `mini_swe_agent_v2` | **6 / 10** | $13.37 | 24m | `msa_team_core_v4` |
| `openhands_sdk` | **5 / 10** | $31.90 | 16m | `oh_team_core` |
| `claude_code` | **5 / 10** | ~$8.5 | 21m | `cc_team_core_v4` |
| `codex` | **5 / 10** | $0* | 21m | `cx_team_core_v4` |

*`gpt-5.5` is not in the local pricing table; codex did do real work
(400 k+ input tokens per agent).

## Per-task pass/fail

Read top-to-bottom by repo, columns are agent frameworks.  `S` =
passed via the solo-agent eval fallback (one agent's patch alone
passed both features); `M` = passed via the merged-tree (naive or
union); `·` = failed.

| Task | `msa` | `oh` | `cc` | `cx` |
|---|---|---|---|---|
| `dottxt_ai_outlines/1655` [1,3] | M | M | M | M |
| `dspy/8563` [1,4]               | · | · | · | · |
| `go_chi/27` [3,4]               | · | · | · | · |
| `llama_index/17244` [5,6]       | M | M | · | M |
| `openai_tiktoken/0` [4,8]       | M | M | S | S |
| `pallets_click/2800` [1,4]      | M | · | · | · |
| `pallets_jinja/1559` [5,8]      | · | · | M | · |
| `pallets_jinja/1621` [6,10]     | M | M | S | S |
| `react_hook_form/153` [2,6]     | · | · | · | · |
| `typst/6554` [2,6]              | S | M | S | S |

Three tasks (`dspy/8563`, `go_chi/27`, `react_hook_form/153`) failed
for every framework — agents on those produced overlapping patches
where neither solo nor the merged tree passed both feature suites.

## What the runs cost to get here

Five reruns plus four re-evals were needed to land at these numbers
— each surfaced and fixed a real bug.  Chronologically:

1. **`msa_team_core` (Modal)** — 0/10.  Every agent died at step 1.
   Modal sandbox terminated on first `exec` because `Sandbox.create`
   wasn't given a long-running command.
2. **`msa_team_core_v2` (Modal, after sleep-infinity fix)** — 3/10.
   Sandboxes survived; real work happened.
3. **`msa_team_core_v3` (Modal, routed msa patches through
   `normalize_patch`)** — *dropped* to 1/10.  `normalize_patch`'s
   own `.strip()` was eating trailing blank-context lines from valid
   `git diff` output, breaking hunks across the board.
4. **`msa_team_core_v4` (Docker, after fixing `normalize_patch` itself
   + adding the solo-agent eval fallback)** — 5/10 from the run,
   6/10 after re-eval.
5. **`cx_team_core` (Docker, c=10)** — 0/10.  Member agents hit the
   120 s `docker run` startup timeout because team mode pairs codex's
   lead with msa's docker env, and 20 parallel container creations
   were too many.
6. **`cx_team_core_v2` (Modal)** — 0/10, 2 h wall.  `codex exec` hangs
   in Modal sandboxes (likely missing tty / auth retry).  Modal is
   not a viable backend for codex today.
7. **`cx_team_core_v3` (Docker, c=3)** — 2/10.  Concurrency low enough
   to avoid the docker-run timeout; failures were all merge-conflict
   union-strategy artifacts.
8. **`cx_team_core_v4` (Docker, c=3, beefed lead prompt)** — 2/10
   from the run, **5/10 after re-eval** with the solo-agent fallback.
9. **`cc_team_core` (Docker, c=10)** — 2/10.  Same docker-startup
   issue as codex (2 tasks died at container creation), but milder
   because cc only spawns one container per agent.
10. **`cc_team_core_v2` (Modal)** — 2/10.  Clean infra; the 2 passes
    were a "real" 2/10 against the original eval.
11. **`cc_team_core_v3` (Docker, c=3, normalize_patch fix)** — 2/10.
    Confirmed `normalize_patch` fix isn't enough on its own.
12. **`cc_team_core_v4` (Docker, c=3, beefed lead prompt)** — 2/10
    from the run, **5/10 after re-eval** with solo-agent fallback.

## Where the numbers ended up

`oh` was the only framework that already passed ≥ 5 / 10 against the
original eval.  All three of the others required the solo-agent
fallback to surface integration work that the agents had genuinely
done but that union-merge had wiped out — see the `S` cells in the
per-task table.  Bug fixes that drove the move are catalogued in
`CHANGELOG.md` under the unreleased entry.
