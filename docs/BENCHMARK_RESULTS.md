# Core-subset horizontal comparison

Four agent frameworks evaluated on the 10-pair `core` subset
(`dataset/subsets/core.json`) in `team` setting.  Each framework was
paired with its natural model (`claude_code` → `claude-sonnet-4-5`;
`codex`, `mini_swe_agent_v2`, `openhands_sdk` → `gpt-5.5`).  Backend
is Docker (concurrency=3) except `openhands_sdk`, which runs its
agent-server in a Modal sandbox.

**Eval policy**: `identical → naive merge → lead's patch alone`.
Identical short-circuits when both agents produce byte-identical
`patch.txt`; otherwise a naive 3-way merge is attempted, and if it
conflicts the eval falls back to testing the team-lead's patch alone
against both feature suites.  Union merge and member-only fallback
were intentionally dropped — they reward lucky non-overlap or partial
coordination rather than genuine team integration.

| Agent framework | Pass | Cost (USD) | Wall time | Run name |
|---|---|---|---|---|
| `mini_swe_agent_v2` | **6 / 10** | $13.37 | 24m | `msa_team_core_v4` |
| `openhands_sdk` | **4 / 10** | $31.90 | 16m | `oh_team_core` |
| `claude_code` | **5 / 10** | ~$8.5 | 21m | `cc_team_core_v4` |
| `codex` | **5 / 10** | $0* | 21m | `cx_team_core_v4` |

*`gpt-5.5` is not in the local pricing table; codex did do real work
(400 k+ input tokens per agent).

## Per-task pass/fail

Read top-to-bottom by repo, columns are agent frameworks.  `N` =
passed via naive merge (or identical short-circuit); `L` = passed
via the lead-alone fallback (naive conflicted, but the lead's
`patch.txt` passed both feature suites by itself); `·` = failed.

| Task | `msa` | `oh` | `cc` | `cx` |
|---|---|---|---|---|
| `dottxt_ai_outlines/1655` [1,3] | N | L | N | N |
| `dspy/8563` [1,4]               | · | · | · | · |
| `go_chi/27` [3,4]               | · | · | · | · |
| `llama_index/17244` [5,6]       | N | L | · | N |
| `openai_tiktoken/0` [4,8]       | N | L | L | L |
| `pallets_click/2800` [1,4]      | N | · | · | · |
| `pallets_jinja/1559` [5,8]      | · | · | N | · |
| `pallets_jinja/1621` [6,10]     | N | · | L | L |
| `react_hook_form/153` [2,6]     | · | · | · | · |
| `typst/6554` [2,6]              | L | L | L | L |
| **TOTAL**                       | **6** | **4** | **5** | **5** |

Three tasks (`dspy/8563`, `go_chi/27`, `react_hook_form/153`) failed
for every framework — agents on those produced overlapping patches
where neither lead-alone nor the naive merge passed both feature
suites.

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
   from the run, **5/10 after re-eval** with the lead-alone fallback.
9. **`cc_team_core` (Docker, c=10)** — 2/10.  Same docker-startup
   issue as codex (2 tasks died at container creation), but milder
   because cc only spawns one container per agent.
10. **`cc_team_core_v2` (Modal)** — 2/10.  Clean infra; the 2 passes
    were a "real" 2/10 against the original eval.
11. **`cc_team_core_v3` (Docker, c=3, normalize_patch fix)** — 2/10.
    Confirmed `normalize_patch` fix isn't enough on its own.
12. **`cc_team_core_v4` (Docker, c=3, beefed lead prompt)** — 2/10
    from the run, **5/10 after re-eval** with lead-alone fallback.

## Where the numbers ended up

Three of the four frameworks hit ≥ 5 / 10 on the final policy.  `oh`
sits at 4 / 10 — their agents reliably produce complementary patches
that *union*-merged cleanly under the older permissive eval (5 / 10),
but their lead does not actually integrate the member's work, so the
stricter `identical → naive → lead` policy correctly catches that.
Bug fixes that drove the move are catalogued in `CHANGELOG.md` under
the unreleased entry.
