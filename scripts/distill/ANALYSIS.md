# Team-Coop Dataset Analysis

## Dataset
- Source: `CooperBench/team-coop` on HuggingFace
- Downloaded to: `data/team-coop/` (gitignored)
- Total trajectory pairs: 1,993
- Successful pairs (correct=true, verified=true): 837
- Training records (one per agent per pair): 1,674 → `data/successful.jsonl`

## Runs in the dataset

| Run | Model | Pass rate | Notes |
|---|---|---|---|
| `cmp-full-team` | gpt-5.5-hao | ~60% | Full team features including protocol |
| `cmp-full-team-noproto` | gpt-5.5-hao | ~60% | All features except protocol |
| `qwen35-cooperdata-team-noproto` | Qwen3.5-9B | ~6% | Small model baseline, almost all fail |
| `qwen35-cooperdata-team-noproto-forced` | Qwen3.5-9B | ~6% | Small model variant |
| `coop/` | Qwen3.5-9B | ~0% | Earlier Qwen runs, all failing |

The useful teacher trajectories come entirely from the two `cmp-full-team` runs (gpt-5.5-hao).

## Scenario Coverage

Threshold for "covered": 20 examples.

| Scenario | Count | Status | Notes |
|---|---|---|---|
| `solo_task_lifecycle` | 1,364 | **Covered** | Agent claims, works, marks done |
| `lead_creates_subtask` | 1,450 | **Covered** | Lead agent creates additional tasks mid-run |
| `parallel_independent` | 1,102 | **Covered** | Both agents work independently, no messaging |
| `cross_agent_dependency` | 718 | **Covered** | Lead waits for member before finishing |
| `blocked_task` | 118 | **Covered** | A task reaches status=blocked |
| `request_respond` | 10 | **GAP** | Only in protocol-enabled run; need ~10 more |
| `claim_after_list` | 0 | **GAP** | Detector may need fixing — shell cmds may not be stored as plain text in trajectory JSON |
| `wait_for_message` | 0 | **GAP** | MCP idle-wait never triggered in successful runs |

## Gaps and next steps

### `request_respond` (10 examples)
Only appears in `cmp-full-team` (protocol on). Close to threshold — a handful of synthetic
examples will cover it. Can also re-run analysis with `--no-require-verified` to pick up
unverified passes from this scenario.

### `claim_after_list` (0 examples)
The detector searches for `coop-task list` as a string inside trajectory steps. Zero hits
likely means trajectory JSON stores tool calls in a structured format (not raw shell strings).
**Before generating synthetic examples**: inspect one trajectory file to confirm the format,
then fix the detector in `analyze_coverage.py`.

### `wait_for_message` (0 examples)
The MCP long-poll tool was never called in any successful trajectory. These runs used Codex
which does have the MCP server registered, but agents never went idle enough to trigger it.
Synthetic examples are the only path to coverage here.

## Recommended next step
Generate ~50 synthetic trajectories covering the three gaps using a teacher model against
a live Redis environment. Priority order: `wait_for_message` (pure gap), `claim_after_list`
(confirm detector first), `request_respond` (almost covered).
