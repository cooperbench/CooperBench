# Spec sufficiency audit

One question per feature:

> Could a supremely intelligent LLM, given **only** `feature.md` and the repository — no
> `tests.patch`, no `feature.patch` — arrive at every behaviour the hidden tests assert?

If yes, the spec is done; a model that fails it failed on its own merits. If no, the spec is
underspecified and gets the **minimal** statement that closes the gap.

## The line between a fix and a leak

Each missed assertion is one of two things:

| | test | action |
|---|---|---|
| **arbitrary** | a perfect reasoner could not have derived it — an identifier, a literal message, a grammar choice, a design decision with several defensible answers | state it in `feature.md`, as a contract |
| **derivable** | a perfect reasoner gets there from what is already written plus ordinary engineering judgement | change nothing |

A statement leaks if it could not have been written **without having seen the reference
implementation**. Contracts describe observable behaviour at a public boundary — names, messages,
accepted inputs, ordering guarantees. Leaks name internal mechanisms, point at which call will
raise, or say where the bug will be.

## Which file to edit

Either `feature.md` or `tests.patch` is fair game. **The gold `feature.patch` must never need to
change** — it is the reference the whole task is built around, and a fix that invalidates it is a
rewrite, not a repair. So:

- **Amend `feature.md`** when the assertion is a legitimate requirement that the spec simply
  failed to state. Keeps test coverage intact. Gold is unaffected by construction.
- **Amend `tests.patch`** when the assertion itself is the problem — it tests nothing (passes on
  the unmodified base), it is computed wrongly, or it demands something no spec should have to
  promise. Relaxing or removing an assertion leaves gold passing; tightening one may not, so
  re-check gold after any test edit.

Prefer amending `feature.md` where both are available: deleting an assertion shrinks what the
benchmark measures, whereas stating the requirement keeps the coverage and makes it winnable.

Worked example of the distinction, from `pallets_jinja/1621` f4. The spec already says *"Invalid
tuple formats should be handled gracefully (treat as regular paths)"*. The graded failure is
`TypeError: expected str, bytes or os.PathLike object, not tuple`. Writing *"do not pass it to
`os.fspath()`, which raises TypeError on a tuple"* would be a leak — it names the call and the
mechanism. It is also unnecessary: "handled gracefully" already forbids raising, so a strong
reasoner covers it. **Verdict: derivable, no change.**

## Status — 199/199 features audited

`OK` 174 · `SPEC` 23 · `TEST` 2

`OK` sufficient, no change · `SPEC` feature.md amended · `TEST` tests.patch amended · `RUN?` defect identified, fix needs a container run · `-` not yet audited

| task | feature | verdict | note |
|---|---|---|---|
| dottxt_ai_outlines_task/task1371 | f1 | `OK  ` | custom filters API named |
|  | f2 | `OK  ` | bare `Exception` in the test is a catch-all any error satisfies |
|  | f3 | `OK  ` | conditional filter loading described |
|  | f4 | `OK  ` | built-in filter extensions named |
| dottxt_ai_outlines_task/task1655 | f1 | `OK  ` | regex type: format described and the type name given |
|  | f10 | `OK  ` | over-complete — spec contains the answer verbatim |
|  | f2 | `OK  ` | regex type: format described and the type name given |
|  | f3 | `OK  ` | regex type: format described and the type name given |
|  | f4 | `OK  ` | regex type: format described and the type name given |
|  | f5 | `OK  ` | regex type: format described and the type name given |
|  | f6 | `OK  ` | regex type: format described and the type name given |
|  | f7 | `OK  ` | every assertion derivable; "xor" is exact |
|  | f8 | `OK  ` | format-only is explicit ('format' x3, XXX-XX-XXXX given) — note 000-00-0000 and 999-99-9999 are valid, so real SSN area rules must NOT be applied |
|  | f9 | `OK  ` | regex type: format described and the type name given |
| dottxt_ai_outlines_task/task1706 | f1 | `OK  ` | '…is not available' is pre-existing outlines text (0 occurrences in gold) |
|  | f2 | `OK  ` | cache_compiled_grammars and the processor API both named |
|  | f3 | `OK  ` | `temperature` named; only pre-existing messages asserted |
|  | f4 | `SPEC` | exact batch-size ValueError text |
|  | f5 | `SPEC` | exact custom_adapter ValueError texts — conditions stated, wording was not |
|  | f6 | `OK  ` | the matched messages come from the test's own validator, not from gold |
|  | f7 | `OK  ` | 'Grammar error' is raised by the test's own stub; fallback API named |
|  | f8 | `OK  ` | all four memory attributes named in the spec |
| dspy_task/task8394 | f1 | `OK  ` | bypass() context manager named and described |
|  | f2 | `OK  ` | over-complete — gives the exact ns_hash formula and key format |
|  | f3 | `OK  ` | every assertion maps to a stated sentence |
|  | f4 | `OK  ` | both candidate gaps derivable; amendment reverted |
|  | f5 | `OK  ` | compression + magic-header format stated in the spec |
| dspy_task/task8563 | f1 | `OK  ` | ToolCall / convert_input_schema_to_tool_args are pre-existing; 'Arg X is invalid' is pre-existing dspy validation |
|  | f2 | `OK  ` | ToolCall / convert_input_schema_to_tool_args are pre-existing; 'Arg X is invalid' is pre-existing dspy validation |
|  | f3 | `OK  ` | ToolCall / convert_input_schema_to_tool_args are pre-existing; 'Arg X is invalid' is pre-existing dspy validation |
|  | f4 | `OK  ` | ToolCall / convert_input_schema_to_tool_args are pre-existing; 'Arg X is invalid' is pre-existing dspy validation |
|  | f5 | `OK  ` | ToolCall / convert_input_schema_to_tool_args are pre-existing; 'Arg X is invalid' is pre-existing dspy validation |
|  | f6 | `OK  ` | ToolCall / convert_input_schema_to_tool_args are pre-existing; 'Arg X is invalid' is pre-existing dspy validation |
| dspy_task/task8587 | f1 | `OK  ` | nothing flagged — streaming API fully named in each spec |
|  | f2 | `OK  ` | nothing flagged — streaming API fully named in each spec |
|  | f3 | `OK  ` | nothing flagged — streaming API fully named in each spec |
|  | f4 | `OK  ` | nothing flagged — streaming API fully named in each spec |
|  | f5 | `OK  ` | nothing flagged — streaming API fully named in each spec |
|  | f6 | `OK  ` | nothing flagged — streaming API fully named in each spec |
| dspy_task/task8635 | f1 | `OK  ` | nothing flagged — proposer params fully named in each spec |
|  | f2 | `OK  ` | nothing flagged — proposer params fully named in each spec |
|  | f3 | `OK  ` | nothing flagged — proposer params fully named in each spec |
|  | f4 | `OK  ` | nothing flagged — proposer params fully named in each spec |
|  | f5 | `OK  ` | nothing flagged — proposer params fully named in each spec |
|  | f6 | `OK  ` | nothing flagged — proposer params fully named in each spec |
| go_chi_task/task26 | f1 | `OK  ` | "alias for chi.URLParam" carries the ordering requirement |
|  | f2 | `SPEC` | added RouteMetric/SimpleMetricsCollector field names |
|  | f3 | `SPEC` | added RouteSelector API — spec named no identifiers at all |
|  | f4 | `SPEC` | added WithPriority + Priority* constants |
| go_chi_task/task27 | f1 | `OK  ` | HTTP method parsing in patterns described |
|  | f2 | `OK  ` | method aliases + case insensitivity stated |
|  | f3 | `SPEC` | added EnableDebugLogging() — gold introduces it, spec named nothing |
|  | f4 | `OK  ` | path prefix support described |
| go_chi_task/task56 | f1 | `OK  ` | Allow header on 405 described |
|  | f2 | `OK  ` | 501 default + configurability stated |
|  | f3 | `OK  ` | request context tracing described |
|  | f4 | `OK  ` | method validation middleware described |
|  | f5 | `OK  ` | CORS options described |
| huggingface_datasets_task/task3997 | f1 | `OK  ` | Features sync semantics stated |
|  | f2 | `SPEC` | added set_custom_decoding_criteria — gold introduces it, spec named nothing |
|  | f3 | `SPEC` | same hook name, needed for the caching feature too |
|  | f4 | `OK  ` | extended update() semantics stated |
|  | f5 | `OK  ` | deep-merge behaviour stated |
| huggingface_datasets_task/task6252 | f1 | `OK  ` | EXIF orientation correction described; decode-time behaviour stated |
|  | f2 | `OK  ` | center-crop-to-square parameter named |
|  | f3 | `OK  ` | max-resolution clamping parameter named |
|  | f4 | `OK  ` | 2x arithmetic follows from "all four sides" |
|  | f5 | `OK  ` | `decode_example` is the repo's existing decode entry point |
|  | f6 | `OK  ` | single assertion, stated verbatim |
| huggingface_datasets_task/task7309 | f1 | `OK  ` | predicate pushdown + filters described |
|  | f2 | `OK  ` | streaming sort described |
| llama_index_task/task17070 | f1 | `OK  ` | NDCG metric change fully described incl. flag semantics |
|  | f2 | `OK  ` | NDCG metric change fully described incl. flag semantics |
|  | f3 | `OK  ` | NDCG metric change fully described incl. flag semantics |
| llama_index_task/task17244 | f1 | `OK  ` | ImageBlock parameter named with its semantics stated |
|  | f2 | `OK  ` | ImageBlock parameter named with its semantics stated |
|  | f3 | `OK  ` | ImageBlock parameter named with its semantics stated |
|  | f4 | `OK  ` | ImageBlock parameter named with its semantics stated |
|  | f5 | `OK  ` | ImageBlock parameter named with its semantics stated |
|  | f6 | `OK  ` | ImageBlock parameter named with its semantics stated |
|  | f7 | `OK  ` | ImageBlock parameter named with its semantics stated |
| llama_index_task/task18813 | f1 | `OK  ` | DocumentBlock is pre-existing; empty-bytes error described |
|  | f2 | `SPEC` | exact 'exceeds maximum allowed size' message — spec gave only the condition |
|  | f3 | `OK  ` | resolve_* variant named with its return shape stated |
|  | f4 | `OK  ` | resolve_* variant named with its return shape stated |
|  | f5 | `OK  ` | resolve_* variant named with its return shape stated |
|  | f6 | `OK  ` | resolve_* variant named with its return shape stated |
| openai_tiktoken_task/task0 | f1 | `OK  ` | encode() parameter named with its post-processing semantics stated |
|  | f10 | `OK  ` | encode() parameter named with its post-processing semantics stated |
|  | f2 | `OK  ` | encode() parameter named with its post-processing semantics stated |
|  | f3 | `OK  ` | encode() parameter named with its post-processing semantics stated |
|  | f4 | `OK  ` | encode() parameter named with its post-processing semantics stated |
|  | f5 | `OK  ` | encode() parameter named with its post-processing semantics stated |
|  | f6 | `OK  ` | composition order stated with an example |
|  | f7 | `OK  ` | encode() parameter named with its post-processing semantics stated |
|  | f8 | `OK  ` | recomputed the dict — exactly "top 3 adjacent pairs" |
|  | f9 | `OK  ` | encode() parameter named with its post-processing semantics stated |
| pallets_click_task/task2068 | f1 | `OK  ` | multi-file edit; `filename` sequence stated, assertions are content round-trips |
|  | f10 | `OK  ` | `working_dir` named; "Editing failed" is pre-existing |
|  | f11 | `OK  ` | spec gives the full callback signature the test asserts, incl. timestamp |
|  | f12 | `OK  ` | `handle_exit_codes`/`exit_code` named; "Editing failed" pre-existing |
|  | f2 | `SPEC` | exact timeout message — and gold does not pluralise, so "after 1 seconds" |
|  | f3 | `SPEC` | exact `Failed to create backup file: {error}` text |
|  | f4 | `OK  ` | `editor_args` named; "Editing failed" is pre-existing click text |
|  | f5 | `OK  ` | `lock_files` named; reuses the pre-existing "Editing failed" message |
|  | f6 | `OK  ` | `process_priority` named; test asserts the attribute round-trips |
|  | f7 | `OK  ` | `escape_shell` named; assertions compare escaped vs unescaped commands |
|  | f8 | `SPEC` | test inspects Popen kwargs for preexec_fn=os.setpgrp; start_new_session is equally correct, so 50/50 |
|  | f9 | `OK  ` | `isolate_env`/`env` named; assertions are on the captured environment |
| pallets_click_task/task2800 | f1 | `OK  ` | over-complete — names function, call sites, construct |
|  | f2 | `OK  ` | exit-hook API named in the spec |
|  | f3 | `OK  ` | resource-limit parameter named and described |
|  | f4 | `OK  ` | 'Test error' comes from the test's own stub |
|  | f5 | `OK  ` | retry messages are test-local; default 0 = no retries is stated, so propagation follows |
|  | f6 | `OK  ` | snapshot/restore API named in the spec |
|  | f7 | `OK  ` | explicit API Specification section |
| pallets_click_task/task2956 | f1 | `OK  ` | `Argument` is pre-existing; `shell_complete` is click's standard completion hook and the spec's subject |
|  | f2 | `OK  ` | 'Custom error' is raised by the test's own validator, not gold |
|  | f3 | `OK  ` | `to_info_dict` is pre-existing click API (context line) |
|  | f4 | `OK  ` | `invoke` is pre-existing click API (context line) |
|  | f5 | `OK  ` | nothing flagged |
|  | f6 | `OK  ` | nothing flagged |
|  | f7 | `OK  ` | nothing flagged |
|  | f8 | `OK  ` | nothing flagged |
| pallets_jinja_task/task1465 | f1 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f10 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f2 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f3 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f4 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f5 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f6 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f7 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f8 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
|  | f9 | `OK  ` | groupby param named with semantics, default and backward-compat all stated |
| pallets_jinja_task/task1559 | f1 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
|  | f10 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
|  | f2 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
|  | f3 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
|  | f4 | `OK  ` | over-complete — error strings quoted verbatim |
|  | f5 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
|  | f6 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
|  | f7 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
|  | f8 | `SPEC` | comma separators + trimmed covering fallback |
|  | f9 | `OK  ` | i18n extension API and tag syntax fully named in the spec |
| pallets_jinja_task/task1621 | f1 | `OK  ` | fallback search paths named and described |
|  | f10 | `OK  ` | aliasing map named and described |
|  | f2 | `OK  ` | normalisation rules stated |
|  | f3 | `OK  ` | validation rules stated |
|  | f4 | `OK  ` | invalid-tuple handling already stated |
|  | f5 | `TEST` | 6/6 passed on base; 3 tests rewritten to observe _path_cache |
|  | f6 | `OK  ` | spec names `path_transform`; the flagged `transform_path` is the test's own local function |
|  | f7 | `OK  ` | expansion of env vars / user paths stated |
|  | f8 | `OK  ` | filtering predicate named |
|  | f9 | `OK  ` | `TemplateNotFound` is jinja's standard exception; monitoring API named |
| pillow_task/task25 | f1 | `OK  ` | readonly-on-matching-filename rule stated |
|  | f2 | `OK  ` | preserve-readonly option named |
|  | f3 | `OK  ` | `OSError` is Pillow's standard save error; spec says dry_run still surfaces format errors |
|  | f4 | `OK  ` | backup-on-overwrite behaviour stated |
|  | f5 | `OK  ` | corner-pixel watermark option named |
| pillow_task/task290 | f1 | `OK  ` | palette-limit semantics stated |
|  | f2 | `OK  ` | maximum colour limit stated |
|  | f3 | `OK  ` | custom palette sort callable named |
|  | f4 | `TEST` | MSE assertions wrapped in uint8 and the reference does not honour the absolute threshold; rewritten to assert the ordering. Verified on Modal: passes with gold, fails without. |
|  | f5 | `SPEC` | lightness_factor <= 0 raises ValueError |
| pillow_task/task68 | f1 | `OK  ` | tuple XMP form stated |
|  | f2 | `OK  ` | metadata-removal option named |
|  | f3 | `OK  ` | orientation transformer callback named |
|  | f4 | `OK  ` | `TypeError` is the pre-existing exif_transpose error the handler replaces |
|  | f5 | `OK  ` | logging hooks named |
| react_hook_form_task/task153 | f1 | `OK  ` | over-complete — callback signature, ordering and target file all stated |
|  | f2 | `OK  ` | over-complete — callback signature, ordering and target file all stated |
|  | f3 | `OK  ` | over-complete — callback signature, ordering and target file all stated |
|  | f4 | `OK  ` | over-complete — callback signature, ordering and target file all stated |
|  | f5 | `OK  ` | over-complete — callback signature, ordering and target file all stated |
|  | f6 | `OK  ` | over-complete — callback signature, ordering and target file all stated |
| react_hook_form_task/task85 | f1 | `OK  ` | over-complete — names the prop, its type, the file and often the implementation shape |
|  | f2 | `OK  ` | over-complete — names the prop, its type, the file and often the implementation shape |
|  | f3 | `OK  ` | over-complete — names the prop, its type, the file and often the implementation shape |
|  | f4 | `OK  ` | over-complete — names the prop, its type, the file and often the implementation shape |
|  | f5 | `OK  ` | over-complete — names the prop, its type, the file and often the implementation shape |
| samuelcolvin_dirty_equals_task/task43 | f1 | `OK  ` | IsIP: version/netmask/ipaddress module all named |
|  | f2 | `OK  ` | IsMac: 'hyphen'/'dot' map 1:1 onto the four named formats |
|  | f3 | `OK  ` | IsEmail: dotted-domain requirement is the universal convention |
|  | f4 | `OK  ` | IsURL: (protocol, domain) order follows the title |
|  | f5 | `SPEC` | country codes are case-insensitive — gold does .upper(), spec silent |
|  | f6 | `SPEC` | issuer tokens are exactly 'Visa'/'Mastercard'/'AmericanExpress' (one word) |
|  | f7 | `OK  ` | IsHash: algorithms and case-preserving repr both stated |
|  | f8 | `OK  ` | IsRegex: repr format given for both flag cases |
|  | f9 | `OK  ` | IsColor: hex/rgb/hsl named in the title |
| typst_task/task6554 | f1 | `OK  ` | default param stated with examples |
|  | f10 | `SPEC` | exact no-words + unknown-unit diagnostics |
|  | f2 | `SPEC` | exact out-of-bounds diagnostic |
|  | f3 | `OK  ` | safe param stated with examples |
|  | f4 | `SPEC` | exact not-enough-characters + repeat diagnostics |
|  | f5 | `SPEC` | exact not-enough-characters + count diagnostics |
|  | f6 | `SPEC` | exact "string contains only whitespace" (new, not pre-existing) |
|  | f7 | `SPEC` | exact no-character-matches diagnostic + check ordering |
|  | f8 | `SPEC` | exact unknown-case-option diagnostic |
|  | f9 | `OK  ` | strip param stated with examples |

## Separately: gradeability — swept across all 30 tasks

Spec sufficiency is meaningless for a feature whose tests never run, so this was checked
mechanically for every task first. Two questions: does the runner actually select each feature's
tests, and can a run that executes nothing be scored as a pass?

**Result: 5 tasks had a real defect, all now fixed. The other 25 are sound.**

| task | defect | fix |
|---|---|---|
| `go_chi/26` | image `golang:1.21-alpine` while feature 1's test carries `//go:build go1.22`, so the file was excluded from the build and `go test -run TestPathValue` matched nothing and exited 0 | bumped to 1.22 |
| `go_chi/26,27,56` | `go test -run <pat>` **exits 0 when the pattern matches nothing**, and the runner read exit 0 as success — a feature could score with no implementation. Filter was also unanchored, so an agent-authored `TestFooExtra` satisfied it | anchored to `^(...)$`; run under `-v` and fail if no `=== RUN` line appears |
| `typst/6554` | runner hardcoded `-- string-first` / `-- string-last`, which only feature 7 uses. Features 2,3,4,5,6,8,9,10 declare `str-*` blocks that were **never selected** | filters derived from the test patch, full-suite fallback for patches that add cases to an existing block, zero-test run fails |
| `dspy/8394` | a missing secondary test target printed "Skipping" and passed | exits 1 |
| `dottxt/1706` | success condition grepped for the bare string `passed`, which also matches `0 passed, N skipped` — an all-skipped run scored as a pass | requires `[1-9][0-9]* passed` |

Checked and **sound**, no change needed:

- **Every Python task** — `pytest` exits 5 on "no tests collected" and 2 on a collection error, both
  non-zero, and every runner is under `set -e` or checks the code explicitly. `dottxt/1706` reads
  `PIPESTATUS[0]` correctly through its `tee`.
- **Both react_hook_form tasks** — jest exits 1 when its path pattern matches no test
  (no `--passWithNoTests`), and all 11 features' tests land in the file the runner names.
- **File targeting across all 199 features** — every feature's `tests.patch` writes to a file the
  task's runner actually executes. The only apparent mismatches were `openai_tiktoken/0`, whose
  runner targets the `tests/` directory rather than named files.

Separately, `--maxfail=1` was removed from `dspy/8394, 8563, 8587, 8635`. It does not mis-grade,
but it stops at the first failure so the reported `tests_failed` is a floor rather than a count —
in one observed run it left 2 of 14 tests unexecuted. `timeout 300` already bounds these.

---

# Per-feature log

## dottxt_ai_outlines/1655 feature 7 — `credit_card` · ✅ no change

**Spec says:** *"Valid credit card numbers will have either no separators, space xor hyphen
separators."* Plus 16-digit, importable from `outlines.types`, and a usage example rejecting a
15-digit string.

**Hidden test asserts 21 cases.** Judged one at a time:

| assertion | derivable from spec + repo? |
|---|---|
| 16 contiguous digits valid | yes — "16-digit", usage example |
| `1234-5678-9012-3456` valid | yes — usage example |
| `1234 5678 9012 3456` valid | yes — "space xor hyphen" |
| all-zeros / all-nines valid | yes — digits are digits |
| **mixed separators invalid** (`1234-5678 9012-3456`) | **yes** — this is what "xor" means, and the word is used correctly |
| 15 and 17 digits invalid | yes — "16-digit"; example rejects 15 |
| incomplete/missing group invalid | yes — follows from 4 groups of 4 |
| dots, underscores invalid | yes — "space xor hyphen" is an exhaustive enumeration |
| letters invalid | yes — "digit" |
| empty string invalid | yes |
| doubled separators invalid (`1234--5678…`) | yes — the natural grouped pattern rejects them |
| matching is `fullmatch`, not `search` | yes — **from the repo**: the pre-existing body of `test_type_regex` in `tests/types/test_custom_types.py` reads `re.fullmatch(regex_str, test_string)`, and agents can read that file |

**Verdict: sufficient, no change.** The earlier automated audit flagged "xor" as thin. I disagree:
it is unusual phrasing but semantically exact, and it is the only word needed to forbid mixing.
Empirically 3 of 5 archived agent regexes scored 21/21 on these cases — and the bar here is a
supremely intelligent reasoner, not the median model, so I am deliberately **not** rewording it
for clarity. Doing so would make the task easier rather than fix a defect.

**Noted, not fixed:** `tests.patch` carries a stale header `# Credit card tests - mixed separators
(should match)` with zero tuples beneath it, two lines above the cases asserting the exact
opposite. Author hygiene from a flipped expectation. Agents never see this file, so it changes no
outcome; worth deleting on any future pass through the test file.

## dottxt_ai_outlines/1655 feature 10 — `hash_sha256` · ✅ no change

**Sufficient — in fact over-complete.** `feature.md` contains
`# Returns: {"type": "string", "pattern": "[a-fA-F0-9]{64}"}`, byte-identical to the gold
implementation `hash_sha256 = Regex(r"[a-fA-F0-9]{64}")`. Every hidden assertion (63/65-char
edges, case-insensitivity, non-hex characters, empty string, whitespace) is additionally spelled
out in the spec's own "Validation Behavior" and "Test Coverage" sections.

**Not changed, deliberately.** This is the opposite failure mode — the spec hands over the answer,
so the feature reduces to copying one string. But an over-complete spec makes a feature *easy*,
not *broken*: it still grades correctly and it discriminates nothing. Removing the pattern would
make a currently-reliable feature harder, which is a change to benchmark difficulty rather than a
repair, and it would destabilise one of the few features that passes consistently. Flagged for a
decision rather than acted on.

## dspy/8394 feature 3 — per-entry TTL · ✅ no change

**Sufficient, and close to over-complete.** The spec names the concrete mechanism at every point
the test probes: `TTLCache(maxsize=memory_max_entries, ttl=ttl)` and where to import it from,
`LRUCache` as the fallback, `disk_cache.set(key, value, expire=ttl)`, `disk_cache.expire()` on the
miss path, `set_ttl` in `__all__`, and `Cache(..., ttl=)` / `configure_cache(ttl=)`.

Every hidden assertion maps onto a stated sentence:

| assertion | where stated |
|---|---|
| `isinstance(memory_cache, TTLCache)` when ttl set | Solution 2 |
| `memory_cache.ttl == ttl_seconds` | Solution 2 |
| entry present before expiry, `None` after, both layers | Considerations: "must expire from both layers after `ttl+ε`" |
| `isinstance(memory_cache, LRUCache)` and **not** `TTLCache` when `ttl=None` | Solution 2 "fall back to LRUCache"; `TTLCache` is not an `LRUCache` subclass in cachetools, so this follows |
| `len(memory_cache) == 0` after expiry | TTLCache semantics |
| `set_ttl` / `configure_cache(ttl=)` / `Cache(ttl=)` all work | Solution 1, 5, 6 and Considerations, individually |

Passed in the teacher-v2 run (10/10 primary + 4 TTL). No change.

## dspy/8394 feature 4 — cache statistics · ✅ no change (earlier amendment reverted)

I had amended this spec, then reverted it. Recording why, since the reverted text is the clearest
illustration of the criterion.

**Amendment 1, "returns a populated dict even when the only event was a miss".** The spec already
says *"Returns `None` when tracking is disabled"*. Read closed-world — and a supremely intelligent
reasoner does read it that way — disabled tracking is the *only* stated `None` case, so tracking
enabled implies non-`None`. The observed implementation invented a second case (`None` when no
events recorded). **Derivable. Reverted.**

**Amendment 2, "tracking applies to any `Cache` instance, not only `dspy.cache`".** The hidden
tests do call `.stats()` on an ordinary `Cache` fixture rather than the singleton, and the spec
only ever writes `dspy.cache.stats()`. But `dspy.cache` *is* a `Cache`, `stats()` is described as a
method, and no reasonable implementation attaches a method to one instance rather than the class.
Instrumenting `Cache.get()` covers every instance for free. **Derivable. Reverted.**

**What actually failed** in the observed run: `CacheUsageTracker.record_hit` / `record_miss` were
defined and never called — `Cache.get()` bumped its own counters and never looked the tracker up,
so `get_stats()` returned `None` and `Prediction._cache_usage` stayed unset. The spec requires the
producer side plainly (*"returns cache events that occurred during that specific prediction's
execution"* — you cannot return events you never recorded). **A model failure, not a spec gap.**

## go_chi/26 feature 1 — `http.Request.PathValue` · ✅ no change

**Sufficient.** The one non-obvious requirement is ordering: the test reads `r.PathValue(pathKey)`
*inside* the handler, so chi must populate the values before dispatching. The spec carries this
with one word — *"`http.Request.PathValue` will act as an **alias** for `chi.URLParam`"*, restated
as *"make `r.PathValue(...)` act as an alias for `chi.URLParam(r, ...)` when routing with chi"*.
`chi.URLParam` exists to be called inside a handler; something that only becomes valid after the
handler returns is not an alias for it. Derivable.

Everything else is stated outright: conditional compilation for pre-1.22 (`path_value.go`,
`path_value_fallback.go` are named in Files Modified), single-segment matches only, and the
`method + pattern` registration form is pre-existing chi behaviour.

This feature was **ungradeable** until the Dockerfile fix — see the gradeability section. Its spec
was never the problem.

## huggingface_datasets/6252 feature 4 — `crop_margin` · ✅ no change

**Sufficient.** Three assertions, all stated:

- `Image(crop_margin=10)` accepted → *"optional `crop_margin` parameter to the `Image` feature"*
- `size == (640-20, 480-20)` → *"crop a fixed number of pixels from **each edge**"* / *"uniform
  margin (specified in pixels) from **all four sides**"*. The 2× per dimension is arithmetic, not
  a hidden requirement.
- applied at decode time → *"during decoding"*

`mode == "RGB"` is unchanged base behaviour. Passed 44/44 in every archived run.

## huggingface_datasets/6252 feature 6 — CMYK→RGB · ✅ no change

**Sufficient.** *"Detect CMYK images during feature decoding and transparently convert them to
RGB"* covers the only real assertion (`img.mode == "RGB"`). Size preservation is implied by a
colour-space conversion and asserted only as a sanity check. Passed 44/44 in every archived run.

## openai_tiktoken/0 feature 6 — `transformers` · ✅ no change

**Sufficient.** Parameter name, list form, and — the only subtle part — order of composition are
all stated: *"applied sequentially in the order given, with each transformer receiving the output
of the previous one"*, with a two-function example. The test exercises exactly that, including
both orderings. Unchanged behaviour without the parameter is ordinary backward compatibility.

## openai_tiktoken/0 feature 8 — `return_repeated_pattern` · ✅ no change

**Sufficient.** The expected dict looked like a magic constant, so I recomputed it from the spec.
Text is `" like love"*99 + " like hate"*20`; token ids are ` like`=1093, ` love`=3021,
` hate`=12491. Adjacent pairs over the whole sequence:

| pair | count |
|---|---|
| (1093, 3021) | 99 |
| (3021, 1093) | 99 — 98 within the `love` block plus 1 at the transition |
| (1093, 12491) | 20 |
| (12491, 1093) | 19 — falls outside the top 3 |

The asserted `{(1093,3021):99, (3021,1093):99, (1093,12491):20}` is precisely *"all adjacent token
pairs … top 3 most frequent"*, both of which the spec says. Ties do not arise. Empty text → `{}`
follows from "top 3" meaning at most 3. The return shape `(tokens, dict)` and the key form
`(token1, token2)` are both in the spec's own example.

**The v2 regression here was not a spec gap.** Agent 2 wrote a correct implementation, published it
twice, then `git push -f` rewound its branch to the base commit and never opened a PR — a
publication failure after six reversals of who owned `def encode(...)`.

## pallets_click/2800 feature 1 — close contexts in completion · ✅ no change

**Sufficient, over-complete.** The spec names the function (`_resolve_context`), all three call
sites to wrap, the exact construct (`with cli.make_context(...) as ctx:`), the symptom
(`ResourceWarning: unclosed file`), and both files. The test asserts only that no warnings are
raised. Effectively an implementation description.

## pallets_click/2800 feature 7 — `validate_nesting` · ✅ no change

**Sufficient, over-complete.** The spec carries an explicit **API Specification** section, and
every hidden assertion has a line in it:

| assertion | stated as |
|---|---|
| `validate_nesting=False` default, normal usage unaffected | *"Default value is `False`"* |
| attribute readable on a Context | *"must be publicly accessible on Context instances"* |
| child inherits `True` from parent | *"all child contexts created via `_make_sub_context()` automatically inherit this setting"* |
| grandchild inherits too | same clause, applied transitively |
| `RuntimeError` if a child is still open at parent close | *"If a parent context is closed while any child contexts remain open, a `RuntimeError` must be raised"* |
| closing children first is fine | *"Child contexts must be removed from parent tracking when they are closed"* |

**Task-design note, not a spec defect.** f7's spec directs the implementer to set
`validate_nesting=False` inside `_resolve_context` — the very call sites f1 is wrapping in `with`
blocks. The two features are therefore *ordered* onto the same statements by their specs. This is
a pairing choice, not something a spec edit can repair.

## pallets_jinja/1559 feature 4 — `trans` metadata · ✅ no change

**Sufficient, over-complete.** The spec states the attribute name (`trans_metadata`), the
empty-dict rule (*"If metadata dict is empty, do **not** add the attribute"*), `hasattr`
accessibility, the three allowed parameter names, and — unusually — the **exact error strings**:
`"metadata parameter '{name}' must be a string literal."` and
`"metadata parameter '{name}' requires a value assignment."`, which the tests then match on. The
remaining assertions (empty string values, non-ASCII, long strings) follow from "string literal".
Interaction with variables / `trimmed` / pluralization is covered by the Integration Requirements.
Passed 61/61 in both runs.

## pillow/290 feature 4 — `error_threshold` · ✅ spec fine · ⚠️ test defect, deferred

**Spec sufficient.** *"maximum allowed mean square error (MSE) between the original image and the
quantized version"*, *"stop adding colors once the error is below this threshold"*, and *"If the
error threshold cannot be met with the specified `colors` limit, it will use the maximum allowed
colors"* cover every behavioural assertion — including the one that reads backwards at first
glance, `low_threshold_colors > with_threshold_colors` (threshold 20 is stricter than 30, so it
needs *more* colours), and `impossible_colors == 10`.

**But two of its assertions compute nothing.** `np.array()` of a PIL RGB image is `uint8`, so
`(image_array - quantized_array) ** 2` wraps twice. Measured directly:

```
channels 100 vs [84, 110, 228]
uint8 : diff [16, 246, 128] -> squares [0, 100, 0]      -> mean    33.3
int   : squares [256, 100, 16384]                        -> mean  5580.0
```

A per-channel error of 16 contributes **0**, and 128 contributes **0**. `with_threshold_mse <=
30.0` and `low_threshold_mse <= 20.0` are therefore comparing a wrapped, arbitrary number against
a threshold — not merely weaker than intended, but capable of passing or failing by luck.

**Fix is `(a.astype(int) - b.astype(int)) ** 2` — deliberately NOT applied yet.** Correcting the
arithmetic raises the computed value by two orders of magnitude, and a 256-colour quantization of
`hopper()` may well exceed the literal `30.0` / `20.0` thresholds. That would break the gold patch,
which the rules forbid. This needs one container run to measure the true MSE and re-tune the two
thresholds together. Queued for the execution pass.

The colour-count assertions carry the real behavioural signal and are unaffected.

## Correction: go_chi/26 feature 2 — "Files Modified" edit reverted

I had removed `metrics_test.go` from Files Modified, on a claim that following it literally makes
`git apply` reject the whole patch. **I verified that claim and it is false.**
`_filter_test_files` (`src/cooperbench/eval/sandbox.py:693`) drops whole `diff --git` sections for
test paths and re-terminates the patch, so a test-file hunk is silently discarded, never rejected.
`pillow/290` f4 lists `Tests/test_image_quantize.py` in its own Files Modified and is a sound
feature, so this is the dataset's normal convention.

Removing it would have told the agent *not* to write tests for its own feature — the opposite of
what we want. Reverted. The API-names half of that edit stands on its own evidence.

## pallets_jinja/1621 feature 5 — path caching · ✅ spec fine · ✏️ tests rewritten

**Spec sufficient.** It names the attribute `_path_cache`, its type
`Dict[str, Optional[str]]`, the `None`-means-searched-and-missing convention, and the cache
check / hit / miss / populate phases individually.

**The tests were the defect: all six passed on the unmodified base.** Not one assertion observed
the cache — they checked that `tmpl1.render() == tmpl2.render()`, that the first search path wins,
and that backward compatibility holds, all of which are true with no cache at all. The
"performance" test closed with `assert cached_time >= 0` and `assert fresh_time >= 0`, which are
vacuous. So the feature scored 5/5 across archived runs regardless of whether it was implemented,
including once for an agent whose patch contained no implementation.

**Replaced the vacuous performance test with three that observe the contract.** Written against
`loader.get_source()` rather than `env.get_template()`, deliberately: the `Environment` keeps its
own template cache, which would mask the loader's behaviour.

| new test | asserts | why base fails |
|---|---|---|
| `..._records_resolved_path` | `_path_cache` starts `{}`; after a successful lookup it holds a real file path | base has **0** occurrences of `_path_cache` in `loaders.py` → `AttributeError` |
| `..._records_miss_as_none` | a failed lookup stores `None`; creating the file afterwards does **not** make it findable, because the cached miss answers without touching the filesystem | same |
| `..._drops_cached_path_when_file_vanishes` | after the cached file is deleted the entry is re-resolved and re-cached as `None`, rather than the stale path being reused | same |

Each traced line-by-line against `feature.patch` before being written — the second encodes gold's
*permanent* negative caching (`if cached_filename is None: raise TemplateNotFound`) and the third
encodes its delete-and-re-resolve path (`open_if_exists` → `None` → `del self._path_cache[...]`).
An earlier automated audit called the permanent negative caching a regression in gold; it is in
fact exactly what the spec prescribes, so the test encodes it rather than contradicting it.

**Verification done:** cloned jinja at the task's base commit `a2920752`, confirmed the rewritten
`tests.patch` applies cleanly to base and that `feature.patch` still applies on top, and
regenerated the hunk header from a real `git diff` rather than hand-editing it. **Not yet done:**
executing the suite — running the cloned repo's tests is blocked in this environment. Queued for
the same execution pass as pillow/290 f4.

## go_chi/26 feature 2 — route metrics · ✏️ amended

**The spec names the types but not the fields the test reads.** It says *"Create `RouteMetric`
struct containing pattern, method, path, duration, and URL parameters"* and *"Provide
`SimpleMetricsCollector` with callback function"* — prose descriptions, not identifiers. The
hidden test reads `collectedMetric.Pattern`, `.Method`, `.Duration`, `.Params["id"]` and
constructs `&SimpleMetricsCollector{OnHit: func(metric RouteMetric) {...}}`.

`Pattern`, `Method` and `Duration` follow from the prose. **`Params` and `OnHit` do not**, and
`Params` is worse than merely unstated: chi's own vocabulary for this concept is `URLParams`
(`RouteParams.URLParams`), which a reasoner reading the repo would naturally follow — straight
into the wrong name. *"callback function"* gives no purchase on `OnHit` at all.

**Added** a Public API block giving the two struct shapes, with a note that it is `Params`, not
`URLParams`. Public identifiers are contract, not implementation — a real PR description states
its API. Nothing about instrumenting `routeHTTP()`, measuring duration, or wiring the collector is
given away.

## pallets_jinja/1559 feature 8 — `ctrans` · ✏️ amended

Two arbitrary grammar decisions, neither derivable:

1. **Commas as separators.** The spec's only grammar line is
   `{% ctrans condition [variables] [trimmed|notrimmed] %}` — space-separated. The test uses
   `{% ctrans use_i18n, count %}`. `grep -ic comma feature.md` → the word never appears. Nothing
   in the repo helps: the tag is new, so there is no existing `ctrans` to imitate.
2. **`trimmed` covering the fallback block.** The test asserts
   `{% ctrans False trimmed %}…{% fallback %}  Fallback  \n  Text  {% endctrans %}` renders
   `"Fallback Text"`. The spec lists `trimmed` among the header modifiers but never says which
   blocks it reaches, and the standard `trans` tag has no fallback block to generalise from.

**Added** both as bullets under the grammar line. **Removed from my own first draft** a
parenthetical explaining that copying `trans`'s comma handling puts the separator one slot too
early — that is diagnostic guidance about where the bug will be, not a requirement. It was a leak
and it is gone.

## pillow/290 feature 5 — `lightness_factor` · ✏️ amended

**One arbitrary requirement.** `feature.md` has zero matches for
`valueerror|raise|invalid|must be|validat|error`; `tests.patch` contains
`with pytest.raises(ValueError)` twice, for `lightness_factor=-0.5` and `=0`. Raising is a design
decision, not a deduction — clamping a non-positive factor to a floor, or letting it produce a
black palette, are both defensible readings of *"a value less than 1.0 makes it darker"*.

The boundary matters and I checked it before writing: the same test requires `0.1`, `0.9`, `1.1`
and `2.0` to **work**. So the rule is exactly `<= 0` raises, with no upper bound. A generic
"validate your parameters" instinct that rejected `0.1` or capped at `1.0` would break passing
assertions — which is why the sentence states the precise boundary.

**Deliberately not added: the `putpalette()` mechanism.** The other half of this feature's failure
is that assigning `im.palette.palette` never reaches the C image, so the adjustment is a silent
no-op. But the spec already promises *"1.1 for 10% brighter"* — observable behaviour at a public
boundary. An implementation that changes nothing violates a promise the spec already makes, so
that is a verification failure by the implementer, not a gap in the spec.

## typst/6554 features 7 and 8 — exact diagnostics · ✏️ amended

typst's test harness compares diagnostic text **verbatim** (`tests/src/run.rs`,
`message != note.message`), so the wording is load-bearing, and neither wording is derivable:

| feature | required message | what the spec said |
|---|---|---|
| 7 | `no character matches pattern "alpha"` | *"propagate the existing … out-of-pattern errors"* |
| 8 | `unknown case option "spongebob"` | *"Invalid values raise a descriptive error"* |

Both strings are introduced *by the reference patch*, so by construction they cannot be found in
the base repo, and the repo's own convention (`unknown variable: {}`, unquoted, colon-separated)
points away from the quoted form.

**Verified against gold before writing**, so that the spec describes what the reference actually
emits: `eco_format!("no character matches pattern \"{}\"", pattern)` and
`eco_format!("unknown case option \"{}\"", other)`. Both statements match exactly.

Feature 7 also gained one ordering sentence — the empty-string check must precede the pattern
check, so `"".last(pattern: "alpha")` still reports `string is empty`. Without it, fixing the
message alone un-masks a second failure.

**Deliberately not stated:** gold's third message, `unknown pattern "{}"` for an unrecognised
pattern name. No test exercises it, and specifying untested behaviour adds obligation without
adding measurement.

The alternative to all of this — relaxing `run.rs` to substring matching — is the more principled
fix, since grading a coding agent on exact prose is measuring the wrong thing. It is not taken
here because `run.rs` lives in the cloned repo rather than the dataset, so changing it would alter
the harness for all 2,354 of typst's own tests.

---

# Per-feature log, part 2 — the remaining 169

The first 30 were judged one at a time by reading every assertion. That does not scale to 169, so
detection was mechanised (`scripts/spec_audit_survey.py`) while the judgement stayed manual. The
survey flags three things per feature and nothing else:

- a **public symbol** the reference introduces, the test uses, and the spec never names
- an **error message** the test matches on that the spec never gives, tagged `GOLD` if the
  reference is where it comes from
- an **exception type** the test expects that the spec never names

Everything it flags is a candidate. **Roughly half turned out to be derivable on inspection**, and
the recurring false positives are worth naming because they would otherwise look like defects:

| false positive | why it is not a gap |
|---|---|
| symbol appears on a **context** line of the patch | pre-existing API — `ToolCall`, `convert_input_schema_to_tool_args`, `to_info_dict`, `invoke`, `Argument`, `DocumentBlock`, `AudioBlock`, `decode_example` |
| message tagged `elsewhere` rather than `GOLD` | raised by the **test's own stub**, so it is not a requirement — click/2800 f4-f5, click/2956 f2, dottxt/1706 f6-f7, jinja/1621 f6 |
| generic exception type | `AssertionError` is dirty-equals' whole idiom; bare `Exception` is a catch-all any error satisfies; `OSError`/`TypeError` were pre-existing library behaviour |
| name collision | jinja/1621 f6 flagged `transform_path`, which is the **test's own local function** passed as `path_transform=` — the parameter the spec does name |

## Amendments made in part 2

| feature | what was not derivable |
|---|---|
| `go_chi/26` f3 | `RouteSelector` interface, `WithRouteSelector`, `NewVersionSelector`, `NewRoleBasedSelector`, `AddHandler` — the spec described dynamic routing in prose and named **no identifier at all** |
| `go_chi/26` f4 | `WithPriority` and the `Priority*` constants — likewise |
| `go_chi/27` f3 | `EnableDebugLogging()` — introduced by the reference, spec said only "adds logging infrastructure" |
| `hf_datasets/3997` f2, f3 | `set_custom_decoding_criteria(custom_criteria=None)` — introduced by the reference, spec described the capability but named no method |
| `typst/6554` f2, f4, f5, f6, f10 | exact diagnostics, verbatim-compared by the harness. f6's is the subtle one: the spec says all-whitespace strings "continue to raise the existing errors", but `string contains only whitespace` is **introduced by the patch** and is not pre-existing |
| `click/2068` f2 | `{editor}: Editing timed out after {timeout} seconds` — and the count is not pluralised, so one second reads "after 1 seconds" |
| `click/2068` f3 | `Failed to create backup file: {error}` |
| `click/2068` f8 | the test inspects `Popen` kwargs for `preexec_fn=os.setpgrp`; `start_new_session=True` is equally correct on POSIX, so a reasoner picks one at 50/50 |
| `dottxt/1706` f4, f5 | exact `ValueError` texts — the conditions were stated, the wording was not |
| `llama_index/18813` f2 | `{method} exceeds maximum allowed size ({size} > {max_bytes})` |
| `dirty_equals/43` f5 | country codes are matched **case-insensitively** — the reference calls `.upper()`, the spec was silent, and the test block is literally commented "Case insensitive country codes" |
| `dirty_equals/43` f6 | issuer tokens are exactly `'Visa'`, `'Mastercard'`, `'AmericanExpress'` — the spec wrote "American Express" with a space, which is not what is accepted |

## One left deliberately unchanged, and why

`dottxt/1655` f8 (`ssn`) accepts `000-00-0000` and `999-99-9999` as **valid**, so it is a pure
`XXX-XX-XXXX` format check rather than real SSN validation, which rejects the 000, 666 and 9xx
area ranges. A strong reasoner might apply the real rules and fail. I left it: the spec says
"format" three times, gives the `XXX-XX-XXXX` template, and never mentions area or group numbers,
so format-only is what is asked. Flagged here because it is the closest call in the whole audit.

## Shape of the corpus

174 of 199 specs needed nothing, and a large fraction are **over-complete** rather than thin —
`react_hook_form` names prop types, file paths and even `React.useRef<NodeJS.Timeout | null>`;
`dspy/8394` f2 gives the exact `sha256(namespace.encode()).hexdigest()[:8]` key formula;
`jinja/1559` f4 quotes its own error strings; `click/2800` f7 ships an API Specification section.

The defects concentrate by **language and by author habit**, not evenly:

- **Go tasks are the worst** for unnamed API. Three of the four `go_chi` features that needed
  amendment described behaviour in prose and named no identifier, and in one case the repo's own
  vocabulary (`URLParams`) actively misleads toward the wrong name.
- **Rust (typst) is the worst for messages**, because its harness compares diagnostics verbatim —
  7 of 10 features there needed the exact text.
- **Python tasks are largely fine**, and where they are not it is error wording rather than API.

---

# Execution pass — both pending items resolved on Modal

Two items were left needing a container run because the fix could have broken the reference. Both
now verified against the real task images. The criterion for a gradeable feature is two runs:
**tests + gold must pass, tests alone must fail.**

## pillow/290 feature 4 — the reference does not honour its own threshold

Measured with the gold patch applied, printing both the wrapped and the correct MSE:

| quantization | wrapped (what was asserted) | correct | colours |
|---|---|---|---|
| `quantize(256)` | 18.43 | 27.16 | 256 |
| `quantize(256, error_threshold=30.0)` | 29.80 | **59.34** | 90 |
| `quantize(256, error_threshold=20.0)` | 19.84 | **29.85** | 203 |

The old assertions were `with_threshold_mse <= 30.0` and `low_threshold_mse <= 20.0`. Under correct
arithmetic those are **59.34** and **29.85** — so the reference implementation does not actually
keep the error under the requested threshold. The assertions passed only because `uint8`
subtraction and squaring wrap, and the wrapped values happened to land just under the limits.

That rules out the obvious repair. Re-tuning to `<= 60` / `<= 30` would preserve gradeability while
asserting a number with no relationship to the stated semantic. What *is* both meaningful and true
of the reference is the ordering: a stricter threshold leaves less error and keeps more colours.

**Rewritten to** `assert high_quality_mse < low_threshold_mse < with_threshold_mse`, on
`int64`-cast arrays, keeping the colour-count assertions untouched. Verified:

```
WITH gold:     1 passed, 8 deselected
WITHOUT gold:  1 failed — TypeError: Image.quantize() got an
               unexpected keyword argument 'error_threshold'
```

**Not fixed, and deliberately so:** the reference's threshold semantics. Correcting it would mean
changing `feature.patch`, which the rules forbid — a task whose reference is wrong is a task-design
question, not a repair. Recorded here so it is not rediscovered as a model failure.

## pallets_jinja/1621 feature 5 — the rewritten tests do what they claim

```
WITH gold:     48 passed
WITHOUT gold:  3 failed, 45 passed
               AttributeError: 'FileSystemLoader' object has no attribute '_path_cache'
               - test_filesystem_loader_records_resolved_path
               - test_filesystem_loader_records_miss_as_none
               - test_filesystem_loader_drops_cached_path_when_file_vanishes
```

Exactly the three tests added, and only those. Before the rewrite all six of this feature's tests
passed on the unmodified base, so it scored 5/5 across archived runs regardless of whether anything
was implemented — including once for an agent whose patch contained no implementation at all.

---

# Gradeability sweep — running every feature's tests with and without gold

The spec audit asks whether a requirement is *knowable*. It cannot tell you whether the tests
actually measure the feature, or whether the reference even passes them. So all 199 were run twice
on Modal through `runner.sh`: **tests alone must FAIL, tests + gold must PASS**
(`scripts/check_gradeable.py`).

**Result: 175 OK, 24 broken (12%).** None of the 24 were findable by reading specs.

## The two dependency-drift failures — both bricked whole tasks

**`dspy/8563`, all 6 features.** On a pristine checkout, with no patches at all:

```
4 failed, 16 passed
dspy/clients/base_lm.py:79:  "usage": dict(response.usage)
AttributeError: 'ModelResponse' object has no attribute 'usage'
```

`ModelResponse` is a litellm type, and dspy declares `litellm>=1.64.0` with **no upper bound**, so
the image floated to a version that dropped `.usage`. The runner executes
`tests/adapters/test_chat_adapter.py` in full, so those four unrelated failures fail every feature
of the task regardless of what an agent writes.

Pinned to `litellm==1.77.1` in all four dspy Dockerfiles → **20 passed**. Note the traceback
surfaces inside `pydantic/main.py`, and pydantic 2.9.2 / 2.10.6 / 2.11.7 all reproduce it
identically — reading the frame above the top one is what found the real culprit.

**`huggingface_datasets/7309`, both features.** `test_parquet_read_geoparquet` fails
`assert 'large_string' == 'string'` on a pristine checkout. Pinning does **not** help: identical at
pyarrow 17.0.0, 18.1.0 and 19.0.1. The test is unrelated to either feature (parquet streaming
filters / sorting), so the runner now deselects it → `51 passed, 1 deselected`. Deselecting shrinks
coverage and is the weaker fix, but the alternative was leaving both features permanently
unscoreable.

Both are the same shape: **a runner that executes whole test files inherits every unrelated
failure in them.** That is also why a feature's score currently depends on its partner not breaking
anything in the same file — a real source of noise in the coop setting, where both patches land in
one tree.

## Deploying a runner or Dockerfile fix

Neither takes effect by editing this repo. `runner.sh` is `COPY`'d into the image at build time, so
until the image is rebuilt and pushed the old copy runs — 9 of the 24 failures were my own fixes
looking like defects for exactly this reason. Three further traps, all hit once:

1. **Architecture.** The published images are multi-arch (amd64 + arm64). A plain `docker build` on
   Apple Silicon produces arm64 only; pushing that breaks every amd64 run. Always `--platform
   linux/amd64,linux/arm64`.
2. **Modal caches by tag.** `Image.from_registry(tag)` is resolved once and cached, so a freshly
   pushed image under an unchanged tag is ignored. Run once with `MODAL_FORCE_BUILD=1` to refresh.
   Without this, a correct fix looks like it did nothing — verified on `go_chi/26`, which only
   went green after the force-build.
3. **Do not full-rebuild for a runner-only change.** It re-resolves every dependency — which is how
   these images drifted into being broken — and takes tens of minutes under emulation. Overlay one
   `COPY` layer on the published image instead. `scripts/rebuild_task_images.sh` picks the mode
   automatically from whether the Dockerfile changed.

`go_chi/26` is the worked example of the whole chain: `golang:1.21 → 1.22` so the build-tagged test
compiles, the runner fix so a zero-match cannot score, multi-arch rebuild, push, force-build —
after which feature 1 is gradeable for the first time (`base_rc=1, gold_rc=0`).

## Two more dependency-drift failures, found on re-verification

Both were invisible until the images were rebuilt, and both broke a whole task.

### `hf/6252` — scipy dragged forward by an unrelated install

Measured by printing versions after each line of `runner.sh`:

| after | numpy | scipy |
|---|---|---|
| image as shipped | 1.26.4 | 1.13.1 |
| `uv pip install torch jax` | **2.5.2** | **1.18.0** |
| `uv pip install "numpy<2.0"` | 1.26.4 | **1.18.0** |

The runner pins numpy but not scipy, so `jax` pulls both forward and only numpy is pulled back.
scipy 1.18 is built for numpy>=2.0 and calls `np.long`, which numpy removed in 1.24 and
reintroduced in 2.0 — so `tests/features/test_image.py` dies with *"module 'numpy' has no
attribute 'long'"* on base **and** gold, and all 6 features read as ungradeable.

Two wrong attempts are worth recording, because both looked plausible:
`scipy>=1.11,<2.0` in the Dockerfile resolved to 1.18 — the newest under 2.0, i.e. exactly the
broken one — and pinning in the Dockerfile at all is futile, since `runner.sh` re-resolves at test
time. **Fixed** by `uv pip install --system "scipy<1.14"` in `runner.sh`, after the numpy
downgrade. Verified `OK 6`.

### `dottxt/1706` — a live download, then a warning escalated to an error

`tests/backends/*.py` build their fixtures from `erwanf/gpt2-mini`. When that download fails in a
fresh sandbox, 10 tests ERROR identically on base and gold, and `runner.sh:83` fails the run on any
`FAILED|ERROR` — so gold can never pass and all 8 features read as ungradeable. It reproduced in a
single sandbox, so it was never a concurrency artefact; it looked like one because a sandbox that
had already cached the model passed.

Baking the model in took it from 10 errors to 1. The survivor was not a download failure at all:
`pyproject.toml:115` sets `filterwarnings = ["error", ...]`, and huggingface_hub's Xet transfer
calls a deprecated `hf_xet` entry point, so *any* download raises `DeprecationWarning` as an ERROR.

**Fixed** with `ENV HF_HUB_DISABLE_XET=1` plus a cache of both fixture models
(`erwanf/gpt2-mini` and the `TinyMistral-248M` GGUF). Verified `OK 8`.

Both fixes live in the image, so they reach a run through Docker Hub rather than through
`dataset/`. Note that `CooperTrain/dataset` is a **separate copy** from `CooperBench/dataset`:
`runner.sh` and `Dockerfile` changes travel inside the image, but `feature.md` and `tests.patch`
are read from the dataset directory at eval time, so that copy has to be kept in sync.

## Still outstanding

Nothing blocking. Every one of the 199 features has been measured base-fails / gold-passes.

The one item deliberately left unfixed is `pillow/290` f4's reference threshold semantics
(above) — gradeability holds, but the reference does not honour its own `error_threshold`, and
correcting it would mean editing `feature.patch`.
