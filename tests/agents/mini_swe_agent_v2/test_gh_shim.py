"""Tests for the `gh` shim that gives coop agents pull requests over the team remote.

These run the real script against real git repositories — no mocks — because every property
worth asserting here is a property of git refs, and a mocked `git` would assert only that the
script calls the commands the test already assumed it would.

The behaviour that matters most is that a PR **tracks the branch**: agents are told to open one
early so a colleague has context, then keep committing. A PR pinned to the commit that existed
at open time would silently submit a fraction of the work, and nothing downstream could tell.
"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[3] / "src/cooperbench/agents/mini_swe_agent_v2/connectors/gh_shim.sh"


def run(cmd: str, cwd: Path, agent: str | None = None, check: bool = True):
    env = {**os.environ, "PATH": os.environ["PATH"]}
    if agent:
        env["COOPERBENCH_AGENT_ID"] = agent
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError(f"{cmd}\nrc={r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


def gh(args: str, cwd: Path, agent: str, check: bool = True):
    return run(f"sh {SHIM} {args}", cwd, agent=agent, check=check)


@pytest.fixture
def team(tmp_path):
    """A bare 'origin' remote plus two agent clones, mirroring the real sandbox topology."""
    server = tmp_path / "server.git"
    subprocess.run(["git", "init", "-q", "--bare", str(server)], check=True)

    seed = tmp_path / "seed"
    seed.mkdir()
    run("git init -q -b main .", seed)
    run("git config user.email a@b.c && git config user.name t", seed)
    (seed / "shared.py").write_text("def quantize(im):\n    return im\n")
    run("git add -A && git commit -qm base", seed)
    run(f"git remote add origin {server} && git push -q origin HEAD:refs/heads/main", seed)

    clones = {}
    for agent in ("agent1", "agent2"):
        d = tmp_path / agent
        run(f"git clone -q {server} {d}", tmp_path)
        run("git config user.email a@b.c && git config user.name t", d)
        run(f"git fetch -q origin && git checkout -q -B {agent} origin/main", d)
        clones[agent] = d
    return clones


def test_pr_tracks_the_branch_not_the_opening_commit(team):
    """The property agents depend on: open early, keep committing, PR follows.

    If the PR froze at open time, an agent that opened a PR after its first commit would
    submit only that commit — and the omission would be invisible to it, to its colleague,
    and to the grader.
    """
    a1, a2 = team["agent1"], team["agent2"]

    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam 'first'", a1)
    gh("pr create --title 'add error_threshold' --body 'wip'", a1, "agent1")

    # More work lands after the PR was opened.
    (a1 / "extra.py").write_text("HELPER = 1\n")
    run("git add -A && git commit -qm 'second'", a1)
    run("git push -q origin HEAD:agent1", a1)

    diff = gh("pr diff agent1", a2, "agent2").stdout
    assert "error_threshold" in diff, "first commit missing from PR"
    assert "extra.py" in diff, "commit made AFTER opening the PR is missing — PR froze"


def test_colleague_sees_the_pr_and_its_description(team):
    a1, a2 = team["agent1"], team["agent2"]
    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam work", a1)
    gh("pr create --title 'error threshold' --body 'touching quantize()'", a1, "agent1")

    listed = gh("pr list", a2, "agent2").stdout
    assert "agent1" in listed and "error threshold" in listed

    view = gh("pr view agent1", a2, "agent2").stdout
    assert "error threshold" in view
    assert "touching quantize()" in view


def test_pr_can_be_opened_before_any_work_exists(team):
    """Opening early is the intended flow, so an empty PR must not be rejected."""
    a1, a2 = team["agent1"], team["agent2"]
    gh("pr create --title 'taking quantize()' --body 'starting now'", a1, "agent1")
    assert "taking quantize()" in gh("pr view agent1", a2, "agent2").stdout


def test_reading_a_pr_that_was_never_opened_fails_clearly(team):
    r = gh("pr diff agent1", team["agent2"], "agent2", check=False)
    assert r.returncode != 0
    assert "has not opened a PR" in r.stderr


def test_unsupported_subcommands_explain_themselves(team):
    """`gh pr merge` and friends must not look like a transient failure."""
    for args in ("pr merge agent1", "pr comment agent1", "issue list"):
        r = gh(args, team["agent1"], "agent1", check=False)
        assert r.returncode == 2, args
        assert "not available in this environment" in r.stderr
        assert "gh pr create" in r.stderr, "error should name what IS available"


def test_create_requires_a_title(team):
    r = gh("pr create --body x", team["agent1"], "agent1", check=False)
    assert r.returncode != 0
    assert "--title is required" in r.stderr


def test_uncommitted_work_is_not_in_the_pr(team):
    """The agent's control over its submission: commit what you mean, leave the rest."""
    a1, a2 = team["agent1"], team["agent2"]
    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam work", a1)
    (a1 / "scratch.py").write_text("print('DEBUG')\n")  # never committed
    gh("pr create --title t --body b", a1, "agent1")

    diff = gh("pr diff agent1", a2, "agent2").stdout
    assert "error_threshold" in diff
    assert "scratch.py" not in diff and "DEBUG" not in diff


def test_checkout_gives_the_peers_code(team):
    a1, a2 = team["agent1"], team["agent2"]
    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam work", a1)
    gh("pr create --title t --body b", a1, "agent1")

    gh("pr checkout agent1", a2, "agent2")
    assert "error_threshold" in (a2 / "shared.py").read_text()


def test_two_agents_open_prs_independently(team):
    """One ref per agent, so simultaneous opens cannot race or clobber each other."""
    a1, a2 = team["agent1"], team["agent2"]
    for d, agent, fn in ((a1, "agent1", "one.py"), (a2, "agent2", "two.py")):
        (d / fn).write_text("x = 1\n")
        run(f"git add -A && git commit -qm {agent}", d)
        gh(f"pr create --title {agent}-work --body b", d, agent)

    listed = gh("pr list", a1, "agent1").stdout
    assert "agent1" in listed and "agent2" in listed
    assert "two.py" in gh("pr diff agent2", a1, "agent1").stdout
    assert "one.py" in gh("pr diff agent1", a2, "agent2").stdout


def test_plain_git_push_reaches_the_agents_own_branch(team, tmp_path):
    """The prompt tells agents to run a bare `git push`, which only works because
    `GitConnector.setup` leaves them on a branch named after their agent id with an
    upstream already set (`git checkout -b <id>` + `git push -u origin <id>`).

    That coupling is invisible: change setup to a detached HEAD or a differently-named
    branch and the prompt's instruction silently stops publishing anything, which is
    precisely the failure mode that produced 0 pushes across 117 trajectories.
    """
    a1 = team["agent1"]
    run("git push -u -q origin agent1", a1)  # what setup() does

    assert run("git rev-parse --abbrev-ref HEAD", a1).stdout.strip() == "agent1"
    assert run("git rev-parse --abbrev-ref @{u}", a1).stdout.strip() == "origin/agent1"

    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam work", a1)
    run("git push", a1)  # exactly what the prompt says

    assert (
        "error_threshold"
        in run("git fetch -q origin && git --no-pager diff origin/main origin/agent1", team["agent2"]).stdout
    )


def test_cannot_open_a_pr_from_a_colleagues_branch(team):
    """`gh pr checkout <peer>` leaves you on their code. Opening a PR from there would
    submit their work as yours, and the merge would see the same edit twice — a wrong
    submission that nothing downstream could detect."""
    a1, a2 = team["agent1"], team["agent2"]
    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam work", a1)
    gh("pr create --title t --body b", a1, "agent1")

    gh("pr checkout agent1", a2, "agent2")  # agent2 now sits on pr-agent1
    r = gh("pr create --title stolen --body b", a2, "agent2", check=False)
    assert r.returncode != 0
    assert "not your own branch" in r.stderr
    assert "git checkout agent2" in r.stderr


def test_solo_can_open_a_pr_against_a_local_bare_repo(tmp_path):
    """Solo has no shared server, but must not get a second submission mechanism.

    Two paths are how solo broke silently while coop was being changed: extraction moved to
    the PR, solo kept being told to write patch.txt, and nothing read it. A bare repo inside
    solo's own sandbox gives it the identical flow.
    """
    work = tmp_path / "repo"
    work.mkdir()
    run("git init -q -b main .", work)
    run("git config user.email a@b.c && git config user.name t", work)
    (work / "shared.py").write_text("def quantize(im):\n    return im\n")
    run("git add -A && git commit -qm base", work)

    # what GitConnector.setup does when there is no shared server
    bare = tmp_path / "solo_team.git"
    run(f"git init -q --bare {bare}", work)
    run(f"git remote add origin {bare}", work)
    run("git push -q origin HEAD:refs/heads/main", work)
    run("git checkout -q -b agent1 && git push -u -q origin agent1", work)

    (work / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam work", work)
    run("git push -q", work)
    gh("pr create --title solo --body b", work, "agent1")

    diff = run("git --no-pager diff origin/main origin/agent1", work).stdout
    assert "error_threshold" in diff
    assert "solo" in gh("pr view agent1", work, "agent1").stdout


def test_submission_survives_someone_moving_main(team):
    """The git daemon has no access control, so either agent can push to `main`.

    Diffing a submission against the movable `origin/main` ref meant one such push — accidental
    or not — silently re-baselined BOTH agents' patches. The base is pinned at setup instead.
    """
    a1, a2 = team["agent1"], team["agent2"]
    base = run("git rev-parse HEAD", a1).stdout.strip()

    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam work", a1)
    run("git push -q origin HEAD:agent1", a1)

    # agent2 moves main to its own work — nothing stops it
    (a2 / "other.py").write_text("x = 1\n")
    run("git add -A && git commit -qm hijack", a2)
    run("git push -q -f origin HEAD:main", a2)
    run("git fetch -q origin", a1)

    against_pinned = run(f"git --no-pager diff {base} origin/agent1", a1).stdout
    against_main = run("git --no-pager diff origin/main origin/agent1", a1).stdout

    assert "error_threshold" in against_pinned, "pinned base must still see agent1's work"
    assert "other.py" in against_main, "diffing against main leaks the mover's changes in"
    assert against_pinned != against_main


def test_detaching_upstream_removes_commits_after_the_task_commit(tmp_path):
    """The clone carries the project's future, and that future may contain the answer.

    A task image runs `git clone <upstream> && git checkout <task-sha>`. Every commit after
    the task commit is still present, reachable via refs/remotes and tags — so for a task
    derived from a real PR, `git log --all` can show the upstream implementation of the
    feature the agent is being asked to write. Dropping the remote alone leaves all of it.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    run("git init -q -b main .", upstream)
    run("git config user.email a@b.c && git config user.name t", upstream)
    (upstream / "shared.py").write_text("def quantize(im):\n    return im\n")
    run("git add -A && git commit -qm 'base'", upstream)
    task_sha = run("git rev-parse HEAD", upstream).stdout.strip()
    # the future: the real implementation, plus a release tag
    (upstream / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im  # THE ANSWER\n")
    run("git commit -qam 'feat: add error_threshold'", upstream)
    run("git tag v2.0", upstream)

    sandbox = tmp_path / "repo"
    run(f"git clone -q {upstream} {sandbox}", tmp_path)
    run(f"git checkout -q {task_sha}", sandbox)

    assert "THE ANSWER" in run("git log --all -p", sandbox).stdout, "fixture should leak"

    # what _detach_upstream does
    run("git remote remove origin", sandbox)
    run("git for-each-ref --format='%(refname)' refs/remotes | xargs -r -n1 git update-ref -d", sandbox)
    run("git tag -l | xargs -r git tag -d", sandbox)
    run(
        "git for-each-ref --format='%(refname:short)' refs/heads | xargs -r -n1 git branch -D 2>/dev/null || true",
        sandbox,
    )
    run("git reflog expire --expire=now --all && git gc --prune=now --quiet", sandbox)

    assert "THE ANSWER" not in run("git log --all -p", sandbox).stdout
    assert run("git tag -l", sandbox).stdout.strip() == ""
    assert run("git remote", sandbox).stdout.strip() == ""
    # history before the task commit survives — that is legitimate context
    assert "base" in run("git log --oneline", sandbox).stdout


def test_extracted_patch_applies_to_the_base_and_carries_the_work(team, tmp_path):
    """End-to-end on the artifact grading actually consumes.

    `submitted_patch` runs `git diff <pinned-base> origin/<agent>` and hands the result to the
    evaluator, which applies it to a fresh checkout of the base. Everything upstream can be
    correct — PR opened, branch pushed — and still produce a diff that does not apply, at
    which point the pair is scored `missing_input` and reads as an agent failure.
    """
    a1 = team["agent1"]
    base = run("git rev-parse HEAD", a1).stdout.strip()

    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    (a1 / "helper.py").write_text("LIMIT = 5\n")
    run("git add -A && git commit -qm work", a1)
    (a1 / "scratch.py").write_text("print('DEBUG')\n")  # never committed
    run("git push -q origin HEAD:agent1", a1)
    gh("pr create --title t --body b", a1, "agent1")

    # exactly what GitConnector.submitted_patch does
    patch = run(f"git --no-pager diff {base} origin/agent1", a1).stdout
    assert patch.strip(), "submission is empty"

    # what the evaluator does: apply it to a clean checkout of the base
    fresh = tmp_path / "fresh"
    run(f"git clone -q {a1} {fresh}", tmp_path)
    run(f"git checkout -q {base}", fresh)
    (fresh / "p.patch").write_text(patch)
    run("git apply p.patch", fresh)  # must not raise

    assert "error_threshold" in (fresh / "shared.py").read_text()
    assert (fresh / "helper.py").exists(), "second changed file missing from submission"
    assert not (fresh / "scratch.py").exists(), "uncommitted scratch leaked into submission"


def test_two_submissions_merge_the_way_the_evaluator_merges_them(team, tmp_path):
    """The evaluator applies each patch to its own branch and merges. Disjoint work must
    merge clean and both features must survive — that is the only path that scores a pass."""
    a1, a2 = team["agent1"], team["agent2"]
    base = run("git rev-parse HEAD", a1).stdout.strip()

    for d, agent, fn, body in ((a1, "agent1", "feat_a.py", "A = 1\n"), (a2, "agent2", "feat_b.py", "B = 2\n")):
        (d / fn).write_text(body)
        run(f"git add -A && git commit -qm {agent}", d)
        run(f"git push -q origin HEAD:{agent}", d)
        gh(f"pr create --title {agent} --body b", d, agent)

    p1 = run(f"git --no-pager diff {base} origin/agent1", a1).stdout
    run("git fetch -q origin", a2)
    p2 = run(f"git --no-pager diff {base} origin/agent2", a2).stdout

    ev = tmp_path / "ev"
    run(f"git clone -q {a1} {ev}", tmp_path)
    run("git config user.email a@b.c && git config user.name t", ev)
    # patches live OUTSIDE the repo: `git add -A` would otherwise commit them as part of the
    # work, and they would disappear on the next checkout
    p1f, p2f = tmp_path / "p1.patch", tmp_path / "p2.patch"
    p1f.write_text(p1)
    p2f.write_text(p2)
    run(f"git checkout -q -B agent1 {base} && git apply {p1f} && git add -A && git commit -qm a1", ev)
    run(f"git checkout -q -B agent2 {base} && git apply {p2f} && git add -A && git commit -qm a2", ev)
    run("git merge --no-commit --no-ff agent1", ev)  # must not conflict

    assert (ev / "feat_a.py").exists() and (ev / "feat_b.py").exists()


def test_conflicting_branch_is_reported_when_the_pr_is_opened(team):
    """Both PRs are merged before either feature is tested, so a collision fails BOTH agents.

    Nothing used to tell them until the run was scored. Measured in a real 10-pair run: a pair
    listed each other's open PRs, saw both, and still shipped conflicting edits to the same
    region. Visibility was not the missing piece -- the merge has to be computed for them.
    """
    a1, a2 = team["agent1"], team["agent2"]

    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam a1", a1)
    run("git push -q origin HEAD:agent1", a1)
    gh("pr create --title 'threshold' --body 'x'", a1, "agent1")

    # agent2 edits the SAME line a different way.
    (a2 / "shared.py").write_text("def quantize(im, palette=None):\n    return im\n")
    run("git commit -qam a2", a2)
    out = gh("pr create --title 'palette' --body 'y'", a2, "agent2")
    combined = out.stdout + out.stderr
    assert "conflicts with agent1" in combined, f"no conflict warning:\n{combined}"


def test_no_conflict_warning_when_the_branches_are_disjoint(team):
    """A warning on every PR would be noise agents learn to ignore."""
    a1, a2 = team["agent1"], team["agent2"]

    (a1 / "one.py").write_text("A = 1\n")
    run("git add -A && git commit -qm a1", a1)
    run("git push -q origin HEAD:agent1", a1)
    gh("pr create --title 'one' --body 'x'", a1, "agent1")

    (a2 / "two.py").write_text("B = 2\n")
    run("git add -A && git commit -qm a2", a2)
    out = gh("pr create --title 'two' --body 'y'", a2, "agent2")
    assert "conflicts with" not in (out.stdout + out.stderr)


def test_body_with_backticks_survives_via_stdin(team):
    """A markdown body containing code is the normal thing to write, and it used to be fatal.

    Inside double quotes bash runs backticks as command substitution, so `gh pr create --body
    "...`parse()`..."` died with rc=137 and "command not found" for every identifier before gh
    was ever reached. `--body-file -` takes the body off the command line entirely.
    """
    a1, a2 = team["agent1"], team["agent2"]
    (a1 / "shared.py").write_text("def quantize(im, error_threshold=0.0):\n    return im\n")
    run("git commit -qam a1", a1)

    run(
        f"sh {SHIM} pr create --title 'adds quantize' --body-file - <<'EOF'\n"
        "## Description\n"
        "Adds `error_threshold` to `quantize()` via $(nothing).\n"
        "EOF\n",
        a1,
        agent="agent1",
    )
    body = gh("pr view agent1", a2, "agent2").stdout
    assert "error_threshold" in body and "quantize()" in body


# --- send_message parsing: malformed calls must not silently reach bash -------------------

_d = importlib.import_module("cooperbench.agents.mini_swe_agent_v2.agents.default")


@pytest.mark.parametrize("cmd", [
    "send_message --wait agent2 <<'MSG'\nhello there\nMSG",
    'send_message agent2 "hello there"',
    "send_message agent2 'hello there'",
    "send_message agent2 --wait <<'MSG'\nhello\nMSG",
])
def test_wellformed_send_message_parses(cmd):
    assert _d._parse_send_messages(cmd), f"should have parsed: {cmd!r}"


@pytest.mark.parametrize("cmd", [
    "send_message --wait agent2",                      # no body at all
    "send_message agent2 <<'MSG'\nunterminated body",  # heredoc never closed
    "send_message agent2 < /tmp/msg.txt",              # body via redirect
    "send_message -t 60 agent2 'hi'",                  # invented flag
])
def test_malformed_send_message_is_not_parsed(cmd):
    """These are the exact shapes agents wrote that fell through to bash.

    Each one reached the sandbox as `send_message: command not found` (rc 127), so the agent
    read a shell error as "messaging is broken" and the message was never sent. 7 such losses
    across two flash_10 runs. The agent loop must return a parse error instead of executing
    them; this test pins the detection half -- that the parser genuinely does not match them.
    """
    assert not _d._parse_send_messages(cmd), f"unexpectedly parsed: {cmd!r}"
    assert re.search(r"\bsend_message\b", cmd), "the fallthrough guard keys off this"
