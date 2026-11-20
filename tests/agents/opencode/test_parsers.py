"""Unit tests for the opencode run-output parser.

opencode's ``run`` command emits human-readable, ANSI-colored streaming
text rather than a structured stream.  ``parse_run_output`` therefore strips
ANSI escapes first and then best-effort scrapes the summary figures (steps,
tokens, cost), defaulting every field to 0 when a pattern is absent.
"""

from cooperbench.agents.opencode.parsers import (
    RunSummary,
    parse_run_output,
    strip_ansi,
)


def test_strip_ansi_removes_color_codes():
    colored = "\x1b[32minput tokens\x1b[0m: \x1b[1m1,234\x1b[0m"
    assert strip_ansi(colored) == "input tokens: 1,234"


def test_empty_output_defaults_to_zero():
    summary = parse_run_output("")
    assert summary == RunSummary(status="Submitted", steps=0, input_tokens=0, output_tokens=0, cost=0.0)


def test_parses_cost():
    assert parse_run_output("total cost: $0.0231").cost == 0.0231
    assert parse_run_output("done — $1.50 spent").cost == 1.50


def test_parses_input_and_output_tokens_with_separators():
    summary = parse_run_output("12,345 input tokens, 6,789 output tokens")
    assert summary.input_tokens == 12345
    assert summary.output_tokens == 6789


def test_parses_tokens_label_form():
    summary = parse_run_output("input: 500\noutput: 250")
    assert summary.input_tokens == 500
    assert summary.output_tokens == 250


def test_total_tokens_fallback():
    summary = parse_run_output("tokens: 999")
    assert summary.input_tokens == 999
    assert summary.output_tokens == 0


def test_parses_steps_count():
    assert parse_run_output("completed in 12 steps").steps == 12
    assert parse_run_output("steps: 7").steps == 7


def test_parsing_works_through_ansi_codes():
    # The whole point of the fix: figures wrapped in color codes must parse.
    raw = "\x1b[1mtotal cost\x1b[0m: \x1b[33m$0.0456\x1b[0m\n\x1b[32m2,000 input tokens\x1b[0m"
    summary = parse_run_output(raw)
    assert summary.cost == 0.0456
    assert summary.input_tokens == 2000


def test_malformed_numbers_do_not_raise():
    # Should never throw regardless of input shape.
    summary = parse_run_output("cost: $abc tokens: xyz steps: ???")
    assert summary.cost == 0.0
    assert summary.input_tokens == 0
    assert summary.steps == 0
