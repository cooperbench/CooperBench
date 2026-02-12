"""Fallback cost calculator for models where litellm doesn't report cost.

Used when the agent returns cost=0 but we have token counts.
Tries litellm's built-in pricing first, then falls back to a manual table.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Manual pricing table: model substring -> (input, output, cache_read, cache_write) per million tokens
# Only needed for models that litellm can't price (e.g., custom endpoints).
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "MiniMax-M2.5": (0.30, 1.20, 0.03, 0.375),
}


def compute_fallback_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Compute cost from token counts when the agent didn't report it.

    Tries litellm's cost_per_token first. Falls back to the manual pricing
    table if litellm returns zero or raises.

    Returns:
        Estimated cost in USD, or None if the model isn't in any pricing source.
    """
    # Try litellm first — it covers most mainstream models
    try:
        from litellm import completion_cost

        cost = completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        if cost and cost > 0:
            return cost
    except Exception:
        pass  # Model not in litellm's pricing DB

    # Fall back to manual table (substring match handles prefixes like "anthropic/MiniMax-M2.5")
    for pattern, (inp_price, out_price, cr_price, cw_price) in _PRICING.items():
        if pattern in model:
            cost = (
                input_tokens * inp_price
                + output_tokens * out_price
                + cache_read_tokens * cr_price
                + cache_write_tokens * cw_price
            ) / 1_000_000
            return cost

    return None
