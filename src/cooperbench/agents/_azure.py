"""Shared Azure OpenAI provider detection for CooperBench agent adapters.

Azure is configured with two host env vars:

  - ``AZURE_OPENAI_API_KEY``   the resource key
  - ``AZURE_OPENAI_ENDPOINT``  the OpenAI-compatible v1 base, e.g.
    ``https://<resource>.cognitiveservices.azure.com/openai/v1``

When both are set, adapters route their model calls at the Azure
deployment (the name passed via ``-m``) instead of the default OpenAI /
Anthropic / Vertex backends.  Azure takes precedence over a plain
``OPENAI_API_KEY``.

For the litellm-backed adapters (mini_swe_agent_v2, swe_agent) and the
OpenHands SDK, Azure's v1 surface is reached through litellm's
*openai-compatible* provider pointed at the endpoint — i.e. model
``openai/<deployment>`` with ``api_base`` + ``api_key`` — which mirrors
how the OpenAI SDK itself is pointed at Azure (``base_url=<v1>``).  This
avoids litellm's native ``azure/`` route and its ``api_version`` dance;
both were verified to work, but the openai-compatible one needs no
version pin.

The codex adapter does *not* use this litellm path (it shells out to the
codex CLI and writes a ``model_provider`` block to ``config.toml``); it
shares only ``resolve_azure_config`` from here.
"""

from __future__ import annotations

import os


def resolve_azure_config() -> dict[str, str] | None:
    """Return ``{"api_key", "endpoint"}`` when Azure is configured, else None.

    ``endpoint`` has any trailing slash stripped (callers append paths).
    """
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if key and endpoint:
        return {"api_key": key, "endpoint": endpoint.rstrip("/")}
    return None


def azure_deployment_name(model_name: str) -> str:
    """The Azure deployment name from the user's ``-m`` value.

    Accepts a bare deployment (``gpt-5.5-hao``) or a provider-prefixed
    form (``azure/gpt-5.5-hao`` / ``openai/gpt-5.5-hao``) and returns the
    last path segment.
    """
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def azure_litellm_model(model_name: str) -> str:
    """litellm model id for an Azure deployment via the openai-compatible
    provider: ``openai/<deployment>``."""
    return f"openai/{azure_deployment_name(model_name)}"
