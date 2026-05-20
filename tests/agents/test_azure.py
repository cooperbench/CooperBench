"""Tests for the shared Azure provider helper (``cooperbench.agents._azure``).

The per-adapter wiring (mini_swe_agent_v2, swe_agent, openhands_sdk, codex)
all funnels through these helpers, so covering them here exercises the
shared contract: detection precedence, endpoint normalization, and the
litellm openai-compatible model id used for Azure deployments.
"""

from __future__ import annotations

import pytest

from cooperbench.agents._azure import (
    azure_deployment_name,
    azure_litellm_model,
    resolve_azure_config,
)


@pytest.fixture(autouse=True)
def _clear_azure_env(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)


class TestResolveAzureConfig:
    def test_returns_config_when_both_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r.cognitiveservices.azure.com/openai/v1/")
        assert resolve_azure_config() == {
            "api_key": "az-key",
            "endpoint": "https://r.cognitiveservices.azure.com/openai/v1",  # trailing slash stripped
        }

    def test_none_when_neither_set(self):
        assert resolve_azure_config() is None

    def test_none_when_only_key(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
        assert resolve_azure_config() is None

    def test_none_when_only_endpoint(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r/openai/v1")
        assert resolve_azure_config() is None

    def test_blank_values_ignored(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "   ")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://r/openai/v1")
        assert resolve_azure_config() is None


class TestDeploymentName:
    def test_bare_name(self):
        assert azure_deployment_name("gpt-5.5-hao") == "gpt-5.5-hao"

    def test_strips_provider_prefix(self):
        assert azure_deployment_name("azure/gpt-5.5-hao") == "gpt-5.5-hao"
        assert azure_deployment_name("openai/gpt-5.5-hao") == "gpt-5.5-hao"


class TestLitellmModel:
    def test_openai_compatible_prefix(self):
        assert azure_litellm_model("gpt-5.5-hao") == "openai/gpt-5.5-hao"

    def test_idempotent_on_prefixed(self):
        # azure/x -> deployment x -> openai/x (no double prefix)
        assert azure_litellm_model("azure/gpt-5.5-hao") == "openai/gpt-5.5-hao"
        assert azure_litellm_model("openai/gpt-5.5-hao") == "openai/gpt-5.5-hao"
