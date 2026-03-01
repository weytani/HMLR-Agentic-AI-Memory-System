# ABOUTME: Tests for Anthropic extended thinking support in ExternalAPIClient.
# ABOUTME: Verifies ModelConfig has thinking budget and that API calls pass thinking params.

import pytest
from hmlr.core.model_config import ModelConfig
from hmlr.core.external_api_client import ExternalAPIClient


def test_model_config_has_thinking_budget():
    """ModelConfig should have THINKING_BUDGET_TOKENS setting."""
    assert hasattr(ModelConfig, 'THINKING_BUDGET_TOKENS')
    assert isinstance(ModelConfig.THINKING_BUDGET_TOKENS, int)
    assert ModelConfig.THINKING_BUDGET_TOKENS > 0


def test_model_config_thinking_models():
    """ModelConfig should identify which operations use extended thinking."""
    assert hasattr(ModelConfig, 'MAIN_USES_THINKING')
    assert hasattr(ModelConfig, 'SYNTHESIS_USES_THINKING')


def test_thinking_params_added_for_opus():
    """Thinking params should be added when model name contains 'opus'."""
    params = {
        "model": "claude-opus-4-20250514",
        "max_tokens": 2000,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "test"}],
    }

    ExternalAPIClient._apply_thinking_params(params, ModelConfig)

    assert "thinking" in params
    assert params["thinking"]["type"] == "enabled"
    assert params["thinking"]["budget_tokens"] == ModelConfig.THINKING_BUDGET_TOKENS
    # Extended thinking requires temperature=1
    assert params["temperature"] == 1
    # max_tokens should be increased to accommodate thinking
    assert params["max_tokens"] >= ModelConfig.THINKING_BUDGET_TOKENS + 2000


def test_thinking_params_not_added_for_sonnet():
    """Thinking params should NOT be added for non-Opus models."""
    params = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2000,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "test"}],
    }

    ExternalAPIClient._apply_thinking_params(params, ModelConfig)

    assert "thinking" not in params
    assert params["temperature"] == 0.1
    assert params["max_tokens"] == 2000


def test_thinking_params_not_added_when_budget_zero(monkeypatch):
    """Thinking params should NOT be added when budget is 0."""
    monkeypatch.setattr(ModelConfig, 'THINKING_BUDGET_TOKENS', 0)

    params = {
        "model": "claude-opus-4-20250514",
        "max_tokens": 2000,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "test"}],
    }

    ExternalAPIClient._apply_thinking_params(params, ModelConfig)

    assert "thinking" not in params
    assert params["temperature"] == 0.1


def test_response_text_extraction_with_multiple_blocks():
    """Response text extraction should concatenate all text blocks."""
    response_text = ExternalAPIClient._extract_anthropic_response_text(
        _FakeResponseMultiBlock()
    )
    assert response_text == "Hello world"


def test_response_text_extraction_with_single_block():
    """Response text extraction should work with a single text block."""
    response_text = ExternalAPIClient._extract_anthropic_response_text(
        _FakeResponseSingleBlock()
    )
    assert response_text == "Just text"


# --- Helpers for response text extraction tests ---

class _FakeThinkingBlock:
    """Simulates an Anthropic thinking content block (has no .text attr)."""
    def __init__(self, thinking):
        self.thinking = thinking


class _FakeTextBlock:
    """Simulates an Anthropic text content block."""
    def __init__(self, text):
        self.text = text


class _FakeResponseMultiBlock:
    """Simulates an Anthropic response with thinking + text blocks."""
    content = [_FakeThinkingBlock("internal reasoning"), _FakeTextBlock("Hello world")]


class _FakeResponseSingleBlock:
    """Simulates a standard Anthropic response with one text block."""
    content = [_FakeTextBlock("Just text")]
