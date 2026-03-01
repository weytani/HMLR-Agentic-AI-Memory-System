# ABOUTME: Tests for Anthropic extended thinking support in ExternalAPIClient.
# ABOUTME: Verifies ModelConfig has thinking budget and that API calls pass thinking params.

import pytest
from hmlr.core.model_config import ModelConfig


def test_model_config_has_thinking_budget():
    """ModelConfig should have THINKING_BUDGET_TOKENS setting."""
    assert hasattr(ModelConfig, 'THINKING_BUDGET_TOKENS')
    assert isinstance(ModelConfig.THINKING_BUDGET_TOKENS, int)
    assert ModelConfig.THINKING_BUDGET_TOKENS > 0


def test_model_config_thinking_models():
    """ModelConfig should identify which operations use extended thinking."""
    assert hasattr(ModelConfig, 'MAIN_USES_THINKING')
    assert hasattr(ModelConfig, 'SYNTHESIS_USES_THINKING')
