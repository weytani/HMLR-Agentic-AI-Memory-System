# ABOUTME: Tests for the claude-cli provider in ExternalAPIClient.
# ABOUTME: Verifies CLI-based LLM calls via subprocess work correctly without an API key.

import pytest
import subprocess
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from hmlr.core.external_api_client import ExternalAPIClient
from hmlr.core.exceptions import ApiConnectionError, ConfigurationError


# ============================================================
# Initialization tests
# ============================================================

def test_load_api_key_returns_empty_for_claude_cli():
    """claude-cli provider should not require an API key."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")
    assert client.api_key == ""


def test_load_api_key_raises_when_claude_not_on_path():
    """claude-cli provider should raise ConfigurationError if claude binary is missing."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(ConfigurationError, match="claude.*not found"):
            ExternalAPIClient(api_provider="claude-cli")


def test_get_base_url_returns_empty_for_claude_cli():
    """claude-cli provider uses subprocess, not HTTP — base_url should be empty."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")
    assert client.base_url == ""


def test_fetch_available_models_returns_empty_for_claude_cli():
    """claude-cli provider should skip the HTTP model-listing call."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")
    assert client.available_models == []


# ============================================================
# Message combining
# ============================================================

def test_messages_combined_into_prompt():
    """System + user messages should be combined into a single prompt for the CLI."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "4"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        result = client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000)

    # Verify the prompt passed via stdin is just the user content
    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs["input"] == "What is 2+2?"

    # Verify system content is passed via --append-system-prompt
    cmd = call_kwargs.args[0]
    idx = cmd.index("--append-system-prompt")
    assert cmd[idx + 1] == "You are helpful."


def test_multiple_user_messages_joined():
    """Multiple user messages should be joined with double newlines."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [
        {"role": "user", "content": "First part."},
        {"role": "user", "content": "Second part."},
    ]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "response"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000)

    call_kwargs = mock_run.call_args
    assert call_kwargs.kwargs["input"] == "First part.\n\nSecond part."


# ============================================================
# Sync API call
# ============================================================

def test_call_claude_cli_api_constructs_correct_command():
    """Verify the subprocess command includes all required flags."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "Hi!"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000)

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "claude"
    assert "--model" in cmd
    assert "claude-sonnet-4-6" in cmd
    assert "-p" in cmd
    assert "--no-session-persistence" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert "--append-system-prompt" in cmd


def test_call_claude_cli_api_returns_normalized_response():
    """Response should match the OpenAI-normalized format."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "test"}]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "  response text  "
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result):
        result = client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000)

    assert result["choices"][0]["message"]["content"] == "response text"
    assert result["model"] == "claude-sonnet-4-6"
    assert result["usage"]["prompt_tokens"] == 0
    assert result["usage"]["completion_tokens"] == 0
    assert result["usage"]["total_tokens"] == 0


def test_call_claude_cli_api_raises_on_nonzero_exit():
    """Non-zero exit code should raise ApiConnectionError."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "test"}]

    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = ""
    fake_result.stderr = "Error: something went wrong"

    with patch("subprocess.run", return_value=fake_result):
        with pytest.raises(ApiConnectionError, match="claude CLI failed"):
            client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000)


def test_call_claude_cli_api_no_system_prompt():
    """When there's no system message, --append-system-prompt should be absent."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "just user"}]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "ok"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000)

    cmd = mock_run.call_args.args[0]
    assert "--append-system-prompt" not in cmd


def test_call_claude_cli_api_uses_timeout():
    """Timeout option should be passed through to subprocess.run."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "test"}]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "ok"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000, timeout=300)

    assert mock_run.call_args.kwargs["timeout"] == 300


# ============================================================
# Async API call
# ============================================================

@pytest.mark.asyncio
async def test_call_claude_cli_api_async_returns_normalized_response():
    """Async variant should return the same normalized response shape."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "async test"}]

    fake_process = MagicMock()
    fake_process.communicate = AsyncMock(return_value=(b"async response", b""))
    fake_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_process):
        result = await client._call_claude_cli_api_async(
            "claude-sonnet-4-6", messages, 2000
        )

    assert result["choices"][0]["message"]["content"] == "async response"
    assert result["model"] == "claude-sonnet-4-6"
    assert result["usage"]["total_tokens"] == 0


@pytest.mark.asyncio
async def test_call_claude_cli_api_async_raises_on_failure():
    """Async variant should raise ApiConnectionError on non-zero exit."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "fail"}]

    fake_process = MagicMock()
    fake_process.communicate = AsyncMock(return_value=(b"", b"async error"))
    fake_process.returncode = 1

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_process):
        with pytest.raises(ApiConnectionError, match="claude CLI failed"):
            await client._call_claude_cli_api_async(
                "claude-sonnet-4-6", messages, 2000
            )


# ============================================================
# Routing integration
# ============================================================

def test_query_external_api_routes_to_claude_cli():
    """query_external_api should route to _call_claude_cli_api for claude-cli provider."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "routed response"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result):
        result = client.query_external_api("Hello", model="claude-sonnet-4-6")

    assert result == "routed response"


@pytest.mark.asyncio
async def test_query_external_api_async_routes_to_claude_cli():
    """query_external_api_async should route to _call_claude_cli_api_async."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    fake_process = MagicMock()
    fake_process.communicate = AsyncMock(return_value=(b"async routed", b""))
    fake_process.returncode = 0

    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=fake_process):
        result = await client.query_external_api_async(
            "Hello", model="claude-sonnet-4-6"
        )

    assert result == "async routed"


# ============================================================
# Environment isolation
# ============================================================

def test_call_claude_cli_api_strips_claudecode_env(monkeypatch):
    """CLAUDECODE env var should be stripped to avoid nested-session detection."""
    monkeypatch.setenv("CLAUDECODE", "1")

    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "test"}]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "ok"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result) as mock_run:
        client._call_claude_cli_api("claude-sonnet-4-6", messages, 2000)

    passed_env = mock_run.call_args.kwargs["env"]
    assert "CLAUDECODE" not in passed_env


# ============================================================
# Extended thinking skip
# ============================================================

def test_thinking_params_skipped_for_claude_cli():
    """_apply_thinking_params should be a no-op when provider is claude-cli."""
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = ExternalAPIClient(api_provider="claude-cli")

    messages = [{"role": "user", "content": "test"}]

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "response"
    fake_result.stderr = ""

    # Even with an opus model, the CLI path shouldn't try to apply thinking params
    with patch("subprocess.run", return_value=fake_result):
        result = client._call_claude_cli_api("claude-opus-4-6", messages, 2000)

    # Should succeed without any thinking-related errors
    assert result["choices"][0]["message"]["content"] == "response"
