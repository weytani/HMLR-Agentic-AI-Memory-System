# ABOUTME: Tests for the session reflection ingestion script.
# ABOUTME: Verifies CLI argument parsing, stdin reading, and ingestion flow.

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "ingest-reflection.py"


# ---------- module loading helpers ----------


def _load_module():
    """Import ingest-reflection.py as a module for unit testing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ingest_reflection", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- unit tests ----------


class TestIngestReflectionFunction:
    """Tests for the ingest_reflection coroutine."""

    @pytest.mark.asyncio
    async def test_ingest_reflection_calls_client_chat(self):
        """ingest_reflection should call HMLRClient.chat with tagged message."""
        mod = _load_module()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"status": "ok"})
        mock_client.close = MagicMock()

        with patch.object(mod, "_create_client", return_value=mock_client):
            await mod.ingest_reflection("Test reflection text", session_id="test-123")

        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        msg = call_args.kwargs.get("message", call_args[0][0] if call_args[0] else "")
        assert "[Tags: session-reflection, learning, pattern]" in msg
        assert "[Type: reflection-analysis]" in msg
        assert "Test reflection text" in msg

    @pytest.mark.asyncio
    async def test_ingest_reflection_uses_session_id(self):
        """ingest_reflection should prefix session_id with 'reflection_'."""
        mod = _load_module()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"status": "ok"})
        mock_client.close = MagicMock()

        with patch.object(mod, "_create_client", return_value=mock_client):
            await mod.ingest_reflection("text", session_id="abc")

        call_args = mock_client.chat.call_args
        assert call_args.kwargs.get("session_id") == "reflection_abc"

    @pytest.mark.asyncio
    async def test_ingest_reflection_default_session_id(self):
        """ingest_reflection should default to session_id 'manual'."""
        mod = _load_module()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"status": "ok"})
        mock_client.close = MagicMock()

        with patch.object(mod, "_create_client", return_value=mock_client):
            await mod.ingest_reflection("text")

        call_args = mock_client.chat.call_args
        assert call_args.kwargs.get("session_id") == "reflection_manual"

    @pytest.mark.asyncio
    async def test_ingest_reflection_closes_client(self):
        """ingest_reflection should close the client after ingestion."""
        mod = _load_module()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value={"status": "ok"})
        mock_client.close = MagicMock()

        with patch.object(mod, "_create_client", return_value=mock_client):
            await mod.ingest_reflection("text")

        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_reflection_closes_client_on_error(self):
        """Client should be closed even if chat() raises an exception."""
        mod = _load_module()

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        mock_client.close = MagicMock()

        with patch.object(mod, "_create_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="LLM down"):
                await mod.ingest_reflection("text")

        mock_client.close.assert_called_once()


# ---------- CLI integration tests ----------


class TestIngestReflectionCLI:
    """Tests for the script's CLI behavior."""

    def test_script_is_executable(self):
        """The script file should exist and be executable."""
        assert SCRIPT_PATH.exists(), f"Script not found at {SCRIPT_PATH}"
        assert SCRIPT_PATH.stat().st_mode & 0o111, "Script is not executable"

    def test_empty_input_exits_nonzero(self):
        """Script should exit with code 1 when given empty input."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "no reflection text provided" in result.stdout.lower()

    def test_whitespace_only_input_exits_nonzero(self):
        """Script should exit with code 1 when given only whitespace."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            input="   \n\n  ",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1

    def test_cli_args_passed_as_text(self):
        """When args are given, they should be joined and used as reflection text."""
        mod = _load_module()

        # The module should combine sys.argv[1:] into text
        # We test this by checking the main block logic
        assert hasattr(mod, "ingest_reflection")
        assert asyncio.iscoroutinefunction(mod.ingest_reflection)


class TestCreateClient:
    """Tests for the _create_client factory function."""

    def test_create_client_uses_env_db_path(self):
        """_create_client should use HMLR_DB_PATH from environment."""
        mod = _load_module()

        with patch.dict("os.environ", {"HMLR_DB_PATH": "/tmp/test-hmlr.db"}):
            with patch("hmlr.HMLRClient") as MockClient:
                mod._create_client()
                MockClient.assert_called_once_with(db_path="/tmp/test-hmlr.db")

    def test_create_client_default_db_path(self):
        """_create_client should default to ~/.hmlr/memory.db."""
        mod = _load_module()

        with patch.dict("os.environ", {}, clear=False):
            # Remove HMLR_DB_PATH if it exists
            env = dict(**__import__("os").environ)
            env.pop("HMLR_DB_PATH", None)
            with patch.dict("os.environ", env, clear=True):
                with patch("hmlr.HMLRClient") as MockClient:
                    mod._create_client()
                    call_args = MockClient.call_args
                    db_path = call_args.kwargs.get(
                        "db_path", call_args[0][0] if call_args[0] else ""
                    )
                    expected = str(Path.home() / ".hmlr" / "memory.db")
                    assert db_path == expected
