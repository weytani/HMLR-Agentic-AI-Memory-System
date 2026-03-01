#!/usr/bin/env python3
# ABOUTME: Ingests session-reflection-analysis output into HMLR memory.
# ABOUTME: Called by Claude Code hook or manually after a session.

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _create_client():
    """Create an HMLRClient with the configured database path.

    Uses HMLR_DB_PATH from environment, falling back to ~/.hmlr/memory.db.
    Separated from ingest_reflection for testability.
    """
    from hmlr import HMLRClient

    db_path = os.getenv("HMLR_DB_PATH", str(Path.home() / ".hmlr" / "memory.db"))
    return HMLRClient(db_path=db_path)


async def ingest_reflection(reflection_text: str, session_id: str = "manual"):
    """Ingest a reflection into HMLR.

    Tags the reflection for gardener processing and stores it via the
    standard HMLRClient.chat pipeline. The message format matches
    the MCP mem_ingest_reflection tool for consistency.

    Args:
        reflection_text: The session-reflection-analysis output text.
        session_id: Identifier for the session that produced the reflection.
    """
    client = _create_client()

    try:
        result = await client.chat(
            message=(
                "[Tags: session-reflection, learning, pattern] "
                f"[Type: reflection-analysis] {reflection_text}"
            ),
            session_id=f"reflection_{session_id}",
        )

        print(f"Reflection ingested: {result.get('status', 'unknown')}")
    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("No reflection text provided")
        sys.exit(1)

    asyncio.run(ingest_reflection(text.strip()))
