#!/usr/bin/env bash
# ABOUTME: Cron wrapper for HMLR memory gardener.
# ABOUTME: Promotes bridge blocks to long-term memory and creates dossiers.

# Install with: crontab -e
# Run nightly at midnight:
# 0 0 * * * /Users/weytani/code/hmlr-memory/scripts/garden-cron.sh

set -euo pipefail

HMLR_DIR="${HMLR_DIR:-$HOME/code/hmlr-memory}"
LOG_DIR="${HOME}/.hmlr/logs"
LOG_FILE="${LOG_DIR}/gardener-$(date +%Y-%m-%d).log"

mkdir -p "$LOG_DIR"

echo "=== Gardener run: $(date) ===" >> "$LOG_FILE"

cd "$HMLR_DIR"

# Run gardener with 'all' to process all active bridge blocks
uv run python -m hmlr.run_gardener all >> "$LOG_FILE" 2>&1

echo "=== Gardener complete: $(date) ===" >> "$LOG_FILE"
