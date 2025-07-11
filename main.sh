#!/usr/bin/env bash
# Root dispatcher for DSO helper scripts.
# Currently delegates to legacy CLI (Bash).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_PATH="$SCRIPT_DIR/tools/bash/cli/cli_legacy.sh"

if [ ! -f "$CLI_PATH" ]; then
  echo "Error: CLI script not found at $CLI_PATH" >&2
  exit 1
fi
bash "$CLI_PATH" "$@"

