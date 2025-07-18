#!/usr/bin/env bash
# Modern setup wrapper for Bash. Calls the Python setup script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the Python modern setup script
python3 "$SCRIPT_DIR/../../python/setup/setup.py" "$@" 