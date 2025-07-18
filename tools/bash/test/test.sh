#!/usr/bin/env bash
# Modern test wrapper for Bash. Calls the Python test script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the Python modern test script
python3 "$SCRIPT_DIR/../../python/test/test.py" "$@"
