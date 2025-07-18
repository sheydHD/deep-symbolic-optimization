#!/usr/bin/env bash
# Modern benchmark wrapper for Bash. Calls the Python benchmark script.

set -euo pipe autofail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Call the Python modern benchmark script
python3 "$SCRIPT_DIR/../../python/benchmark/benchmark.py" "$@"
