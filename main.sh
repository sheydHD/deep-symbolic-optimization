#!/usr/bin/env bash
# Root launcher for Deep Symbolic Optimization using the modern toolchain.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=-1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR              # child scripts can rely on this

MODERN="$ROOT_DIR/tools/python/run.py"

usage() {
  cat <<EOF
Usage: ./main.sh [args …]
       ./main.sh            # interactive menu
EOF
}

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then usage; exit 0; fi

# Execute the modern toolchain directly
exec python3 "$MODERN" "$@"
