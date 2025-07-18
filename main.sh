#!/usr/bin/env bash
# Root launcher for Deep Symbolic Optimization helpers.
# Will call legacy (Bash) or modern (Python) tool-chain that already lives in tools/.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR              # child scripts can rely on this

LEGACY="$ROOT_DIR/tools/bash/run.sh"
MODERN="$ROOT_DIR/tools/python/run.py"

usage() {
  cat <<EOF
Usage: ./main.sh [legacy|modern] [args …]
       ./main.sh            # interactive menu
EOF
}

if [[ ${1:-} =~ ^(-h|--help)$ ]]; then usage; exit 0; fi

case "${1:-}" in
  legacy)  shift; exec bash "$LEGACY" "$@";;
  modern)  shift; exec python3 "$MODERN" "$@";;
  "")      ;;                        # fall through to interactive
  *)       # no sub-command given → treat as modern passthrough
           exec python3 "$MODERN" "$@";;
esac

# Interactive (“graphical”) menu
while true; do
  echo -e "\n\033[1mDeep Symbolic Optimization\033[0m"
  echo "  1) Legacy   (Py3.6  +  TensorFlow-1)"
  echo "  2) Modern   (Py3.11 +  TensorFlow-2)"
  echo "  3) Quit"
  read -r -p "Choice [1-3]: " ans
  case "$ans" in
    1) bash "$LEGACY";;
    2) python3 "$MODERN";;
    3) echo "Bye!"; exit 0;;
    *) echo "Invalid.";  ;;
  esac
done
