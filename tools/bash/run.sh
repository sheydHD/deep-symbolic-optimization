#!/usr/bin/env bash
# Central dispatcher for legacy Bash scripts.

set -euo pipefail

# shellcheck source=./lib/utils.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="${SCRIPT_DIR}/lib"
source "${UTILS_DIR}/utils.sh"

print_menu() {
  echo -e "\n${BOLD}Deep Symbolic Optimization – Legacy CLI${RESET}"
  echo "Select an option:"
  echo "  1) Setup"
  echo "  2) Run tests"
  echo "  3) Run benchmark (Nguyen-5)"
  echo "  4) Quit"
}

main_loop() {
  while true; do
    print_menu
    read -r -p "Enter choice [1-4]: " CHOICE
    case "$CHOICE" in
      1) bash "$SCRIPT_DIR/setup/setup_legacy.sh" setup ;;
      2)
        echo "Running tests..."
        # Set PYTHONPATH to include the parent directory of dso for proper module discovery
        PYTHONPATH="$SCRIPT_DIR/../dso:$PYTHONPATH" \
        pytest -q "$@"
        ;;
      3) bash "$SCRIPT_DIR/benchmark/benchmark_legacy.sh" ;;
      4) log_info "Bye!"; exit 0 ;;
      *) log_warn "Invalid choice" ;;
    esac
  done
}

main_loop "$@" 