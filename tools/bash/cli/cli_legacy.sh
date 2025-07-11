#!/usr/bin/env bash
# Interactive menu for legacy DSO environment management.
# Exposed options: Setup, Test, Benchmark, Quit

set -euo pipefail

# shellcheck source=../../lib/utils.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="${SCRIPT_DIR%/*}/lib"
source "${UTILS_DIR}/utils.sh"

SETUP_SCRIPT="${SCRIPT_DIR%/*}/setup/setup_legacy.sh"

print_menu() {
  echo -e "\n${BOLD}Deep Symbolic Optimization – Legacy CLI${RESET}"
  echo "Select an option:"
  echo "  1) Setup"
  echo "  2) Run tests"
  echo "  3) Run benchmark (Nguyen-5)"
  echo "  4) Quit"
}

run_setup_legacy() {
  echo "Choose installation profile:"
  echo "  1) Regression-only (light)"
  echo "  2) Full legacy (all extras)"
  read -r -p "Profile [1/2]: " PROFILE_CHOICE
  case "$PROFILE_CHOICE" in
    2) bash "$SETUP_SCRIPT" full ;;
    *) bash "$SETUP_SCRIPT" regression ;;
  esac
}

run_setup_menu() {
  while true; do
    echo -e "\nSetup options:"
    echo "  1) Legacy stack (Python 3.6 + TF1)"
    echo "  2) Modern stack (placeholder)"
    echo "  3) Back"
    read -r -p "Enter choice [1-3]: " S_CHOICE
    case "$S_CHOICE" in
      1) run_setup_legacy ;;
      2) log_warn "Modern setup not implemented yet." ;;
      3) return ;;
      *) log_warn "Invalid choice" ;;
    esac
  done
}

run_tests() {
  if [ -d ".venv_36" ]; then
    # shellcheck disable=SC1091
    source .venv_36/bin/activate
  fi
  if ! command -v pytest >/dev/null 2>&1; then
    log_error "pytest not found – run setup first"
    return 1
  fi
  log_info "Running tests..."
  pytest -q
  log_success "Tests completed"
}

run_benchmark_menu() {
  if [ -d ".venv_36" ]; then
    # shellcheck disable=SC1091
    source .venv_36/bin/activate
  fi
  if ! python -c "import dso" >/dev/null 2>&1; then
    log_error "dso package not found – run setup first"
    return 1
  fi
  log_info "Executing benchmark (Nguyen-5)"
  python -m dso.run dso/config/examples/regression/Nguyen-2.json --benchmark Nguyen-5
  log_success "Benchmark finished"
}

main_loop() {
  while true; do
    print_menu
    read -r -p "Enter choice [1-4]: " CHOICE
    case "$CHOICE" in
      1) run_setup_menu ;;
      2) run_tests ;;
      3) run_benchmark_menu ;;
      4) log_info "Bye!"; exit 0 ;;
      *) log_warn "Invalid choice" ;;
    esac
  done
}

main_loop
