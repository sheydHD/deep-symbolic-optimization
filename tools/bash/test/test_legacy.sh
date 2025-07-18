#!/usr/bin/env bash
# Legacy test runner for Bash.

set -euo pipefail

# shellcheck source=../lib/utils.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="${SCRIPT_DIR%/*}/lib"
source "${UTILS_DIR}/utils.sh"

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

run_tests "$@"
