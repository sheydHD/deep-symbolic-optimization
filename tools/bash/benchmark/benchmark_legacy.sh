#!/usr/bin/env bash
# Legacy benchmark runner for Bash.

set -euo pipefail

# shellcheck source=../lib/utils.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="${SCRIPT_DIR%/*}/lib"
source "${UTILS_DIR}/utils.sh"

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

run_benchmark_menu "$@"
