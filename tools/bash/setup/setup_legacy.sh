#!/usr/bin/env bash
# Automated installer for the DSO legacy environment (Python 3.6.15 + TF 1.14).
# This script is called by cli_legacy.sh and MUST be idempotent.

set -euo pipefail

# shellcheck source=../lib/utils.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="${SCRIPT_DIR%/*}/lib" # one level up then /lib
source "${UTILS_DIR}/utils.sh"

#######################################
# Globals
#######################################
PY_VERSION="3.6.15"
VENV_NAME=".venv_36"

# Selected profile: regression | full
INSTALL_PROFILE="regression"

#######################################
# Functions
#######################################

check_pyenv() {
  if ! command -v pyenv >/dev/null 2>&1; then
    log_warn "pyenv not found – installing..."
    curl https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    log_success "pyenv installed"
  else
    log_info "pyenv found: $(pyenv --version)"
  fi
}

install_python() {
  local PY_DIR="$(pyenv root)/versions/${PY_VERSION}"
  if [ -d "$PY_DIR" ]; then
    log_info "Python ${PY_VERSION} already installed via pyenv"
    return
  fi

  log_info "Installing Python ${PY_VERSION} (this may take a while)"
  # --skip-existing avoids prompt if partially installed; older pyenv may not recognise; ignore errors.
  pyenv install --skip-existing "${PY_VERSION}" || pyenv install "${PY_VERSION}"
}

create_venv() {
  if [ -d "${VENV_NAME}" ]; then
    log_info "Virtualenv ${VENV_NAME} already exists"
  else
    log_info "Creating venv ${VENV_NAME} using Python ${PY_VERSION}"
    # Use the desired Python version explicitly without relying on interactive shell integration
    log_info "Creating virtualenv via pyenv exec (PYENV_VERSION=${PY_VERSION})"
    abort_on_error env PYENV_VERSION="${PY_VERSION}" pyenv exec python -m venv "${VENV_NAME}"
    log_success "Virtualenv created"
  fi

  # Activate for subsequent commands
  # shellcheck disable=SC1090
  source "${VENV_NAME}/bin/activate"
  log_info "Activated venv (python: $(python -V))"

  # Upgrade packaging tools
  pip install --upgrade pip setuptools wheel >/dev/null
}

install_dependencies() {
  local REQUIREMENTS_FILE
  case "$INSTALL_PROFILE" in
    regression) REQUIREMENTS_FILE="configs/requirements/base_legacy.txt" ;;
    full)       REQUIREMENTS_FILE="configs/requirements/base_legacy_all.txt" ;;
    *)          log_error "Unknown profile: $INSTALL_PROFILE"; exit 1 ;;
  esac

  log_info "Installing dependencies from $REQUIREMENTS_FILE"
  if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
  else
    log_error "Requirements file not found: $REQUIREMENTS_FILE"; exit 1
  fi

  # Install local package (editable). Extras flag adds correct entry-point for users;
  # we retain --no-deps so pinned versions from requirements file stay intact.
  local EXTRAS_SUFFIX
  if [ "$INSTALL_PROFILE" = "full" ]; then
    EXTRAS_SUFFIX="[all]"
  else
    EXTRAS_SUFFIX="[regression]"
  fi

  pip install -e "./dso${EXTRAS_SUFFIX}" --no-deps
  log_success "Dependencies installed"
}

verify_install() {
  python - <<'PY'
import importlib, sys
for pkg in ("tensorflow", "dso"):
    try:
        importlib.import_module(pkg)
        print(f"{pkg} import ✔")
    except ImportError:
        print(f"{pkg} import ✖", file=sys.stderr)
        sys.exit(1)
PY
  log_success "Sanity import tests passed"
}

run_tests() {
  log_info "Running unit tests (profile: $INSTALL_PROFILE)"
  if [ "$INSTALL_PROFILE" = "regression" ]; then
    python -m pytest -q dso/dso/task/regression/
  else
    python -m pytest -q
  fi
  log_success "Tests passed"
}

run_benchmark() {
  log_info "Running benchmark (Nguyen-5, single seed)"
  local CONFIG_PATH="dso/dso/config/examples/regression/Nguyen-2.json"
  python -m dso.run "$CONFIG_PATH" --benchmark Nguyen-5
  log_success "Benchmark finished"
}

#######################################
# Main
#######################################

main() {
  # Profile selection comes as first argument
  if [[ ${1:-} == "full" ]]; then
    INSTALL_PROFILE="full"
  else
    INSTALL_PROFILE="regression"
  fi

  check_pyenv
  install_python
  create_venv
  install_dependencies
  verify_install

  if ask_yes_no "Run unit tests now?"; then
    run_tests
  fi

  if ask_yes_no "Run benchmark now? (Nguyen-5)"; then
    run_benchmark
  fi

  log_success "Legacy setup complete. Remember to 'source ${VENV_NAME}/bin/activate' next time."
}

main "$@"