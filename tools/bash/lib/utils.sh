#!/usr/bin/env bash
# Utility helpers for Bash scripts in the DSO project.
# Provides coloured logging, confirm prompts, and robust path handling.

# Resolve directory of the current script, regardless of symlinks.
get_script_dir() {
  local SOURCE DIR
  SOURCE="$BASH_SOURCE"
  while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if SOURCE was relative symlink, resolve it relative to DIR
  done
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  echo "$DIR"
}

# Colours
RESET='\033[0m'
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'

log_info()   { echo -e "${BLUE}[INFO]${RESET} $*"; }
log_warn()   { echo -e "${YELLOW}[WARN]${RESET} $*"; }
log_error()  { echo -e "${RED}[ERROR]${RESET} $*"; }
log_success(){ echo -e "${GREEN}[ OK ]${RESET} $*"; }

# Ask a yes/no question (default No). Returns 0 for yes, 1 for no.
ask_yes_no() {
  local PROMPT=${1:-"Proceed"}
  read -r -p "$PROMPT [y/N]: " REPLY
  case "$REPLY" in
    [Yy]*) return 0 ;;
    *)     return 1 ;;
  esac
}

# Abort on error helper
abort_on_error() {
  "$@" || { log_error "Command failed: $*"; exit 1; }
} 