#!/usr/bin/env python3
"""
Modern helper entry for Deep Symbolic Optimization (Python ≥3.11 + TF-2).

Sub-commands
============
setup            — bootstrap or upgrade the virtual-env in .venv
test  [pytest…]  — run dso/test with optional pytest flags
bench CONFIG     — run benchmark with given JSON config
menu             — interactive menu
"""

from __future__ import annotations
import argparse, os, subprocess, sys, textwrap
from pathlib import Path

# --------------------------------------------------------------------------- paths
THIS_FILE  = Path(__file__).resolve()
TOOLS_DIR  = THIS_FILE.parent
PROJECT    = TOOLS_DIR.parents[1]          # deep-symbolic-optimization/
DSO_PKG    = PROJECT / "dso"

# --------------------------------------------------------------------------- helpers
def run(cmd: list[str], **kw) -> None:
    """Let subprocess inherit stdout/stderr (live logs)."""
    print("➜", *cmd, flush=True)
    subprocess.run(cmd, check=True, **kw)

def python_exe() -> str:
    venv_py = PROJECT / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable

# --------------------------------------------------------------------------- sub-commands
def cmd_setup(ns: argparse.Namespace) -> None:
    setup_path = TOOLS_DIR / "setup" / "setup.py"
    # Change to project directory for setup
    os.chdir(PROJECT)
    run([python_exe(), str(setup_path)] + ns.forward)
    
    # If we just created the venv, re-exec this script inside it
    if (PROJECT / ".venv" / "bin" / "python").exists():
        os.execv(str(PROJECT / ".venv" / "bin" / "python"),
                 [str(PROJECT / ".venv" / "bin" / "python"), *sys.argv])

def cmd_test(ns: argparse.Namespace) -> None:
    target = PROJECT / "dso" / "dso" / "test"
    # Change to project directory to ensure relative imports work
    os.chdir(PROJECT)
    run([python_exe(), "-m", "pytest", str(target), "-q"] + ns.forward)

def cmd_bench(ns: argparse.Namespace) -> None:
    # Check if config file exists
    config_path = Path(ns.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {ns.config}")
        print("Available configs:")
        config_dir = PROJECT / "dso" / "dso" / "config" / "examples"
        if config_dir.exists():
            for cfg in config_dir.rglob("*.json"):
                print(f"  - {cfg.relative_to(PROJECT)}")
        return
    
    # Run the DSO training script with the config
    os.chdir(PROJECT)
    # Call dso.run with config path as a positional argument, not with --config flag
    run([python_exe(), "-m", "dso.run", str(config_path)])

def interactive_menu() -> None:
    while True:
        print("\n-- DSO Menu --")
        print("  1) Setup environment (.venv + uv)")
        print("  2) Run tests")
        print("  3) Run benchmark")
        print("  4) Quit")
        choice = input("Choice [1-4]: ").strip()
        if   choice == "1": cmd_setup(argparse.Namespace(forward=[]))
        elif choice == "2": cmd_test (argparse.Namespace(forward=[]))
        elif choice == "3":
            cfg = input("Path to JSON config: ").strip() or \
                  "dso/dso/config/examples/regression/Nguyen-2.json"
            cmd_bench(argparse.Namespace(config=cfg))
        elif choice == "4": return
        else: print("Invalid.")

# --------------------------------------------------------------------------- CLI parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(__doc__))
sub = parser.add_subparsers(dest="cmd", required=True)

sub.add_parser("setup")  .set_defaults(func=lambda _ns, *_: None)  # handled in dispatch
sub.add_parser("test")   .set_defaults(func=lambda _ns, *_: None)  # handled in dispatch
sub.add_parser("menu")   .set_defaults(func=lambda _ns, *_: interactive_menu())
sub.add_parser("bench")  .add_argument("config")

# note: `parse_known_args()` lets us collect *unknown* flags and hand them off
# --------------------------------------------------------------------------- dispatch
if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_menu(); sys.exit(0)

    ns, forward = parser.parse_known_args()
    if ns.cmd == "setup":
        cmd_setup(argparse.Namespace(forward=forward))
    elif ns.cmd == "test":
        cmd_test (argparse.Namespace(forward=forward))
    elif ns.cmd == "bench":
        cmd_bench(ns)                 # bench validates its own args
    else:                             # menu
        ns.func(ns)
