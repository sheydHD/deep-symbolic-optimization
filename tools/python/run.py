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

import sys, pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from utils.env import PROJECT, python_exe, run
from utils import setup_env, test_runner, benchmark_runner

# --------------------------------------------------------------------------- helpers
DSO_PKG    = PROJECT / "dso"

# --------------------------------------------------------------------------- sub-commands
def cmd_setup(ns: argparse.Namespace) -> None:
    os.chdir(PROJECT)
    setup_env.main(ns.forward)

def cmd_test(ns: argparse.Namespace) -> None:
    test_runner.run_tests(ns.forward)

def cmd_bench(ns: argparse.Namespace) -> None:
    benchmark_runner.run_benchmark(ns.config, getattr(ns, "benchmark", None))

def interactive_menu() -> None:
    while True:
        print("\n-- DSO Menu --")
        print("  1) Setup environment (.venv + uv)")
        print("  2) Run tests")
        print("  3) Run benchmark")
        print("  h) Help")
        print("  q) Quit")
        choice = input("Choice [1-3,h,q]: ").strip().lower()
        if   choice == "1":
            cmd_setup(argparse.Namespace(forward=[]))
        elif choice == "2":
            cmd_test(argparse.Namespace(forward=[]))
        elif choice == "3":
            cfg = input("Path to JSON config: ").strip() or "dso/dso/config/examples/regression/Nguyen-2.json"
            cmd_bench(argparse.Namespace(config=cfg))
        elif choice == "h":
            print("\nHelp:")
            print("  setup      - create / update virtual environment (.venv) and install deps")
            print("  test [args]- run pytest suite (additional pytest args optional)")
            print("  bench CFG  - run benchmark with config JSON (use --benchmark if supported)")
            print("  menu       - interactive menu (this)")
            print("  quit       - exit menu")
            print("\nExamples:")
            print("  ./main.sh setup (--fresh)")
            print("  ./main.sh test (-k regression)")
            print("  ./main.sh bench dso/dso/config/examples/regression/Nguyen-2.json")
        elif choice == "q":
            return
        else:
            print("Invalid.")

# --------------------------------------------------------------------------- CLI parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(__doc__))
sub = parser.add_subparsers(dest="cmd", required=True)

sub.add_parser("setup")  .set_defaults(func=lambda _ns, *_: None)  # handled in dispatch
sub.add_parser("test")   .set_defaults(func=lambda _ns, *_: None)  # handled in dispatch
sub.add_parser("menu")   .set_defaults(func=lambda _ns, *_: interactive_menu())
bench_p = sub.add_parser("bench")
bench_p.add_argument("config", help="Path to benchmark config JSON")
bench_p.add_argument("--benchmark", help="Optional benchmark selector passed through to dso.run")

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
