#!/usr/bin/env python3
"""Developer CLI for DSO’s *modern* branch.

Sub-commands:
  setup        – bootstrap / upgrade env (wraps setup_env.py)
  test [ARGS]  – run pytest against dso/test
  bench CFG    – run benchmark with given JSON config
"""
from __future__ import annotations
import argparse, importlib, os, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # deep-symbolic-optimization/
sys.path.insert(0, str(PROJECT))               # import dso.* while editable

def _venv_exec(module_args: list[str]) -> None:
    """Run `python -m <args>` inside the project’s venv if present."""
    py = PROJECT / ".venv" / "bin" / "python"
    exe = str(py) if py.exists() else sys.executable
    subprocess.run([exe, "-m", *module_args], check=True)

def cmd_setup(args: argparse.Namespace) -> None:
    _venv_exec(["scripts.python.setup_env", *args.forward])

def cmd_test(args: argparse.Namespace) -> None:
    target = PROJECT / "dso" / "test"
    _venv_exec(["pytest", str(target), *args.forward, "-q"])

def cmd_bench(args: argparse.Namespace) -> None:
    if not Path(args.config).exists():
        sys.exit(f"Config file not found: {args.config}")
    import importlib
    importlib.import_module("dso.benchmark").main(config_file=args.config)

# --------------------------------------------------------------------------- CLI parser
p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)

sp = sub.add_parser("setup");   sp.set_defaults(func=cmd_setup);  sp.add_argument("forward", nargs=argparse.REMAINDER)
sp = sub.add_parser("test");    sp.set_defaults(func=cmd_test);   sp.add_argument("forward", nargs=argparse.REMAINDER)
sp = sub.add_parser("bench");   sp.set_defaults(func=cmd_bench);  sp.add_argument("config")

# --------------------------------------------------------------------------- dispatch
if __name__ == "__main__":
    ns = p.parse_args()
    ns.func(ns)
