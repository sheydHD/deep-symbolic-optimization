"""Test execution helper separated from run.py."""
from __future__ import annotations
import argparse, os
from .env import PROJECT, python_exe, run

def run_tests(extra: list[str] | None = None) -> None:
    os.chdir(PROJECT)
    target = PROJECT / "dso" / "dso" / "test"
    cmd = [python_exe(), "-m", "pytest", str(target), "-q", "-rs"]
    if extra:
        cmd.extend(extra)
    run(cmd)

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run test suite")
    ap.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Args passed through to pytest")
    ns = ap.parse_args(argv)
    run_tests(ns.pytest_args)

if __name__ == "__main__":  # pragma: no cover
    main()
