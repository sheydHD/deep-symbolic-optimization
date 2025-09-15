"""Benchmark invocation helper split from run.py."""
from __future__ import annotations
import argparse, os
from pathlib import Path
from .env import PROJECT, python_exe, run

def run_benchmark(config: str, benchmark: str | None = None) -> None:
    cfg_path = Path(config)
    if not cfg_path.exists():
        print(f"❌ Config file not found: {config}")
        cfg_dir = PROJECT / "dso_pkg" / "dso" / "config" / "examples"
        if cfg_dir.exists():
            print("Available configs:")
            for c in cfg_dir.rglob("*.json"):
                print(f"  - {c.relative_to(PROJECT)}")
        return
    os.chdir(PROJECT)
    cmd = [python_exe(), "-m", "dso.run", str(cfg_path)]
    if benchmark:
        cmd.extend(["--benchmark", benchmark])
    run(cmd)

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a benchmark config")
    ap.add_argument("config")
    ap.add_argument("--benchmark")
    ns = ap.parse_args(argv)
    run_benchmark(ns.config, ns.benchmark)

if __name__ == "__main__":  # pragma: no cover
    main()
