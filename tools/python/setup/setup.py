#!/usr/bin/env python3
"""
Bootstrap or upgrade the virtual-env (.venv) for Deep Symbolic
Optimization.  Expects the new requirements trio:

    configs/requirements/{core,extras,dev}.in   (sources)
    configs/requirements/{core,extras,dev}.txt  (optional lock files)

If a .txt lock file is missing, it will be compiled on the fly.
"""

from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path

HERE    = Path(__file__).resolve()
PROJECT = HERE.parents[3]                       # deep-symbolic-optimization/
REQDIR  = PROJECT / "configs" / "requirements"

def run(cmd: list[str]) -> None:
    """Run a subprocess with live output and hard-fail on non-zero exit."""
    print("➜", *cmd, flush=True)
    subprocess.run(cmd, check=True)

def ensure_uv(python: str) -> None:
    try:
        subprocess.run(["uv", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        run([python, "-m", "pip", "install", "--upgrade", "pip", "uv"])

def compile_if_needed(name: str) -> Path:
    """
    Ensure REQDIR / f"{name}.txt" exists.
    If only the .in file is present, compile to .txt with uv pip compile.
    """
    txt = REQDIR / f"{name}.txt"
    if txt.exists():
        return txt

    src = REQDIR / f"{name}.in"
    if not src.exists():
        raise FileNotFoundError(
            f"Neither {txt.name} nor {src.name} found in {REQDIR}")
    print(f"⚙️  Compiling {src.name} → {txt.name}")
    run(["uv", "pip", "compile", str(src), "--output-file", str(txt)])
    return txt

# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Create / upgrade .venv for Deep Symbolic Optimization")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete any existing .venv first")
    args = ap.parse_args(argv)

    python = sys.executable
    ensure_uv(python)

    venv = PROJECT / ".venv"
    if args.fresh and venv.exists():
        shutil.rmtree(venv)
    if not venv.exists():
        run(["uv", "venv", "-p", python])

    # ---------------------------------------------------------------- install deps
    for layer in ("core", "extras", "dev"):
        req = compile_if_needed(layer)
        run(["uv", "pip", "install", "-r", str(req)])

    # ---------------------------------------------------------------- install package
    # Install DSO package without extras to avoid version conflicts
    dso_path = PROJECT / "dso"
    run(["uv", "pip", "install", "-e", str(dso_path)])

    # Show platform-specific activation command
    if sys.platform == "win32":
        activate_cmd = "call .venv\Scripts\activate.bat"
    else:
        activate_cmd = "source .venv/bin/activate"
    print(f"\n✓ Environment ready — activate with:  {activate_cmd}")

if __name__ == "__main__":
    main()
