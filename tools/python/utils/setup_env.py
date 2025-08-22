"""Virtual environment bootstrap / upgrade logic split from run.py.

Provides main() callable used by CLI front-end.
"""
from __future__ import annotations

import argparse, shutil, subprocess, sys
from pathlib import Path
from .env import PROJECT, run

REQDIR = PROJECT / "configs" / "requirements"

def ensure_uv(python: str) -> None:
    try:
        subprocess.run(["uv", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        run([python, "-m", "pip", "install", "--upgrade", "pip", "uv"])

def compile_requirements(name: str) -> Path:
    src = REQDIR / "in_files" / f"{name}.in"
    if not src.exists():
        raise FileNotFoundError(f"Source file missing: {src}")
    out = REQDIR / f"{name}.txt"
    print(f"⚙️  Compiling {src.name} → {out.name}")
    run(["uv", "pip", "compile", str(src), "--output-file", str(out)])
    return out

def install_layers() -> None:
    pyproject = PROJECT / "pyproject.toml"
    if pyproject.exists():
        # Use uv sync groups (core + optional groups dev & extras)
        # First sync core deps
        try:
            run(["uv", "pip", "install", "numpy>=1.26,<2.0"])  # ensure numpy headers for cython
            run(["uv", "sync"])  # install core deps + build
            # install optional extras & dev toolchain via extras bracket
            run(["uv", "pip", "install", "-e", ".[extras,dev]"])  # editable with extras
            return
        except subprocess.CalledProcessError:
            print("[setup-env] uv sync failed. Falling back to editable install via inner setup.py (core only)", file=sys.stderr)
            run(["uv", "pip", "install", "-e", "dso"])  # minimal fallback
            return
    # Legacy path (.in compilation)
    for layer in ("core", "extras", "dev"):
        req = compile_requirements(layer)
        run(["uv", "pip", "install", "-r", str(req)])
    run(["uv", "pip", "install", "-e", str(PROJECT / "dso")])

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Create / upgrade .venv for DSO")
    ap.add_argument("--fresh", action="store_true", help="Recreate .venv from scratch")
    # Enforce Python 3.11 runtime early (matches pyproject constraint)
    if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
        print(f"[setup-env] Python 3.11 required; current interpreter: {sys.version.split()[0]}", file=sys.stderr)
        print("Recreate venv with: uv venv -p $(pyenv which python3.11) && source .venv/bin/activate", file=sys.stderr)
        sys.exit(1)
    args = ap.parse_args(argv)

    python = sys.executable
    ensure_uv(python)
    venv = PROJECT / ".venv"
    if args.fresh and venv.exists():
        shutil.rmtree(venv)
    if not venv.exists():
        run(["uv", "venv", "-p", python])
    install_layers()
    activate = "call .venv\\Scripts\\activate.bat" if sys.platform == "win32" else "source .venv/bin/activate"
    print(f"\n✓ Environment ready — activate with: {activate}")

if __name__ == "__main__":  # pragma: no cover
    main()
