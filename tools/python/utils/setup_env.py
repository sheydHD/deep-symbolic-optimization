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
    venv_python = PROJECT / ".venv" / "bin" / "python"
    venv_pip = PROJECT / ".venv" / "bin" / "pip"
    if pyproject.exists():
        # Use uv sync groups (core + optional groups dev & extras)
        run(["uv", "pip", "install", "numpy>=1.26,<2.0"])  # ensure numpy headers for cython
        try:
            run(["uv", "sync"])  # install core deps + build
        except Exception as e:
            # On Windows, tensorflow-io-gcs-filesystem may fail to install, but is not required
            if sys.platform == "win32":
                print("[setup-env] Detected Windows and tensorflow-io-gcs-filesystem install error. Attempting to continue...")
                # Try to uninstall if partially installed
                try:
                    run(["uv", "pip", "uninstall", "-y", "tensorflow-io-gcs-filesystem"])
                except Exception:
                    pass
                # Try to continue with the rest of the install
            else:
                raise e
        # Always use the venv pip for editable install
        # Use 'uv pip install' instead of direct pip path for uv venv compatibility
        run(["uv", "pip", "install", "-e", ".[extras,dev]"])
        return
    # If pyproject.toml is missing, raise an error (legacy .in/.txt not supported)
    raise RuntimeError("pyproject.toml not found. Please ensure you are in the project root and have a valid pyproject.toml.")

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
