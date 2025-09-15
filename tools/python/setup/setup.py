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

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[3]  # deep-symbolic-optimization/
REQDIR = PROJECT / "configs" / "requirements"


def run(cmd: list[str]) -> None:
    """Run a subprocess with live output and hard-fail on non-zero exit."""
    print("➜", *cmd, flush=True)
    subprocess.run(cmd, check=True)


def ensure_uv(python: str) -> None:
    try:
        subprocess.run(
            ["uv", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        run([python, "-m", "pip", "install", "--upgrade", "pip", "uv"])


def compile_requirements(name: str) -> Path:
    """
    Compile REQDIR / f"{name}.in" to REQDIR / f"{name}.txt" using uv pip compile.
    Handles platform-specific exclusions for Windows/CUDA if necessary.
    """
    src = REQDIR / "in_files" / f"{name}.in"  # Corrected path
    if not src.exists():
        raise FileNotFoundError(f"Source file {src.name} not found in {REQDIR}")

    txt = REQDIR / f"{name}.txt"
    temp_src_path = None

    if sys.platform == "win32" and name in ["dev", "extras"]:
        # For Windows, exclude CUDA-related dependencies from dev.in and extras.in
        # Create a temporary .in file without CUDA dependencies
        original_content = src.read_text()
        filtered_lines = []

        # Add explicit numpy pinning for Windows to avoid conflicts
        # This ensures that all .in files contribute to a consistent numpy version
        # when compiling to .txt files.
        filtered_lines.append("numpy==1.26.0")

        for line in original_content.splitlines():
            # Exclude lines that specifically mention cuda or tensorflow with cuda
            if (
                "cuda" in line.lower()
                or "tensorflow[and-cuda]" in line.lower()
                or "tensorrt" in line.lower()
                or "nvidia" in line.lower()
            ):  # Added more aggressive filtering
                filtered_lines.append(f"# {line} # Excluded for Windows (CUDA)")
            elif (
                line.strip().startswith("numpy") and "==" not in line
            ):  # Exclude other numpy lines if they are not explicitly pinned
                filtered_lines.append(
                    f"# {line} # Excluded for Windows (Numpy handled by explicit pin)"
                )
            else:
                filtered_lines.append(line)
        filtered_content = "\n".join(filtered_lines)

        temp_src_path = src.with_name(f"{src.name}.tmp")
        temp_src_path.write_text(filtered_content)
        src = temp_src_path
        print(
            f"WARNING: CUDA-related dependencies (via {name}.in) will be excluded from {txt.name} compilation on Windows.",
            file=sys.stderr,
        )
        print(
            "If you need CUDA support, please install a compatible CUDA toolkit and NVIDIA cuDNN manually.",
            file=sys.stderr,
        )

    print(f"⚙️  Compiling {src.name} → {txt.name}")
    run(["uv", "pip", "compile", str(src), "--output-file", str(txt)])

    if temp_src_path and temp_src_path.exists():
        temp_src_path.unlink()  # Clean up temporary file

    return txt


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Create / upgrade .venv for Deep Symbolic Optimization"
    )
    ap.add_argument(
        "--fresh", action="store_true", help="Delete any existing .venv first"
    )
    args = ap.parse_args(argv)

    python = sys.executable
    ensure_uv(python)

    venv = PROJECT / ".venv"
    if args.fresh and venv.exists():
        shutil.rmtree(venv)
    if not venv.exists():
        run(["uv", "venv", "-p", python])

    # ---------------------------------------------------------------- compile and install deps
    for layer in ("core", "extras", "dev"):
        req = compile_requirements(layer)  # Use the new compile_requirements function
        try:
            run(["uv", "pip", "install", "-r", str(req)])
        except subprocess.CalledProcessError as e:
            if layer == "extras" and sys.platform == "win32":
                print("\n" + "*" * 80, file=sys.stderr)
                print(
                    "ERROR: Failed to install 'extras.txt' dependencies.",
                    file=sys.stderr,
                )
                print(
                    "This often happens on Windows if you are missing system-level build tools.",
                    file=sys.stderr,
                )
                print(
                    "Please ensure you have the following installed and configured:",
                    file=sys.stderr,
                )
                print("", file=sys.stderr)
                print(
                    "  1. SWIG (Simplified Wrapper and Interface Generator):",
                    file=sys.stderr,
                )
                print(
                    "     Download from: [https://sourceforge.net/projects/swig/](https://sourceforge.net/projects/swig/)",
                    file=sys.stderr,
                )
                print(
                    "     Add the extracted SWIG directory (e.g., C:\Program Files\swigwin) to your system's PATH environment variable.",
                    file=sys.stderr,
                )
                print("", file=sys.stderr)
                print(
                    "  2. Microsoft Visual C++ Build Tools (version 14.0 or greater):",
                    file=sys.stderr,
                )
                print(
                    "     Download from: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)",
                    file=sys.stderr,
                )
                print(
                    "     Ensure to select 'Desktop development with C++' workload during installation.",
                    file=sys.stderr,
                )
                print("", file=sys.stderr)
                print(
                    "After installing these tools, open a NEW terminal and run setup again.",
                    file=sys.stderr,
                )
                print("*" * 80 + "\n", file=sys.stderr)
                sys.exit(1)  # Exit to indicate failure requiring manual intervention
            else:
                raise e  # Re-raise if not the specific error we're handling

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
