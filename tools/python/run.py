#!/usr/bin/env python3
"""
Modern helper entry for Deep Symbolic Optimization (Python ≥3.11 + TF-2).

Sub-commands
============
setup            — bootstrap or upgrade the virtual-env in .venv
test  [pytest…]  — run all tests including MIMO with optional pytest flags
bench-miso CFG   — run MISO (Multiple Input Single Output) benchmark
bench-mimo CFG   — run MIMO (Multiple Input Multiple Output) benchmark
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
DSO_PKG = PROJECT / "dso_pkg"

# --------------------------------------------------------------------------- sub-commands
def cmd_setup(ns: argparse.Namespace) -> None:
    os.chdir(PROJECT)
    setup_env.main(ns.forward)

def cmd_test(ns: argparse.Namespace) -> None:
    """Run all tests including MIMO tests."""
    test_runner.run_tests(ns.forward)

def cmd_bench_miso(ns: argparse.Namespace) -> None:
    """Run MISO benchmark - Multiple inputs, single output."""
    print("="*60)
    print("RUNNING MISO BENCHMARK")
    print("="*60)
    
    # Use default MISO benchmark if no config provided
    if not ns.config:
        ns.config = "dso_pkg/dso/config/examples/regression/Nguyen-2.json"
        print(f"Using default MISO config: {ns.config}")
    
    print("MISO: Multiple Inputs → Single Output")
    print("This tests traditional symbolic regression with multiple variables")
    print("="*60)
    
    benchmark_runner.run_benchmark(ns.config, getattr(ns, "benchmark", None))

def cmd_bench_mimo(ns: argparse.Namespace) -> None:
    """Run MIMO benchmark - Multiple inputs, multiple outputs."""
    print("="*60)
    print("RUNNING MIMO BENCHMARK")
    print("="*60)
    
    # Use default MIMO benchmark if no config provided
    if not ns.config:
        ns.config = "dso_pkg/dso/config/examples/regression/MIMO-benchmark.json"
        print(f"Using default MIMO config: {ns.config}")
    
    print("MIMO: Multiple Inputs → Multiple Outputs")
    print("Dataset: MIMO-benchmark (3 inputs → 3 outputs)")
    print("Expressions: [x1*x2, sin(x3), x1+x2*x3]")
    print("="*60)
    
    # For MIMO, we need to use the fixed implementation
    os.chdir(PROJECT)
    cmd = [python_exe(), "-c", f"""
import sys
sys.path.insert(0, {repr(str(DSO_PKG))})
from dso.core_fixed import DeepSymbolicOptimizerFixed
from dso.config import load_config

# Load configuration
config = load_config('{ns.config}')

# Ensure required sections
if "state_manager" not in config:
    config["state_manager"] = {{
        "type": "hierarchical", 
        "observe_parent": True,
        "observe_sibling": True,
        "observe_action": False,
        "observe_dangling": False,
        "embedding": False,
        "embedding_size": 8
    }}

if "checkpoint" not in config:
    config["checkpoint"] = {{"save": False}}

if "logging" not in config:
    config["logging"] = {{
        "save_summary": True,
        "save_all_iterations": False
    }}

# Reduce samples for benchmark demo
config["training"]["n_samples"] = min(5000, config["training"].get("n_samples", 5000))
config["training"]["batch_size"] = min(100, config["training"].get("batch_size", 100))

print('Starting MIMO symbolic regression...')
print(f'Configuration: {ns.config}')

try:
    # Create and setup DSO with MIMO support
    dso = DeepSymbolicOptimizerFixed(config)
    dso.setup()
    
    # Verify initialization
    from dso.program import Program
    print(f'\\n✓ Initialized with {{Program.library.L}} tokens')
    print(f'✓ Input shape: {{Program.task.X_train.shape}}')
    print(f'✓ Output shape: {{Program.task.y_train.shape}}')
    
    # Run training
    print('\\nRunning training steps...')
    for step in range(10):
        result = dso.train_one_step()
        if result is not None:
            print(f'\\nTraining completed at step {{step+1}}!')
            if 'r' in result:
                print(f'Best reward: {{result["r"]:.6f}}')
            if 'expression' in result:
                print(f'Expression: {{result["expression"]}}')
            break
        elif step % 2 == 0:
            if hasattr(dso.trainer, 'p_r_best') and dso.trainer.p_r_best:
                print(f'  Step {{step+1}}: Best reward = {{dso.trainer.p_r_best.r:.6f}}')
    
    print('\\n✅ MIMO BENCHMARK COMPLETED SUCCESSFULLY!')
    
except Exception as e:
    print(f'\\n❌ MIMO BENCHMARK FAILED: {{e}}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""]
    run(cmd)

def interactive_menu() -> None:
    """Interactive menu for DSO operations."""
    pre_req_msg = """
IMPORTANT: Before running setup, ensure the following are installed and added to your PATH:

- Python 3.11 (https://www.python.org/downloads/release/python-3110/)
- SWIG (Simplified Wrapper and Interface Generator)
    - Linux:   sudo apt-get update && sudo apt-get install swig
    - Windows: Download swig from the internet, extract, and add swig.exe to your PATH
- Microsoft C++ Build Tools (Windows only)
    - Download and install from https://visualstudio.microsoft.com/visual-cpp-build-tools/
    - During install, select "Desktop development with C++"

You can verify installation by running:
- python --version
- swig -version
- (Windows) cl.exe (should print Microsoft C/C++ compiler info)
"""
    print(pre_req_msg)
    while True:
        print("==================================================")
        print("DEEP SYMBOLIC OPTIMIZATION - MAIN MENU")
        print("==================================================")
        print("  1) Setup environment")
        print("  2) Run tests (including MIMO)")
        print("  3) Run MISO benchmark (Multiple Input, Single Output)")
        print("  4) Run MIMO benchmark (Multiple Input, Multiple Output)")
        print("  5) Quit")
        print("--------------------------------------------------")
        choice = input("Select option [1-5]: ").strip()
        if choice == "1":
            print("\n🔧 Setting up environment...")
            cmd_setup(argparse.Namespace(forward=[]))
        elif choice == "2":
            print("\n🧪 Running all tests...")
            cmd_test(argparse.Namespace(forward=[]))
        elif choice == "3":
            print("\n📊 MISO Benchmark")
            print("Examples of MISO problems:")
            print("  - Nguyen-1: x^3 + x^2 + x")
            print("  - Nguyen-2: x^4 + x^3 + x^2 + x")
            print("  - Custom: Your own multi-variable expression")
            cfg = input("\nConfig path (or Enter for default): ").strip()
            if not cfg:
                cfg = "dso_pkg/dso/config/examples/regression/Nguyen-2.json"
            cmd_bench_miso(argparse.Namespace(config=cfg))
        elif choice == "4":
            print("\n📊 MIMO Benchmark")
            print("Examples of MIMO problems:")
            print("  - MIMO-simple: 2 inputs → 2 outputs")
            print("  - MIMO-benchmark: 3 inputs → 3 outputs")
            print("  - MIMO-easy: Simple 2x2 case")
            cfg = input("\nConfig path (or Enter for default): ").strip()
            if not cfg:
                cfg = "dso_pkg/dso/config/examples/regression/MIMO-benchmark.json"
            cmd_bench_mimo(argparse.Namespace(config=cfg))
        elif choice == "5":
            print("\n👋 Goodbye!")
            return

# --------------------------------------------------------------------------- CLI parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent(__doc__))
sub = parser.add_subparsers(dest="cmd", required=False)

# Setup command
setup_p = sub.add_parser("setup", help="Setup/update virtual environment")
setup_p.set_defaults(func=lambda _ns, *_: None)

# Test command
test_p = sub.add_parser("test", help="Run all tests including MIMO")
test_p.set_defaults(func=lambda _ns, *_: None)

# MISO benchmark command
miso_p = sub.add_parser("bench-miso", help="Run MISO benchmark")
miso_p.add_argument("config", nargs="?", help="Path to config JSON (optional)")
miso_p.add_argument("--benchmark", help="Optional benchmark selector")

# MIMO benchmark command
mimo_p = sub.add_parser("bench-mimo", help="Run MIMO benchmark")
mimo_p.add_argument("config", nargs="?", help="Path to config JSON (optional)")

# Menu command
menu_p = sub.add_parser("menu", help="Interactive menu")
menu_p.set_defaults(func=lambda _ns, *_: interactive_menu())

# --------------------------------------------------------------------------- dispatch
if __name__ == "__main__":
    # If no arguments, show interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        sys.exit(0)
    
    # Parse arguments
    ns, forward = parser.parse_known_args()
    
    if ns.cmd == "setup":
        cmd_setup(argparse.Namespace(forward=forward))
    elif ns.cmd == "test":
        cmd_test(argparse.Namespace(forward=forward))
    elif ns.cmd == "bench-miso":
        cmd_bench_miso(ns)
    elif ns.cmd == "bench-mimo":
        cmd_bench_mimo(ns)
    elif ns.cmd == "menu":
        interactive_menu()
    else:
        # No command specified, show menu
        interactive_menu()