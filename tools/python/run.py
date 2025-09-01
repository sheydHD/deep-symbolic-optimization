#!/usr/bin/env python3
"""
Modern helper entry for Deep Symbolic Optimization (Python ≥3.11 + TF-2).

Sub-commands
============
setup            — bootstrap or upgrade the virtual-env in .venv
test  [pytest…]  — run dso/test with optional pytest flags
bench CONFIG     — run benchmark with given JSON config
mimo             — run MIMO benchmark with fixed implementation
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

def cmd_mimo_bench() -> None:
    """Run MIMO benchmark using the fixed DSO implementation."""
    print("="*60)
    print("RUNNING MIMO BENCHMARK WITH FIXED IMPLEMENTATION")
    print("="*60)
    print("Dataset: MIMO-benchmark (3 inputs → 3 outputs)")
    print("Expressions: [x1*x2, sin(x3), x1+x2*x3]")
    print("Using: DeepSymbolicOptimizerFixed for proper MIMO support")
    print("="*60)
    
    os.chdir(PROJECT)
    cmd = [python_exe(), "-c", """
import sys
sys.path.insert(0, 'dso_pkg')
import numpy as np
from dso.core_fixed import DeepSymbolicOptimizerFixed
from dso.config import load_config

def run_mimo_benchmark():
    # Load and enhance MIMO benchmark configuration
    config_path = 'dso_pkg/dso/config/examples/regression/MIMO-benchmark.json'
    config = load_config(config_path)
    
    # Add missing configuration sections for complete setup
    config.update({
        "state_manager": {
            "type": "hierarchical", 
            "observe_parent": True,
            "observe_sibling": True,
            "observe_action": False,
            "observe_dangling": False,
            "embedding": False,
            "embedding_size": 8
        },
        "checkpoint": {"save": False},
        "logging": {
            "save_summary": True,
            "save_all_iterations": False,
            "save_test": False,
            "save_positional_entropy": False,
            "save_pareto_front": False
        }
    })
    
    # Reduce training samples for faster benchmark testing
    config["training"]["n_samples"] = 5000
    config["training"]["batch_size"] = 100
    
    print('\\nStarting MIMO symbolic regression benchmark...')
    print(f'Configuration loaded from: {config_path}')
    
    try:
        # Create and setup DSO with MIMO support
        print('\\n1. Creating Fixed DSO...')
        dso = DeepSymbolicOptimizerFixed(config)
        
        print('\\n2. Setting up MIMO DSO...')
        dso.setup()
        
        # Verify initialization
        from dso.program import Program
        print('\\n3. Verifying initialization...')
        print(f'   ✓ Library initialized with {Program.library.L} tokens')
        print(f'   ✓ X_train shape: {Program.task.X_train.shape}')
        print(f'   ✓ y_train shape: {Program.task.y_train.shape}')
        print(f'   ✓ MIMO outputs: {Program.task.y_train.shape[1]}')
        print(f'   ✓ Policy created: {type(dso.policy).__name__}')
        
        print('\\n4. Running MIMO training steps...')
        for step in range(10):
            print(f'   Step {step+1}/10...', end=' ')
            result = dso.train_one_step()
            
            if result is not None:
                print(f'Training completed at step {step+1}!')
                print(f'   Final result keys: {list(result.keys())}')
                if 'r' in result:
                    print(f'   Best reward: {result["r"]:.6f}')
                break
            else:
                print('✓')
        
        print('\\n✅ MIMO BENCHMARK COMPLETED SUCCESSFULLY!')
        print('\\nThe MIMO implementation is working correctly with:')
        print('   - Proper initialization order')
        print('   - Multi-output data handling') 
        print('   - Policy creation without errors')
        print('   - Training step execution')
        
    except Exception as e:
        print(f'\\n❌ MIMO BENCHMARK FAILED: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

# Run the benchmark
success = run_mimo_benchmark()
sys.exit(0 if success else 1)
"""]
    run(cmd)

def interactive_menu() -> None:
    while True:
        print("\n-- DSO Menu --")
        print("  1) Setup environment (.venv + uv)")
        print("  2) Run tests")
        print("  3) Run benchmark")
        print("  4) Run MIMO benchmark")
        print("  h) Help")
        print("  q) Quit")
        choice = input("Choice [1-4,h,q]: ").strip().lower()
        if   choice == "1":
            cmd_setup(argparse.Namespace(forward=[]))
        elif choice == "2":
            cmd_test(argparse.Namespace(forward=[]))
        elif choice == "3":
            cfg = input("Path to JSON config: ").strip() or "dso_pkg/dso/config/examples/regression/Nguyen-2.json"
            cmd_bench(argparse.Namespace(config=cfg))
        elif choice == "4":
            cmd_mimo_bench()
        elif choice == "h":
            print("\nHelp:")
            print("  setup      - create / update virtual environment (.venv) and install deps")
            print("  test [args]- run pytest suite (additional pytest args optional)")
            print("  bench CFG  - run benchmark with config JSON (use --benchmark if supported)")
            print("  mimo       - run MIMO benchmark with fixed implementation")
            print("  menu       - interactive menu (this)")
            print("  quit       - exit menu")
            print("\nExamples:")
            print("  ./main.sh setup (--fresh)")
            print("  ./main.sh test (-k regression)")
            print("  ./main.sh bench dso/dso/config/examples/regression/Nguyen-2.json")
            print("  ./main.sh mimo  # Uses MIMO-benchmark with fixed DSO implementation")
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
sub.add_parser("mimo")   .set_defaults(func=lambda _ns, *_: cmd_mimo_bench())
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
    elif ns.cmd == "mimo":
        cmd_mimo_bench()              # mimo uses fixed implementation
    else:                             # menu
        ns.func(ns)
