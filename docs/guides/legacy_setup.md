# Legacy Compatibility Guide (Python 3.6 + TensorFlow 1.14)

> **Version 1.1 • Last updated: 2025-07-11**

Deep Symbolic Optimization (DSO) was originally written for Python 3.6 and TensorFlow 1.14. While the mainline code now targets modern Python & TF versions, we keep **branch `compat-py36`** plus frozen dependency files so that older experiments remain reproducible.

There are **two ways** to get the legacy stack running:

1. **Interactive CLI (recommended – zero manual steps)**

```bash
# Unix/macOS
./main.sh          # opens menu – choose Setup → Legacy

# Windows
main.bat           # same menu in cmd.exe / PowerShell
```

The CLI will:

1. Install/verify **pyenv** + Python 3.6.15 (or prompt on Windows for pyenv-win).
2. Create `.venv_36` inside the repo.
3. Ask whether you want the light **Regression-only** or **Full legacy** install.
4. Offer to run the unit tests and a quick Nguyen-5 benchmark.

5. **Manual setup** – follow the steps below if you prefer full control or need to script parts of the process.

The remainder of this document keeps the original manual instructions (Steps 2-8). Feel free to skip straight to **Step 5** after running the CLI.

---

## 1 When should I use the legacy stack?

- You need **bit-exact replication** of results published with TF 1.x.
- You must run DSO on hardware / OS images that cannot be upgraded.
- You are comparing algorithmic changes against the historical baseline.

If none of the above apply, use the modern installation described in <kbd>Guides → Installation</kbd>.

---

## 2 Install Python 3.6.15 via `pyenv`

```bash
# Install pyenv (skip if already installed)
curl https://pyenv.run | bash
# Restart shell or run the commands printed by the installer, then:
pyenv install 3.6.15
pyenv virtualenv 3.6.15 dso-legacy36
pyenv activate dso-legacy36   # or: pyenv shell dso-legacy36
```

> We pin **3.6.15** because it is the last CPython 3.6 patch and matches our pre-compiled C-extensions.

---

## 3 Clone & check out the compatibility branch

```bash
git clone https://github.com/your-org/dso.git
cd dso
git switch compat-py36      # contains minor shims for TF 1.14 / NumPy 1.19
```

---

## 4 Create the project venv & install dependencies

You are already inside `pyenv`'s **global** Python 3.6.15. We now create a **project-local** venv so packages stay sandboxed.

```bash
python -m venv .venv          # classic venv; uv doesn't support 3.6
source .venv/bin/activate

# Upgrade packaging tooling
pip install --upgrade pip setuptools wheel
```

### 4A Regression-only environment (💡 fastest & lightest)

If you **only care about symbolic-regression experiments** (Nguyen, Feynman, etc.) run:

```bash
pip install -e ./dso
```

This locks:

- TensorFlow **1.14.0** (GPU flavour optional)
- NumPy **1.19.5** (latest binary compatible with TF 1.14)
- Legacy SciPy, SymPy, Pandas, Scikit-learn, etc.

### 4B Classification / Control-task extras (optional, heavier)

Reinforcement-learning or control benchmarks (e.g. MountainCar, LunarLander) require extra physics environments.

```bash
# Requirements pinned to the last Py 3.6-compatible versions
pip install -r configs/requirements/requirements-legacy36.txt
```

---

## 5 Run the unit tests

All commands assume **repo root** (`dso/`).

### 5A Regression tests ✅

```bash
pytest -q dso/task/regression/
```

(Thousands of DeprecationWarnings are normal with TF 1.x.)

### 5B Classification / Control tests (optional)

After installing the extras in 4B:

```bash
pytest -q dso/task/control/
```

If you prefer to run _all_ tests (regression + control) in one go:

```bash
pytest -q
```

---

## 6 Run a quick symbolic-regression benchmark

We ship miniature "Nguyen" datasets under `dso/dso/task/regression/data/`.
Try a **1-seed** run on _Nguyen-5_ (≈ 20 s on CPU):

```bash
cd /dso
python -m dso.run dso/config/examples/regression/Nguyen-2.json \
    --benchmark Nguyen-5          # picks Nguyen-5.csv / Nguyen-5_test.csv
```

Script prints progress every few iterations and writes logs to `runs/<timestamp>/…`.
Interrupt with <kbd>Ctrl-C</kbd> any time; partial results are preserved.

Need multiple seeds in parallel?

```bash
python -m dso.run dso/config/examples/regression/Nguyen-2.json \
    --runs 4 --n-cores-task 4 --seed 123
```

---

## 7 Control-task example (optional)

If you installed the _[control]_ extras:

```bash
python -m dso.run dso/config/examples/control/MountainCar-v0.json
```

The first run downloads the Gym environment & weights (≈ 50 MB).

---

## 8 Troubleshooting

| Symptom                                                                 | Fix                                                                             |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `ImportError: numpy.core.multiarray failed to import`                   | Ensure NumPy 1.19.x; _never_ upgrade to ≥1.20 in the legacy env.                |
| `ModuleNotFoundError: No module named 'tensorflow.compat.v1'`           | You installed TF 2.x by mistake; wipe venv and reinstall (Step 4A).             |
| `ModuleNotFoundError: No module named 'gym'` when running control tests | Install extras (Step 4B) or skip control tests (Step 5A only).                  |
| Random segfaults on macOS 11+                                           | Use an x86_64 Python build under Rosetta, or switch to the modern TF 2.x stack. |

---

🎉 You now have a fully-reproducible legacy environment!
If you hit any snags, open an issue or drop by the discussion board.
