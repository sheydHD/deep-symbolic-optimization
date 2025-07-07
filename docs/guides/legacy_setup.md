# Legacy Compatibility Guide (Python 3.6 + TensorFlow 1.14)

> Version: 1.0 • Last updated: 2025-07-07

Deep Symbolic Optimization was originally written for Python 3.6 and TensorFlow 1.14.  
While the mainline code now targets modern Python & TF versions, we keep **branch `compat-py36`** plus frozen dependency files so that older experiments remain reproducible.

This page walks you through creating an isolated legacy environment, running the unit tests, and reproducing a small benchmark run.

---

## 1 When should I use the legacy stack?

- You need **bit-exact replication** of results published with TF 1.x.
- You must run DSO on hardware / OS images that cannot be upgraded.
- You are comparing algorithmic changes against the historical baseline.

If none of the above apply, use the modern installation described in [Guides → Installation](installation.md).

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

> We pin 3.6.15 because it is the last CPython 3.6 patch and matches our pre-compiled C-extensions.

---

## 3 Clone & check-out the compatibility branch

```bash
git clone https://github.com/your-org/dso.git
cd dso
git switch compat-py36      # contains minor shims for TF 1.14/NumPy 1.19
```

---

## 4 Create the virtual environment & install deps

```bash
# inside repo root, still in the dso-legacy36 shell
python -m venv .venv          # classic venv; uv doesn't support 3.6
source .venv/bin/activate

# Upgrade packaging tooling
pip install --upgrade pip setuptools wheel

# Install pinned legacy requirements
pip install -r configs/requirements/requirements-legacy36.txt

# Editable install so changes reflect immediately
pip install -e ./dso
```

The long `requirements-legacy36.txt` locks:

- TensorFlow **1.14.0** (GPU flavour optional)
- NumPy **1.19.5** (latest binary compatible with TF 1.14)
- Legacy versions of SciPy, SymPy, Gym, etc.

---

## 5 Run the unit tests

From the repository **root**:

```bash
# Option A – change into package dir first (simplest)
cd dso/dso
pytest -q                     # 35 tests, ~3 min on CPU

# Option B – stay in root and give pytest a rootdir
pytest -q --rootdir dso/dso dso/dso/test
```

On a fresh install you should see

```
35 passed, 0 failed, X warnings in ~180 s
```

(Thousands of DeprecationWarnings are normal with TF 1.x.)

---

## 6 Run a quick symbolic-regression benchmark

We ship miniature "Nguyen" datasets under `dso/dso/task/regression/data/`.
Try a 1-seed run on _Nguyen-5_ (≈20 s on CPU):

```bash
python -m dso.run dso/config/examples/regression/Nguyen-2.json \
    --benchmark Nguyen-5          # picks Nguyen-5.csv / Nguyen-5_test.csv
```

The script prints progress every few iterations and writes logs to
`runs/<timestamp>/…`.  
Interrupt with `Ctrl-C` any time; partial results are still saved.

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

The first run downloads the Gym environment & weights (~ 50 MB).

---

## 8 Troubleshooting

| Symptom                                                       | Fix                                                                   |
| ------------------------------------------------------------- | --------------------------------------------------------------------- |
| `ImportError: numpy.core.multiarray failed to import`         | Ensure NumPy 1.19.x; _never_ upgrade to 1.20+ in the legacy env.      |
| `ModuleNotFoundError: No module named 'tensorflow.compat.v1'` | You installed TF 2.x by mistake; wipe venv and reinstall.             |
| Random segfaults on macOS 11+                                 | Use an x86_64 Python build under Rosetta, or the modern TF 2.x stack. |

---

🎉 You now have a fully-reproducible legacy environment! If you hit any snags, open an issue or drop by the discussion board.
