from pathlib import Path
from setuptools import setup, find_packages, Extension
import numpy as np

# --------------------------------------------------------------------------- paths
HERE     = Path(__file__).parent.resolve()               # …/deep-symbolic-optimization/dso
SRC_DIR  = HERE / "dso"                                  # ★ inner package dir
REQ_DIR  = HERE.parent / "configs" / "requirements"

# --------------------------------------------------------------------------- helpers (unchanged)
def _read(fname: str, fallback=None):
    path = REQ_DIR / fname
    if not path.is_file():
        return fallback or []
    with path.open() as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith(("#", "-e"))]

core_requires   = _read("core.in")
extras_requires = _read("extras.in")

if not core_requires:  # sdist / CI fallback
    core_requires = [
        "numpy>=1.26","cython>=3.0","tensorflow>=2.13,<2.16",
        "pandas>=1.5","scikit-learn>=1.4","sympy>=1.12","click>=8.1",
        "tqdm>=4.66","pyyaml>=6.0","prettytable>=3.9","deap>=1.4","pathos>=0.3",
    ]

extras = {
    "control": extras_requires or [
        "gymnasium[box2d]>=0.29","pybullet>=3.2","stable-baselines3>=2.3",
    ],
    "docs": ["mkdocs-material[imaging]>=9.2","mkdocstrings[python]>=0.23",
             "mkdocs-awesome-pages-plugin>=2.9","pymdown-extensions>=10"],
    "dev":  ["pytest>=8","pytest-cov>=4","black>=23","isort>=5.13",
             "flake8>=7","mypy>=1.10","pip-tools>=7","uv>=0.1"],
}
extras["all"] = sorted({p for group in extras.values() for p in group})

# --------------------------------------------------------------------------- cython
extensions = [
    Extension(
        name="dso.cyfunc",
        sources=[str(SRC_DIR / "cyfunc.pyx")],            # ★ corrected path
        include_dirs=[np.get_include()],
        language="c++",
    )
]

from Cython.Build import cythonize
ext_modules = cythonize(
    extensions,
    language_level="3",
    compiler_directives={"boundscheck": False, "wraparound": False},
)

# --------------------------------------------------------------------------- setup()
setup(
    name="dso",
    version="2.0.0",
    description="Deep Symbolic Optimization (TF-2 branch)",
    author="LLNL",
    packages=find_packages(where="dso", include=["dso", "dso.*"]),
    package_dir={"dso": "dso"},          # import name → inner folder
    python_requires=">=3.11",
    install_requires=core_requires,
    extras_require=extras,
    ext_modules=ext_modules,
    include_package_data=True,
    entry_points={"console_scripts": ["dso = dso.cli:main"]},
    zip_safe=False,
)
