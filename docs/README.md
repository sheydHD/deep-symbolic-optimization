# DSO Documentation Hub

> Version: 1.0 • Last updated: 2025-07-07

Welcome to the **Deep Symbolic Optimization (DSO)** documentation hub.
This site follows an industry-standard hierarchy that separates _guides_, _tutorials_, _reference_, and _design_ documents to serve both new users and core developers.

---

## Sections

| Category     | Purpose                                         |
| ------------ | ----------------------------------------------- |
| Overview     | High-level introduction and system context      |
| Guides       | Step-by-step how-tos and operational procedures |
| Tutorials    | End-to-end examples & notebooks                 |
| Reference    | Auto-generated API & CLI references             |
| Architecture | Design decisions, diagrams, and ADRs            |
| Operations   | Release, maintenance, and on-call runbooks      |
| Rules        | Project-wide policies and standards             |
| Legal        | Code of Conduct, security, and licensing info   |

---

## Navigation

```
/docs
├── README.md                # You are here
├── overview.md              # Product summary & scope
├── guides/                  # User & operator guides
├── tutorials/               # Hands-on walkthroughs
├── reference/               # API / CLI reference docs (auto-generated)
├── architecture/            # Design docs & ADRs
├── operations/              # Release & maintenance docs
├── rules/                   # Policies (coding, git, docs…)
└── legal/                   # Governance, security, conduct
```

Each sub-directory contains its own `README.md` to aid discoverability (except where a single document suffices).

---

## Building Docs Locally

```bash
uv run mkdocs serve  # Live-reload at http://127.0.0.1:8000/
```

If MkDocs isn't installed yet, run:

```bash
uv pip install mkdocs-material mkdocstrings[python]
```

---

For contribution guidelines, see [`docs/rules`](rules/).
