# Deep Symbolic Optimization Documentation

> Version: 1.0 • Last updated: 2025-07-07

<p align="center">
<img src="attachments/images/banner.png" width=750/>
</p>

Welcome to the **Deep Symbolic Optimization (DSO)** documentation hub.
This site follows an industry-standard hierarchy that separates _guides_, _tutorials_, _reference_, and _design_ documents to serve both new users and core developers.

---

## Sections

| Section                     | Content                           |
| --------------------------- | --------------------------------- |
| [Guides](docs/guides)       | Installation, getting started     |
| [Rules](docs/rules)         | Code style, git policies, testing |
| [Structure](docs/structure) | Project layout and architecture   |
| [Tasks](docs/tasks)         | Development plans and roadmaps    |

---

## Navigation

```bash
/docs
├── README.md                # You are here
├── overview.md              # Product summary & scope
├── guides/                  # User & operator guides
├── rules/                   # Release & maintenance docs
├── structure/               # Policies (coding, git, docs…)
└── tasks/                   # Governance, security, conduct
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

For contribution guidelines, see the [Rules overview](rules/README.md).
