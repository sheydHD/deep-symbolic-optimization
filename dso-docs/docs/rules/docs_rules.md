# Documentation Rules

> Version: 1.0 • Last updated: 2025-07-07

These guidelines apply to READMEs, API docs, wikis, and internal guides.

## 1. Structure

- Use **Markdown** (`.md`) for all docs.
- Begin with a clear _Title_ and _Purpose_ section.
- Use headings in logical hierarchy (`#`, `##`, `###`).
- Include a Table of Contents if file > 2 pages.
- Parent directories (levels 1–2) require a `README.md`; deeper levels **do not**.

## 2. Language & Style

- Clear, concise, professional English.
- Active voice; avoid jargon or define it.
- Short sentences and paragraphs.

## 3. Formatting

- Bullet or numbered lists to improve readability.
- Bold or `inline code` for key terms.
- Code blocks (```python) for multi-line snippets.
- Tables for configuration values or comparisons.

## 4. Versioning & Changelog

- Add _Version_ and _Last updated_ at top.
- Maintain project-wide `CHANGELOG.md`.

## 5. Code Documentation

- Each script: header comment with purpose, author, usage.
- Public functions/classes: Google-style docstrings.
- Link to external references where helpful.

## 6. Review & Ownership

- Docs reviewed in every PR.
- Assign a documentation owner for each repo area.
- Keep docs current with codebase.

## 7. Git Practices

- Commits touching docs use `docs:` Conventional Commit type.
- Docs must accompany significant code changes.

## 8. Related Policies

- Source code conventions: [`code_style.md`](code_style.md)
- Git workflow & commits: [`git_rules.md`](git_rules.md)
