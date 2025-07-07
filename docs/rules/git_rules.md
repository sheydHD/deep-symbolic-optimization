# Git Rules

> Version: 1.0 • Last updated: 2025-07-07

## 1. Commit Conventions

- Use **Conventional Commits**; linted by commit-lint hook.
- Keep subject ≤ 72 chars; use present tense: `fix(core): handle None input`.
- Body lines wrapped at 80 chars; explain _what_ and _why_.
- Include issue reference (`Fixes #123`) when applicable.

## 2. Grouping Changes

- One logical change per commit.
- Run `git add -p` to stage hunks selectively.
- Unrelated file updates must go into separate commits or PRs.

## 3. Branching & Merging

- Follow [`branching_model.md`](branching_model.md).
- Use **Squash & Merge** to keep history linear.
- Delete remote branches after merge to reduce clutter.

## 4. Signatures & DCO

- Sign commits where possible (`git commit -S`).
- Ensure `Signed-off-by` trailer for external contributions.

## 5. Large Files

- Use Git LFS for binaries >100 KB (if needed).
- Never commit datasets; rely on DVC or external storage.

## 6. Hooks & Automation

- Pre-commit runs lint, format, security, and type checks.
- CI blocks merges on failing checks.
