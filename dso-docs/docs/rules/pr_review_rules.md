# Pull Request Creation & Review Rules

> Version: 1.0 • Last updated: 2025-07-07

These rules standardise how we collaborate on GitHub.

## 1. Opening a PR

1. **Branch naming**: `<type>/<short-description>` – see branching model (e.g., `feat/gp-tuner`, `fix/memory-leak`).
2. **Title** uses Conventional Commits: `<type>(scope): message`  
   Example: `feat(core): parallel pool optimisations`.
3. Fill out the PR template:
   - Motivation & Context
   - Linked Issue(s) (`Fixes #123`)
   - Checklist
   - Screenshots / Benchmarks (if UI or perf related)

## 2. Author Checklist

- [ ] Lint passes (`make lint`)
- [ ] Tests pass (`make test`)
- [ ] Coverage ≥ target (`make coverage`)
- [ ] Docs updated (if applicable)
- [ ] No secrets committed (`git secrets --scan`)
- [ ] Changelog entry added (`CHANGELOG.md`)

## 3. Review Process

- **≥1** approval for docs / test-only PRs.
- **≥2** approvals for code touching `dso/`.
- Reviewers focus on correctness, readability, security, performance, and docs.
- Use GitHub **Suggested Changes** when possible.

## 4. Resolving Feedback

- Address every comment or mark as _won't fix_ with rationale.
- Re-request review after pushing updates.

## 5. Merging

- Use **Squash & Merge** to keep `main` linear.
- Rebase on latest `main` before merging: `git pull --rebase origin main`.
- Delete the feature branch after merge.

## 6. Reverts & Hotfixes

- Use `revert:` prefix in commit and PR title; raise with high priority.
