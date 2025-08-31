# Branching Model

> Version: 1.0 • Last updated: 2025-07-07

The project uses **trunk-based development** with short-lived topic branches.

## 1. Branch Types

| Type    | Prefix     | Purpose                               |
| ------- | ---------- | ------------------------------------- |
| Main    | `main`     | Always deployable & protected         |
| Feature | `feat/`    | New functionality                     |
| Fix     | `fix/`     | Bug fixes                             |
| Chore   | `chore/`   | Maintenance tasks (docs, build, deps) |
| Hotfix  | `hotfix/`  | Critical patch directly off `main`    |
| Release | `release/` | Version stabilisation (rare)          |

## 2. Workflow

1. Create branch: `git switch -c feat/<short-topic>`
2. Commit following Conventional Commits.
3. Open draft PR early; update as work progresses.
4. Rebase frequently on `main`.
5. Squash-merge once CI passes and approvals met.

## 3. Tags & Releases

- Semantic version tags (`v2.3.0`) applied on `main` after release PR merges.

## 4. Hotfix Policy

- Branch `hotfix/<issue>` from `main`.
- Fast-track reviews, merge, then tag `vX.Y.Z+1`.
