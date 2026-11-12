---
name: manage-git-workflow
description: "Manages the team's Git workflow, branch and commit naming, merge requests, stacked MRs, Semantic Versioning, Keep a Changelog, repository naming, README expectations, releases, tags, and Git LFS. Use when planning or performing branch work, preparing commits or MRs, splitting large changes, choosing release versions, updating changelogs, organizing repositories, or handling large binary artifacts."
---

# Manage Git Workflow

Use the team's simplified flow with permanent main and test branches. Preserve user changes and never rewrite shared history by default.

## Workflow

1. Read repository instructions and inspect status, current branch, relevant diff, and recent message style.
2. Read [the Git workflow](references/git-workflow.md) for branches, commits, MRs, repositories, and LFS.
3. For release, version, or changelog work, also read [the versioning rules](references/versioning.md).
4. Choose the smallest coherent branch, commit, MR, or release plan.
5. Apply only the Git mutations requested by the user.
6. Verify the resulting diff, metadata, and release consistency.

## Rules

- Base commit and changelog text on the actual diff.
- Write commit summaries in Russian unless repository precedent requires another language.
- Do not add Co-authored-by trailers.
- Keep one MR to one feature or fix; split changes over roughly 500 lines into logical stacked MRs.
- Choose SemVer from user-visible compatibility rather than the commit label.
- Never push, tag remotely, merge, rebase shared work, or force-push without explicit authorization.

## Adjacent skills

Hand off when the skill is installed; otherwise apply this skill's own rules and continue: $configure-gitlab-ci (release, publish, and deployment pipelines), $configure-python-quality (repository hygiene and pre-commit).
