---
name: configure-python-quality
description: "Configures, updates, runs, troubleshoots, and reviews Python quality tooling using the team's Ruff, mypy, pre-commit, gitleaks, bidirectional-Unicode protection, notebook cleanup, gitignore, and Git LFS conventions. Use when bootstrapping a Python repository, adding or changing lint and type checks, setting up pre-commit hooks, fixing quality-tool failures, standardizing project configs, or improving repository hygiene."
---

# Configure Python Quality

Configure quality checks as executable project policy. Take the configs from the team's linter-config repository, which holds the current versions, and adapt them to the project rather than copying blindly.

## Workflow

1. Read Python version, source/test layout, pyproject.toml, existing tool configs, CI jobs, and repository instructions.
2. Read [the quality tooling reference](references/quality-tooling.md) for the config source and per-tool rules.
3. Take only the required configs from the linter-config repository, then replace project-specific paths and first-party package names.
4. Add compatible dev dependencies through uv and preserve the existing lock strategy.
5. Run each tool directly before running the full pre-commit suite.
6. Fix configuration errors, then code findings; do not weaken rules merely to make checks green.
7. Ensure the same commands run locally and in CI.

## Rules

- The linter-config repository is the only source of the configs. This skill ships no copies.
- When that repository is unreachable, report it and limit changes to what the project already contains. Do not reconstruct the configs from memory.
- Normalize the Python target to the team baseline when the source config targets an older version.
- Pin hook revisions and update them intentionally.

## Adjacent skills

Hand off when the skill is installed; otherwise apply this skill's own rules and continue: $develop-python (the code changes exposed by these checks), $manage-git-workflow (Git LFS decisions).
