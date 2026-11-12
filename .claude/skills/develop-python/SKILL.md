---
name: develop-python
description: "Implements, modifies, debugs, tests, refactors, and reviews Python 3.12+ code using the team's engineering conventions for typing, uv dependency management, Pydantic configuration, Ruff, mypy, pytest, logging, docstrings, immutability, imports, and maintainability. Use for Python application or library development, project setup, feature work, bug fixes, refactoring, tests, notebooks, and Python code review."
---

# Develop Python

Apply the team baseline to Python work while respecting more specific repository instructions and pinned tool configuration.

## Workflow

1. Read repository instructions, project metadata, dependency configuration, and relevant tests.
2. Define the requested behavior and inspect affected callers, data contracts, resources, and error paths.
3. Read [the Python standard](references/python-standard.md).
4. Implement the smallest complete change with explicit types, Pydantic configuration where appropriate, module logging, and focused tests.
5. Run the repository's Ruff, mypy, and pytest commands in proportion to the change.
6. Review the final diff for regressions, accidental dependency churn, secret exposure, and unrelated edits.

## Rules

- Preserve existing architecture unless the task requires changing it.
- Prefer simple domain types over deeply nested generic dictionaries.
- Keep sync I/O out of async execution paths.
- Close resources deterministically and retain exception context.
- Add tests for changed behavior, boundaries, and failure paths.
- During review-only tasks, do not modify files and report actionable findings by severity with file and line evidence.

## Adjacent skills

Hand off when the skill is installed; otherwise apply this skill's own rules and continue: $configure-python-quality (Ruff, mypy, pre-commit, repository hygiene), $develop-ml-systems (model, RAG, or data-pipeline behavior), $build-fastapi-services (FastAPI architecture), $secure-applications (security-sensitive changes).
