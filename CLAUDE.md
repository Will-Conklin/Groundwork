# CLAUDE.md

## Commit Hygiene

Use **conventional commits** for every commit in this repository.

### Format

```
<type>(<scope>): <short description>

[optional body — explain WHY, not WHAT]
```

### Types

| Type | Use for |
|---|---|
| `feat` | New module, notebook, or function that adds capability |
| `test` | Adding or updating tests |
| `fix` | Bug fix |
| `chore` | Scaffolding, config files, pyproject.toml, .gitignore, justfile |
| `docs` | README, CLAUDE.md, FUTURE_ENHANCEMENTS.md, inline docstrings |
| `ci` | GitHub Actions workflow changes |
| `refactor` | Code reorganization with no behavior change |
| `style` | Formatting-only (ruff auto-fix, whitespace) |

### Scope

Use the file or module name: `features`, `io`, `model`, `train`, `batch-score`, `ci`, `docker`.

### Examples

```
chore(scaffold): initialize project structure and pyproject.toml
docs: add CLAUDE.md with commit conventions
test(features): add select_features tests — red
feat(features): implement select_features with column validation
test(io): add load_csv and save_csv round-trip tests — red
feat(io): implement load_csv and save_csv with loguru logging
test(model): add FraudModel load and predict_proba tests — red
feat(model): implement FraudModel wrapping lgb.Booster
feat(train): add marimo training notebook with reactive sliders
feat(batch-score): add marimo batch scoring notebook
chore(tooling): add justfile, Makefile, and pre-commit config
ci: add GitHub Actions workflow
chore(docker): add Dockerfile.batch for headless batch scoring
docs: add README and FUTURE_ENHANCEMENTS.md
```

### Rules

- **One logical change per commit** — test commits land before implementation commits (red before green)
- Never `git add .` — stage specific files by name
- Never `--no-verify` unless a hook has a known false positive; document why in the commit body
- Subject line ≤ 72 characters; imperative mood ("add", "implement", "fix" — not "added", "fixed")
