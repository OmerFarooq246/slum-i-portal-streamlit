# Contributing

## Workflow

1. Do not push directly to `main`. Create a branch and open a pull request.
2. Ensure every required CI job passes before merge.
3. Address review feedback before merge.
4. Keep changes focused. Do not combine unrelated features, fixes, and refactors.

The repository installs a local `pre-push` hook that rejects direct pushes to `main`. GitHub branch
protection or a repository ruleset should also require pull requests and successful CI checks.

Follow the technical requirements in [Data pipeline](docs/data-pipeline.md) and
[Development](docs/development.md) when preparing a pull request. See [README.md](README.md) for the
project overview and quick start.

## Branch names

Use this format:

```text
<type>/<short-kebab-description>
```

Recommended types:

| Type | Purpose |
| --- | --- |
| `feat` | New functionality |
| `fix` | Bug fix |
| `refactor` | Internal restructuring without an intended behavior change |
| `test` | Tests or testing infrastructure |
| `docs` | Documentation changes |
| `ci` | CI or GitHub Actions changes |
| `chore` | Tooling, dependency, or maintenance work |
| `perf` | Data-pipeline or training performance improvements |

Examples:

```text
feat/model-training
fix/validation-split-rounding
refactor/dataset-configuration
test/image-decoding
docs/dataset-setup
ci/notebook-checks
chore/update-tensorflow
```

## Pull request titles

Use Conventional Commit style:

```text
<type>(<scope>): <summary>
```

Useful scopes include:

```text
config
data
loader
model
notebook
tests
deps
ci
docs
```

Examples:

```text
feat(loader): add RGB augmentation pipeline
fix(config): calculate test split percentage correctly
test(data): cover malformed metadata rows
docs(setup): explain Nutrition5K directory layout
ci(quality): enforce clean notebook outputs
chore(deps): update TensorFlow lock
```

Use lowercase for the type and scope. Keep the summary concise and suitable for the final squash
commit on `main`.

## Merge strategy

Use **Squash and merge** for pull requests. Avoid unnecessary merge commits.

Before merging, confirm that:

- All required CI jobs pass.
- Required review feedback has been addressed.
- The pull request title follows the repository convention.
- The squash commit title accurately describes the change.

## Text policy

Do not use em dashes anywhere in the repository.

This applies to Python, comments, docstrings, Markdown, notebooks, commit messages, pull request
titles, and pull request descriptions. Use a hyphen, comma, colon, parentheses, or rewrite the
sentence instead.

The policy is enforced by the managed Git hooks and CI.

## Code review

Use priority notation for actionable review findings:

| Priority | Meaning | Merge policy |
| --- | --- | --- |
| `P0` | Critical data loss, security, or destructive behavior | Must fix immediately. Do not merge. |
| `P1` | Serious correctness, leakage, security, or reliability defect | Must fix before merge. |
| `P2` | Real defect with limited impact or an important maintainability issue | Normally fix before merge unless explicitly deferred. |
| `P3` | Minor improvement, cleanup, or readability issue | Non-blocking. |
| `P4` | Optional polish or preference | Non-blocking. |

Each actionable finding should state what is wrong, why it matters, and a practical way to fix it.
The overall review priority is the highest unresolved finding.
