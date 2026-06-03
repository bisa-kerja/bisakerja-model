# Project Structure

This repository uses a model-focused workspace structure. It does not contain Backend API source. The Backend API lives in a separate repository: <https://github.com/bisa-kerja/bisakerja-api>.

The layout is conservative on purpose: existing tests, notebooks, artifact manifests, and release evidence depend on current model-side paths.

## Top-Level Layout

```text
.
|-- docs/          # Cross-workspace docs and runbooks
|-- training/      # Notebook-first model training
|-- model_api/     # FastAPI model serving package
|-- scripts/       # Verification and report-generation scripts
|-- tests/         # Python unittest suites
|-- artifacts/     # Durable generated artifacts and fixtures
|-- reports/       # Generated reports and release evidence
|-- legacy/        # Historical snapshot only
|-- requirements.txt
|-- REQUIREMENT.md
|-- GAP_MODEL_TRAINING.md
|-- RUNNING_STEPS.md
`-- TODOS.md
```

## Ownership Rules

| Path         | Owner                 | Mutable?        | Notes                                                                |
| ------------ | --------------------- | --------------- | -------------------------------------------------------------------- |
| `training/`  | Model training        | Yes             | Notebook-first workflow. No training package entrypoints by default. |
| `model_api/` | Model serving         | Yes             | FastAPI runtime. No DB ownership.                                    |
| `scripts/`   | Verification tooling  | Yes             | Keep outputs deterministic and repo-root relative.                   |
| `tests/`     | Verification coverage | Yes             | Keep path expectations aligned with scripts/artifacts.               |
| `artifacts/` | Generated evidence    | Limited         | Update only through notebooks/scripts or explicit evidence refresh.  |
| `reports/`   | Generated evidence    | Limited         | Prefer generated writes through verification scripts.                |
| `legacy/`    | Historical snapshot   | No, except docs | Do not use for active runtime.                                       |

## Backend Boundary

Backend API source and deployment config belong to <https://github.com/bisa-kerja/bisakerja-api>. This model repo may consume Backend-generated contracts or fixtures as release inputs, but it must not treat Backend source as an in-repo workspace.

## Safe Restructuring Policy

To avoid breaking current project behavior:

1. Keep active import paths stable: `model_api.*`, `scripts.*`, and `tests.*`.
2. Keep root governance files at root: `TODOS.md`, `REQUIREMENT.md`, `GAP_MODEL_TRAINING.md`, `RUNNING_STEPS.md`.
3. Keep artifact paths stable because notebooks and manifests record SHA-256 and relative paths.
4. Add documentation and README files freely when they do not alter runtime behavior.
5. Move files only when every script, notebook, manifest, README, and test reference is updated and verified.

## Dependency Direction

```text
training notebooks -> artifacts/reports
model_api -> artifacts + exported Backend contract fixtures
scripts/tests -> model_api + artifacts + reports + root docs
external Backend API repo -> contract producer only
legacy -> no active dependency
```

Model API must not depend on training notebooks at runtime. Training may produce artifacts consumed by Model API, but runtime serving must use exported files only.

## Documentation Placement

- Root README: workspace overview and quick start.
- Workspace README: purpose, layout, commands, and boundary for that folder.
- `docs/architecture/`: cross-service ownership and integration contracts.
- `docs/runbooks/`: operational steps, release gates, and troubleshooting.
- Generated reports stay in `reports/`; do not hand-edit generated evidence unless the script explicitly writes it.
