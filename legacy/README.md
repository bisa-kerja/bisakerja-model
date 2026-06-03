# Legacy Snapshot

Historical snapshot of the older Bisakerja model project before the notebook-first production rebuild.

## Purpose

This folder exists for audit, comparison, and recovery context only. It is not the active training or serving workspace.

## Contents

| Path                                        | Purpose                                         |
| ------------------------------------------- | ----------------------------------------------- |
| `api/`                                      | Older FastAPI prototype.                        |
| `models/`                                   | Older model artifacts.                          |
| `dataset/`                                  | Older cleaned datasets.                         |
| `artifacts/`, `cache/`, `reports/`, `logs/` | Older generated training and inference outputs. |
| `bisakerja_model_training_FINAL.ipynb`      | Previous all-in-one training notebook.          |
| `docs/`                                     | Historical product/training notes.              |
| `tasks/`                                    | Historical task notes.                          |

## Active Replacements

| Legacy concern      | Active location                                              |
| ------------------- | ------------------------------------------------------------ |
| Training workflow   | `../training/`                                               |
| Production serving  | `../model_api/`                                              |
| Current artifacts   | `../artifacts/`                                              |
| Current reports     | `../reports/`                                                |
| Release gates       | `../scripts/` and `../tests/`                                |
| Backend integration | External repo: <https://github.com/bisa-kerja/bisakerja-api> |

## Rules

- Do not use this folder for active runtime or new model training.
- Do not move legacy artifacts into production paths without explicit validation.
- Prefer adding notes to active docs instead of extending legacy docs.
- Keep this folder path-stable because old reports may reference it.
