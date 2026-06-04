# Tests

Python `unittest` coverage for Model API layout, runtime behavior, contract hardening, and release gates.

## Layout

| Path                 | Purpose                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| `model_api/`         | Tests for Model API package layout and artifact-backed behavior.                                     |
| `test_phase_27_*.py` | Release evidence, notebook hygiene, requirement matrix, model card, and validation expansion checks. |
| `test_phase_28_*.py` | Backend/Model API contract realignment checks.                                                       |
| `test_phase_29_*.py` | Model API hardening and deterministic error behavior.                                                |
| `test_phase_31_*.py` | Release gate, readiness, security, privacy, and runbook coverage checks.                             |
| `test_training_step_1_*.py` | Training stabilization Step 1 scope, baseline evidence, and dirty-state audit checks.         |

## Common Commands

Use the dedicated training venv when the default terminal Python is not `3.13.x`:

```powershell
training\.tf-venv-3.13\Scripts\python.exe -m pytest tests\test_training_step_1_scope_baseline.py
training\.tf-venv-3.13\Scripts\python.exe -m pytest tests\test_training_steps_3_8_audit.py
```

```bash
python -m unittest tests.model_api.test_phase_26_layout
python -m unittest tests.test_training_step_1_scope_baseline
python -m unittest tests.test_phase_29_model_api_hardening
python -m unittest tests.test_phase_31_release_gate
python -m unittest discover tests
```

Some tests require TensorFlow/Keras artifacts, E5 dependencies, generated reports, or live Model API execution. If a check needs live services, use the matching script/runbook first.

## Rules

- Keep tests repo-root relative.
- Do not depend on production secrets.
- Do not log raw CV text, service tokens, DB URLs, or unrelated PII.
- Update tests when contracts, env vars, or release-gate acceptance criteria change.
