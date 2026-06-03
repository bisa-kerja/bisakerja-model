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
| `test_phase_36_*.py` | AI CV Analyzer staging readiness gate evidence checks.                                               |

## Common Commands

```bash
python -m unittest tests.model_api.test_phase_26_layout
python -m unittest tests.test_phase_29_model_api_hardening
python -m unittest tests.test_phase_31_release_gate
python -m unittest tests.test_phase_36_ai_cv_analyzer_staging_gate
python -m unittest discover tests
```

Some tests require TensorFlow/Keras artifacts, E5 dependencies, generated reports, or live Model API execution. If a check needs live services, use the matching script/runbook first.

## Rules

- Keep tests repo-root relative.
- Do not depend on production secrets.
- Do not log raw CV text, service tokens, DB URLs, or unrelated PII.
- Update tests when contracts, env vars, or release-gate acceptance criteria change.
