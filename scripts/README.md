# Scripts

Verification, report-generation, and release-gate utilities for the model workspace.

## Scope

Scripts should be deterministic, repo-root relative, and safe to run locally unless explicitly documented otherwise.

## Common Commands

```bash
python scripts/generate_training_audit_v1.py
python scripts/verify_training_phase1.py
python scripts/verify_phase_27_1_27_2_release_gate.py --write
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
python scripts/verify_phase_27_9_model_api_production_smoke.py --write --run-live
python scripts/verify_phase_31_release_gate.py
```

## Rules

- Keep outputs deterministic and path-stable.
- Write generated evidence under `reports/` or `artifacts/`.
- Avoid hidden network calls unless a script is clearly a live smoke gate.
- Do not require production secrets for ordinary local checks.
- Preserve exact failure semantics used by tests.
