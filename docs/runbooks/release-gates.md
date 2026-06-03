# Release Gates

Production-ready claims require durable evidence. Local notebook success is not enough.

## Main Gate Commands

```bash
python scripts/verify_phase_27_1_27_2_release_gate.py --write
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
python scripts/verify_phase_27_7_clean_kernel_export.py --write
python scripts/verify_phase_27_8_model_card_manifest_refresh.py --write
python scripts/verify_phase_27_9_model_api_production_smoke.py --write --run-live
python scripts/verify_phase_27_10_requirement_matrix.py --write
python scripts/verify_phase_28_contract_realignment.py --write
python scripts/verify_phase_31_release_gate.py
python scripts/verify_phase_36_ai_cv_analyzer_staging_gate.py
```

## Evidence Areas

| Area                      | Evidence                                                                                                                                      |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Training                  | TensorFlow model export, custom component, custom training loop, TensorBoard logs, model card.                                                |
| Data and labels           | Human-label policy, validation expansion, slice coverage, ATS benchmark, recommendation fixtures.                                             |
| Model API                 | Runtime loading, artifact verification, strict schemas, PDF parsing, E5 features, deterministic errors.                                       |
| External Backend contract | OpenAPI/Prisma owner matrix, internal contract fixtures, Backend-owned hydration boundary from <https://github.com/bisa-kerja/bisakerja-api>. |
| Security/privacy          | No DB credentials in Model API, no raw CV logs, service-token routing, upload and retention notes.                                            |

## Expected Failure Modes

- invalid PDF -> `422` validation response
- parse failure -> deterministic low-confidence parser result
- empty candidates -> Backend no-recommendation policy
- Model API timeout -> `504` or AI-unavailable mapping
- TensorFlow load failure -> `503` readiness failure
- E5 failure -> `503` readiness or inference failure
- GenAI wrapper failure -> Backend fallback copy while preserving model scores/order

## Release Notes

- Run gates from a clean or well-understood git state.
- Do not claim production readiness with ignored TensorBoard logs only; release logs must be recorded under artifact manifests with SHA-256 and byte size.
- Do not bypass human-label requirements with weak labels.
- Do not use the small handoff fixture as the only recommendation-quality evidence.
- Keep Model API private/internal and service-token protected.
