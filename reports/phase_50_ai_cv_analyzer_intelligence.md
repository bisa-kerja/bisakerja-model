# AI CV Analyzer Intelligence Optimization

Status: complete

## Checks

- [x] `richer_cv_extraction`
- [x] `feature_based_ats_scoring`
- [x] `stronger_job_fit_evidence`
- [x] `evidence_based_actionable_inputs_without_backend_fields`
- [x] `benchmark_and_safety_tests`
- [x] `genai_safe_wrapper_boundary_preserved`
- [x] `unit_tests_pass`

## Contract safety

- Backend public schema remains `cv-analysis-v2`.
- Model API response stays `model-core-cv-analysis-v1`.
- Model API does not return backend-owned fields or raw CV text.
- Model API makes no external GenAI call.

## Evaluation gates

- parser signal extraction fixture
- ATS missing-metrics/formatting-risk fixture
- model-core evidence + no backend-owned fields + no raw-CV leakage fixture
