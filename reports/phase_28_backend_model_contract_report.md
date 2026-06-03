# Phase 28 Backend/Model API Contract Realignment Report

Final decision: `passed`

## Checks

- [x] `contract_route_is_internal_multipart`
- [x] `positive_fixtures_cover_compare_sources`
- [x] `positive_fixtures_cover_input_modes`
- [x] `negative_fixtures_cover_duplicate_empty_missing_evidence`
- [x] `model_core_response_schema_is_v1`
- [x] `model_core_response_has_required_sections`
- [x] `model_core_response_excludes_public_wrapper_fields`
- [x] `model_recommendations_are_backend_candidate_members`
- [x] `owner_matrix_covers_required_entities`
- [x] `enum_mapping_is_frozen`
- [x] `openapi_contains_public_cv_analysis_v2`
- [x] `prisma_contains_persistence_entities`
- [x] `docs_reference_contract_and_fixtures`

## Fixtures

Positive: direct-upload-pdf-job-search, active-cv-reference-bookmark, direct-job-detail-candidate
Negative: duplicate-job-ids, empty-pdf-parse, missing-candidate-evidence
