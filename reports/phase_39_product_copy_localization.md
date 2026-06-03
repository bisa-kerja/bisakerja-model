# Phase 39 AI CV Analyzer Product Copy and Localization

Decision: `go`

## Language Policy
- Staging default: `english_safe_copy`
- `id`: return English public copy until Indonesian localization is product-approved
- `en`: return English public copy

## Checks
- PASS `language_policy_explicit`
- PASS `fallback_templates_cover_public_fields`
- PASS `backend_uses_approved_template_copy`
- PASS `localization_tests_cover_id_en_behavior`
- PASS `copy_safety_filters_and_pii_tests_present`
- PASS `schema_length_limits_match_public_contract`
- PASS `approved_copy_has_no_localization_drift_or_sensitive_literals`
- PASS `genai_boundary_remains_validated_and_optional`

## Before / After
- Before: CV shows fit through TypeScript, PostgreSQL, with gaps in Docker.
- After: Your CV shows relevant evidence in TypeScript, PostgreSQL. Strengthen proof for Docker to improve role fit.

## Remaining Limitations
- English-only public copy for id requests during staging/demo.
- Deterministic copy cannot add richer personalization beyond sanitized model evidence and backend job metadata.
- Indonesian localization still needs product-approved templates or validated GenAI output.
