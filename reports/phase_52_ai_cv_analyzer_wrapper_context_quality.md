# Phase 52 AI CV Analyzer Wrapper Context Quality

Generated: `2026-06-04T21:04:38Z`

Passed: `True`

## Gates

- wrapper_evidence_schema_present: PASS
- required_context_fields_present: PASS
- years_not_missing_skills: PASS
- section_evidence_grounded: PASS
- fallback_no_bad_years_copy: PASS
- prompt_rules_cover_evidence_only_and_invariants: PASS
- validator_rejects_bad_wrapper: PASS
- no_contact_value_in_wrapper: PASS
- unit_tests_passed: PASS

## Bad Copy Regression

Before: `Add one measurable bullet or project example that proves maximum 3 years of experience`

After examples:

- Add concrete CV evidence for required skill `sql` using one project, tool, or outcome bullet.
- Address requirement `Remote work in Jakarta` with specific, verifiable CV evidence or mark it as unclear.

## Requirement Coverage Examples

- `python` -> `skill` / `matched`
- `sql` -> `skill` / `missing`
- `maximum 3 years of experience` -> `experience_years` / `matched`
- `minimum 1 years of experience` -> `experience_years` / `matched`
- `Bachelor degree` -> `education` / `matched`
- `Remote work in Jakarta` -> `other` / `unclear`
- `junior` -> `experience_years` / `matched`
