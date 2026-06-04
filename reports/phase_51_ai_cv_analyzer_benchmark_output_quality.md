# AI CV Analyzer Benchmark Output Quality

Status: complete

Scope: `steps_51_1_51_7`

## Checks

- [x] `benchmark_fixture_exists`
- [x] `required_upload_matrix_locked`
- [x] `parser_snapshots_match_real_pdfs`
- [x] `score_ranges_recorded`
- [x] `english_only_policy_declared`
- [x] `english_output_normalizer_implemented`
- [x] `role_specific_job_fit_evidence`
- [x] `real_pdf_parser_repairs_implemented`
- [x] `benchmark_regression_tests_exist`
- [x] `grounded_ats_issue_copy_implemented`
- [x] `grounded_overall_impression_templates_implemented`
- [x] `phase51_5_51_6_regression_tests_exist`
- [x] `phase51_7_regression_tests_exist`
- [x] `no_raw_cv_text_in_fixture`
- [x] `benchmark_english_only_outputs`
- [x] `benchmark_role_specific_evidence_gate`
- [x] `benchmark_non_generic_copy_gate`
- [x] `benchmark_score_spread_gate`
- [x] `benchmark_no_raw_cv_leakage_gate`
- [x] `benchmark_ats_issue_precision_gate`
- [x] `benchmark_schema_compatibility_gate`
- [x] `benchmark_bounded_latency_gate`
- [x] `benchmark_no_hardcoded_role_only_behavior_gate`
- [x] `unit_tests_pass`

## Benchmark examples

- `salman-abdurrahman-ats`: parseQuality=partial, atsScore=67, sections=['summary', 'experience', 'education', 'skills'], issueCount=2
- `agil-2026`: parseQuality=partial, atsScore=65, sections=['education'], issueCount=2
- `dzikri-albantani`: parseQuality=partial, atsScore=42, sections=[], issueCount=4

## Benchmark quality gate examples

- `salman-abdurrahman-ats` / `Software Engineer`: jobFitScore=65, matched=['docker', 'javascript', 'node.js', 'react', 'sql'], missing=['collaborate with product and design', 'debug production issues', 'develop maintainable software'], latencyMs=5
- `salman-abdurrahman-ats` / `Product Manager`: jobFitScore=64, matched=[], missing=['agile', 'communication skills', 'coordinate cross-functional delivery'], latencyMs=4
- `salman-abdurrahman-ats` / `Data Analyst`: jobFitScore=64, matched=['sql'], missing=['build kpi dashboards', 'clean datasets', 'dashboard reporting'], latencyMs=5
- `agil-2026` / `Software Engineer`: jobFitScore=64, matched=[], missing=['collaborate with product and design', 'debug production issues', 'develop maintainable software'], latencyMs=4
- `agil-2026` / `Product Manager`: jobFitScore=64, matched=[], missing=['agile', 'communication skills', 'coordinate cross-functional delivery'], latencyMs=4
- `agil-2026` / `Data Analyst`: jobFitScore=64, matched=[], missing=['build kpi dashboards', 'clean datasets', 'dashboard reporting'], latencyMs=4
- `dzikri-albantani` / `Software Engineer`: jobFitScore=64, matched=[], missing=['collaborate with product and design', 'debug production issues', 'develop maintainable software'], latencyMs=4
- `dzikri-albantani` / `Product Manager`: jobFitScore=64, matched=[], missing=['agile', 'communication skills', 'coordinate cross-functional delivery'], latencyMs=4
- `dzikri-albantani` / `Data Analyst`: jobFitScore=64, matched=[], missing=['build kpi dashboards', 'clean datasets', 'dashboard reporting'], latencyMs=4

## Remaining Phase 51 steps

- None
