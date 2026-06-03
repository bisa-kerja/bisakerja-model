# Phase 27.5-27.6 Validation Expansion Gate

Generated at: `2026-06-03T10:41:38.226690+00:00`
Final decision: **production-ready**

## ATS validation

Status: **PASS**
Fixture: `artifacts/phase_27_validation_expansion/ats_release_cv_benchmark.csv`
Documents: `72`
Parse coverage: `0.889`
Bucket agreement: `0.889`
Macro issue precision: `0.900`
Macro issue recall: `0.933`

## Recommendation validation

Status: **PASS**
Fixture: `artifacts/phase_27_validation_expansion/recommendation_release_candidate_sets.json`
Candidate sets: `12`
Candidates: `120`
Model NDCG@5: `1.000`
Model NDCG@10: `1.000`
Model MAP@10: `1.000`
NDCG@10 uplift: `0.378`
Constraint violation rate: `0.000`

## Blockers

- None
