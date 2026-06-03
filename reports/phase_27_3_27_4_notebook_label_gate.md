# Phase 27.3-27.4 Notebook and Label Evidence Gate

Generated at: `2026-06-03T10:37:04.454345+00:00`
Final decision: **blocked**

## Notebook hygiene

Status: **PASS**
Active notebooks scanned: `27`
Notebooks with saved error outputs: `0`
Active notebooks with unexecuted code cells: `0`

Retired notebooks:
- `training/notebooks/phase_13_data_snapshot_contract_freezing.ipynb` — Phase 13 execution evidence is frozen in reports/phase_13_*.json; executable cells are retired for production notebook hygiene.

## Production label policy

Policy artifact: `artifacts/manual_validation/phase_27_production_label_policy.json`
Frozen label file: `artifacts/manual_validation/phase_16_human_labels_frozen.csv`
Status: **FAIL**
Production score claims allowed: `False`
Unique review items: `120`
Reviewer count: `2`
Minimum reviewers per item: `2`

Current blockers:
- frozen human validation item count 120 < required 600
- score band low has 40 items < required 150
- score band medium has 40 items < required 150
- score band high has 40 items < required 150
- required slice dimension role_family has under-covered buckets: {'backend': 12, 'web': 13, 'other': 19, 'cloud': 21, 'security': 11, 'data': 22, 'software_engineering': 17, 'frontend': 3, 'fullstack': 2}
- required slice dimension language missing from frozen labels
- required slice dimension experience_band missing from frozen labels
- required slice dimension pair_type has under-covered buckets: {'high_fit_positive': 40, 'hard_negative': 14, 'same_role_different_seniority': 8, 'cross_role_confusing': 10, 'random_negative': 8, 'medium_fit': 40}

Weak labels remain allowed only as bootstrap/training support. Human/recruiter-reviewed frozen labels are mandatory for production score claims.
