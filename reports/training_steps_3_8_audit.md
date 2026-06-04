# Training Steps 3-8 Audit

Generated at: `2026-06-04T03:31:44.071474+00:00`
Final decision: **implemented-with-warnings**

## Step 3. Notebook hygiene

Status: **PASS**

No blockers or warnings.

## Step 4. Data snapshot and contract input

Status: **PASS**

No blockers or warnings.

## Step 5. Embedding and feature contract

Status: **WARN**

Warnings:
- Phase 14 keeps historical local-hash embedding evidence; Phase 25 production contract forbids fallback backends.

## Step 6. Label governance

Status: **WARN**

Warnings:
- frozen human validation item count 120 < required 600
- score band low has 40 items < required 150
- score band medium has 40 items < required 150
- score band high has 40 items < required 150
- required slice dimension role_family has under-covered buckets: {'backend': 12, 'web': 13, 'other': 19, 'cloud': 21, 'security': 11, 'data': 22, 'software_engineering': 17, 'frontend': 3, 'fullstack': 2}
- required slice dimension language has under-covered buckets: {'EN': 67, 'UNKNOWN': 52, 'ID': 1}
- required slice dimension experience_band has under-covered buckets: {'profile:entry|job:entry': 13, 'profile:senior|job:junior': 4, 'profile:junior|job:junior': 12, 'profile:entry|job:junior': 19, 'profile:senior|job:senior': 3, 'profile:junior|job:entry': 8, 'profile:senior|job:mid': 12, 'profile:mid|job:entry': 10, 'profile:mid|job:junior': 13, 'profile:senior|job:entry': 5, 'profile:entry|job:lead': 7, 'profile:junior|job:senior': 1, 'profile:junior|job:lead': 5, 'profile:entry|job:senior': 1, 'profile:mid|job:senior': 1, 'profile:mid|job:lead': 1, 'profile:junior|job:mid': 4, 'profile:mid|job:mid': 1}
- required slice dimension pair_type has under-covered buckets: {'high_fit_positive': 40, 'hard_negative': 14, 'same_role_different_seniority': 8, 'cross_role_confusing': 10, 'random_negative': 8, 'medium_fit': 40}
- Label validation slices are present but remain under release-scale coverage thresholds.

## Step 7. Pair generation and split

Status: **PASS**

No blockers or warnings.

## Step 8. Baseline before model selection

Status: **WARN**

Warnings:
- Phase 25 TensorFlow candidate does not yet pass strict production selection against Phase 18 preservation checks.
- Prototype tradeoffs remain recorded in phase_25_baseline_selection_gate.json.
