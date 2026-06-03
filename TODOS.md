# TODOS — Notebook-First Training Model Phases

Source reference: `GAP_MODEL_TRAINING.md`  
Model/API contract reference: `references/docs/generated/openapi.json`

This TODO resets the training work to **Phase 0** and switches the training workflow to readable `.ipynb` notebooks. Existing script-package extraction work is no longer the active phase plan.

## Scope

Model/training owns:

1. `jobFitAlignment`
   - score `0-100`
   - grounded alignment signals
   - missing or matched skill signals

2. `atsFriendliness`
   - score `0-100`
   - detected ATS issues

3. `overallImpression`
   - grounded summary signal that the API wrapper can render as product copy

Backend/API wrapper owns:

- `topActionables`
- `sectionReviews`
- final job detail hydration
- auth, persistence, request validation, and OpenAI wrapper orchestration

## Documentation Rule

Each notebook step must start with complete English Markdown documentation before any code is added. Required Markdown sections per step:

- Purpose
- Required input
- Action
- Expected output
- Verification

## Phase List

### Phase 0 — Reproducibility Snapshot

Notebook: `training/notebooks/phase_00_reproducibility_snapshot.ipynb`  
Status: Complete

Goal: Capture the current model, dataset, artifact, and environment state before any new experiment begins.

Tasks:

- [x] Step 0.1: Artifact inventory — List every model, cache, pair dataset, report, and notebook artifact. Record file path, format, owner, and whether the artifact is required for inference or only for training.
- [x] Step 0.2: Environment snapshot — Document Python version, notebook runtime, package versions, hardware assumptions, and random seeds that affect repeatability.
- [x] Step 0.3: Dataset snapshot — Count raw jobs, profiles, CV samples, generated pairs, missing values, language distribution, and duplicated identifiers.
- [x] Step 0.4: Baseline metric capture — Copy existing metrics exactly as reported. Mark any metric that came from weak labels or prototype-only gates.
- [x] Step 0.5: Risk register — Write known risks before new work starts, including label quality, data leakage, missing high-fit samples, stale jobs, and runtime incompatibilities.

Acceptance Criteria:

- [x] Inventory covers all required training artifacts.
- [x] Baseline metrics are recorded before new experiments.
- [x] Known blockers are visible before Phase 1 starts.

---

### Phase 1 — Data Audit and Output Contracts

Notebook: `training/notebooks/phase_01_data_audit_contracts.ipynb`  
Status: Complete

Goal: Understand source data quality and align model outputs with product/API contracts before feature or label work.

Tasks:

- [x] Step 1.1: Source schema review — Document every source column, data type, null policy, example value, and whether it is safe to use during training.
- [x] Step 1.2: Identifier integrity check — Describe checks for profile IDs, job IDs, duplicates, stale records, inactive jobs, and leakage-prone joins.
- [x] Step 1.3: Output contract mapping — Map training outputs to product fields: jobFitAlignment, atsFriendliness, overallImpression, and recommendation scores.
- [x] Step 1.4: Boundary definition — Separate model-owned outputs from wrapper-owned outputs such as topActionables, sectionReviews, and hydrated job details.
- [x] Step 1.5: Data readiness decision — State whether the available data can support the next phase or whether synthetic/manual labels are needed first.

Acceptance Criteria:

- [x] Every training input field has a documented purpose.
- [x] Model-owned outputs are separated from API-wrapper outputs.
- [x] Data readiness decision is explicit.

---

### Phase 2 — Label Schema and Baseline Definitions

Notebook: `training/notebooks/phase_02_label_schema_baselines.ipynb`  
Status: Complete

Goal: Define transparent labels and baseline models before building stronger training experiments.

Tasks:

- [x] Step 2.1: Job-fit label decomposition — Define skill overlap, semantic similarity, experience match, role match, requirement coverage, and optional location or work-preference components.
- [x] Step 2.2: ATS label decomposition — Define parseability, section completeness, contact detection, date detection, metric evidence, and formatting-risk components.
- [x] Step 2.3: Score band policy — Document low, medium, and high bands with numeric thresholds and explain what each band means for product users.
- [x] Step 2.4: Baseline model catalog — Define constant mean, constant median, skill-overlap-only, cosine-only, and simple regression baselines for future comparison.
- [x] Step 2.5: Manual validation plan — Describe how to sample low, medium, and high cases for human review without leaking validation labels into training.

Acceptance Criteria:

- [x] Label components are documented before pair generation.
- [x] Baselines are defined before model training.
- [x] Manual validation sample plan exists.

---

### Phase 3 — Normalization and Feature Design

Notebook: `training/notebooks/phase_03_normalization_feature_design.ipynb`  
Status: Complete

Goal: Design stable normalization and feature rules for skills, experience, language, roles, and text before generating pairs.

Tasks:

- [x] Step 3.1: Experience normalization — Define explicit mappings for Fresher, 1-2 years, 3-5 years, 5+ years, ENTRY_LEVEL, JUNIOR, MID_LEVEL, SENIOR, LEAD, and MANAGER.
- [x] Step 3.2: Skill normalization — Document lowercase rules, punctuation trimming, alias handling, Indonesian/English variants, framework variants, and unknown-skill handling.
- [x] Step 3.3: Language normalization — Define ID, EN, MIXED, and UNKNOWN assignment rules and the minimum evidence needed for each category.
- [x] Step 3.4: Text feature construction — Describe how profile text, CV text, job text, role text, requirements, and skills are combined before embedding.
- [x] Step 3.5: Missing and unknown report — Define metrics for unknown language rate, unknown experience rate, empty skills rate, and empty text rate.

Acceptance Criteria:

- [x] No known experience value falls through silently.
- [x] Skill alias policy is documented.
- [x] Feature-quality metrics are defined.

---

### Phase 4 — Pair Generation and Split Strategy

Notebook: `training/notebooks/phase_04_pair_generation_splits.ipynb`  
Status: Complete

Goal: Design balanced training pairs and leakage-safe splits for job-fit and recommendation ranking experiments.

Tasks:

- [x] Step 4.1: Pair type taxonomy — Define high-fit positives, medium-fit pairs, hard negatives, random negatives, same-role different-seniority pairs, and cross-role confusing pairs.
- [x] Step 4.2: Target distribution — Document target score distribution across 0-1 and minimum high-fit coverage needed for meaningful training.
- [x] Step 4.3: Group isolation — Define split isolation by profile ID and optional job family to reduce leakage between training and validation.
- [x] Step 4.4: Pair metadata — List required columns such as pair_type, label components, final score, language, role family, and split name.
- [x] Step 4.5: Distribution diagnostics — Plan charts for score distribution, score by pair type, score by role, score by language, and split balance.

Acceptance Criteria:

- [x] Pair categories cover positive, medium, and negative cases.
- [x] Split rules prevent obvious profile leakage.
- [x] Distribution diagnostics are specified.

---

### Phase 5 — Baseline Evaluation Notebook

Notebook: `training/notebooks/phase_05_baseline_evaluation.ipynb`  
Status: Complete

Goal: Evaluate simple baselines before training complex models, so model improvement can be proven.

Tasks:

- [x] Step 5.1: Regression baseline metrics — Document MAE, RMSE, R-squared, Spearman, and score-band agreement for constant and simple feature baselines.
- [x] Step 5.2: Ranking baseline metrics — Document NDCG@5, NDCG@10, MAP@10, and candidate ordering examples when recommendation labels are available.
- [x] Step 5.3: Slice analysis — Report metrics by role, language, experience band, pair type, and score band to expose weak spots.
- [x] Step 5.4: Error inspection — Describe representative false-high and false-low examples with probable root causes.
- [x] Step 5.5: Training readiness gate — State whether labels and splits are strong enough to start model training or whether pair generation must be fixed first.

Acceptance Criteria:

- [x] Best baseline is known before model training.
- [x] Weak slices are documented.
- [x] Training readiness decision is explicit.

---

### Phase 6 — JobFitAlignment Training Experiments

Notebook: `training/notebooks/phase_06_jobfit_training_experiments.ipynb`  
Status: Complete

Goal: Plan and compare model experiments for jobFitAlignment scoring and ranking signals.

Tasks:

- [x] Step 6.1: Experiment matrix — Define models to compare: linear baseline, cosine-only baseline, neural scorer, feature-augmented scorer, and optional ranking objective.
- [x] Step 6.2: Input feature contract — Document required inputs: profile/CV embedding, job embedding, normalized skill overlap, experience gap, role match, and requirement coverage.
- [x] Step 6.3: Output signal contract — Define model-owned outputs: score, summarySignals, missingSignals, matchedSkills, missingSkills, and confidence notes.
- [x] Step 6.4: Training protocol — Document split use, seeds, batch strategy, early stopping policy, checkpoint naming, and failure recovery expectations.
- [x] Step 6.5: Selection criteria — Define minimum improvement over the best baseline, including MAE improvement, R-squared, Spearman, and slice stability.

Acceptance Criteria:

- [x] Experiment matrix is complete before code is run.
- [x] Output signal contract matches product boundary.
- [x] Model selection gates are explicit.

---

### Phase 7 — ATS Friendliness Scoring

Notebook: `training/notebooks/phase_07_ats_friendliness_scoring.ipynb`  
Status: Complete

Goal: Design ATS friendliness scoring so CV quality is evaluated separately from job-fit alignment.

Tasks:

- [x] Step 7.1: Benchmark coverage — Document required CV cases: normal PDF, scanned PDF, multi-column PDF, DOCX, table-heavy CV, very short CV, and overly long CV.
- [x] Step 7.2: Issue taxonomy — Define parseable text, section completeness, contact detection, date detection, metric evidence, excessive formatting risk, and empty parse risk.
- [x] Step 7.3: Scoring rule design — Describe a transparent rule-based baseline before any classifier is considered.
- [x] Step 7.4: Evaluation policy — Define precision, recall, score-bucket agreement, empty-text rate, and failure-case review workflow.
- [x] Step 7.5: Output contract — Define model/core output fields: atsFriendliness.score and atsFriendliness.detectedIssues.

Acceptance Criteria:

- [x] ATS scoring is separate from job-fit scoring.
- [x] Supported file cases are documented.
- [x] Evaluation metrics are defined before implementation.

---

### Phase 8 — Overall Impression Signals

Notebook: `training/notebooks/phase_08_overall_impression_signals.ipynb`  
Status: Complete

Goal: Define grounded overallImpression signals that can be rendered by an API wrapper without inventing unsupported claims.

Tasks:

- [x] Step 8.1: Input signal policy — List which signals can be used for an impression and which fields must never be inferred without evidence.
- [x] Step 8.2: Grounded summary design — Document deterministic summary templates that mention only observed strengths, gaps, and ATS risks.
- [x] Step 8.3: Fallback handling — Define safe summaries for empty CV parse, missing job target, unknown language, and low-confidence model output.
- [x] Step 8.4: Wrapper boundary — Explain how the model/core output can be converted by the API wrapper into the OpenAPI overallImpression string.
- [x] Step 8.5: Quality checks — Define checks that prevent hallucinated skills, unsupported seniority claims, and language mismatch.

Acceptance Criteria:

- [x] Impression rules are grounded in model signals.
- [x] Fallback policy exists.
- [x] Wrapper boundary is clear.

---

### Phase 9 — Candidate Job Reranking

Notebook: `training/notebooks/phase_09_candidate_reranking.ipynb`  
Status: Complete

Goal: Align recommendation training with backend-owned candidate retrieval instead of static job-index recommendation.

Tasks:

- [x] Step 9.1: Input contract — Document that backend provides jobCandidates and the model scores only those candidates.
- [x] Step 9.2: Output contract — Define output fields: jobId, matchScore, matchLevel, matchedSkills, missingSkills, rankingSignals, and optional nextSteps.
- [x] Step 9.3: Constraint validation — Define checks for no unknown jobId, no duplicate jobId, and every output jobId belonging to input candidates.
- [x] Step 9.4: Ranking evaluation — Define NDCG@5, NDCG@10, MAP@10, and comparison against best ranking baseline.
- [x] Step 9.5: Staleness boundary — Document that backend remains owner of job detail, visibility, availability, and hydration.

Acceptance Criteria:

- [x] Recommendation flow uses backend candidate sets.
- [x] Output never invents jobs.
- [x] Ranking metrics and constraints are defined.

---

### Phase 10 — Calibration, Model Card, and Artifact Manifest

Notebook: `training/notebooks/phase_10_calibration_model_card.ipynb`  
Status: Complete

Goal: Define score semantics, calibration evidence, model card contents, and artifact manifest requirements.

Tasks:

- [x] Step 10.1: Score semantics — Define what score ranges mean for jobFitAlignment, atsFriendliness, and recommendation matchScore.
- [x] Step 10.2: Calibration diagnostics — Plan calibration plots, bucket accuracy, bucket error, and confidence notes for each output score.
- [x] Step 10.3: Model card fields — Document model name, version, date, dataset hash, label version, split seed, feature config, metrics, limitations, and intended use.
- [x] Step 10.4: Artifact manifest fields — Document model paths, embedding/cache paths, schema versions, file hashes, and reproducibility references.
- [x] Step 10.5: Deployment readiness notes — Describe which artifacts are required for inference and which artifacts are training-only.

Acceptance Criteria:

- [x] Score meanings are not hardcoded without report evidence.
- [x] Model card template is complete.
- [x] Artifact manifest template is complete.

---

### Phase 11 — Final Gate Review

Notebook: `training/notebooks/phase_11_final_gate_review.ipynb`  
Status: Complete

Goal: Run a go/no-go review for model readiness using documented metrics, artifacts, risks, and product boundaries.

Tasks:

- [x] Step 11.1: Full pipeline review — Verify that data audit, labels, pairs, baselines, training, ATS scoring, recommendation reranking, calibration, and artifacts are complete.
- [x] Step 11.2: Gate metric review — Compare final metrics against required thresholds for job fit, ATS friendliness, recommendations, robustness, and language slices.
- [x] Step 11.3: Contract review — Confirm model outputs remain within product/API boundaries and do not own wrapper-only or backend-owned fields.
- [x] Step 11.4: Risk review — Document remaining data quality, label quality, calibration, localization, and deployment risks.
- [x] Step 11.5: Readiness decision — Choose one status: prototype-only, staging-ready, or production-ready, with evidence and required next actions.

Acceptance Criteria:

- [x] Final readiness status is evidence-based.
- [x] Remaining risks are documented.
- [x] Next actions are clear.

---

## Implementation Phases — Production Training Track

The completed Phase 0-11 notebooks are design and gate-review artifacts. The phases below keep training execution notebook-first while making each notebook runnable, reproducible, auditable, and export-ready. Training model work must stay inside versioned `.ipynb` notebooks. Do not add Python package extraction or `training/*.py` entrypoints for training execution.

### Phase 12 — Repository Hygiene and Notebook Runtime Bootstrap

Notebook: `training/notebooks/phase_12_repository_hygiene_runtime_bootstrap.ipynb`  
Status: Complete

Goal: Keep training notebook-first while removing local environment noise and adding reproducible notebook runtime conventions.

Tasks:

- [x] Step 12.1: Remove or relocate committed virtual environments from `training/notebooks/**/venv`; keep only `.gitignore` rules for local environments.
- [x] Step 12.2: Create the Phase 12 notebook with documented setup cells for repo path resolution, dependency checks, deterministic seeds, logging, and artifact hashing.
- [x] Step 12.3: Define versioned notebook config blocks for `jobfit_v2`, `ats_quality_v1`, and shared feature/schema settings inside the notebook, with optional JSON report export under `reports/`.
- [x] Step 12.4: Add notebook verification cells that check required notebooks, configs, source artifacts, dependency availability, and model-load compatibility without relying on hidden kernel state.
- [x] Step 12.5: Document the notebook-only training policy, including where generated reports, model cards, manifests, and exported artifacts should be written.

Acceptance Criteria:

- [x] Phase 12 notebook runs top-to-bottom from a clean kernel.
- [x] Verification cells fail when required notebooks, config blocks, source artifacts, or dependencies are missing.
- [x] No committed virtual environment is required for training.
- [x] All promoted notebook configs include schema version, model version, label version, split seed, and artifact paths.
- [x] No new `training/*.py` package entrypoints are created for training execution.

---

### Phase 13 — Data Snapshot, Versioning, and Contract Freezing

Notebook: `training/notebooks/phase_13_data_snapshot_contract_freezing.ipynb`  
Status: Complete

Goal: Build immutable data snapshots and schema contracts in a reproducible notebook before new labels, pairs, or features are generated.

Tasks:

- [x] Step 13.1: Define source schemas in notebook config cells for jobs, profiles, CV text, parsed CV sections, candidate job sets, and manual labels.
- [x] Step 13.2: Generate dataset snapshot manifests from notebook cells with row counts, column profiles, null rates, duplicate IDs, language distribution, and SHA-256 hashes.
- [x] Step 13.3: Freeze model-owned output schemas in notebook cells for `jobFitAlignment`, `atsFriendliness`, `overallImpression`, and recommendation ranking outputs.
- [x] Step 13.4: Add notebook validation cells that reject missing IDs, duplicate IDs, unsupported languages, invalid scores, and unsafe fields.
- [x] Step 13.5: Store snapshot reports under `reports/` and reference them from the model card template.

Acceptance Criteria:

- [x] Every training input file has hash, row count, schema version, and owner.
- [x] Dataset snapshot can be regenerated deterministically by running the notebook top-to-bottom from a clean kernel.
- [x] Model-core schema is aligned with `references/docs/generated/openapi.json` boundaries.
- [x] Unsafe wrapper/backend-owned fields are blocked from model-core training outputs.

---

### Phase 14 — Normalization and Feature Builder Implementation

Notebook: `training/notebooks/phase_14_normalization_feature_builder.ipynb`  
Status: Complete

Goal: Implement reusable notebook feature cells for skills, experience, language, text construction, and embeddings.

Tasks:

- [x] Step 14.1: Implement notebook cells for experience normalization covering `Fresher`, `1-2 years`, `3-5 years`, `5+ years`, `ENTRY_LEVEL`, `JUNIOR`, `MID_LEVEL`, `SENIOR`, `LEAD`, and `MANAGER`.
- [x] Step 14.2: Implement notebook cells for versioned skill alias normalization with Indonesian/English aliases, punctuation cleanup, framework variants, and unknown-skill reporting.
- [x] Step 14.3: Implement notebook cells for language normalization for `ID`, `EN`, `MIXED`, and `UNKNOWN` with slice-level reports.
- [x] Step 14.4: Implement notebook cells for text builders for profile/CV text, job text, role text, requirement text, and section-aware CV text.
- [x] Step 14.5: Implement notebook cells for embedding generation/cache validation with embedding model version, shape, dtype, finite-value checks, and hash manifest.

Acceptance Criteria:

- [x] No observed experience value falls through silently.
- [x] Feature-quality report includes unknown language, unknown experience, empty skills, empty text, and unknown skill token rates.
- [x] Feature cells are exercised by notebook verification fixtures and produce exported reports under `reports/`.
- [x] Embedding cache cannot be reused with mismatched model version, row count, or hash.

---

### Phase 15 — Balanced Pair Generation v2 and Leakage-Safe Splits

Notebook: `training/notebooks/phase_15_balanced_pair_generation_splits.ipynb`  
Status: Complete

Goal: Materialize a full-range job-fit training dataset from notebook cells with pair metadata and split safety.

Tasks:

- [x] Step 15.1: Implement notebook pair-generation cells for high-fit positives, medium-fit pairs, hard negatives, random negatives, same-role different-seniority pairs, and cross-role confusing pairs.
- [x] Step 15.2: Generate `pairs_v2.parquet` with required metadata: `pair_id`, `profile_id`, `job_id`, `pair_type`, `split`, `score_band`, `job_fit_score`, label components, language, role family, and experience band.
- [x] Step 15.3: Enforce profile-level split isolation and report leakage checks.
- [x] Step 15.4: Enforce high-fit validation/test minimum coverage and low/medium/high score-band coverage in every split.
- [x] Step 15.5: Publish distribution diagnostics by score, pair type, role family, language, experience band, and split.

Acceptance Criteria:

- [x] Validation and test splits each contain enough high-fit examples for score-band evaluation.
- [x] No `profile_id` appears in more than one split.
- [x] Every required `pair_type` is present overall and in validation/test, or has a documented blocker.
- [x] Legacy weak-label `pairs.parquet` is no longer used as production training evidence.

---

### Phase 16 — Human Validation Labels and Label Governance

Notebook: `training/notebooks/phase_16_human_validation_label_governance.ipynb`  
Status: Complete

Goal: Add trusted validation evidence so model quality is not judged only against weak labels.

Tasks:

- [x] Step 16.1: Create reviewer guidelines for job-fit score bands, ATS issue labels, recommendation relevance, and unsupported-claim rejection.
- [x] Step 16.2: Sample balanced low/medium/high job-fit cases from validation/test splits without leaking labels into training features.
- [x] Step 16.3: Collect human labels with reviewer ID, timestamp, label version, evidence notes, and disagreement flags.
- [x] Step 16.4: Calculate reviewer agreement such as weighted kappa or equivalent score-band agreement.
- [x] Step 16.5: Freeze manual validation labels as immutable evaluation-only artifacts.

Acceptance Criteria:

- [x] Human-labeled validation set covers low, medium, and high score bands.
- [x] Manual labels are never used as model input features.
- [x] Reviewer agreement is reported before production-readiness claims.
- [x] Label version and label manifest hash appear in model card inputs.

---

### Phase 17 — Baseline Evaluation v2

Notebook: `training/notebooks/phase_17_baseline_evaluation_v2.ipynb`  
Status: Complete

Goal: Recompute transparent baselines on `pairs_v2` inside a notebook before training complex models, using `intfloat/e5-base-v2` as the default English embedding model.

Tasks:

- [x] Step 17.1: Implement notebook baseline cells for constant mean, constant median, skill-overlap-only, E5 cosine-only, simple regression, and feature-regression baselines with `intfloat/e5-base-v2` embeddings.
- [x] Step 17.2: Report MAE, RMSE, R-squared, Spearman, score-band agreement, and high-fit recall.
- [x] Step 17.3: Report slice metrics by role family, language, experience band, pair type, and score band.
- [x] Step 17.4: Add ranking proxy metrics where candidate groups exist: NDCG@5, NDCG@10, and MAP@10.
- [x] Step 17.5: Write error inspection examples for false-high and false-low predictions.

Acceptance Criteria:

- [x] Best baseline is selected and stored as the model-improvement floor.
- [x] Complex training is blocked if validation/test high-fit coverage is insufficient.
- [x] Slice regressions and weak data ranges are visible in `reports/`.
- [x] Training gate fails unless later models beat the best baseline by required margins.

---

### Phase 18 — JobFitAlignment Model Training v2

Notebook: `training/notebooks/phase_18_jobfit_training_v2.ipynb`  
Status: Complete

Goal: Train and select a calibrated English-focused job-fit scorer in a notebook that beats baselines and produces grounded alignment signals, using `intfloat/e5-base-v2` as the default frozen embedding model.

Tasks:

- [x] Step 18.1: Train candidate models: linear baseline, cosine baseline, neural embedding scorer, feature-augmented scorer, and optional ranking objective.
- [x] Step 18.2: Keep model inputs limited to approved features: `intfloat/e5-base-v2` embeddings, skill overlap, experience gap, role match, requirement coverage, language, and pair metadata allowed for slices.
- [x] Step 18.3: Export model-owned outputs: `score`, `summarySignals`, `missingSignals`, `matchedSkills`, `missingSkills`, and confidence notes.
- [x] Step 18.4: Evaluate against baselines on validation, test, human-labeled validation, and required slices.
- [x] Step 18.5: Save checkpoints, training curves, final model artifact, experiment config, and run manifest.

Acceptance Criteria:

- [x] MAE improves at least 20% versus best baseline or documented stricter gate.
- [x] R-squared is positive and meets release threshold on validation/test.
- [x] Spearman and score-band agreement meet release thresholds.
- [x] Model uses `intfloat/e5-base-v2` query/passage embeddings and does not emit unsupported skills, seniority claims, or wrapper-owned copy.

---

### Phase 19 — ATS Friendliness Benchmark and Scorer

Notebook: `training/notebooks/phase_19_ats_friendliness_benchmark_scorer.ipynb`  
Status: Complete

Goal: Make `atsFriendliness` measurable in a notebook with parser benchmarks, issue labels, and score calibration.

Tasks:

- [x] Step 19.1: Create locked CV benchmark manifest covering normal PDF, scanned PDF, multi-column PDF, table-heavy PDF, short CV, long CV, and any supported non-PDF format if contract expands.
- [x] Step 19.2: Implement parser benchmark metrics: extraction coverage, empty-text rate, section detection accuracy, and critical parse failure recall.
- [x] Step 19.3: Label ATS issue taxonomy: parseability, section completeness, contact detection, date detection, metric evidence, formatting risk, and empty parse risk.
- [x] Step 19.4: Implement transparent ATS scorer/classifier in notebook cells and compare against rule-based baseline.
- [x] Step 19.5: Report precision, recall, bucket agreement, issue-level errors, and fallback behavior.

Acceptance Criteria:

- [x] ATS precision, recall, and bucket agreement meet release thresholds.
- [x] Empty-text rate on supported documents stays under release threshold.
- [x] Unsupported/scanned/empty parse cases trigger safe fallback outputs.
- [x] `atsFriendliness.score` and `detectedIssues` are evidence-backed.

---

### Phase 19.5 — Pre-Phase 20 JobFit Blocker Remediation

Notebook: `training/notebooks/phase_19_5_jobfit_blocker_remediation.ipynb`  
Status: Complete

Goal: Remove current jobFitAlignment blockers before Overall Impression work starts, especially real E5 embedding usage, Phase 18 gate consistency, skill-signal cleanliness, and minimum evidence quality.

Tasks:

- [x] Step 19.5.1: Fix E5 runtime — Install/pin `sentence-transformers`, load/cache `intfloat/e5-base-v2`, regenerate Phase 17/18 embedding manifests with `query:` profile/CV prefixes, `passage:` job prefixes, normalized embeddings, and `production_eligible_e5=true`.
- [x] Step 19.5.2: Remove TF-IDF production fallback — Update Phase 17/18 notebook gates so TF-IDF or local-hash embedding fallback is allowed only for local plumbing and fails any production/staging readiness gate.
- [x] Step 19.5.3: Rerun Phase 17 and Phase 18 with real E5 — Recompute baselines, model metrics, slice metrics, human-label evaluation, output examples, checkpoints, and run manifests from clean kernels.
- [x] Step 19.5.4: Enforce Phase 18 selection floor — Block selected models unless they meet MAE improvement and preserve or improve best-baseline high-fit recall and score-band agreement on validation/test, or document an explicit approved trade-off.
- [x] Step 19.5.5: Clean skill signals — Filter `matchedSkills` and `missingSkills` so outputs contain skill/requirement evidence only, not marketing prose, benefits, location copy, or unsupported seniority claims.
- [x] Step 19.5.6: Audit human validation evidence — Verify Phase 16 labels are independent evaluation evidence, not copied weak labels; document reviewer independence, timestamps, evidence notes, disagreement policy, and any need for more labels.
- [x] Step 19.5.7: Review weak slices before Phase 20 — Report language and role-family gaps, especially `ID`, `MIXED`, `UNKNOWN`, small role slices, and high-fit recall regressions such as software engineering/security/other.

Acceptance Criteria:

- [x] Phase 17 and Phase 18 embedding manifests show real `intfloat/e5-base-v2` backend, correct query/passage prefixes, normalized embeddings, no E5 load exception, and `production_eligible_e5=true`.
- [x] Phase 18 status is no longer blocked by the E5 gate.
- [x] Selected Phase 18 model meets MAE, R-squared, Spearman, score-band agreement, and high-fit recall gates versus the best baseline, or the trade-off is explicitly documented as prototype-only.
- [x] `matchedSkills` and `missingSkills` output examples contain no benefits, marketing copy, generic prose, unsupported seniority claims, or wrapper-owned copy.
- [x] Human-label audit report states whether labels are trusted enough for Phase 20 evidence use.
- [x] Slice-risk report lists blockers and non-blocking risks before Phase 20 starts.

---

### Phase 20 — Overall Impression Signal Implementation

Notebook: `training/notebooks/phase_20_overall_impression_signals.ipynb`  
Status: Complete

Goal: Generate grounded impression signals in a notebook without hallucinated claims or unsupported product copy.

Tasks:

- [x] Step 20.1: Implement notebook cells for deterministic evidence ledger from job-fit, ATS, matched/missing skills, role, experience, language, and parser confidence.
- [x] Step 20.2: Implement notebook cells for safe summary templates for `ID` and `EN` that only mention observed evidence.
- [x] Step 20.3: Implement notebook cells for fallback summaries for empty CV parse, missing target job, unknown language, low confidence, and sparse evidence.
- [x] Step 20.4: Add validation checks for hallucinated skills, unsupported seniority, hiring-outcome claims, and language mismatch.
- [x] Step 20.5: Export core `overallImpression.score`, `summary`, evidence keys, and confidence notes for wrapper rendering.

Acceptance Criteria:

- [x] Every generated impression can be traced to evidence keys.
- [x] Unsupported claims are rejected before export.
- [x] ID and EN summary templates are tested with representative examples.
- [x] Wrapper-owned `topActionables` and `sectionReviews` remain outside model-core ownership.

---

### Phase 21 — Backend Candidate Reranking Implementation

Notebook: `training/notebooks/phase_21_backend_candidate_reranking.ipynb`  
Status: Complete

Goal: Replace static job-index recommendations with notebook scoring/ranking for backend-provided candidate jobs.

Tasks:

- [x] Step 21.1: Define candidate-set input schema with request ID, candidate set ID, profile/CV features, and backend-provided `jobCandidates`.
- [x] Step 21.2: Implement notebook scorer cells that rank only input candidates and return `jobId`, `matchScore`, `matchLevel`, matched/missing skills, and ranking signals.
- [x] Step 21.3: Enforce constraint checks for no unknown job IDs, no duplicate job IDs, score range `0-100`, and maximum item count.
- [x] Step 21.4: Build candidate-set relevance labels from backend-like groups or manual review.
- [x] Step 21.5: Evaluate NDCG@5, NDCG@10, MAP@10, baseline uplift, and constraint violation rates.

Acceptance Criteria:

- [x] Model never invents jobs outside backend input candidates.
- [x] Static `job_index.json` is not required for production recommendation output.
- [x] Ranking metrics beat baseline by release threshold.
- [x] Backend remains owner of title, company, visibility, availability, reason copy, and hydration.

---

### Phase 22 — Calibration, Model Card, and Artifact Export

Notebook: `training/notebooks/phase_22_calibration_model_card_export.ipynb`  
Status: Complete

Goal: Turn notebook-trained outputs into deployable, versioned, auditable artifacts.

Tasks:

- [x] Step 22.1: Calibrate `jobFitAlignment.score`, `atsFriendliness.score`, and `recommendations[].matchScore` into `0-100` semantics.
- [x] Step 22.2: Generate calibration tables for buckets `0-20`, `21-40`, `41-60`, `61-80`, and `81-100`.
- [x] Step 22.3: Report ECE, MCE, bucket agreement, within-10-points rate, bucket MAE, and slice calibration.
- [x] Step 22.4: Export final model artifacts, feature config, label manifest, schema files, model card, and artifact manifest with SHA-256 hashes.
- [x] Step 22.5: Run runtime load checks for exported artifacts from a clean notebook kernel and document any environment requirement.

Acceptance Criteria:

- [x] Score meanings are backed by calibration evidence.
- [x] Model card includes git commit, dataset hash, label version, split seed, feature config, metrics, limitations, intended use, and blocked use.
- [x] Artifact manifest lists every inference-required and training-only artifact with hash and schema version.
- [x] Exported model loads without hidden notebook state or local-only paths.

---

### Phase 23 — Model API Contract and Integration Validation

Notebook: `training/notebooks/phase_23_model_api_contract_validation.ipynb`  
Status: Complete

Goal: Prove notebook-exported model-core outputs can be safely consumed by the backend/API wrapper contract.

Tasks:

- [x] Step 23.1: Define model-core request/response schemas for CV analysis and candidate reranking.
- [x] Step 23.2: Add notebook contract fixtures aligned with `references/docs/generated/openapi.json` and `references/docs/modules/ai-cv-analyzer.md`.
- [x] Step 23.3: Validate score bounds, required fields, language handling, candidate membership, duplicate rejection, max item counts, and model metadata.
- [x] Step 23.4: Validate safe fallbacks for timeout, model unavailable, invalid model output, empty CV parse, and low confidence.
- [x] Step 23.5: Produce integration readiness report that separates model-core outputs from backend/wrapper-owned response fields.

Acceptance Criteria:

- [x] Model-core output can be mapped to `cv-analysis-v2` without exposing raw internals.
- [x] Invalid model output is rejected before persistence or user response.
- [x] Notebook contract checks include positive and negative fixtures.
- [x] Backend/wrapper-only fields remain outside training artifacts.

---

### Phase 24 — CI, Reproducibility, and Final Production Gate

Notebook: `training/notebooks/phase_24_reproducibility_final_gate.ipynb`  
Status: Complete

Goal: Automate notebook readiness checks and make production promotion evidence-based.

Tasks:

- [x] Step 24.1: Add CI-safe notebook execution checks for schema validation, data snapshot verification, baseline evaluation smoke test, and artifact manifest validation.
- [x] Step 24.2: Add reproducibility run that rebuilds features, pairs, metrics, model card, and export manifest from frozen notebook configs.
- [x] Step 24.3: Add production gate report that summarizes job-fit, ATS, recommendation, robustness, language, calibration, and contract results.
- [x] Step 24.4: Add release checklist for artifact promotion, rollback, monitoring hooks, and model version registration.
- [x] Step 24.5: Mark final status as `prototype-only`, `staging-ready`, or `production-ready` based only on passed gates.

Acceptance Criteria:

- [x] CI fails on missing required reports, hashes, schemas, or gates.
- [x] Full training/evaluation/export pipeline can be rerun from notebook configs in clean-kernel order.
- [x] Production gate passes only when all required thresholds are met.
- [x] Final readiness decision links to immutable reports and artifacts.

---

## Production Delivery Phase — Single Notebook TensorFlow Training Track

The completed Phase 12-24 notebooks provide implementation evidence, but the current selected model is exported as `joblib` and does not fully satisfy `REQUIREMENT.md` for TensorFlow deep learning, custom training loop, TensorBoard, and `.keras`/`SavedModel` deployment. The phase below creates one final training-only notebook that consolidates the useful production-track work while meeting the assignment and model-api handoff requirements.

### Phase 25 — Single Notebook TensorFlow Training Delivery and Model API Handoff

Notebook: `training/notebooks/phase_25_tensorflow_training_delivery.ipynb`
Status: Complete

Goal: Build one clean, runnable, production-grade training notebook that uses the best current Phase 12-24 data/feature/evaluation contracts, replaces the `joblib` scorer with a TensorFlow model, satisfies `REQUIREMENT.md` model-training requirements, and exports artifacts that a separate FastAPI/Flask Model API can load safely.

Scope boundary:

- Notebook owns training, evaluation, calibration, TensorBoard logging, TensorFlow export, model card, artifact manifest, and inference smoke tests.
- Notebook must not implement the long-running FastAPI/Flask server, direct backend DB access, auth, persistence, job hydration, or OpenAI/GenAI wrapper runtime.
- API runtime remains separate source code, but this phase must export schemas, examples, and handoff fixtures aligned with `references/docs/generated/openapi.json` so the API can consume the model artifact without hidden notebook state.

Notebook writing style:

- Use concise English Markdown, similar to `legacy/bisakerja_model_training_FINAL.ipynb`, but shorter and more production-focused.
- Each section should explain only purpose, required input, action, expected output, and gate in a few bullets before code.
- Avoid long theory, verbose background, and duplicated explanations; implementation cells should start quickly after the short Markdown context.
- Make outputs human-readable: compact tables, metric summaries, gate badges/pass-fail rows, clear artifact paths, and pretty-printed JSON snippets.
- Keep raw debug logs, huge dataframe dumps, embedding arrays, and long stack traces out of saved notebook outputs unless needed to explain a failed gate.

Tasks:

- [x] Step 25.1: Requirement and contract matrix — Add a concise English notebook section that maps every `REQUIREMENT.md` item to implemented notebook cells or explicit non-notebook API deliverables. Include the OpenAPI boundary for `AnalyzeCvMultipartRequest`, `CvAnalysis.analysisResult`, `jobFitAlignment`, `atsFriendliness`, `overallImpression`, and `jobRecommendations`.
- [x] Step 25.2: Single-notebook reproducibility setup — Define deterministic seeds, repo-root resolver, dependency checks, runtime versions, clean-worktree warning, artifact paths, config versions, dataset hashes, label versions, split seed, and fail-fast checks in the first executable cells. Display a compact human-readable setup summary table.
- [x] Step 25.3: Data and feature reuse from current notebooks — Load or rebuild Phase 13-16 snapshots, `pairs_v2.parquet`, feature-quality reports, human validation labels, ATS benchmark labels, and candidate-set fixtures. Reuse Phase 14 normalization rules and Phase 15 split isolation; reject stale hashes, missing required columns, leakage, unsupported languages, and empty required evidence.
- [x] Step 25.4: E5 embedding contract — Generate or validate `intfloat/e5-base-v2` embeddings with `query:` profile/CV prefixes, `passage:` job prefixes, normalized embeddings, fixed shape, finite-value checks, cache hash validation, and explicit failure when TF-IDF/local-hash fallback is used for staging or production evidence.
- [x] Step 25.5: TensorFlow architecture — Implement a TensorFlow Functional API or Model Subclassing architecture for job-fit scoring using approved numeric features and E5-derived similarity features. The model must produce a calibrated score compatible with `0-100` API semantics while training on a normalized `0-1` target.
- [x] Step 25.6: Required custom component — Implement at least one advanced TensorFlow custom component used by the selected model, such as `CosineInteractionLayer`, `WeightedHuberLoss`, or `ProductionGateCallback`. Register serialization with `get_config()` so exported `.keras` or `SavedModel` can reload without notebook-only objects.
- [x] Step 25.7: Custom training and evaluation loop — Train and evaluate with `tf.GradientTape` as the primary loop. Do not use `model.fit()` for the main training path. Track train/validation/test loss, MAE, RMSE, R², Spearman, score-band agreement, high-fit recall, calibration metrics, and slice metrics.
- [x] Step 25.8: TensorBoard monitoring — Write bounded TensorBoard logs for train/eval metrics, learning rate, loss curves, gate metrics, and selected histograms under a versioned path such as `artifacts/tensorboard/phase_25_tensorflow_training_delivery/`. Include log path and hash references in the artifact manifest.
- [x] Step 25.9: Baseline and strict selection gate — Compare the TensorFlow model against Phase 17 best baseline and Phase 18 selected scorer. Production selection must improve or preserve MAE, high-fit recall, and score-band agreement, pass R²/Spearman thresholds, and meet `REQUIREMENT.md` regression target `MAE <= 0.02` on normalized `0-1` scale (`<= 2.0` points on `0-100` scale). If any prototype trade-off remains, final readiness must be capped at `staging-ready`.
- [x] Step 25.10: Calibration and model-card export — Calibrate `jobFitAlignment.score`, `atsFriendliness.score`, and `recommendations[].matchScore` buckets `0-20`, `21-40`, `41-60`, `61-80`, `81-100`; export calibration tables, model card, feature config, label manifest, dataset manifest, and artifact manifest with SHA-256 hashes.
- [x] Step 25.11: TensorFlow artifact export — Export the selected TensorFlow model as `.keras` or `SavedModel`, reload it from a clean notebook cell with custom objects registered, and run inference smoke tests without hidden kernel state or local-only paths.
- [x] Step 25.12: Model API handoff fixtures — Export JSON fixtures for CV analysis and candidate reranking that include model-core outputs only, reject wrapper/backend-owned fields, enforce score bounds, enforce language `id/en`, enforce candidate membership for recommendations, and map cleanly to `cv-analysis-v2` wrapper response fields from OpenAPI.
- [x] Step 25.13: Training-only GenAI boundary — Document that GenAI is a secondary API-wrapper feature, not a training signal. Provide deterministic placeholder summary signals and a handoff contract for the API wrapper; do not call external GenAI services from training cells.
- [x] Step 25.14: Final clean-kernel gate — Run the notebook top-to-bottom from a clean kernel, save without error outputs, verify all exported hashes, verify TensorBoard logs exist, verify `.keras`/`SavedModel` reload, verify API handoff fixtures, and write `reports/phase_25_tensorflow_training_delivery.json` with final status: `prototype-only`, `staging-ready`, or `production-ready`.
- [x] Step 25.15: Human-readable final report — End the notebook with one compact final summary containing requirement compliance, key metrics, gate status, artifact paths, model version, API handoff files, known limitations, and next action. This summary should be readable without scrolling through all code cells.

Acceptance Criteria:

- [x] One notebook contains the full training implementation needed for model development and export; no other notebook state is required to reproduce the selected TensorFlow model.
- [x] Notebook Markdown is concise English and implementation-first; each section has short context followed directly by executable cells.
- [x] Saved notebook outputs are human-readable, bounded, and production-focused: compact metrics, pass/fail gates, artifact paths, and JSON examples without excessive logs or large dumps.
- [x] `REQUIREMENT.md` sections 1.1-1.5 and 2.1-2.2 are fully satisfied by the notebook: TensorFlow architecture, custom component, `tf.GradientTape` training/evaluation loop, TensorBoard logs, target MAE gate, `.keras`/`SavedModel` export, and inference smoke test.
- [x] `REQUIREMENT.md` sections 3.x and 4.x are not embedded as notebook runtime, but the notebook exports validated artifacts, schemas, examples, and handoff fixtures required for a separate FastAPI/Flask Model API and GenAI wrapper implementation.
- [x] The selected TensorFlow model beats or preserves Phase 17/18 gates without prototype-only trade-offs, or final readiness is capped below production.
- [x] The final artifact export is from a clean or explicitly documented worktree; production-ready status is blocked when `git_dirty_at_export=true`.
- [x] Exported TensorFlow model reloads without hidden notebook state and produces bounded JSON-compatible outputs for model-core inference.
- [x] Model-core output aligns with `references/docs/generated/openapi.json` while wrapper-owned fields (`topActionables`, `sectionReviews`, generated CV copy, job title/company hydration, auth, persistence) remain outside the training artifact.
- [x] No direct backend database integration is required or performed by the notebook; backend candidate jobs are represented by fixtures and model output only scores/ranks provided candidate IDs.
- [x] TensorBoard log references, model artifact hash, dataset hash, label hash, feature-config hash, calibration report, and model card are all recorded under `reports/` or `artifacts/`.

---

## Model API Delivery Phase — Loader and Inference Endpoint

Phase 25 exports the selected TensorFlow `.keras` artifact and handoff contracts. The next phase turns those artifacts into a small serving API that satisfies `REQUIREMENT.md` sections 3.x and the inference-code deliverable without retraining or embedding backend responsibilities.

### Phase 26 — Model API Loader and Inference Endpoint

Source artifacts:

- `artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras`
- `artifacts/phase_25_tensorflow_training_delivery/tensorflow_feature_config.json`
- `artifacts/phase_25_tensorflow_training_delivery/feature_config.json`
- `artifacts/phase_25_tensorflow_training_delivery/score_calibration.json`
- `artifacts/phase_25_tensorflow_training_delivery/model_card.json`
- `artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json`
- `artifacts/phase_25_tensorflow_training_delivery/export/inference_smoke_fixture.json`
- `artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_fixtures.json`
- `artifacts/phase_25_tensorflow_training_delivery/export/model_api_handoff_validation.json`
- `artifacts/phase_25_tensorflow_training_delivery/export/genai_wrapper_handoff_contract.json`
- `references/docs/generated/openapi.json`

Status: In progress

Goal: Implement a FastAPI or Flask Model API that loads the Phase 25 TensorFlow model once at startup, builds the approved six-feature inference vector from user/CV and backend-provided candidate jobs, runs batch inference, and returns bounded JSON model-core outputs aligned with `cv-analysis-v2` handoff rules.

Scope boundary:

- API owns model loading, artifact validation, feature construction, inference, response validation, health checks, and smoke tests.
- API must not retrain models, mutate Phase 25 artifacts, access backend DB directly, perform auth/persistence, hydrate job details, or call external GenAI by default.
- Backend provides sanitized CV/profile signals and `jobCandidates`; Model API only scores/ranks candidate IDs from that set.
- GenAI remains wrapper-owned secondary behavior via `genai_wrapper_handoff_contract.json`; model scores and candidate membership must never be changed by GenAI output.

Tasks:

- [x] Step 26.1: API package layout — Create production source layout for the serving app, config module, schemas, artifact loader, feature builder, inference service, response validators, and tests. Keep legacy prototype code separate unless migrated intentionally.
- [x] Step 26.2: Artifact manifest verification — Resolve artifact paths from config/env, load `artifact_manifest.json`, verify SHA-256/size for every runtime-required Phase 25 artifact, and fail startup when model/config/contract hashes are missing or stale.
- [x] Step 26.3: Custom object registration — Move or wrap Phase 25 custom-object definitions from `registered_custom_objects_smoke.py` into reusable API code, register `CosineInteractionLayer`, `WeightedHuberLoss`, `ProductionGateCallback`, and `HighRecallCalibrationLayer`, then load `.keras` with `compile=False`.
- [x] Step 26.4: Startup model loader — Load TensorFlow/Keras model once during FastAPI lifespan or Flask app startup, expose model name/version/artifact hash/readiness, and return `503` for inference while loader is not ready.
- [x] Step 26.5: Request schemas — Define strict request schemas for CV/profile scoring and candidate reranking: `requestId`, `language` enum `id/en`, sanitized profile/CV text or extracted signals, and `jobCandidates[]` with backend-owned metadata separated from model-owned scoring inputs.
- [x] Step 26.6: Feature builder — Build exactly the Phase 25 approved feature vector in order: `e5_cosine`, `skill_overlap`, `requirement_coverage`, `role_match`, `experience_match`, `experience_gap_years_clipped`. Use `intfloat/e5-base-v2` with `query:` and `passage:` prefixes, validate finite values, normalize via `tensorflow_feature_config.json`, and reject TF-IDF/local-hash fallback for staging/production mode.
- [x] Step 26.7: Batch inference service — Run model prediction for all candidate pairs, transform `score_0_1` to integer `0-100`, clamp bounds, apply calibration policy from `score_calibration.json` when required, preserve candidate IDs, sort by score descending, and limit recommendations to OpenAPI max item count.
- [x] Step 26.8: Inference endpoints — Implement `GET /health`, `GET /model-info`, and `POST /inference/cv-analysis` or equivalent REST route that returns model-core JSON for `jobFitAlignment`, `atsFriendliness`, `overallImpression` evidence keys/placeholders, and `candidateReranking.recommendations[]` without wrapper/backend-owned fields.
- [x] Step 26.9: Contract validators — Validate every response against Phase 25 handoff rules: score integers `0-100`, language `id/en`, unique recommendations, candidate membership enforced, max recommendations `5`, no fields such as `title`, `companyName`, `reason`, `nextStep`, `topActionables`, `sectionReviews`, auth, persistence, or hydrated job data.
- [x] Step 26.10: Error and fallback behavior — Return deterministic JSON errors for invalid language, empty candidate set, duplicate candidate IDs, missing text/signals, unsupported artifact version, embedding/model failures, and timeout. Do not silently return rule-based scores when TensorFlow inference fails in staging/production.
- [x] Step 26.11: Smoke and contract tests — Add tests that load the real Phase 25 `.keras` artifact, run `inference_smoke_fixture.json`, verify bounded predictions, run positive and negative cases from `model_api_handoff_fixtures.json`, and compare validation behavior with `model_api_handoff_validation.json`.
- [x] Step 26.12: API docs and runbook — Document install/run commands, required Python/TensorFlow versions, env vars, artifact paths, example requests/responses, health checks, smoke-test command, deployment notes, and known non-goals. Update README or `RUNNING_STEPS.md`.
- [x] Step 26.13: Packaging and dependencies — Add or update `requirements.txt` for serving runtime only, pin TensorFlow/Keras/SentenceTransformers-compatible versions, and keep notebook/training-only dependencies out unless required for inference.
- [x] Step 26.14: Final API gate report — Write a compact report under `reports/` summarizing loader status, artifact hashes, contract tests, endpoint smoke output, latency sample, model version, and readiness decision.

Acceptance Criteria:

- [ ] REST API starts from clean process, verifies Phase 25 artifact hashes, registers custom objects, and loads `selected_jobfit_tf_phase25.keras` without notebook state.
- [ ] Inference endpoint accepts sanitized user/CV signals plus backend-provided `jobCandidates`, runs TensorFlow prediction, and returns JSON with bounded scores.
- [ ] Feature vector order, normalization, E5 prefix policy, and score scaling match Phase 25 `tensorflow_feature_config.json` and handoff fixtures.
- [ ] Response contains model-core fields only; wrapper/backend-owned fields are rejected before response.
- [ ] Invalid language, duplicate/unknown candidate IDs, out-of-range scores, missing model, stale hashes, and malformed input have deterministic tests.
- [ ] `REQUIREMENT.md` sections 2.2 and 3.x are satisfied by source code, tests, docs, and a passing local smoke command.
- [ ] GenAI integration remains secondary and wrapper-owned; no external GenAI call is required for core model inference.

---

## Production Readiness Closure Phase — Final Training and Runtime Gate

Phase 25 currently proves TensorFlow training delivery and handoff at staging level. Phase 26 implements the Model API source boundary but still records runtime loader verification as pending in the current interpreter. This phase closes the remaining production blockers before any `production-ready` claim is allowed.

### Phase 27 — Production Readiness Remediation, Clean Freeze, and Release Gate

Source evidence:

- `REQUIREMENT.md`
- `GAP_MODEL_TRAINING.md`
- `reports/phase_25_tensorflow_training_delivery.json`
- `reports/phase_25_human_readable_final_report.md`
- `artifacts/phase_25_tensorflow_training_delivery/model_card.json`
- `artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json`
- `artifacts/phase_25_tensorflow_training_delivery/export/selected_jobfit_tf_phase25.keras`
- `reports/phase_26_final_api_gate_report.json`
- `model_api/README.md`

Status: Planned

Goal: Convert the current staging-ready training and source/contract-ready API evidence into a production-ready release by removing dirty-worktree, untracked-monitoring, weak-label-only, small-fixture, stale-notebook, and runtime-verification blockers.

Scope boundary:

- Training remains notebook-first and centered on `training/notebooks/phase_25_tensorflow_training_delivery.ipynb` unless `REQUIREMENT.md` changes.
- Production-ready means the final model artifact, TensorBoard logs, model card, calibration, contract fixtures, API loader, tests, and docs are all reproducible from a clean commit.
- Model-core output remains limited to scores, evidence signals, and ranking for backend-provided candidate IDs.
- Backend auth, persistence, DB hydration, frontend rendering, and external GenAI prose generation remain outside training readiness.

Tasks:

- [ ] Step 27.1: Freeze a clean release baseline — Start from a clean worktree, record final git commit, remove or commit all generated Phase 25/26 changes intentionally, and make `git_dirty_at_setup=false` and `git_dirty_at_export=false` mandatory for production status.
- [ ] Step 27.2: Make TensorBoard evidence release-visible — Store Phase 25 TensorBoard event files or a reproducible TensorBoard archive under a tracked release path, record SHA-256/byte-size in the artifact manifest, and ensure `REQUIREMENT.md` section 1.4 is satisfied without relying on ignored local files.
- [x] Step 27.3: Repair notebook hygiene — Remove saved error outputs from older notebooks such as Phase 8, execute or intentionally retire unexecuted code cells such as Phase 13, update stale notebook indexes/README files, and add a report proving all active production notebooks are saved without error outputs.
- [ ] Step 27.4: Upgrade label evidence beyond weak-label-only training — Define the production label policy, add or import a larger frozen human/recruiter-reviewed validation set, keep weak labels only as bootstrap/training support, and block production score claims when human labels are too small, single-reviewer, or not slice-covered.
- [x] Step 27.5: Expand ATS validation with real or sanitized CV documents — Replace synthetic-only ATS readiness with a mixed benchmark covering normal PDF, scanned PDF/OCR fallback, multi-column PDF, table-heavy PDF, DOCX, short CV, long CV, Indonesian CV, and English CV; report parse coverage, issue precision/recall, bucket agreement, and safe fallback behavior.
- [x] Step 27.6: Expand recommendation validation — Evaluate candidate reranking on a larger backend-like candidate-set fixture or real anonymized relevance labels, preserve candidate membership, prove NDCG/MAP uplift over backend order, and block production if metrics are based only on the current 3-set/15-candidate fixture.
- [ ] Step 27.7: Rerun Phase 25 clean-kernel production export — Re-execute the final TensorFlow training notebook top-to-bottom in Python 3.13, verify Functional API/custom components/`tf.GradientTape`/no `model.fit()`/MAE target/TensorBoard/export/inference smoke, and write a final report whose status is `production-ready` only when every strict gate passes.
- [x] Step 27.8: Refresh model card and artifact manifest — Update dataset hash, label hash, feature config hash, TensorBoard hash, model hash, calibration hash, git commit, split seed, intended use, blocked use, limitations, and score semantics so no stale Phase 22/25 hash or staging-only status remains.
- [ ] Step 27.9: Run real Model API production smoke — In a Python 3.13 serving environment, install root `requirements.txt`, load the real `.keras` artifact with registered custom objects, run unskipped Phase 26 tests, start FastAPI, hit `/health`, `/model-info`, and `/inference/cv-analysis`, and record real TensorFlow/E5 latency.
- [ ] Step 27.10: Verify `REQUIREMENT.md` end-to-end — Produce a requirement matrix proving sections 1.1-1.5, 2.1-2.2, 3.1-3.2, 4.1, and deliverables are satisfied by tracked source, artifacts, tests, logs, docs, and smoke evidence.
- [ ] Step 27.11: Write final production gate report — Create `reports/phase_27_production_readiness_release_gate.json` and a short Markdown summary with final decision, blockers, metrics, artifact paths, runtime versions, tests run, latency, and release checklist.

Acceptance Criteria:

- [ ] Final training report status is `production-ready`, not `staging-ready` or `prototype-only`.
- [ ] Final production gate fails if worktree is dirty, TensorBoard logs are missing/untracked, notebook error outputs remain, artifact hashes are stale, or API runtime smoke is skipped.
- [ ] `REQUIREMENT.md` is fully satisfied with tracked evidence: TensorFlow architecture, custom component, `tf.GradientTape` training/evaluation loop, TensorBoard logs, MAE target, `.keras`/`SavedModel` export, inference code, REST API, GenAI secondary boundary, requirements files, and docs.
- [ ] Model card no longer relies on weak-label-only evidence for production score claims; human/reviewer validation coverage and known limits are explicit.
- [ ] ATS and recommendation metrics are based on release-scale fixtures or approved anonymized real data, not only small synthetic/current handoff fixtures.
- [ ] Phase 26 API readiness decision becomes production-ready with real TensorFlow/Keras/FastAPI runtime verification, not `runtime-loader-verification-pending`.
- [ ] Public output contract remains safe: model-core fields only, bounded scores, `id/en` language, max 5 recommendations, no invented jobs, no backend-owned hydrated fields, no auth/persistence/GenAI mutation inside core inference.
- [ ] A reviewer can reproduce the release from a clean commit using documented commands and verify all required hashes.

---

## Backend + Model API Production Integration Phases

These phases correct the current boundary mismatch: Backend OpenAPI receives PDF CV uploads, Backend owns auth/DB/persistence/final `cv-analysis-v2`, and Model API owns PDF parsing plus model-core scoring/ranking only. The preferred architecture is Backend-as-orchestrator; Model API must not connect directly to the production database.

### Phase 28 — Backend/Model API Contract Realignment

Status: Planned

Goal: Freeze a safe internal contract where Backend sends a sanitized multipart PDF plus backend-selected job candidates to Model API, and Model API returns model-core output that Backend maps to the public OpenAPI `cv-analysis-v2` response.

Scope boundary:

- Backend owns auth, file ownership, storage, candidate retrieval, DB hydration, persistence, idempotency, public response shape, and GenAI prose wrapper.
- Model API owns PDF parsing, CV evidence extraction, ATS issue signals, E5 feature building, TensorFlow scoring, candidate reranking, health checks, and deterministic model-core errors.
- Model API must not receive raw DB credentials, write backend data, generate final public hydrated job fields, or decide auth/ownership.
- Backend may send `backendMetadata` for trace/debug only; Model API must never trust it for candidate identity outside `jobId` membership.

Tasks:

- [ ] Step 28.1: Define final internal request contract — Document `POST /internal/model/cv-analysis` as multipart with `requestId`, `language`, `inputMode`, `compareSource`, `jobRoles[]`, `cvFile`, `jobCandidates` JSON, and `rankingPolicy` JSON.
- [ ] Step 28.2: Define final internal response contract — Document `model-core-cv-analysis-v1` containing `parsedCv`, `jobFitAlignment`, `atsFriendliness`, `overallImpression` evidence, `candidateReranking.recommendations[]`, `model`, and timestamps; exclude public wrapper fields.
- [ ] Step 28.3: Map OpenAPI/Prisma fields to contract owners — Create a matrix from `references/docs/generated/openapi.json` and `references/prisma/schema.prisma` covering `CvAnalysisResult`, `JobRecommendationRun`, `JobRecommendationItem`, `JobListing`, `JobRequirement`, and `JobSkill`.
- [ ] Step 28.4: Define language and enum mapping — Freeze `id/en` for public/model API, `ID/EN` for Prisma, `UPLOAD/REFERENCE`, `BOOKMARK/JOB_SEARCH/DIRECT_JOB_DETAIL`, and `strong/good/stretch` to Prisma `STRONG/GOOD/STRETCH`.
- [ ] Step 28.5: Define failure contracts — Specify deterministic backend behavior for Model API `422`, `503`, `504`, parse failure, empty candidate set, stale artifact, timeout, and GenAI wrapper failure.
- [ ] Step 28.6: Add contract fixtures — Create positive/negative fixtures for direct upload PDF, active CV reference, bookmarked candidates, job-search candidates, direct-job-detail candidate, duplicate job IDs, empty PDF parse, and missing candidate evidence.
- [ ] Step 28.7: Update durable docs — Update `model_api/README.md`, backend integration docs under `references/docs/integrations/model-api.md`, and module docs so future implementation follows one contract.

Acceptance Criteria:

- [ ] One internal contract clearly separates model-owned fields from backend/wrapper-owned fields.
- [ ] Backend can map every Model API response field into final `CvAnalysis.analysisResult` without guessing or inventing jobs.
- [ ] Model API rejects backend-owned public fields such as `title`, `companyName`, `reason`, `nextStep`, `topActionables`, `sectionReviews`, `generatedCv`, auth, persistence, and hydrated DB objects in model-core output.
- [ ] Contract fixtures cover all public compare sources and both upload/reference CV modes.
- [ ] Reviewer can verify the contract against OpenAPI and Prisma without reading implementation code.

---

### Phase 29 — Model API PDF Parsing and Model-Core Inference Hardening

Status: Planned

Goal: Upgrade Model API from JSON-only sanitized signals to production-safe PDF intake, deterministic parsing, ATS evidence extraction, TensorFlow scoring, and model-core output for backend-provided candidates.

Scope boundary:

- Model API accepts PDF bytes only from trusted Backend internal calls.
- Model API does not persist uploaded files after request completion and does not store raw CV text beyond request-scoped logs/metrics.
- Model API still supports JSON-only `/inference/cv-analysis` for tests/backward compatibility, but production flow uses multipart PDF endpoint.
- No external GenAI call is allowed in core inference.

Tasks:

- [ ] Step 29.1: Add safe env/config defaults — Provide `.env.example` for Model API with `MODEL_API_ENV`, artifact paths, `MODEL_API_SERVICE_TOKEN`, `MODEL_API_MAX_PDF_BYTES`, `MODEL_API_MAX_PDF_PAGES`, `MODEL_API_TIMEOUT_MS`, `SENTENCE_TRANSFORMERS_HOME`, `MODEL_API_ENABLE_GENAI_WRAPPER=false`, and OpenRouter vars disabled by default.
- [ ] Step 29.2: Add internal auth middleware — Require `Authorization: Bearer <MODEL_API_SERVICE_TOKEN>` for inference endpoints in staging/production while allowing explicit local/test bypass only through safe config.
- [ ] Step 29.3: Implement multipart PDF endpoint — Add `POST /internal/model/cv-analysis` that validates content type, file size, PDF magic bytes, one-file-only policy, required form fields, candidate JSON shape, and request ID.
- [ ] Step 29.4: Implement deterministic PDF parser — Extract text with a pinned parser, detect page count, empty/scanned PDFs, section names, contact/date signals, formatting risk, and parse quality; never hallucinate missing CV content.
- [ ] Step 29.5: Build profile/CV signals from parsed PDF — Convert parsed text into `SanitizedProfileInput` using `cvText`, `targetRoles`, detected sections, normalized skills, language, role family, and optional experience evidence.
- [ ] Step 29.6: Replace ATS placeholder logic — Compute `atsFriendliness.score` and `detectedIssues` from parser evidence with transparent penalties and safe fallback for empty/scanned/failed parse.
- [ ] Step 29.7: Fix top-candidate evidence — Derive `jobFitAlignment.matchedSkills/missingSkills` and recommendation skill evidence from the top-ranked candidate, not from the first input candidate.
- [ ] Step 29.8: Add candidate reranking endpoint — Add `POST /inference/candidate-reranking` for backend jobs-only scoring with the same membership, max-item, score-bound, and response-contract validators.
- [ ] Step 29.9: Add hard runtime guards — Enforce parse, embedding, and model inference timeouts; reject fallback embedding backends in staging/production; warm up TensorFlow and E5 at startup when configured.
- [ ] Step 29.10: Expand tests — Cover PDF parser edge cases, multipart validation, auth missing/invalid token, ATS scoring, top-candidate evidence, reranking endpoint, timeout errors, and no backend-owned fields.

Acceptance Criteria:

- [ ] Model API can parse a normal PDF CV sent by Backend and build valid Phase 25 features without backend pre-parsed `cvText`.
- [ ] Empty/scanned/malformed PDFs return deterministic safe errors or low-confidence fallback signals, never fabricated CV text.
- [ ] All model-core scores are bounded integers `0-100`; recommendations are max `5`, unique, and restricted to backend-supplied job IDs.
- [ ] Staging/production cannot run with missing service token, fallback embeddings, unverified artifacts, or skipped model loader.
- [ ] Tests prove PDF endpoint, JSON endpoint, reranking endpoint, and response validators work without leaking backend-owned fields.

---

### Phase 30 — Backend Orchestrator, Candidate Retrieval, GenAI Wrapper, and Persistence

Status: Planned

Goal: Update Backend API so `/api/v1/ai/cv-analyzer` orchestrates CV file resolution, DB candidate retrieval, Model API call, GenAI wrapper copy generation, job hydration, persistence, and public `cv-analysis-v2` response formatting.

Scope boundary:

- Backend owns the public OpenAPI contract and must return exactly `CvAnalysis` envelope shape.
- Backend retrieves candidates from Prisma DB; Model API receives only those candidates and cannot invent or hydrate jobs.
- Backend wrapper may use GenAI for prose fields, but must be grounded in model-core evidence and DB data.
- If GenAI fails, Backend returns deterministic fallback copy from model-core evidence; scores/ranks must not be changed by GenAI.

Tasks:

- [ ] Step 30.1: Extend AI CV Analyzer repository — Add candidate retrieval methods for `BOOKMARK`, `JOB_SEARCH`, and `DIRECT_JOB_DETAIL`, filtering visible/active jobs, excluding expired/hidden jobs, and including company, requirements, skills, work type, location, experience, and update timestamps.
- [ ] Step 30.2: Build candidate scoring payloads — Map `JobListing`, `JobRequirement`, and `JobSkill` records into `jobCandidates[].scoringInput` with `titleText`, `descriptionText`, `requirementSummary`, `requiredSkills`, `requirements`, `roleFamily`, `experienceLevel`, `workType`, and optional numeric signals.
- [ ] Step 30.3: Send multipart Model API request — Update Model API client to support multipart PDF upload/reference file streaming plus `jobCandidates` JSON, service token auth, request ID propagation, timeout, and deterministic error mapping.
- [ ] Step 30.4: Validate model-core response in Backend — Add Zod schemas for `model-core-cv-analysis-v1` and reject unknown schema versions, out-of-range scores, unknown candidate IDs, duplicate recommendations, unsupported language, and backend-owned fields.
- [ ] Step 30.5: Hydrate recommendations from DB — Convert model recommendation IDs into final public `jobRecommendations[]` with `jobId`, `title`, `companyName`, `matchScore`, `reason`, and `nextStep`, preserving model order and excluding stale/non-visible jobs.
- [ ] Step 30.6: Implement grounded GenAI wrapper — Generate `jobFitAlignment.summary`, `atsFriendliness.summary`, `overallImpression`, `topActionables`, `sectionReviews`, recommendation `reason`, and `nextStep` from model-core evidence + DB job context only; enforce JSON schema, max lengths, language, and no unsupported claims.
- [ ] Step 30.7: Add deterministic wrapper fallback — If GenAI is disabled or unavailable, build localized rule-based copy from model-core evidence so Backend still returns valid `cv-analysis-v2` without changing model scores/ranks.
- [ ] Step 30.8: Persist full analysis snapshot — Store `CvAnalysisResult`, optional `JobRecommendationRun`, and `JobRecommendationItem` records with model name/version, input summary, candidate count, recommendation count, request ID/idempotency data, and safe audit metadata.
- [ ] Step 30.9: Preserve upload/reference flow — Ensure direct PDF upload, explicit `cvFileId`, and active CV fallback work with ownership checks, temporary upload cleanup, retention, and no file path traversal.
- [ ] Step 30.10: Add backend tests — Cover repository candidate filters, payload mapping, Model API client multipart request, response validation, GenAI fallback, hydration, persistence, OpenAPI response shape, and failure modes.

Acceptance Criteria:

- [ ] `/api/v1/ai/cv-analyzer` returns public `cv-analysis-v2` exactly as OpenAPI expects for upload and reference flows.
- [ ] Candidate jobs always originate from Backend DB and are filtered for visibility/status/expiry before Model API scoring.
- [ ] Final `jobRecommendations[]` are hydrated from Backend DB, preserve model score/order, and never include unknown/non-visible jobs.
- [ ] GenAI wrapper cannot alter numeric scores, candidate IDs, model order, or backend-owned persistence fields.
- [ ] Backend returns valid deterministic fallback response when GenAI or Model API prose wrapper behavior fails within defined policy.

---

### Phase 31 — End-to-End Safety, Observability, and Release Gate

Status: Planned

Goal: Prove the full Backend + Model API system is safe, observable, reproducible, and ready for staging/production traffic after Phases 28-30 are implemented.

Scope boundary:

- This phase validates the whole local/staging flow; it does not deploy or mutate production systems.
- Production secrets must never be committed; only `.env.example`/template files may be tracked.
- Logs/metrics must avoid raw CV text, tokens, DB URLs, and unrelated PII.

Tasks:

- [ ] Step 31.1: Add env templates and secret safety checks — Update root/backend/model API `.env.example` files, ensure `.env` is ignored, document required service tokens, and add tests/scripts that fail if secrets or DB URLs appear in tracked files.
- [ ] Step 31.2: Add observability — Record request ID, model version, artifact hash, candidate count, parse quality, parse latency, embedding latency, TensorFlow latency, wrapper latency, total latency, error code, and fallback reason without raw CV text.
- [ ] Step 31.3: Add health/readiness checks — Backend health must report Model API reachability; Model API readiness must verify artifacts, TensorFlow model, E5 backend, PDF parser dependency, and service-token config.
- [ ] Step 31.4: Add e2e contract tests — Run Backend API upload → Model API parse/inference → Backend wrapper/hydration/persistence → OpenAPI response validation using fixture PDFs and fixture jobs.
- [ ] Step 31.5: Add load and timeout smoke — Test candidate counts up to 50, max PDF size, slow parser, slow E5, slow TensorFlow, slow GenAI, and ensure deterministic `503/504/422` behavior.
- [ ] Step 31.6: Add security and privacy review — Verify auth boundaries, service-token rotation guidance, no Model API DB access, no raw CV logs, upload cleanup, retention, path traversal defense, and CORS/non-public Model API routing.
- [ ] Step 31.7: Update docs and runbooks — Document local run, staging run, env vars, curl examples, expected response, troubleshooting, rollback, and how to run Backend + Model API tests.
- [ ] Step 31.8: Write integration release report — Create a report summarizing implemented contracts, tests, latency, fallback coverage, env readiness, security findings, remaining risks, and final go/no-go decision.

Acceptance Criteria:

- [ ] Backend and Model API pass e2e fixture tests with real PDF parsing and real TensorFlow/E5 runtime in Python 3.13.
- [ ] No production DB credentials are required by Model API; Backend remains sole DB owner.
- [ ] `.env` files are ignored; only safe examples/templates are tracked.
- [ ] Logs and reports include operational metadata but not raw CV text, tokens, DB URLs, or unrelated PII.
- [ ] Failure modes are deterministic and documented for invalid PDF, parse failure, empty candidates, Model API timeout, TensorFlow load failure, E5 failure, and GenAI wrapper failure.
- [ ] Final public response validates against `references/docs/generated/openapi.json` and persisted data validates against `references/prisma/schema.prisma` expectations.

---

## Suggested Execution Order

1. Treat Phase 0-11 as completed design and audit baseline.
2. Start implementation from Phase 12 and complete phases sequentially as notebooks unless a blocker requires a narrower spike.
3. Do not train complex models before Phase 15 pair generation and Phase 17 baselines pass.
4. Complete Phase 19.5 before starting Phase 20 so overallImpression is built only on trusted E5-backed job-fit evidence and clean skill signals.
5. Do not claim production readiness before Phase 22 calibration/export and Phase 24 final gate pass.
6. Implement Phase 25 as the final single-notebook TensorFlow training delivery when `REQUIREMENT.md` compliance is required.
7. Implement Phase 26 after Phase 25 artifacts exist and pass handoff validation.
8. Complete Phase 27 before making any final production-ready claim for training model or Model API runtime.
9. Complete Phase 28 before changing Backend or Model API integration code.
10. Implement Phase 29 and Phase 30 together behind tests because the PDF/candidate contract spans both services.
11. Complete Phase 31 before staging/production traffic.
12. Keep wrapper/backend work out of training notebooks unless it changes the model-output contract or integration validation fixtures.

## Out of Scope for Model-Core Training Notebooks

- Backend auth, persistence, DB hydration, and public response formatting.
- Backend GenAI wrapper implementation.
- Frontend rendering.
- Production deployment or release mutation.
- Direct Model API access to production database credentials.
