# TODOS — Notebook-First Training Model Phases

Source reference: `GAP_MODEL_TRAINING.md`  
Backend/API contract source: <https://github.com/bisa-kerja/bisakerja-api> (use exported contract snapshots or release fixtures when needed)

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
- [x] Model-core schema is aligned with exported Backend API contract boundaries from <https://github.com/bisa-kerja/bisakerja-api>.
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
- [x] Step 23.2: Add notebook contract fixtures aligned with exported Backend API contracts from <https://github.com/bisa-kerja/bisakerja-api>.
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
- API runtime remains separate source code, but this phase must export schemas, examples, and handoff fixtures aligned with the external Backend API contract so the API can consume the model artifact without hidden notebook state.

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
- [x] Model-core output aligns with the external Backend API contract while wrapper-owned fields (`topActionables`, `sectionReviews`, generated CV copy, job title/company hydration, auth, persistence) remain outside the training artifact.
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
- Exported Backend OpenAPI snapshot from <https://github.com/bisa-kerja/bisakerja-api>

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

Status: Complete

Goal: Freeze a safe internal contract where Backend sends a sanitized multipart PDF plus backend-selected job candidates to Model API, and Model API returns model-core output that Backend maps to the public OpenAPI `cv-analysis-v2` response.

Scope boundary:

- Backend owns auth, file ownership, storage, candidate retrieval, DB hydration, persistence, idempotency, public response shape, and GenAI prose wrapper.
- Model API owns PDF parsing, CV evidence extraction, ATS issue signals, E5 feature building, TensorFlow scoring, candidate reranking, health checks, and deterministic model-core errors.
- Model API must not receive raw DB credentials, write backend data, generate final public hydrated job fields, or decide auth/ownership.
- Backend may send `backendMetadata` for trace/debug only; Model API must never trust it for candidate identity outside `jobId` membership.

Tasks:

- [x] Step 28.1: Define final internal request contract — Document `POST /internal/model/cv-analysis` as multipart with `requestId`, `language`, `inputMode`, `compareSource`, `jobRoles[]`, `cvFile`, `jobCandidates` JSON, and `rankingPolicy` JSON.
- [x] Step 28.2: Define final internal response contract — Document `model-core-cv-analysis-v1` containing `parsedCv`, `jobFitAlignment`, `atsFriendliness`, `overallImpression` evidence, `candidateReranking.recommendations[]`, `model`, and timestamps; exclude public wrapper fields.
- [x] Step 28.3: Map OpenAPI/Prisma fields to contract owners — Create a matrix from exported Backend OpenAPI/Prisma snapshots covering `CvAnalysisResult`, `JobRecommendationRun`, `JobRecommendationItem`, `JobListing`, `JobRequirement`, and `JobSkill`.
- [x] Step 28.4: Define language and enum mapping — Freeze `id/en` for public/model API, `ID/EN` for Prisma, `UPLOAD/REFERENCE`, `BOOKMARK/JOB_SEARCH/DIRECT_JOB_DETAIL`, and `strong/good/stretch` to Prisma `STRONG/GOOD/STRETCH`.
- [x] Step 28.5: Define failure contracts — Specify deterministic backend behavior for Model API `422`, `503`, `504`, parse failure, empty candidate set, stale artifact, timeout, and GenAI wrapper failure.
- [x] Step 28.6: Add contract fixtures — Create positive/negative fixtures for direct upload PDF, active CV reference, bookmarked candidates, job-search candidates, direct-job-detail candidate, duplicate job IDs, empty PDF parse, and missing candidate evidence.
- [x] Step 28.7: Update durable docs — Update `model_api/README.md` and external Backend integration docs in <https://github.com/bisa-kerja/bisakerja-api> so future implementation follows one contract.

Acceptance Criteria:

- [x] One internal contract clearly separates model-owned fields from backend/wrapper-owned fields.
- [x] Backend can map every Model API response field into final `CvAnalysis.analysisResult` without guessing or inventing jobs.
- [x] Model API rejects backend-owned public fields such as `title`, `companyName`, `reason`, `nextStep`, `topActionables`, `sectionReviews`, `generatedCv`, auth, persistence, and hydrated DB objects in model-core output.
- [x] Contract fixtures cover all public compare sources and both upload/reference CV modes.
- [x] Reviewer can verify the contract against OpenAPI and Prisma without reading implementation code.

---

### Phase 29 — Model API PDF Parsing and Model-Core Inference Hardening

Status: Complete

Goal: Upgrade Model API from JSON-only sanitized signals to production-safe PDF intake, deterministic parsing, ATS evidence extraction, TensorFlow scoring, and model-core output for backend-provided candidates.

Scope boundary:

- Model API accepts PDF bytes only from trusted Backend internal calls.
- Model API does not persist uploaded files after request completion and does not store raw CV text beyond request-scoped logs/metrics.
- Model API still supports JSON-only `/inference/cv-analysis` for tests/backward compatibility, but production flow uses multipart PDF endpoint.
- No external GenAI call is allowed in core inference.

Tasks:

- [x] Step 29.1: Add safe env/config defaults — Provide `.env.example` for Model API with `MODEL_API_ENV`, artifact paths, `MODEL_API_SERVICE_TOKEN`, `MODEL_API_MAX_PDF_BYTES`, `MODEL_API_MAX_PDF_PAGES`, `MODEL_API_TIMEOUT_MS`, `SENTENCE_TRANSFORMERS_HOME`, `MODEL_API_ENABLE_GENAI_WRAPPER=false`, and OpenRouter vars disabled by default.
- [x] Step 29.2: Add internal auth middleware — Require `Authorization: Bearer <MODEL_API_SERVICE_TOKEN>` for inference endpoints in staging/production while allowing explicit local/test bypass only through safe config.
- [x] Step 29.3: Implement multipart PDF endpoint — Add `POST /internal/model/cv-analysis` that validates content type, file size, PDF magic bytes, one-file-only policy, required form fields, candidate JSON shape, and request ID.
- [x] Step 29.4: Implement deterministic PDF parser — Extract text with a pinned parser, detect page count, empty/scanned PDFs, section names, contact/date signals, formatting risk, and parse quality; never hallucinate missing CV content.
- [x] Step 29.5: Build profile/CV signals from parsed PDF — Convert parsed text into `SanitizedProfileInput` using `cvText`, `targetRoles`, detected sections, normalized skills, language, role family, and optional experience evidence.
- [x] Step 29.6: Replace ATS placeholder logic — Compute `atsFriendliness.score` and `detectedIssues` from parser evidence with transparent penalties and safe fallback for empty/scanned/failed parse.
- [x] Step 29.7: Fix top-candidate evidence — Derive `jobFitAlignment.matchedSkills/missingSkills` and recommendation skill evidence from the top-ranked candidate, not from the first input candidate.
- [x] Step 29.8: Add candidate reranking endpoint — Add `POST /inference/candidate-reranking` for backend jobs-only scoring with the same membership, max-item, score-bound, and response-contract validators.
- [x] Step 29.9: Add hard runtime guards — Enforce parse, embedding, and model inference timeouts; reject fallback embedding backends in staging/production; warm up TensorFlow and E5 at startup when configured.
- [x] Step 29.10: Expand tests — Cover PDF parser edge cases, multipart validation, auth missing/invalid token, ATS scoring, top-candidate evidence, reranking endpoint, timeout errors, and no backend-owned fields.

Acceptance Criteria:

- [x] Model API can parse a normal PDF CV sent by Backend and build valid Phase 25 features without backend pre-parsed `cvText`.
- [x] Empty/scanned/malformed PDFs return deterministic safe errors or low-confidence fallback signals, never fabricated CV text.
- [x] All model-core scores are bounded integers `0-100`; recommendations are max `5`, unique, and restricted to backend-supplied job IDs.
- [x] Staging/production cannot run with missing service token, fallback embeddings, unverified artifacts, or skipped model loader.
- [x] Tests prove PDF endpoint, JSON endpoint, reranking endpoint, and response validators work without leaking backend-owned fields.

---

### Phase 30 — Backend Orchestrator, Candidate Retrieval, GenAI Wrapper, and Persistence

Status: Complete

Goal: Update Backend API so `/api/v1/ai/cv-analyzer` orchestrates CV file resolution, DB candidate retrieval, Model API call, GenAI wrapper copy generation, job hydration, persistence, and public `cv-analysis-v2` response formatting.

Scope boundary:

- Backend owns the public OpenAPI contract and must return exactly `CvAnalysis` envelope shape.
- Backend retrieves candidates from Prisma DB; Model API receives only those candidates and cannot invent or hydrate jobs.
- Backend wrapper may use GenAI for prose fields, but must be grounded in model-core evidence and DB data.
- If GenAI fails, Backend returns deterministic fallback copy from model-core evidence; scores/ranks must not be changed by GenAI.

Tasks:

- [x] Step 30.1: Extend AI CV Analyzer repository — Add candidate retrieval methods for `BOOKMARK`, `JOB_SEARCH`, and `DIRECT_JOB_DETAIL`, filtering visible/active jobs, excluding expired/hidden jobs, and including company, requirements, skills, work type, location, experience, and update timestamps.
- [x] Step 30.2: Build candidate scoring payloads — Map `JobListing`, `JobRequirement`, and `JobSkill` records into `jobCandidates[].scoringInput` with `titleText`, `descriptionText`, `requirementSummary`, `requiredSkills`, `requirements`, `roleFamily`, `experienceLevel`, `workType`, and optional numeric signals.
- [x] Step 30.3: Send multipart Model API request — Update Model API client to support multipart PDF upload/reference file streaming plus `jobCandidates` JSON, service token auth, request ID propagation, timeout, and deterministic error mapping.
- [x] Step 30.4: Validate model-core response in Backend — Add Zod schemas for `model-core-cv-analysis-v1` and reject unknown schema versions, out-of-range scores, unknown candidate IDs, duplicate recommendations, unsupported language, and backend-owned fields.
- [x] Step 30.5: Hydrate recommendations from DB — Convert model recommendation IDs into final public `jobRecommendations[]` with `jobId`, `title`, `companyName`, `matchScore`, `reason`, and `nextStep`, preserving model order and excluding stale/non-visible jobs.
- [x] Step 30.6: Implement grounded GenAI wrapper — Generate `jobFitAlignment.summary`, `atsFriendliness.summary`, `overallImpression`, `topActionables`, `sectionReviews`, recommendation `reason`, and `nextStep` from model-core evidence + DB job context only; enforce JSON schema, max lengths, language, and no unsupported claims.
- [x] Step 30.7: Add deterministic wrapper fallback — If GenAI is disabled or unavailable, build localized rule-based copy from model-core evidence so Backend still returns valid `cv-analysis-v2` without changing model scores/ranks.
- [x] Step 30.8: Persist full analysis snapshot — Store `CvAnalysisResult`, optional `JobRecommendationRun`, and `JobRecommendationItem` records with model name/version, input summary, candidate count, recommendation count, request ID/idempotency data, and safe audit metadata.
- [x] Step 30.9: Preserve upload/reference flow — Ensure direct PDF upload, explicit `cvFileId`, and active CV fallback work with ownership checks, temporary upload cleanup, retention, and no file path traversal.
- [x] Step 30.10: Add backend tests — Cover repository candidate filters, payload mapping, Model API client multipart request, response validation, GenAI fallback, hydration, persistence, OpenAPI response shape, and failure modes.

Acceptance Criteria:

- [x] `/api/v1/ai/cv-analyzer` returns public `cv-analysis-v2` exactly as OpenAPI expects for upload and reference flows.
- [x] Candidate jobs always originate from Backend DB and are filtered for visibility/status/expiry before Model API scoring.
- [x] Final `jobRecommendations[]` are hydrated from Backend DB, preserve model score/order, and never include unknown/non-visible jobs.
- [x] GenAI wrapper cannot alter numeric scores, candidate IDs, model order, or backend-owned persistence fields.
- [x] Backend returns valid deterministic fallback response when GenAI or Model API prose wrapper behavior fails within defined policy.

---

### Phase 31 — End-to-End Safety, Observability, and Release Gate

Status: Complete

Goal: Prove the full Backend + Model API system is safe, observable, reproducible, and ready for staging/production traffic after Phases 28-30 are implemented.

Scope boundary:

- This phase validates the whole local/staging flow; it does not deploy or mutate production systems.
- Production secrets must never be committed; only `.env.example`/template files may be tracked.
- Logs/metrics must avoid raw CV text, tokens, DB URLs, and unrelated PII.

Tasks:

- [x] Step 31.1: Add env templates and secret safety checks — Update root/backend/model API `.env.example` files, ensure `.env` is ignored, document required service tokens, and add tests/scripts that fail if secrets or DB URLs appear in tracked files.
- [x] Step 31.2: Add observability — Record request ID, model version, artifact hash, candidate count, parse quality, parse latency, embedding latency, TensorFlow latency, wrapper latency, total latency, error code, and fallback reason without raw CV text.
- [x] Step 31.3: Add health/readiness checks — Backend health must report Model API reachability; Model API readiness must verify artifacts, TensorFlow model, E5 backend, PDF parser dependency, and service-token config.
- [x] Step 31.4: Add e2e contract tests — Run Backend API upload → Model API parse/inference → Backend wrapper/hydration/persistence → OpenAPI response validation using fixture PDFs and fixture jobs.
- [x] Step 31.5: Add load and timeout smoke — Test candidate counts up to 50, max PDF size, slow parser, slow E5, slow TensorFlow, slow GenAI, and ensure deterministic `503/504/422` behavior.
- [x] Step 31.6: Add security and privacy review — Verify auth boundaries, service-token rotation guidance, no Model API DB access, no raw CV logs, upload cleanup, retention, path traversal defense, and CORS/non-public Model API routing.
- [x] Step 31.7: Update docs and runbooks — Document local run, staging run, env vars, curl examples, expected response, troubleshooting, rollback, and how to run Backend + Model API tests.
- [x] Step 31.8: Write integration release report — Create a report summarizing implemented contracts, tests, latency, fallback coverage, env readiness, security findings, remaining risks, and final go/no-go decision.

Acceptance Criteria:

- [x] Backend and Model API pass e2e fixture tests with real PDF parsing and real TensorFlow/E5 runtime in Python 3.13.
- [x] No production DB credentials are required by Model API; Backend remains sole DB owner.
- [x] `.env` files are ignored; only safe examples/templates are tracked.
- [x] Logs and reports include operational metadata but not raw CV text, tokens, DB URLs, or unrelated PII.
- [x] Failure modes are deterministic and documented for invalid PDF, parse failure, empty candidates, Model API timeout, TensorFlow load failure, E5 failure, and GenAI wrapper failure.
- [x] Final public response validates against exported Backend OpenAPI expectations and persisted data validates against exported Backend Prisma expectations.

---

### Phase 32 — AI CV Analyzer Contract Drift Audit and Canonical Schema Freeze

Status: Complete

Goal: Re-open the AI CV Analyzer integration after the current drift finding and freeze one canonical Backend ↔ Model API contract before code changes.

Scope boundary:

- Focus only on AI CV Analyzer.
- Backend API public source of truth is `references/docs/generated/openapi.json` path `POST /api/v1/ai/cv-analyzer` and schema `CvAnalysis`.
- Backend currently only consumes AI CV Analyzer and AI CV Generate from Model API; AI CV Generate remains out of this corrective scope.
- Frontend must only receive Backend public envelope `{ success, message, data, meta }`; Model API remains internal-only.

Tasks:

- [x] Step 32.1: Extract public OpenAPI contract — Generate a durable JSON/Markdown summary for `POST /api/v1/ai/cv-analyzer`, including request fields, success envelope, `CvAnalysis.analysisResult`, required fields, max/min items, nullable fields, score ranges, and error envelopes.
- [x] Step 32.2: Extract backend Model API client contract — Document `references/src/shared/integrations/model-api.schema.ts`, `model-api.client.ts`, and `ai-cv-analyzer.service.ts` expectations for multipart request, model-core response, wrapper mapping, hydration, persistence, and errors.
- [x] Step 32.3: Extract current Model API contract — Document actual `model_api/app.py`, `schemas.py`, and `validators.py` request/response behavior for `/internal/model/cv-analysis`, including envelope/raw payload, `parsedCv`, `jobFitAlignment`, `atsFriendliness`, `overallImpression`, `candidateReranking`, `model`, and timestamps.
- [x] Step 32.4: Write drift matrix — Record every mismatch: response envelope, `createdAt` vs `analyzedAt`, `parsedCv.status/detectedSections` vs `sectionNames`, `parseQuality` enum, `requirements` object vs string, `numericSignals` vs `numericFeatures`, `backendMetadata.locationDisplay`, `jobRoles[]` parsing, extra strict-object fields, and model artifact metadata exposure.
- [x] Step 32.5: Choose canonical internal response shape — Decide whether `/internal/model/cv-analysis` returns raw model-core payload or Backend unwraps `data`; update docs/fixtures so exactly one behavior is allowed.
- [x] Step 32.6: Freeze language policy — Default product-facing prose must be English for current staging; request `language` remains explicit `id|en`, but fallback/wrapper output must not silently switch to Indonesian.
- [x] Step 32.7: Mark AI CV Generate separate — Add a short note that AI CV Generate compatibility will be audited later and must not block AI CV Analyzer closure.

Acceptance Criteria:

- [x] One canonical AI CV Analyzer contract exists and cites `references/docs/generated/openapi.json` `POST /api/v1/ai/cv-analyzer`.
- [x] Drift matrix covers request payload, response payload, wrapper output, error mapping, language behavior, and security/privacy fields.
- [x] Internal Model API response ownership is clear: model-core fields only; Backend owns public `cv-analysis-v2` envelope and prose.
- [x] No implementation starts until the canonical schema and drift matrix are reviewed.

---

### Phase 33 — Model API AI CV Analyzer Request Compatibility Fix

Status: Complete

Goal: Make `/internal/model/cv-analysis` accept exactly the multipart payload produced by Backend AI CV Analyzer without weakening validation or leaking sensitive data.

Scope boundary:

- Model API still owns PDF parsing, ATS evidence, model-core scoring, and candidate reranking only.
- Backend still owns auth, CV ownership, DB candidate retrieval, job hydration, persistence, public response formatting, and GenAI wrapper prose.
- Model API must not accept frontend/public payloads, DB credentials, tokens, storage keys as trusted scoring input, or hydrated job objects.

Tasks:

- [x] Step 33.1: Fix `jobRoles[]` multipart parsing — Support repeated form fields from Backend `FormData.append("jobRoles", role)` and reject empty/oversized role lists according to Backend OpenAPI limits.
- [x] Step 33.2: Align `jobCandidates[].scoringInput.requirements` — Accept Backend requirement objects `{ type, value, priority }[]` and safely derive scoring text from `value`; reject unknown unsafe shapes with deterministic `contract_validation_error`.
- [x] Step 33.3: Align numeric signals — Accept Backend `numericSignals` or explicitly remove it from Backend fixture; do not allow arbitrary numeric keys to bypass approved Phase 25 feature order.
- [x] Step 33.4: Align backend metadata allowlist — Accept only safe hydration hints currently sent by Backend (`title`, `companyName`, `locationDisplay`, `sourceUpdatedAt`) plus documented source fields; never trust metadata for candidate membership or scoring identity.
- [x] Step 33.5: Enforce candidate policy — Keep max 50 candidates, unique `jobId`, required scoring evidence, `rankingPolicy.backendOwnsHydration=true`, `requireCandidateJobIds=true`, `deduplicateByJobId=true`, `maxRecommendations<=5`.
- [x] Step 33.6: Harden PDF parse failures — Return deterministic validation/parse error or low-confidence parser evidence for empty/scanned PDFs; never fabricate CV text, skills, sections, or experience.
- [x] Step 33.7: Preserve internal auth — Require bearer service token for staging/production; local bypass must remain disabled when `MODEL_API_ENV=staging|production`.
- [x] Step 33.8: Add request contract tests — Cover Backend-produced multipart fixtures for `UPLOAD`, `REFERENCE`, `BOOKMARK`, `JOB_SEARCH`, `DIRECT_JOB_DETAIL`, repeated `jobRoles`, requirements objects, duplicate candidates, empty candidates, empty PDF, malformed PDF, and oversized PDF.

Acceptance Criteria:

- [x] Backend-produced AI CV Analyzer multipart payload parses successfully without backend source changes beyond agreed canonical contract.
- [x] Invalid candidate membership, duplicate IDs, missing scoring evidence, unsafe metadata, and malformed PDFs fail closed with stable errors.
- [x] Model API never trusts backend metadata for job identity and never receives or uses Backend DB credentials.
- [x] Tests prove request compatibility against exported fixtures.

---

### Phase 34 — Model API AI CV Analyzer Response Compatibility Fix

Status: Complete

Goal: Return model-core response that Backend Zod schemas can validate and map into public `cv-analysis-v2` exactly as OpenAPI expects.

Scope boundary:

- Model API response must be internal model-core, not public `CvAnalysis`.
- Backend public response must match `references/docs/generated/openapi.json` `CvAnalysis` under success envelope `{ success, message, data, meta }`.
- Model API must not return backend-owned public prose fields: `topActionables`, `sectionReviews`, hydrated `jobRecommendations`, `generatedCv`, `reason`, `nextStep`, `title`, or `companyName`.

Tasks:

- [x] Step 34.1: Fix raw/envelope behavior — Match Backend client expectation exactly: either return raw model-core payload from `/internal/model/cv-analysis` or update Backend client to unwrap `data`; add a contract test that fails on mismatch.
- [x] Step 34.2: Align `parsedCv` — Return `status: parsed|empty_text|parse_failed`, `pageCount`, `textLength`, `detectedSections`, and optional `extractionEvidence` exactly as Backend schema expects.
- [x] Step 34.3: Align `jobFitAlignment` — Return `score`, `matchedSignals`, `missingSignals`, `matchedSkills`, `missingSkills`, and optional `evidence`; remove or map `summarySignals/confidenceNotes` before Backend validation.
- [x] Step 34.4: Align `atsFriendliness` — Return `score`, `detectedIssues`, `parseQuality: high|medium|low|failed`, and optional `evidence`; map parser qualities from Model API internal values safely.
- [x] Step 34.5: Align `overallImpression` — Return `score` and `evidence` array only; no final prose from Model API core.
- [x] Step 34.6: Align `candidateReranking` — Return `recommendations[]` only if Backend strict schema requires it, or update Backend schema explicitly; each item must contain only `jobId`, `matchScore`, `matchLevel`, `matchedSkills`, `missingSkills`, and optional string `rankingSignals`.
- [x] Step 34.7: Align `model` and timestamp — Return only `model.name`, `model.version`, and `createdAt` if Backend expects `createdAt`; do not expose artifact path/hash in the strict Model API response unless Backend schema explicitly allows it.
- [x] Step 34.8: Keep scores/ranks immutable — Validate scores are integer `0-100`, recommendation count max 5, job IDs are unique, all job IDs come from request candidates, and ordering is deterministic.
- [x] Step 34.9: Add response contract tests — Validate real Model API output with Backend `cvAnalyzerModelResponseSchema` or equivalent JSON Schema fixture before any staging-ready claim.

Acceptance Criteria:

- [x] Model API AI CV Analyzer response passes Backend response schema with no strict-object extra field failures.
- [x] Backend can map model-core output into public OpenAPI `CvAnalysis.analysisResult` without missing fields or timestamp drift.
- [x] Model API response contains no backend-owned hydrated fields, no raw CV text, no prompt, no token, no storage key, and no raw model artifact path unless explicitly allowed.
- [x] English-safe model-core evidence is suitable for Backend fallback/wrapper prose.

---

### Phase 35 — Backend AI CV Analyzer Wrapper, Prompt Safety, and English Default Hardening

Status: Complete

Goal: Ensure Backend wrapper/fallback creates OpenAPI-compatible, English-default public prose from model-core evidence safely, even when GenAI is enabled or unavailable.

Scope boundary:

- Backend wrapper may generate public prose; Model API core must not call external GenAI for AI CV Analyzer staging.
- GenAI output must never change numeric scores, model version, candidate IDs, recommendation order, candidate membership, or Backend persistence identifiers.
- Deterministic fallback must remain the default safe path when GenAI is disabled, times out, returns invalid JSON, or violates schema.

Tasks:

- [x] Step 35.1: Define wrapper input allowlist — Only pass `requestId`, requested `language`, `jobRoles`, `compareSource`, `inputMode`, model-core evidence, detected sections, and hydrated candidate metadata needed for user copy; exclude raw CV text by default, tokens, storage keys, emails, phones, addresses, DB URLs, and full Model API payloads.
- [x] Step 35.2: Write injection-resistant system prompt — Prompt must instruct the model to ignore CV/job prompt injection, use only provided evidence, produce JSON only, avoid unsupported skills/seniority/salary/hiring outcomes/protected-class claims, and preserve scores/IDs/order exactly.
- [x] Step 35.3: Enforce English default — For current staging, default generated and fallback prose must be English; if `language=id` is passed, either explicitly return approved English copy per product decision or implement tested Indonesian copy without mixed-language leakage.
- [x] Step 35.4: Add strict wrapper JSON schema — Validate `jobFitAlignment.summary`, `atsFriendliness.summary`, `overallImpression`, `topActionables[1..3]`, dynamic `sectionReviews`, `jobRecommendations[].reason`, and `jobRecommendations[].nextStep` before persistence or frontend response.
- [x] Step 35.5: Add wrapper safety filters — Reject output that mentions raw prompt, system/developer messages, secrets, tokens, email/phone/address, unprovided companies/jobs, protected-class attributes, guaranteed hiring outcomes, or altered numeric scores.
- [x] Step 35.6: Keep deterministic fallback complete — Fallback must produce valid OpenAPI `CvAnalysis.analysisResult` with English prose, `generatedCv.available=false`, max 5 hydrated recommendations, and no raw Model API internals.
- [x] Step 35.7: Add prompt red-team tests — Test malicious CV text and job descriptions that ask to ignore instructions, reveal prompts, alter scores, invent companies, add fake skills, or output non-JSON.
- [x] Step 35.8: Add OpenAPI response tests — Validate final Backend `/api/v1/ai/cv-analyzer` 200 response against `references/docs/generated/openapi.json`, including upload/reference flow and all compare sources.

Acceptance Criteria:

- [x] Backend final response matches public OpenAPI `CvAnalysis` success envelope exactly.
- [x] Wrapper prompt and fallback cannot alter model scores, IDs, order, model metadata, or persistence data.
- [x] Generated/fallback prose is English by default for current staging and never exposes raw CV text, prompt, tokens, storage keys, or unrelated PII.
- [x] Invalid GenAI output is rejected and replaced with deterministic fallback, not returned to frontend.

---

### Phase 36 — AI CV Analyzer Staging Readiness Gate

Status: Complete

Goal: Prove AI CV Analyzer is safe to enable in staging after request, response, wrapper, language, and privacy gaps are closed.

Scope boundary:

- This gate covers AI CV Analyzer only.
- AI CV Generate compatibility and prompt safety must be a later separate phase.
- This gate does not deploy or mutate production.

Tasks:

- [x] Step 36.1: Run Model API tests — Run unit/contract tests for schemas, multipart parsing, PDF parser, candidate membership, response validation, auth, timeout, artifact readiness, and no backend-owned fields.
- [x] Step 36.2: Run Backend tests — Run Backend model-api schema/client tests, AI CV Analyzer service tests, route tests, OpenAPI schema validation, persistence tests, hydration tests, and wrapper prompt/fallback tests.
- [x] Step 36.3: Run cross-repo e2e fixture — Execute Backend upload/reference request with fixture PDFs and fixture jobs through Model API to final `CvAnalysis` response; assert frontend only sees Backend public envelope.
- [x] Step 36.4: Verify failure mapping — Cover invalid file type, file too large, no active CV, job not found, bookmark not owned, empty candidates, Model API 422, Model API invalid response, timeout, model not ready, GenAI invalid JSON, and GenAI timeout.
- [x] Step 36.5: Verify security/privacy — Confirm service-token auth, non-public Model API routing, CORS only on Backend, no raw CV/prompt/token/storage key in logs/responses/persistence, upload cleanup, retention, and no Model API DB access.
- [x] Step 36.6: Verify language behavior — Confirm current staging default prose is English and snapshots record requested language consistently without frontend-facing mixed-language drift.
- [x] Step 36.7: Write readiness report — Produce JSON/Markdown report with contract versions, test commands, pass/fail evidence, latency notes, fallback coverage, secret-scan result, remaining risks, and staging go/no-go decision.

Acceptance Criteria:

- [x] AI CV Analyzer e2e response validates against `references/docs/generated/openapi.json` `POST /api/v1/ai/cv-analyzer`.
- [x] Backend accepts Model API output without Zod/schema errors and rejects invalid Model API output with `502 DOWNSTREAM_ERROR`.
- [x] Model API accepts Backend multipart payload without contract drift and rejects unsafe input with deterministic errors.
- [x] Prompt/GenAI path is schema-bound, injection-resistant, English-default, and safely replaceable by deterministic fallback.
- [x] Final decision is explicit: `go` only if all blockers are closed; otherwise `no-go` with exact remaining fixes.

---

### Phase 37 — Backend AI CV Analyzer GenAI Wrapper Provider Integration

Status: Complete

Goal: Wire an optional Backend-owned GenAI provider for AI CV Analyzer public prose while keeping Model API deterministic and preserving deterministic fallback as the safe default.

Scope boundary:

- Model API must keep `MODEL_API_ENABLE_GENAI_WRAPPER=false` for AI CV Analyzer core inference and must not call external GenAI.
- Backend owns the optional GenAI call, prompt assembly, JSON parsing, OpenAPI validation, fallback, and observability.
- GenAI must never receive raw CV bytes/text by default, service tokens, storage keys, DB URLs, auth headers, or full internal payloads.
- GenAI output must never change model scores, model metadata, candidate IDs, recommendation order, candidate membership, or persistence identifiers.

Current gap:

- Backend has `buildCvAnalyzerWrapperInput`, `cvAnalyzerWrapperSystemPrompt`, wrapper output schema validation, and deterministic fallback.
- Backend does not yet have a provider client, env config, timeout policy, or service integration that passes `wrapperResponse` into `buildPublicCvAnalysisResponse`.
- Current public prose is deterministic fallback only; no external LLM/OpenRouter call is wired.

Tasks:

- [x] Step 37.1: Add Backend GenAI config — Add explicit Backend env vars such as `AI_CV_ANALYZER_GENAI_ENABLED`, provider base URL, model name, API key env, timeout, and max retries; default disabled in local/test/staging unless intentionally enabled.
- [x] Step 37.2: Implement provider client — Create a Backend-owned OpenAI-compatible/OpenRouter client with request timeout, abort handling, JSON-only response parsing, sanitized logging, and no retry for expensive unsafe inference by default.
- [x] Step 37.3: Wire analyzer service — In `AiCvAnalyzerService.analyzeCv`, build wrapper input from model-core response, call provider only when enabled, pass provider output into `buildPublicCvAnalysisResponse`, and fallback deterministically on timeout, invalid JSON, schema drift, or safety rejection.
- [x] Step 37.4: Preserve safety invariants — Enforce that generated output cannot alter scores, IDs, order, model name/version, `createdAt/analyzedAt`, recommendation count, or candidate membership.
- [x] Step 37.5: Add red-team tests — Cover prompt injection in CV evidence, job descriptions, skills, company names, malformed JSON, markdown output, raw secret leakage, PII-like output, score mutation, candidate mutation, and provider timeout.
- [x] Step 37.6: Add OpenAPI contract tests — Validate both generated and fallback responses against `references/docs/generated/openapi.json` `CvAnalysis` and public response envelope.
- [x] Step 37.7: Update operations docs — Document when to enable GenAI, required secrets, latency/cost impact, fallback behavior, and rollback to deterministic fallback.

Acceptance Criteria:

- [x] Backend can optionally generate higher-quality AI CV Analyzer prose through a provider while deterministic fallback remains the default safe path.
- [x] Provider failures never fail successful model-core inference unless product policy explicitly chooses fail-closed.
- [x] Generated output validates against OpenAPI and cannot mutate model-owned scores, candidate IDs/order, model metadata, or timestamps.
- [x] No raw CV text, uploaded file bytes, tokens, storage keys, DB URLs, prompts, or auth headers are sent to provider or logged.

---

### Phase 38 — AI CV Analyzer Staging Runtime Warmup and Performance Gate

Status: In Progress

Goal: Make staging/demo behavior reliable after the live Model API smoke passes by adding warmup, readiness, and latency acceptance gates for TensorFlow and E5 first-load behavior.

Scope boundary:

- This phase does not retrain the model.
- This phase does not change public Backend API contracts.
- This phase focuses on operational readiness for AI CV Analyzer live staging traffic.

Current gap:

- Live Model API smoke now proves Python 3.13, TensorFlow/Keras, E5, artifact loading, custom objects, and endpoint behavior.
- First live `/inference/cv-analysis` call can take around 20 seconds because E5 model weights load lazily.
- Backend readiness currently checks Model API `/health`; strict staging should also verify Model API `/ready` and a warmed inference path before demo traffic.

Tasks:

- [x] Step 38.1: Add Model API warmup command/runbook — Document and script a safe warmup that loads TensorFlow and E5 using a sanitized fixture before routing demo/staging traffic.
- [x] Step 38.2: Add readiness strictness — Update Backend readiness or staging gate to verify Model API `/ready.ready=true`, not only `/health` reachability.
- [x] Step 38.3: Add latency budget evidence — Record cold-start and warm inference latency for `/internal/model/cv-analysis`, including parse, embedding, TensorFlow, and total latency.
- [x] Step 38.4: Add timeout alignment check — Ensure Backend `MODEL_API_TIMEOUT_MS` is greater than cold/warm expected latency or warmup is mandatory before traffic.
- [x] Step 38.5: Add HF/E5 cache guidance — Document `SENTENCE_TRANSFORMERS_HOME`, optional `HF_TOKEN` for rate limits, and cache persistence for container/staging deployments.
- [ ] Step 38.6: Add staging smoke script — Run a Backend-to-Model fixture through public `/api/v1/ai/cv-analyzer` after warmup and assert latency, response shape, persistence behavior, and no private field leakage.
- [x] Step 38.7: Add rollback checklist — Document fallback to deterministic prose, traffic disable switch, Model API service-token rotation, and artifact path rollback.

Acceptance Criteria:

- [x] Staging runbook includes a repeatable warmup path and expected cold/warm latency numbers.
- [x] Backend readiness or staging gate fails when Model API `/ready` is false or E5/TensorFlow are not warmed according to policy.
- [x] Demo/staging latency budget is explicit and verified with live fixture evidence.
- [x] E5 model cache and optional HF token behavior are documented for macOS/Linux local and container staging.

---

### Phase 39 — Backend AI CV Analyzer Product-Copy Quality and Localization Review

Status: Complete

Goal: Decide whether deterministic fallback copy is sufficient for staging/demo or whether GenAI/approved localized templates are required before broader user testing.

Scope boundary:

- This phase covers final user-visible copy quality only, not model scoring correctness.
- Model API remains model-core only.
- Backend owns language policy, copy templates, GenAI wrapper output, and fallback text.

Current gap:

- Current deterministic fallback is safe and OpenAPI-compatible but generic.
- Current staging policy defaults to English even when request language is `id`, based on previous safety decision.
- Indonesian copy or richer personalized copy requires explicit product decision and tests.

Tasks:

- [x] Step 39.1: Define product language policy — Decide whether `language=id` should return Indonesian copy, English copy, or bilingual-safe copy for staging/demo.
- [x] Step 39.2: Review fallback copy quality — Create representative CV/job fixtures and evaluate whether deterministic summaries/actionables are useful enough without GenAI.
- [x] Step 39.3: Add approved templates — If GenAI remains disabled, add richer deterministic templates for job fit, ATS, overall impression, actions, section reviews, and recommendation reasons.
- [x] Step 39.4: Add localization tests — Verify `id`/`en` behavior, no mixed-language drift, no PII leakage, and OpenAPI-compatible text lengths.
- [x] Step 39.5: Add product acceptance report — Produce a concise demo readiness report with before/after sample responses and remaining copy limitations.

Acceptance Criteria:

- [x] Product-facing language behavior is explicit and tested.
- [x] Fallback copy is either accepted for staging/demo or replaced by approved templates/GenAI provider output.
- [x] Public responses stay schema-valid, safe, and free of raw CV text, prompt text, tokens, storage keys, and unrelated PII.

---

### Phase 40 — AI CV Analyzer Backend Regression Closure and Candidate Policy Hardening

Status: Complete

Goal: Close the remaining Backend AI CV Analyzer regressions found after the Phase 39 copy update, and ensure Backend handles candidate edge cases before calling Model API.

Scope boundary:

- Focus only on AI CV Analyzer Backend code and tests under `references/`.
- Do not change Model API model-core response contract unless a new drift is proven.
- Do not rollback approved Phase 39 English-safe fallback copy.
- Backend remains owner of public `cv-analysis-v2`, candidate retrieval policy, and error mapping.

Current gap:

- `references/tests/integration/routes/ai-cv-analyzer.test.ts` still expects pre-Phase-39 fallback copy, so Backend route test suite fails even though actual response uses approved richer templates.
- Backend can still call Model API with an empty `jobCandidates` set, although integration docs say Backend should resolve empty candidates before Model API call.
- Multipart upload limits in `createCvUploadMiddleware` may reject valid requests with up to 10 repeated `jobRoles[]` fields plus analyzer metadata.

Tasks:

- [x] Step 40.1: Refresh route fallback assertions — Update AI CV Analyzer route tests to expect approved Phase 39 English-safe copy templates for `jobFitAlignment`, `atsFriendliness`, `overallImpression`, `topActionables`, `sectionReviews`, and recommendation `reason/nextStep`.
- [x] Step 40.2: Keep generated-wrapper tests aligned — Ensure GenAI success/failure tests still prove provider output is accepted only when schema-safe, while provider failure falls back to approved Phase 39 deterministic copy.
- [x] Step 40.3: Add empty-candidate policy — In Backend service, handle empty candidate retrieval before Model API call with deterministic product-approved behavior and no fake production scores.
- [x] Step 40.4: Map candidate-not-found errors — For `DIRECT_JOB_DETAIL`, return `404 JOB_NOT_FOUND`; for `BOOKMARK`, hide ownership with `404 BOOKMARK_NOT_FOUND` or approved no-candidate response; for `JOB_SEARCH`, use approved no-recommendation/validation policy.
- [x] Step 40.5: Expand candidate edge tests — Cover `DIRECT_JOB_DETAIL` missing job, unowned bookmark/no bookmark candidates, empty job search result, and assert Model API client is not called when Backend resolves no candidates.
- [x] Step 40.6: Fix multipart field limits — Increase or redesign Multer `fields`/`parts` limits so a valid request with 10 `jobRoles` plus metadata fields is accepted without weakening file count and size limits.
- [x] Step 40.7: Add max-role route test — Submit 10 valid repeated `jobRoles` fields through `/api/v1/ai/cv-analyzer` and assert success or the agreed validation behavior.
- [x] Step 40.8: Run Backend regression suite — Run `cd references && bun test --preload ./tests/preload-env.ts tests/unit/ai-cv-analyzer tests/unit/shared/model-api.schema.test.ts tests/unit/shared/model-api.client.test.ts tests/integration/routes/ai-cv-analyzer.test.ts tests/integration/contracts/fixture-contracts.test.ts` and record result.

Acceptance Criteria:

- [x] Backend AI CV Analyzer route tests pass with Phase 39 approved fallback copy, not stale pre-Phase-39 strings.
- [x] Backend never calls Model API with empty `jobCandidates` unless an explicitly documented no-candidate model-core contract exists.
- [x] Public errors for missing direct job/bookmark ownership remain ownership-safe and match documented error codes.
- [x] Valid multipart requests with the documented max `jobRoles` do not fail due to infrastructure limits.
- [x] No raw CV text, storage keys, service tokens, prompt text, DB URLs, or raw Model API internals leak in responses, logs, or persisted snapshots.

Verification:

- `cd references && bun test --preload ./tests/preload-env.ts tests/unit/ai-cv-analyzer tests/unit/shared/model-api.schema.test.ts tests/unit/shared/model-api.client.test.ts tests/integration/routes/ai-cv-analyzer.test.ts tests/integration/contracts/fixture-contracts.test.ts` — 51 pass, 0 fail.

---

### Phase 41 — AI CV Analyzer Public Staging Smoke and Payload Hygiene Closure

Status: Completed

Goal: Complete the missing public staging smoke evidence and remove internal payload ambiguity before broader AI CV Analyzer staging/demo traffic.

Scope boundary:

- Focus on AI CV Analyzer staging readiness only.
- Do not deploy or mutate production.
- Do not audit AI CV Generate in this phase.
- Keep Model API internal-only and Backend public-only.

Current gap:

- Phase 38 still has incomplete Step 38.6 for a real Backend-to-Model public staging smoke.
- The staging runbook curl example is incomplete for the current public analyzer schema because it omits required `jobRoles` and `inputMode` fields.
- Backend Model API client sends `cv` metadata with `storageKey` even though the canonical internal multipart contract only requires the PDF file and safe analyzer fields.

Tasks:

- [x] Step 41.1: Add public staging smoke script — Create a repeatable script that warms Model API, calls Backend `/api/v1/ai/cv-analyzer` with a sanitized PDF fixture, valid auth, `jobRoles`, `language`, `inputMode`, `compareSource`, and optional `persistResult`, then writes a report.
- [x] Step 41.2: Seed or fixture candidate jobs — Ensure the smoke has deterministic visible/active Backend job candidates for `JOB_SEARCH` and optionally `DIRECT_JOB_DETAIL`/`BOOKMARK` without production data dependency.
- [x] Step 41.3: Validate public response shape — Assert the public envelope `{ success, message, data, meta }`, `cv-analysis-v2`, max 5 hydrated recommendations, `generatedCv.available=false`, model metadata, and OpenAPI-compatible field limits.
- [x] Step 41.4: Validate persistence behavior — When `persistResult=true`, assert sanitized `CvAnalysisResult`, recommendation run/items, and CV file metadata are written without raw CV text, file bytes, storage keys, prompts, tokens, or full internal payloads.
- [x] Step 41.5: Validate latency budget — Capture Backend total latency plus Model API observability where available, and compare against cold/warm staging budgets from the Phase 38 runbook.
- [x] Step 41.6: Update staging runbook curl — Fix `docs/runbooks/ai-cv-analyzer-staging-runtime.md` example to include required `jobRoles` and `inputMode`, plus expected auth and persistence notes.
- [x] Step 41.7: Remove or justify `cv.storageKey` in Model API multipart — Either stop sending the unused `cv` metadata field/storage key from Backend client, or document why it remains internal-only and add tests proving it is never exposed/logged/trusted.
- [x] Step 41.8: Add payload hygiene tests — Assert Backend-to-Model multipart excludes unneeded sensitive metadata when possible, and public responses never include `storageKey`, uploaded bytes, raw CV text, auth headers, service tokens, prompts, DB URLs, or artifact paths.
- [x] Step 41.9: Close Phase 38.6 evidence — Update `TODOS.md`, `reports/phase_38_ai_cv_analyzer_runtime_gate.*`, or a new Phase 41 report with smoke command, pass/fail result, latency, and no-private-field evidence.

Acceptance Criteria:

- [x] A reviewer can run one documented public staging smoke command after warmup and get a valid Backend `CvAnalysis` response.
- [x] Smoke report records latency, persistence choice, contract validation result, model version, and private-field leak checks.
- [x] Staging runbook curl example is executable against the current public schema.
- [x] Backend-to-Model multipart payload contains only necessary internal fields, and any retained internal metadata is justified and non-public.
- [x] Phase 38.6 is either completed or explicitly superseded by Phase 41 evidence.

---

### Phase 42 — Backend-Owned AI CV Generate Implementation

Status: Complete

Goal: Implement AI CV Generate as a Backend API feature using the existing public `/api/v1/ai/cv-generate` contract, without adding CV generation responsibility to Model API.

Scope boundary:

- Focus on Backend API code under `references/`.
- Keep Model API model-core only; do not add `/cv-generate` or GenAI generation to `model_api/` for this phase.
- Keep Frontend out of scope except for preserving the existing OpenAPI response contract.
- Backend owns auth, CV ownership, storage read, prompt orchestration, GenAI provider call, output safety, and public response envelope.
- Model API remains owner of CV analysis/scoring/reranking only.

Current gap:

- Backend already exposes `POST /api/v1/ai/cv-generate`, validates request body, checks `cvFileId` ownership, and sanitizes returned markdown.
- Backend currently delegates generation to `modelApiClient.generateCvMarkdown`, but Model API in this repository has no `/cv-generate` route and intentionally does not own GenAI calls.
- Backend sends `storageKey` in the CV generate payload, but Model API should not read backend storage keys or own file storage access.
- `templateHtml` is required by schema/OpenAPI, while module docs still contain contradictory optional wording.
- Output generation needs CV evidence/text, summary, and template handling owned by Backend, similar to the AI CV Analyzer GenAI wrapper pattern.

Tasks:

- [x] Step 42.1: Freeze backend-owned architecture — Document that AI CV Generate is implemented in Backend API, not Model API, and that Frontend must never call Model API directly.
- [x] Step 42.2: Replace Model API dependency — Remove or bypass `modelApiClient.generateCvMarkdown` from `AiCvGenerateService` and introduce a Backend-owned generation path.
- [x] Step 42.3: Add CV storage access — Inject `CvFileStorage` into AI CV Generate service and reuse the AI CV Analyzer storage-read pattern to load ownership-checked CV bytes without exposing `storageKey` publicly.
- [x] Step 42.4: Build safe CV evidence input — Extract deterministic CV text/signals from the stored PDF or reuse the latest sanitized CV analysis snapshot when available; never persist raw prompt or raw CV text unless explicitly approved.
- [x] Step 42.5: Add GenAI client wrapper — Create an `AiCvGenerateGenAiClient` similar to `ai-cv-analyzer.genai.ts`, with timeout, retry policy, request id, provider error mapping, and no raw provider details in public errors.
- [x] Step 42.6: Add generation prompt contract — Define a constrained prompt/input contract using sanitized CV evidence, user `summary`, `templateHtml`, language/default policy, and explicit instruction to return only safe markdown HTML.
- [x] Step 42.7: Add deterministic fallback policy — Decide whether provider failure returns `503 SERVICE_UNAVAILABLE` only, or a minimal deterministic template-rendered markdown fallback; do not fabricate unsupported CV claims.
- [x] Step 42.8: Harden input/output safety — Sanitize `templateHtml`, validate markdown is non-empty and within max length, reject executable HTML patterns, and ensure frontend still sanitizes before render.
- [x] Step 42.9: Fix docs and OpenAPI metadata — Update AI CV Generate docs so `templateHtml` is consistently required, add missing `AI CV Generate` top-level OpenAPI tag if still absent, and document backend-owned generation boundary.
- [x] Step 42.10: Add Backend unit tests — Cover ownership checks, cross-user 404, missing/expired CV, storage read failure, provider timeout/error, invalid provider response, unsafe output rejection, and no Model API call.
- [x] Step 42.11: Add Backend route/contract tests — Cover successful `POST /api/v1/ai/cv-generate`, validation errors, auth errors, response envelope, `markdown` only response data, no `storageKey`/raw CV/prompt/token leakage, and OpenAPI schema alignment.
- [x] Step 42.12: Run targeted Backend regression suite — Run AI CV Generate, shared GenAI/model integration, response safety, and affected route tests from `references/`; record command and result.

Acceptance Criteria:

- [x] `POST /api/v1/ai/cv-generate` succeeds without requiring any Model API `/cv-generate` endpoint.
- [x] Backend performs CV ownership check and reads CV evidence internally before generation.
- [x] Public response remains exactly `{ success, message, data: { markdown }, meta }` for success.
- [x] Public responses and logs do not expose raw CV text, storage keys, prompts, provider raw payloads, auth headers, service tokens, DB URLs, or Model API internals.
- [x] Unsafe or empty generated markdown is rejected with the documented error behavior.
- [x] Docs clearly state Backend owns AI CV Generate orchestration, while Model API remains model-core only.
- [x] Targeted Backend tests pass and verification command is recorded.

Verification:

- [x] `cd references && bun run typecheck && bun test --preload ./tests/preload-env.ts tests/unit/ai-cv-generate tests/integration/routes/ai-cv-generate.test.ts tests/unit/shared/model-api.client.test.ts tests/unit/shared/model-api.schema.test.ts` — passed 2026-06-04.

---

## Embedding Runtime Migration Phases — multilingual E5 Small Performance Track

These phases define a safe migration from `intfloat/e5-base-v2` to `intfloat/multilingual-e5-small` for staging performance. This is a model-version change, not a drop-in runtime tweak. The migration must preserve contract safety, artifact reproducibility, calibration semantics, backend integration behavior, and rollback ability.

### Phase 43 — multilingual-e5-small Migration Decision, Baseline Capture, and Risk Register

Status: Complete

Goal: Freeze the current Phase 25 E5-base behavior and decide whether `intfloat/multilingual-e5-small` is acceptable for a staging performance experiment before any model or serving code changes.

Scope boundary:

- Do not overwrite Phase 25 artifacts.
- Do not change production/staging Model API env to a new embedding model yet.
- Do not claim quality equivalence between E5-base and multilingual-E5-small without measured evidence.
- This phase produces decision evidence only.

Tasks:

- [x] Step 43.1: Capture current runtime baseline — Record VPS/container startup time, `/live`, `/health`, `/ready`, first `/internal/model/cv-analysis` latency, warm inference latency, CPU, RAM, disk, Docker image size, and E5 cache size for `intfloat/e5-base-v2`.
- [x] Step 43.2: Capture current quality baseline — Export current Phase 25 validation metrics, calibration tables, score distribution, candidate reranking examples, and representative public AI CV Analyzer responses.
- [x] Step 43.3: Define migration hypothesis — State expected benefits of `intfloat/multilingual-e5-small`: lower startup latency, lower RAM, faster embedding, smaller cache, better Indonesian/English coverage, and lower timeout risk.
- [x] Step 43.4: Define migration risks — Document cosine distribution drift, score calibration drift, ranking changes, possible loss of English-only semantic quality, multilingual tokenization behavior, and user-facing recommendation changes.
- [x] Step 43.5: Define go/no-go thresholds — Freeze staging acceptance thresholds for runtime, quality, calibration, contract, and rollback before training starts.
- [x] Step 43.6: Define artifact namespace — Reserve a new artifact root such as `artifacts/phase_43_multilingual_e5_small_migration/` or a later selected training delivery root; do not reuse `phase_25_tensorflow_training_delivery`.
- [x] Step 43.7: Define model version naming — Reserve names such as `jobfit_tf_phase43_multilingual_e5_small_v1` and `model-core-cv-analysis-v1` compatibility notes.
- [x] Step 43.8: Document stakeholder decision — Record that this track is approved for staging experiment only until all validation gates pass.

Acceptance Criteria:

- [x] Current E5-base runtime and quality baselines are recorded with commands, timestamps, model version, artifact hash, and environment.
- [x] `multilingual-e5-small` migration risks and go/no-go thresholds are explicit before implementation.
- [x] New artifact/model version namespace is defined so Phase 25 remains rollback-safe.
- [x] Reviewer can compare old vs new behavior without guessing which artifacts were used.

Verification:

- [x] `python scripts/verify_phase_43_multilingual_e5_small_migration.py --write` — passed 2026-06-04.
- [x] `python -m unittest tests.test_phase_43_multilingual_e5_small_migration` — 4 tests passed 2026-06-04.

---

### Phase 44 — Embedding Compatibility Audit and Feature Drift Study

Status: Complete

Goal: Measure how `intfloat/multilingual-e5-small` changes text embeddings and `e5_cosine` features before retraining TensorFlow.

Scope boundary:

- This phase evaluates embeddings and feature distributions only.
- Do not train the selected TensorFlow scorer yet.
- Do not modify Model API production defaults.

Tasks:

- [x] Step 44.1: Verify embedding model metadata — Record model name, library version, embedding dimension, normalized embedding behavior, prefix policy (`query:` and `passage:`), license/reference URL, cache path, and runtime hardware.
- [x] Step 44.2: Confirm dimension and finite values — Generate embeddings for representative profile/CV/job texts and verify expected dimension, finite values, deterministic normalization, and no fallback backend.
- [x] Step 44.3: Regenerate paired embedding cache — Build a small-to-full cache for the same frozen pair set used by Phase 25 using `intfloat/multilingual-e5-small`, stored under a new cache path.
- [x] Step 44.4: Compare cosine distributions — Compare E5-base vs multilingual-E5-small `e5_cosine` mean, std, min/max, percentiles, and histogram by split, language, role family, pair type, and score band.
- [x] Step 44.5: Analyze rank correlation — Compute Spearman/Pearson correlation between old and new cosine values and identify worst drift examples.
- [x] Step 44.6: Analyze multilingual slices — Compare Indonesian, English, mixed, and unknown-language examples; flag improvements/regressions for ID CVs and English job descriptions.
- [x] Step 44.7: Check feature normalization impact — Recompute the approved six-feature vectors and identify whether Phase 25 mean/std normalization remains invalid for new cosine distribution.
- [x] Step 44.8: Decide retrain vs recalibrate-only — Block direct artifact swap if cosine drift exceeds threshold or feature normalization changes materially.

Acceptance Criteria:

- [x] Embedding dimension, normalization, prefix behavior, and runtime dependency behavior are verified.
- [x] Old-vs-new `e5_cosine` drift is quantified across important slices.
- [x] Direct replacement without retraining is explicitly rejected unless drift evidence proves it safe.
- [x] Recommendation for retraining/recalibration is recorded with data, not assumptions.

Verification:

- [x] `training/.tf-venv-3.13/bin/python scripts/verify_phase_44_embedding_compatibility_audit.py --write --allow-download --regenerate` — passed 2026-06-04.
- [x] `.venv/bin/python -m unittest tests.test_phase_44_embedding_compatibility_audit` — 4 tests passed 2026-06-04.

---

### Phase 45 — TensorFlow Retraining with multilingual-e5-small Features

Status: Complete

Goal: Train a new TensorFlow scorer using multilingual-E5-small-derived features, preserving the Phase 25 model-core contract while producing a new artifact version.

Scope boundary:

- Training remains notebook-first.
- Do not mutate Phase 25 notebooks/artifacts except for docs linking to this migration track.
- The selected model must still consume the approved feature contract or explicitly version any changed feature contract.

Tasks:

- [x] Step 45.1: Create migration notebook — Add a new notebook such as `training/notebooks/phase_45_multilingual_e5_small_training_delivery.ipynb` with required English Markdown sections: Purpose, Required input, Action, Expected output, Verification.
- [x] Step 45.2: Freeze inputs — Use the same frozen dataset, labels, splits, pair IDs, human validation labels, ATS benchmark, and backend candidate fixtures as Phase 25 unless a documented update is approved.
- [x] Step 45.3: Generate approved features — Rebuild `e5_cosine`, `skill_overlap`, `requirement_coverage`, `role_match`, `experience_match`, and `experience_gap_years_clipped` using multilingual-E5-small embeddings.
- [x] Step 45.4: Recompute normalization stats — Generate new train-split mean/std for every approved feature and write a new `tensorflow_feature_config.json` with embedding model metadata.
- [x] Step 45.5: Train TensorFlow model — Use TensorFlow Functional API or subclassing plus existing custom components and `tf.GradientTape`; do not use `model.fit()` for the main training path.
- [x] Step 45.6: Compare baselines — Compare against constant, skill-only, E5-base Phase 25 scorer, cosine-only multilingual-E5-small, and previous selected scorer.
- [x] Step 45.7: Evaluate quality metrics — Record MAE, RMSE, R², Spearman, score-band agreement, high-fit recall, NDCG/MAP for candidate reranking, and slice metrics by role/language/experience/pair type.
- [x] Step 45.8: Evaluate Indonesian behavior — Include dedicated ID and mixed-language CV/job examples to ensure multilingual migration improves or preserves practical staging behavior.
- [x] Step 45.9: Export TensorBoard evidence — Write bounded TensorBoard logs under the new artifact namespace and record hash/byte-size metadata.
- [x] Step 45.10: Select or reject model — Select the new model only if it meets predefined thresholds; otherwise keep E5-base as selected and record rejection evidence.

Acceptance Criteria:

- [x] New training notebook runs from a clean Python 3.13 kernel without hidden state.
- [x] New feature config records `intfloat/multilingual-e5-small` and new normalization stats.
- [x] TensorFlow model is trained and evaluated with the same or stricter gates as Phase 25.
- [x] New model is selected only when quality and slice metrics pass staging thresholds.

Verification:

- [x] `training/.tf-venv-3.13/bin/python scripts/verify_phase_45_multilingual_e5_small_training_delivery.py --write` — passed 2026-06-04.
- [x] `training/.tf-venv-3.13/bin/python - <<'PY' ... nbclient.NotebookClient(...).execute()` — clean-kernel notebook execution passed 2026-06-04.
- [x] `.venv/bin/python scripts/verify_phase_45_multilingual_e5_small_training_delivery.py` — passed 2026-06-04.
- [x] `.venv/bin/python -m unittest tests.test_phase_45_multilingual_e5_small_training_delivery` — 4 tests passed 2026-06-04.

---

### Phase 46 — Calibration, Model Card, Artifact Manifest, and Handoff Fixtures Refresh

Status: Complete

Goal: Produce a complete artifact package for the multilingual-E5-small model so Model API can load it safely without stale Phase 25 metadata.

Scope boundary:

- New artifacts must be self-contained and hash-verified.
- Do not point Model API to mixed Phase 25/Phase 45 files.
- Do not reuse old calibration unless validated against new score distribution.

Tasks:

- [x] Step 46.1: Recalibrate scores — Rebuild calibration tables for `jobFitAlignment.score`, `atsFriendliness.score`, and `recommendations[].matchScore` buckets `0-20`, `21-40`, `41-60`, `61-80`, and `81-100`.
- [x] Step 46.2: Validate calibration quality — Record ECE, MCE, bucket MAE, within-10-points rate, score-band agreement, and slice calibration.
- [x] Step 46.3: Export new TensorFlow artifact — Save selected model as `.keras`, reload it in a clean cell with registered custom objects, and run inference smoke without notebook state.
- [x] Step 46.4: Export refreshed configs — Write new `tensorflow_feature_config.json`, `feature_config.json`, `score_calibration.json`, `label_manifest.json`, and dataset manifest.
- [x] Step 46.5: Export model card — Document embedding model change, intended use, blocked use, metrics, slice performance, runtime benefits, known risks, and rollback artifact.
- [x] Step 46.6: Export artifact manifest — Include SHA-256, byte size, schema version, runtime/training-only classification, embedding model metadata, and TensorBoard references.
- [x] Step 46.7: Refresh handoff fixtures — Regenerate CV analysis and candidate reranking fixtures using the new model and prove response shape stays `model-core-cv-analysis-v1` compatible.
- [x] Step 46.8: Refresh validation report — Validate score bounds, candidate membership, language handling, duplicate rejection, max recommendations, and no backend-owned fields.
- [x] Step 46.9: Write migration report — Produce a concise Markdown/JSON report comparing E5-base vs multilingual-E5-small runtime and quality.

Acceptance Criteria:

- [x] No stale Phase 25 hash/config/calibration is used by the new artifact package.
- [x] Model card explicitly says this is a multilingual-E5-small model version.
- [x] Artifact manifest verifies every runtime-required file.
- [x] Handoff fixtures remain Backend-compatible and model-core-only.

Verification:

- [x] `training/.tf-venv-3.13/bin/python scripts/verify_phase_46_calibration_model_card_manifest_handoff_refresh.py --write` — passed 2026-06-04.
- [x] `training/.tf-venv-3.13/bin/python - <<'PY' ... nbclient.NotebookClient(...).execute()` — clean-kernel notebook execution passed 2026-06-04.
- [x] `.venv/bin/python scripts/verify_phase_46_calibration_model_card_manifest_handoff_refresh.py` — passed 2026-06-04.
- [x] `.venv/bin/python -m unittest tests.test_phase_46_calibration_model_card_manifest_handoff_refresh` — 5 tests passed 2026-06-04.

---

### Phase 47 — Model API Runtime Support for Versioned Embedding Artifacts

Status: Complete

Goal: Update Model API so it can safely load the multilingual-E5-small artifact package through config/env without hardcoded E5-base assumptions or mixed artifacts.

Scope boundary:

- Model API must still reject fallback embeddings in staging/production.
- Frontend must not call Model API directly.
- Public Backend API contract must not change unless a separate Backend phase approves it.

Tasks:

- [x] Step 47.1: Make embedding model configurable from artifact config — Read the approved embedding model name from `tensorflow_feature_config.json` or `feature_config.json` instead of a hardcoded runtime constant.
- [x] Step 47.2: Validate artifact/runtime embedding match — Fail startup when env/model card says multilingual-E5-small but feature config/model card/manifest disagree.
- [x] Step 47.3: Preserve prefix policy — Keep `query:` for CV/profile text and `passage:` for job text; record prefix policy in model info.
- [x] Step 47.4: Update E5 backend validation — Allow only the embedding model declared by the loaded artifact package; reject local-hash/TF-IDF/fallback and reject undeclared model swaps.
- [x] Step 47.5: Support artifact root switching — Add env/runbook support for selecting the new artifact root, model path, feature config, calibration, model card, and manifest together.
- [x] Step 47.6: Update `/model-info` — Return embedding model name, artifact phase, model version, artifact hash, and readiness so backend/staging can confirm the correct model is deployed.
- [x] Step 47.7: Add runtime tests — Cover E5-base artifact load, multilingual-E5-small artifact load, mismatch rejection, fallback rejection, missing config, and response contract stability.
- [x] Step 47.8: Add performance smoke — Measure startup, `/ready`, first inference, warm inference, memory, and cache behavior for the new artifact on VPS.

Acceptance Criteria:

- [x] Model API cannot silently run a different embedding model than the artifact declares.
- [x] Both old and new artifact packages can be selected explicitly for rollback/testing.
- [x] `/ready` and `/model-info` expose enough metadata to verify deployment correctness.
- [x] Runtime tests prove no backend-owned fields leak and no fallback embeddings are used.

Verification:

- [x] `python -m unittest tests.model_api.test_phase_26_layout tests.test_phase_29_model_api_hardening tests.test_phase_33_cv_analyzer_request_compatibility tests.test_phase_34_cv_analyzer_response_compatibility tests.test_phase_38_ai_cv_analyzer_runtime_gate tests.test_phase_47_model_api_runtime_support` — 65 tests passed, 2 skipped 2026-06-04.
- [x] `python scripts/verify_phase_47_model_api_runtime_support.py` — report complete 2026-06-04.

---

### Phase 48 — Backend/Staging Integration, Shadow Comparison, and Rollback Plan

Status: Repo-side Complete / Staging Execution Pending

Goal: Validate the multilingual-E5-small model behind the existing Backend AI CV Analyzer flow before broader staging/demo traffic.

Scope boundary:

- Backend public response shape remains unchanged.
- Database persistence and public recommendation hydration remain Backend-owned.
- New model rollout starts in staging only.
- Live staging deployment, token smoke, shadow comparison, and rollback drill require operator credentials and approval.

Tasks:

- [x] Step 48.1: Deploy Model API staging revision — Freeze staging deploy env, Docker Compose command, versioned artifact root, expected embedding model, and persistent cache guidance.
- [x] Step 48.2: Verify readiness and metadata — Add `/live`, `/health`, `/ready`, and `/model-info` gates for multilingual-E5-small metadata and artifact hashes.
- [x] Step 48.3: Warm runtime — Add warmup command that loads TensorFlow, multilingual-E5-small, PDF parser, and runtime caches before Backend traffic.
- [x] Step 48.4: Run direct Model API smoke — Add sanitized multipart smoke for `/internal/model/cv-analysis` with latency and response-shape evidence capture.
- [x] Step 48.5: Run Backend public smoke — Add Backend public smoke for `POST /api/v1/ai/cv-analyzer` with `cv-analysis-v2`, persistence, hydration, and private-leakage checks.
- [x] Step 48.6: Shadow compare old vs new — Add old-vs-new fixture comparison for Phase 25 E5-base vs multilingual-E5-small score deltas, rank swaps, skills, ATS stability, and rollout review flags.
- [x] Step 48.7: Define allowed deltas — Define score delta threshold, top-rank swap review, match-level review, ATS stability review, language regression blocker, and high-fit recall blocker.
- [x] Step 48.8: Validate failure behavior — Document failure behavior for model not ready, timeout, invalid PDF, empty candidates, invalid token, and rollback artifact path.
- [x] Step 48.9: Freeze rollback commands — Document env/artifact changes needed to switch back to Phase 25 E5-base within one deploy.

Acceptance Criteria:

- [x] Repo-side Backend public AI CV Analyzer staging smoke command and checks are ready for operator execution.
- [x] Shadow comparison explains output differences and blocks rollout on unsafe deltas.
- [x] Rollback to E5-base is documented with frozen env and Docker commands.
- [x] Static staging report includes latency gates, resource/cache notes, response contracts, model metadata, and known live-evidence limitations.
- [ ] Live staging report proves Backend public AI CV Analyzer works end-to-end with the new Model API artifact.
- [ ] Live rollback drill to E5-base is executed and recorded.

Verification:

- [x] `python -m unittest tests.test_phase_48_backend_staging_integration` — 4 tests passed 2026-06-04.
- [x] `python scripts/verify_phase_48_backend_staging_integration.py` — static gate passed 2026-06-04 with status `ready_for_staging_execution`.

---

### Phase 49 — Staging Promotion Decision and Production Guardrails

Status: Complete

Goal: Decide whether multilingual-E5-small should become the staging default, remain an experiment, or be rejected, and define production guardrails before any paid/real-user rollout.

Scope boundary:

- This phase does not force production rollout.
- Production-ready claims still require Phase 27-quality evidence and any updated human/reviewer validation gates.
- If staging-only, docs and model card must say staging-only.

Tasks:

- [x] Step 49.1: Compile final migration evidence — Gather Phase 43-48 reports, artifact hashes, runtime metrics, quality metrics, shadow comparison, and smoke results.
- [x] Step 49.2: Make readiness decision — Mark the new model as `staging-default`, `staging-experiment-only`, or `rejected` with reasons.
- [x] Step 49.3: Update deployment docs — Document exact env vars/artifact paths for new default and rollback, including VPS Docker, Nginx, Cloud Run if used, and HF staging caveats if relevant.
- [x] Step 49.4: Update monitoring checklist — Monitor timeout rate, `MODEL_NOT_READY`, inference latency, memory, CPU, score distribution drift, recommendation count, and backend downstream errors.
- [x] Step 49.5: Add production blockers — List remaining blockers before production: human/reviewer validation scale, slice coverage, calibration confidence, privacy review, cost/resource monitoring, and rollback drill.
- [x] Step 49.6: Update Suggested Execution Order — Ensure future agents do not skip migration validation by changing runtime constants directly.
- [x] Step 49.7: Archive rejected artifacts if needed — If rejected, keep reports but prevent accidental deployment by documenting status and not using those artifact paths in default env examples.

Acceptance Criteria:

- [x] Final decision is evidence-based and recorded in TODOs/reports/model card.
- [x] Staging default, experiment-only, or rejected status is unambiguous.
- [x] Production rollout remains blocked unless quality, calibration, human validation, contract, runtime, and rollback gates pass.
- [x] Future maintainers can reproduce or rollback the migration without reading chat history.

Verification:

- [x] `python -m unittest tests.test_phase_49_staging_promotion_guardrails` — 4 tests passed 2026-06-04.
- [x] `python scripts/verify_phase_49_staging_promotion_guardrails.py --write` — report complete 2026-06-04 with decision `staging-experiment-only`.

---

### Phase 50 — AI CV Analyzer Intelligence Optimization Priorities

Status: Complete

Goal: Make `/api/v1/ai/cv-analyzer` feel smarter while keeping output contract stable, evidence-grounded, and safe.

Scope boundary:

- Backend public response shape remains `cv-analysis-v2`.
- Model API must not return backend-owned fields or raw CV text.
- Fast wins may use deterministic parser/rule improvements before full retraining.
- GenAI prose must preserve model-core scores, recommendation IDs, order, and safety filters.

Tasks:

- [x] Priority 1: Richer CV input extraction — Improve PDF/text parser to extract normalized sections, skills, role titles, companies, project evidence, education, certifications, languages, contact/timeline signals, seniority hints, quantified impact, and estimated years of experience.
- [x] Priority 2: Feature-based ATS scoring — Replace coarse placeholder ATS logic with transparent features: required sections, heading clarity, contact/timeline presence, parse quality, quantified-impact evidence, length/readability, image/table/multi-column risk, and keyword visibility.
- [x] Priority 3: Stronger job-fit scoring — Improve ranking features and calibration with required-vs-nice-to-have skill weights, role-family mapping, seniority/experience matching, project/domain matching, multilingual skill aliases, and score calibration so `jobFitAlignment.score` is trustworthy.
- [x] Priority 4: Evidence-based actionables — Generate `topActionables`, `sectionReviews`, recommendation `reason`, and `nextStep` from missing skills, ATS issues, low experience match, missing metrics, weak role summary, and candidate-specific job evidence instead of generic advice.
- [x] Priority 5: Dataset and labels — Build/collect labeled CV-job pairs with human match score, missing skills, ATS quality, seniority/experience fit, top improvements, language, and acceptance/rejection outcome when available.
- [x] Priority 6: Automated evaluation gates — Add fixed benchmark fixtures and metrics for score MAE/calibration, ranking NDCG/MRR, missing-skill precision/recall, ATS agreement, schema safety, no raw-CV leakage, and latency/resource limits.
- [x] Priority 7: Safer prose wrapper optimization — Improve wrapper prompt/templates so public copy is specific, language-consistent, evidence-only, non-hallucinated, privacy-safe, and backed by deterministic fallback when GenAI fails.

Acceptance Criteria:

- [x] Public analyzer output is more specific without contract drift.
- [x] Scores and recommendations can be explained from deterministic evidence and model features.
- [x] ATS/actionable feedback identifies concrete CV issues instead of generic improvement text.
- [x] Benchmark and safety gates block regressions before staging rollout.

Verification:

- [x] `python -m unittest tests.test_phase_50_ai_cv_analyzer_intelligence tests.test_phase_29_model_api_hardening tests.test_phase_34_cv_analyzer_response_compatibility` — 16 tests passed 2026-06-04.
- [x] `python scripts/verify_phase_50_ai_cv_analyzer_intelligence.py --write --run-tests` — report complete 2026-06-04.

---

### Phase 51 — AI CV Analyzer Benchmark-Driven Output Quality

Status: Complete

Goal: Improve `overallImpression`, `jobFitAlignment`, and `atsFriendliness` using real CV benchmark files so `/api/v1/ai/cv-analyzer` produces specific, role-aware, English-only, evidence-grounded analysis instead of generic template copy.

Benchmark scope:

- Use benchmark CVs from `cv_examples/` with `inputMode=UPLOAD`.
- Required benchmark files:
  - `cv_examples/CV Salman Abdurrahman ATS.pdf`
  - `cv_examples/Agil's CV New 2026.pdf`
  - `cv_examples/CV_DZIKRIALBANTANI (4).pdf`
- Optional additional benchmark file:
  - `cv_examples/CV TASYA ANGGRAENI FIRDAUS (kyknya fix).pdf`
- Run each benchmark CV against target roles:
  - `Software Engineer`
  - `Product Manager`
  - `Data Analyst`
- These three roles are benchmark probes only, not hardcoded coverage limits. Phase 51 implementation must generalize to every supported job role by deriving evidence from target-role inputs, candidate job requirements, normalized role families, and CV evidence.

Scope boundary:

- Keep public response contract stable.
- Model API must not return raw CV text, backend-owned fields, or unsupported claims.
- Parser/rule improvements are allowed before retraining.
- GenAI or wrapper prose must preserve model-core scores, evidence, recommendation IDs, and safety filters.
- User-facing analyzer output must be English only, even when CV text or extracted signals are Indonesian.
- Do not special-case only `Software Engineer`, `Product Manager`, or `Data Analyst`; use them to prove behavior across technical, product, and analytical role families, then keep the analyzer safe for all existing and future roles.

Tasks:

- [x] Step 51.1: Lock benchmark matrix — Add fixed benchmark fixtures for every required CV and role combination in `UPLOAD` mode, including expected parse evidence, score ranges, language policy, and regression snapshots.
- [x] Step 51.2: Enforce English-only output policy — Normalize or translate Indonesian signal labels into canonical English for `overallImpression`, `jobFitAlignment`, `missingSkills`, `matchedSkills`, `atsFriendliness.detectedIssues`, recommendation evidence, and wrapper-ready copy. Proper nouns such as names, schools, and company names may remain unchanged.
- [x] Step 51.3: Make job-fit evidence role-specific and generalizable — Ensure the same CV produces different matched/missing evidence for `Software Engineer`, `Product Manager`, and `Data Analyst` benchmark probes, while using generic role-family and requirement-driven logic that remains safe for every supported role. Do not emit finance or forecasting gaps unless the selected target role or candidate job requires them.
- [x] Step 51.4: Repair real-PDF extraction weaknesses — Improve compact text, glyph-encoded text, email/phone reconstruction, no-space section headings, role title detection, company detection, date/timeline detection, and parser confidence reporting for the benchmark PDFs.
- [x] Step 51.5: Improve ATS issue copy — Replace raw or overly generic issue lists with grouped English issues that explain what was detected, what is weak, and how to fix it. Keep issue text grounded in parser evidence.
- [x] Step 51.6: Implement grounded Overall Impression templates — Generate specific summaries that mention available role-relevant strengths, ATS risks, parser confidence, and concrete next improvements without hallucinated skills, seniority, companies, or hiring outcomes.
- [x] Step 51.7: Add benchmark quality gates — Add automated checks for English-only output, role-specific evidence, non-generic copy, score spread across benchmark target roles, no raw CV leakage, ATS issue precision, schema compatibility, bounded latency, and no hardcoded role-only behavior.

Acceptance Criteria:

- [x] All benchmark analyzer outputs are English-only, excluding proper nouns and literal credential/company names.
- [x] `overallImpression` mentions at least two CV-specific signals when parse evidence is usable, or clearly states low parser confidence when extraction is weak.
- [x] `jobFitAlignment` matched and missing evidence changes with selected target role and does not collapse to the same generic gaps across `Software Engineer`, `Product Manager`, and `Data Analyst`; the implementation also remains requirement-driven and safe for all supported roles beyond the benchmark set.
- [x] `atsFriendliness` identifies concrete CV issues such as missing measurable impact, unclear role titles, weak company evidence, missing contact/timeline signals, or parser/formatting risk without false missing-contact claims when contact evidence exists.
- [x] Benchmark scores do not collapse into identical low-score patterns for all CV/role combinations unless evidence justifies it and the low-confidence reason is explicit.
- [x] Public response shape remains `cv-analysis-v2` compatible and does not expose raw CV text or backend-owned fields.
- [x] Phase 51 completion proves the analyzer is not hardcoded to the benchmark roles and is safe for current and future job roles through role-family normalization, candidate requirement evidence, fallback behavior, and regression gates.

Verification:

- [x] Add and run benchmark regression tests for Phase 51 CV/role matrix.
- [x] Add and run a static verification script that writes a Phase 51 report with output examples, quality-gate results, and remaining blockers.

Phase 51 verification:

- [x] `python -m unittest tests.test_phase_51_ai_cv_analyzer_benchmark_output_quality` — 9 tests passed 2026-06-04.
- [x] `python -m unittest tests.test_phase_51_ai_cv_analyzer_benchmark_output_quality tests.test_phase_34_cv_analyzer_response_compatibility` — 12 tests passed 2026-06-04.
- [x] `python scripts/verify_phase_51_ai_cv_analyzer_benchmark_output_quality.py --write --run-tests` — report complete for steps 51.1-51.7 with all Phase 51 quality gates passing.

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
12. Because contract drift was found after Phase 31, complete corrective Phases 32-36 before any new AI CV Analyzer staging-ready claim.
13. Keep AI CV Generate out of Phases 32-36 except as a documented future scope; focus current corrective work on AI CV Analyzer only.
14. Keep wrapper/backend work out of training notebooks unless it changes the model-output contract or integration validation fixtures.
15. Complete Phase 37 only if product needs LLM-generated AI CV Analyzer prose; otherwise keep deterministic fallback as the safe default.
16. Complete Phase 38 before broader staging/demo traffic so E5/TensorFlow cold-start latency and readiness behavior are explicit.
17. Complete Phase 39 before user-facing copy review, localization, or broader beta testing.
18. Complete Phase 40 before any new AI CV Analyzer staging-ready claim, because Backend route regressions and empty-candidate policy are current blockers.
19. Complete Phase 41 before broader staging/demo traffic, because public Backend-to-Model smoke evidence and payload hygiene must be reproducible.
20. Implement Phase 42 for AI CV Generate after Phase 41, keeping generation backend-owned and leaving Model API unchanged unless a separate architecture decision approves Model API GenAI ownership.
21. Start Phase 43 before changing any embedding runtime constant; capture E5-base runtime/quality baseline first.
22. Complete Phase 44 before retraining so embedding drift is measured and direct replacement is not assumed safe.
23. Complete Phases 45-46 to retrain, recalibrate, export, and document a complete multilingual-E5-small artifact package.
24. Complete Phase 47 before deploying the new artifact so Model API verifies artifact-declared embedding model instead of relying on hardcoded assumptions.
25. Complete Phase 48 in staging with shadow comparison and rollback before making multilingual-E5-small the default.
26. Complete Phase 49 before broader staging/demo or production claims, and keep production blocked until human/reviewer validation, calibration, contract, runtime, monitoring, and rollback gates are satisfied.
27. Do not change runtime constants or default env examples directly for embedding migration. Promote only through `MODEL_API_ARTIFACT_ROOT` and `MODEL_API_EXPECTED_EMBEDDING_MODEL`, then verify `/ready` and `/model-info` metadata.
28. Implement Phase 51 before any new user-facing AI CV Analyzer copy demo, because benchmark outputs must be English-only, role-specific, non-generic, and grounded in real CV parser evidence.

## Out of Scope for Model-Core Training Notebooks

- Backend auth, persistence, DB hydration, and public response formatting.
- Backend GenAI wrapper implementation.
- Frontend rendering.
- Production deployment or release mutation.
- Direct Model API access to production database credentials.
