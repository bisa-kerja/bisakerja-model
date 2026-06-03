# Bisakerja Notebook-First Model Training

This directory is reset to a notebook-first training workflow. The old script/package extraction scaffold has been removed from this directory.

## Notebook phases

Run and review notebooks in numeric order:

1. `notebooks/phase_00_reproducibility_snapshot.ipynb`
2. `notebooks/phase_01_data_audit_contracts.ipynb`
3. `notebooks/phase_02_label_schema_baselines.ipynb`
4. `notebooks/phase_03_normalization_feature_design.ipynb`
5. `notebooks/phase_04_pair_generation_splits.ipynb`
6. `notebooks/phase_05_baseline_evaluation.ipynb`
7. `notebooks/phase_06_jobfit_training_experiments.ipynb`
8. `notebooks/phase_07_ats_friendliness_scoring.ipynb`
9. `notebooks/phase_08_overall_impression_signals.ipynb`
10. `notebooks/phase_09_candidate_reranking.ipynb`
11. `notebooks/phase_10_calibration_model_card.ipynb`
12. `notebooks/phase_11_final_gate_review.ipynb`
13. `notebooks/phase_12_repository_hygiene_runtime_bootstrap.ipynb`
14. `notebooks/phase_13_data_snapshot_contract_freezing.ipynb`
15. `notebooks/phase_14_normalization_feature_builder.ipynb`
16. `notebooks/phase_15_balanced_pair_generation_splits.ipynb`
17. `notebooks/phase_16_human_validation_label_governance.ipynb`
18. `notebooks/phase_17_baseline_evaluation_v2.ipynb`
19. `notebooks/phase_18_jobfit_training_v2.ipynb`
20. `notebooks/phase_19_ats_friendliness_benchmark_scorer.ipynb`
21. `notebooks/phase_19_5_jobfit_blocker_remediation.ipynb`
22. `notebooks/phase_20_overall_impression_signals.ipynb`
23. `notebooks/phase_21_backend_candidate_reranking.ipynb`
24. `notebooks/phase_22_calibration_model_card_export.ipynb`
25. `notebooks/phase_23_model_api_contract_validation.ipynb`
26. `notebooks/phase_24_reproducibility_final_gate.ipynb`
27. `notebooks/phase_25_tensorflow_training_delivery.ipynb`

Phase 13 executable cells are intentionally retired for production notebook hygiene; durable evidence remains in `reports/phase_13_*.json`. Phase 25 is the current TensorFlow training delivery notebook for release evidence.

## Default embedding model

For Phase 17+ English-focused training, use `intfloat/e5-base-v2` as the default frozen embedding model.

- Profile/CV text prefix: `query:`
- Job text prefix: `passage:`
- Use normalized embeddings.
- Regenerate embedding cache and manifests; do not reuse legacy `all-MiniLM-L6-v2` cache.
- Use `all-mpnet-base-v2` / `all-MiniLM-L6-v2` only as comparator baselines.

## TensorFlow notebook runtime

Use a TensorFlow-compatible Python runtime for the final training notebook. Local verification currently uses Python `3.13` with dependencies from `training/requirements.txt`.

Do not install Phase 25 dependencies into an active Python `3.14` notebook virtual environment. If `pip -V` points to `training/notebooks/venv/lib/python3.14`, deactivate that environment first and create a dedicated TensorFlow runtime:

```bash
deactivate 2>/dev/null || true
PYENV_VERSION=3.13.11 pyenv exec python -m venv training/.tf-venv-3.13
source training/.tf-venv-3.13/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
python - <<'PY'
import sys
import tensorflow as tf
print(sys.version)
print(tf.__version__)
PY
```

Python `3.14` is not a reliable TensorFlow runtime for this project unless compatible TensorFlow wheels are available.

## Production release evidence gates

Before any production-ready claim, verify release evidence from a clean git state:

```bash
python scripts/verify_phase_27_1_27_2_release_gate.py --write
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
```

The release gates record the current git commit, exit non-zero when production status is blocked, verify TensorBoard event evidence under `artifacts/phase_25_tensorflow_training_delivery/tensorboard_release/`, verify active notebook hygiene, and enforce the production human-label policy. TensorBoard logs in ignored local run directories are not sufficient for `REQUIREMENT.md` section 1.4 unless copied to the release path and recorded in `artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json` with SHA-256 and byte size. Weak labels remain allowed only as bootstrap/training support; production score claims require larger frozen human/recruiter-reviewed slice-covered validation evidence. ATS readiness requires a release-scale real or sanitized CV benchmark covering normal PDF, scanned/OCR fallback, multi-column PDF, table-heavy PDF, DOCX, short CV, long CV, Indonesian CV, and English CV. Recommendation readiness requires a larger backend-like candidate-set fixture or anonymized relevance labels, candidate membership preservation, NDCG/MAP uplift over backend order, and no reliance on the small 3-set/15-candidate handoff fixture.

## Notebook-only rule

Training implementation stays in versioned `.ipynb` notebooks. Do not add `training/*.py` package entrypoints for training execution.

## Documentation rule

Every notebook step starts with English Markdown documentation before any future code cell is added. Each step explains purpose, inputs, action, expected output, and verification.
