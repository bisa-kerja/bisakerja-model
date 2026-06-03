# Training Workspace

Notebook-first model training workspace for Bisakerja Model API.

Training owns feature design, weak-label and human-label governance, baseline comparison, TensorFlow delivery, calibration, model-card evidence, and artifact export. Model serving lives in `../model_api/`. Backend API is a separate repository at <https://github.com/bisa-kerja/bisakerja-api>.

## Boundary

Training is responsible for producing model-core evidence and artifacts for:

- `jobFitAlignment`
- `atsFriendliness`
- `overallImpression`
- candidate reranking scores for Backend-provided job IDs

Training does not own Backend auth, persistence, final public response formatting, hydrated job details, or GenAI wrapper prose.

## Layout

```text
training/
|-- README.md
|-- requirements.txt              # Training/notebook dependencies
|-- notebooks/                    # Versioned notebook-first workflow
|-- .gitignore
`-- .tf-venv-3.13/                # Local ignored TensorFlow runtime when created
```

## Notebook-First Rule

Training implementation stays in versioned `.ipynb` notebooks. Do not add `training/*.py` package entrypoints for training execution unless the workflow is explicitly redesigned.

Every notebook step must start with English Markdown before code. Each step documents:

- Purpose
- Required input
- Action
- Expected output
- Verification

## Notebook Order

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

Phase 13 executable cells are intentionally retired for production notebook hygiene; durable evidence remains in `../reports/phase_13_*.json`.

## Runtime

Use Python `3.13.11` for the TensorFlow training runtime. Python `3.14` is not reliable for this project unless compatible TensorFlow wheels are available.

```bash
deactivate 2>/dev/null || true
PYENV_VERSION=3.13.11 pyenv exec python -m venv training/.tf-venv-3.13
source training/.tf-venv-3.13/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
python -m pip install ipykernel jupyterlab
python -m ipykernel install --user --name bisakerja-model-tf313 --display-name "Bisakerja Model TF 3.13"
```

Verify runtime:

```bash
python - <<'PY'
import sys
import tensorflow as tf
import keras
from sentence_transformers import SentenceTransformer

print('python:', sys.version.split()[0])
print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)
print('sentence_transformers: import-ok')
PY
```

Expected Python: `3.13.x`. Phase 25 smoke evidence records TensorFlow `2.21.0` and Keras `3.14.1`.

## Default Embedding Model

For production-track English-focused training, use `intfloat/e5-base-v2` as the frozen embedding model.

- CV/profile text prefix: `query:`
- Job text prefix: `passage:`
- Use normalized embeddings.
- Regenerate embedding cache and manifests when the embedding model changes.
- Use `all-mpnet-base-v2` or `all-MiniLM-L6-v2` only as comparator baselines.

## Backend Contract Inputs

When training notebooks or validation gates need Backend API contracts, use generated snapshots or fixtures exported from <https://github.com/bisa-kerja/bisakerja-api>. Backend source is not part of this repository.

## Production Evidence Gates

Before any production-ready claim, verify release evidence from a clean or well-understood git state:

```bash
python scripts/verify_phase_27_1_27_2_release_gate.py --write
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
```

Release gates record git commit, notebook hygiene, TensorBoard release evidence, model-card/manifest consistency, validation expansion, and human-label policy. TensorBoard logs in ignored local directories are not enough for `REQUIREMENT.md` section 1.4 unless copied to the release path and recorded with SHA-256 and byte size.

Weak labels are allowed only as bootstrap/training support. Production score claims require larger frozen human/recruiter-reviewed validation evidence with slice coverage.

## Artifact Outputs

Production-track notebooks write durable outputs under:

```text
../artifacts/phase_25_tensorflow_training_delivery/
../reports/phase_25_*.json
../reports/phase_27_*.json
```

Do not mutate exported artifacts at runtime. If an artifact changes, update the relevant manifest and rerun verification.
