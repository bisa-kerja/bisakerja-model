# Running Steps — Bisakerja Model Workspace

This runbook explains how to run the Bisakerja model workspace from terminal setup to TensorFlow notebooks, Model API runtime, integration checks, release gates, troubleshooting, and rollback.

Backend API is a separate repository: <https://github.com/bisa-kerja/bisakerja-api>. Keep Backend source, env files, DB config, and generated Backend artifacts outside this model repository unless a specific release fixture is intentionally copied into `artifacts/`.

## Release Gate Runbook

### Local run

1. Copy the Model API env template to an untracked local file:

   ```bash
   cp model_api/.env.example model_api/.env
   ```

2. Optionally copy the root model-workspace env template:

   ```bash
   cp .env.example .env
   ```

   Root `.env.example` is optional. Model API runtime primarily uses `model_api/.env.example`. Root `.env.example` only stores model-workspace orchestration defaults and the external Backend repo URL.

3. Set `MODEL_API_SERVICE_TOKEN` to a local-only value when testing authenticated internal routes.
4. Keep `MODEL_API_ENABLE_GENAI_WRAPPER=false` unless testing Backend wrapper fallback behavior.
5. Start Model API with Python `3.13.x` TensorFlow/E5 runtime.
6. When Backend integration is in scope, open <https://github.com/bisa-kerja/bisakerja-api> outside this repo and start it with `MODEL_API_BASE_URL=http://localhost:8000` plus a matching service token.
7. Run Model API tests:

   ```bash
   python -m unittest tests.test_phase_31_release_gate
   python -m unittest tests.test_phase_29_model_api_hardening
   ```

8. Run Backend health, route, and contract tests from the external Backend repository when integration testing is required.

### Staging run

1. Use staging-only secrets from deployment secret store; do not commit `.env` files.
2. Verify Model API `/health` checks:
   - artifacts verified
   - TensorFlow model loaded
   - E5 backend configured
   - PDF parser available
   - service-token configured
3. Verify Backend API readiness from <https://github.com/bisa-kerja/bisakerja-api> when integration testing is required.
4. Run upload contract flow through Backend API `/api/v1/ai/cv-analyzer` with fixture PDFs and fixture jobs.
5. Validate public `CvAnalysis` response shape, persistence records, and `JobRecommendationRun` / `JobRecommendationItem` records in Backend repository checks.
6. Confirm Frontend UI never calls Model API directly.

### Failure modes

- invalid PDF -> deterministic `422` validation response.
- parse failure -> deterministic low-confidence parse fallback, no fabricated raw CV.
- empty candidates -> Backend deterministic no-recommendation policy before Model API call.
- Model API timeout -> deterministic `504` or AI-unavailable mapping.
- TensorFlow load failure -> deterministic `503` readiness failure.
- E5 failure -> deterministic `503` readiness or inference failure.
- GenAI wrapper failure -> deterministic Backend fallback copy without changing model scores or ordering.

### Troubleshooting summary

- If Model API readiness fails, inspect artifact paths, manifest hashes, E5 backend setup, TensorFlow load logs, PDF parser dependency, and `MODEL_API_SERVICE_TOKEN` presence.
- If Backend integration fails, inspect the external Backend repo config, `MODEL_API_BASE_URL`, service-token mismatch, Redis, PostgreSQL, and Model API reachability.
- Verify security/privacy review items: service-token rotation guidance, raw CV log exclusion, upload cleanup, retention, path traversal defense, and non-public Model API routing.
- Logs must include request ID, model version, artifact hash, candidate count, parse quality, latency fields, error code, and fallback reason only. Do not log raw CV text, service-token values, DB URLs, or unrelated PII.

### Rollback

1. Disable public analyzer traffic from Backend API or route to deterministic fallback.
2. Keep Backend as DB owner; do not give Model API production DB credentials.
3. Restore previous model artifact paths.
4. Rerun Model API readiness and contract/smoke tests.
5. Rerun Backend integration tests from <https://github.com/bisa-kerja/bisakerja-api> when relevant.
6. Rotate service-token values if any boundary exposure is suspected.

## Target Runtime

Current target runtime:

- Python: `3.13.11`
- Training virtual environment: `training/.tf-venv-3.13`
- Model API virtual environment: `.venv`
- Jupyter kernel: `Bisakerja Model TF 3.13`
- Main Phase 25 notebook: `training/notebooks/phase_25_tensorflow_training_delivery.ipynb`
- TensorFlow: `2.21.0`
- Keras: `3.14.1`
- NumPy: `2.1.3` for Python `3.13.11`

Use Python `3.13.11` for both notebooks and Model API serving. Do not use Python `3.14` for Phase 25 or live Model API smoke. TensorFlow `2.21.0` is not available/reliable in that runtime in this project environment. Do not pin NumPy `1.26.x` on Python `3.13`; `ml-dtypes` requires NumPy `2.1+` there.

## 1. Enter Repository Root

Open a new terminal and enter the project folder:

```bash
cd /path/to/bisakerja-model
```

Check current directory:

```bash
pwd
```

Expected path ends with:

```text
bisakerja-model
```

Check important files:

```bash
ls TODOS.md REQUIREMENT.md GAP_MODEL_TRAINING.md training/requirements.txt
```

If any file returns `No such file`, you are not in repository root.

## 2. Deactivate Old Virtual Environment

If terminal prompt shows `notebooks Py`, `.venv`, `venv`, or another active environment, deactivate it first:

```bash
deactivate 2>/dev/null || true
```

Check visible global Python:

```bash
which python
python -V
```

It is okay if this still shows Python `3.14`; the next step creates a dedicated Python `3.13` virtual environment.

## 3. Create TensorFlow Python 3.13 Virtual Environment

Run:

```bash
PYENV_VERSION=3.13.11 pyenv exec python -m venv training/.tf-venv-3.13
```

Activate it:

```bash
source training/.tf-venv-3.13/bin/activate
```

Check again:

```bash
which python
python -V
python -m pip -V
```

Expected:

```text
.../bisakerja-model/training/.tf-venv-3.13/bin/python
Python 3.13.11
.../bisakerja-model/training/.tf-venv-3.13/lib/python3.13/...
```

If it still points to `training/notebooks/venv/lib/python3.14`, repeat from step 2.

## 4. Install Training Dependencies

With `training/.tf-venv-3.13` active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
python -m pip install ipykernel jupyterlab
```

Why:

- `training/requirements.txt` contains training dependencies including TensorFlow.
- `ipykernel` makes the virtual environment available as a notebook kernel.
- `jupyterlab` runs notebook UI from the same virtual environment.

## 5. Validate TensorFlow and Main Dependencies

Run:

```bash
python - <<'PY'
import sys
import pandas as pd
import tensorflow as tf
import keras
from sentence_transformers import SentenceTransformer

print('python:', sys.executable)
print('python_version:', sys.version.split()[0])
print('pandas:', pd.__version__)
print('tensorflow:', tf.__version__)
print('keras:', keras.__version__)
print('sentence_transformers: import-ok')
PY
```

Expected important lines:

```text
python: .../training/.tf-venv-3.13/bin/python
python_version: 3.13.11
tensorflow: 2.21.0
keras: 3.14.1
```

If TensorFlow fails with `No matching distribution`, pip is almost certainly using Python `3.14`. Repeat steps 2 and 3.

## 6. Register Correct Jupyter Kernel

Run:

```bash
python -m ipykernel install --user --name bisakerja-model-tf313 --display-name "Bisakerja Model TF 3.13"
```

Check kernel list:

```bash
jupyter kernelspec list
```

Expected kernel name:

```text
bisakerja-model-tf313
```

Remove old confusing project kernel if it exists:

```bash
jupyter kernelspec uninstall -f bisakerja-model-venv 2>/dev/null || true
```

Do not select default `python3` kernel for Phase 25 because it often points to global Python `3.14`.

## 7. Stop Old Jupyter Servers

Check active servers:

```bash
jupyter server list
jupyter notebook list
```

If output only shows:

```text
Currently running servers:
```

then no server is active.

If old servers exist, inspect the port from output, for example `http://localhost:8888/...`, then stop it:

```bash
jupyter server stop 8888
```

Repeat for other ports. Check again:

```bash
jupyter server list
```

## 8. Start Jupyter Lab from Correct Virtual Environment

Make sure virtual environment is still active:

```bash
which python
python -V
```

Start Jupyter Lab from repository root:

```bash
python -m jupyter lab --notebook-dir .
```

Browser should open automatically. If not, copy the URL from terminal.

Keep this terminal running while using notebooks.

## 9. Select Correct Kernel in Notebook

In Jupyter Lab:

1. Open notebook.
2. Click **Kernel**.
3. Choose **Change Kernel**.
4. Select **Bisakerja Model TF 3.13**.
5. Click **Restart Kernel**.

Check from notebook cell:

```python
import sys
import tensorflow as tf
print(sys.executable)
print(tf.__version__)
```

Expected:

```text
.../training/.tf-venv-3.13/bin/python
2.21.0
```

If path differs, kernel is wrong. Change again to **Bisakerja Model TF 3.13**.

## 10. Run Phase 25 Notebook

Main notebook for TensorFlow delivery requirement:

```text
training/notebooks/phase_25_tensorflow_training_delivery.ipynb
```

Safe run flow:

1. Open Phase 25 notebook.
2. Select kernel **Bisakerja Model TF 3.13**.
3. Click **Kernel -> Restart Kernel**.
4. Click **Run -> Run All Cells**.
5. Wait until complete.
6. Save notebook.

After completion, check Phase 25 reports:

```bash
ls reports/phase_25_*.json
```

Check compact report status:

```bash
python - <<'PY'
import json
from pathlib import Path

for p in sorted(Path('reports').glob('phase_25_*.json')):
    d = json.loads(p.read_text())
    print(p.name, 'passed=', d.get('passed'), 'status=', d.get('status'))
PY
```

Check TensorBoard logs:

```bash
ls artifacts/tensorboard/phase_25_tensorflow_training_delivery
```

Open TensorBoard from the environment that has TensorBoard installed:

```bash
tensorboard --logdir artifacts/tensorboard/phase_25_tensorflow_training_delivery
```

If `tensorboard` command is not available, reinstall training dependencies from active venv:

```bash
python -m pip install -r training/requirements.txt
```

## 11. Notebook Order When Starting from Scratch

Run notebook-first track in numeric order:

```text
phase_00_reproducibility_snapshot.ipynb
phase_01_data_audit_contracts.ipynb
phase_02_label_schema_baselines.ipynb
phase_03_normalization_feature_design.ipynb
phase_04_pair_generation_splits.ipynb
phase_05_baseline_evaluation.ipynb
phase_06_jobfit_training_experiments.ipynb
phase_07_ats_friendliness_scoring.ipynb
phase_08_overall_impression_signals.ipynb
phase_09_candidate_reranking.ipynb
phase_10_calibration_model_card.ipynb
phase_11_final_gate_review.ipynb
phase_12_repository_hygiene_runtime_bootstrap.ipynb
phase_13_data_snapshot_contract_freezing.ipynb
phase_14_normalization_feature_builder.ipynb
phase_15_balanced_pair_generation_splits.ipynb
phase_16_human_validation_label_governance.ipynb
phase_17_baseline_evaluation_v2.ipynb
phase_18_jobfit_training_v2.ipynb
phase_19_ats_friendliness_benchmark_scorer.ipynb
phase_19_5_jobfit_blocker_remediation.ipynb
phase_20_overall_impression_signals.ipynb
phase_21_backend_candidate_reranking.ipynb
phase_22_calibration_model_card_export.ipynb
phase_23_model_api_contract_validation.ipynb
phase_24_reproducibility_final_gate.ipynb
phase_25_tensorflow_training_delivery.ipynb
```

For current TensorFlow delivery verification, focus on Phase 25 first.

## 12. Validate E5 Embedding Model

Phase 17+ and Phase 25 use `intfloat/e5-base-v2`.

Run once from active venv:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('intfloat/e5-base-v2')
emb = model.encode(
    ['query: backend developer with python', 'passage: python backend job'],
    normalize_embeddings=True,
)
print('shape:', emb.shape)
print('norms:', np.linalg.norm(emb, axis=1).round(4).tolist())
PY
```

Expected:

```text
shape: (2, 768)
norms: [1.0, 1.0]
```

If download/cache fails, retry when internet is stable.

## 13. Model API Runtime

Model API is the serving package under `model_api/`. Use Python `3.13.11`, same as the TensorFlow notebook runtime.

Create and activate serving venv if not already active:

```bash
PYENV_VERSION=3.13.11 pyenv exec python -m venv .venv
source .venv/bin/activate
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you do not use `pyenv`, make sure `python -V` prints `3.13.11` before creating `.venv`.

Copy Model API env template:

```bash
cp model_api/.env.example model_api/.env
```

Start Model API:

```bash
export MODEL_API_ENV=local
export MODEL_API_SERVICE_TOKEN=replace-with-local-service-token
uvicorn model_api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Check health/readiness:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -H "authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" http://127.0.0.1:8000/model-info
```

Run targeted tests:

```bash
python -m unittest tests.model_api.test_phase_26_layout
python -m unittest tests.test_phase_29_model_api_hardening
python -m unittest tests.test_phase_31_release_gate
```

## 14. Backend integration run

Use this only when Backend integration is in scope.

1. Clone/open Backend repo outside this model repository:

   ```text
   https://github.com/bisa-kerja/bisakerja-api
   ```

2. Configure Backend API with:

   ```text
   MODEL_API_BASE_URL=http://localhost:8000
   MODEL_API_SERVICE_TOKEN=<same local token as Model API>
   ```

3. Start Backend API from Backend repository.
4. Run Backend health/route/contract tests from Backend repository.
5. Run upload contract flow through Backend `/api/v1/ai/cv-analyzer` with fixture PDFs and fixture jobs.
6. Keep Backend env files and database config outside this model repository.

## 15. Release Gate Commands

Run from repository root:

```bash
python scripts/verify_phase_27_1_27_2_release_gate.py --write
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
python scripts/verify_phase_27_7_clean_kernel_export.py --write
python scripts/verify_phase_27_8_model_card_manifest_refresh.py --write
python scripts/verify_phase_27_9_model_api_production_smoke.py --write --run-live
python scripts/verify_phase_27_10_requirement_matrix.py --write
python scripts/verify_phase_28_contract_realignment.py --write
python scripts/verify_phase_31_release_gate.py
```

Notes:

- `--run-live` requires Model API live runtime with dependencies installed.
- Some gates require TensorFlow/Keras artifact reload.
- Some gates require exported Backend contract fixtures from <https://github.com/bisa-kerja/bisakerja-api>.
- Do not hand-edit generated reports when a script owns them.

## 16. Troubleshooting Details

### Error: `No matching distribution found for tensorflow==2.21.0`

Common cause: pip uses Python `3.14`.

Check:

```bash
python -V
python -m pip -V
```

If output contains `python3.14` or `training/notebooks/venv/lib/python3.14`, fix:

```bash
deactivate 2>/dev/null || true
source training/.tf-venv-3.13/bin/activate
python -V
python -m pip -V
python -m pip install -r training/requirements.txt
```

### Notebook package not found

Check notebook cell:

```python
import sys
print(sys.executable)
```

If it is not `training/.tf-venv-3.13/bin/python`, kernel is wrong.

Fix: **Kernel -> Change Kernel -> Bisakerja Model TF 3.13**.

### Too many kernels are confusing

List kernels:

```bash
jupyter kernelspec list
```

For Phase 25, use only:

```text
bisakerja-model-tf313
```

Remove old project kernel if it appears:

```bash
jupyter kernelspec uninstall -f bisakerja-model-venv
```

### Too many Jupyter servers are confusing

List servers:

```bash
jupyter server list
```

Stop servers by port:

```bash
jupyter server stop 8888
jupyter server stop 8889
```

Run one server only:

```bash
python -m jupyter lab --notebook-dir .
```

### Reports do not change after Run All

Check:

- notebook uses correct kernel
- no error cell exists
- notebook was saved
- `reports/phase_25_*.json` timestamp changed

Command:

```bash
python - <<'PY'
from pathlib import Path
from datetime import datetime
for p in sorted(Path('reports').glob('phase_25_*.json')):
    print(datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'), p)
PY
```

### Model API readiness fails

Check:

- artifact path env vars
- `artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json`
- TensorFlow/Keras import
- custom object registration
- E5 dependency and cache
- PDF parser dependency
- `MODEL_API_SERVICE_TOKEN` for staging/production

### Backend integration fails

Check in external Backend repository:

- `MODEL_API_BASE_URL`
- service token match
- Redis and PostgreSQL connectivity
- Backend readiness endpoint
- Backend route tests
- Model API reachability from Backend process

## 17. Project Rules

- Training execution stays in versioned `.ipynb` notebooks.
- Do not add `training/*.py` entrypoints for training execution unless workflow is intentionally redesigned.
- Phase 25 uses TensorFlow Functional API / custom training flow.
- Embedding default for Phase 17+: `intfloat/e5-base-v2`.
- Profile/CV prefix: `query:`.
- Job prefix: `passage:`.
- Embeddings must be normalized.
- TF-IDF/local-hash fallback is allowed only for local smoke/plumbing, not staging/production evidence.
- Backend/wrapper-owned outputs stay outside model core: `topActionables`, `sectionReviews`, hydration, auth, persistence.
- Model API owns no DB credentials.
- Do not log raw CV text, service tokens, DB URLs, auth headers, uploaded file bytes, or unrelated PII.
- Do not mutate artifacts at runtime.
- Keep Backend source outside this model repository.
