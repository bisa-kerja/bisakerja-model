# Local Development Runbook

## Prerequisites

- Python `3.13.11`
- TensorFlow-compatible environment for training and serving checks
- Repository root as working directory
- Local Model API env file copied from template

Backend API is not part of this repository. Use <https://github.com/bisa-kerja/bisakerja-api> when integration testing requires Backend API.

## Environment Files

Primary Model API template:

```bash
cp model_api/.env.example model_api/.env
```

Optional root template:

```bash
cp .env.example .env
```

Root `.env.example` is not required for Model API runtime. It only holds model-workspace orchestration defaults. Keep concrete `.env` files untracked.

## Model API Runtime

Use Python `3.13.11`, same as the TensorFlow notebook runtime:

```bash
PYENV_VERSION=3.13.11 pyenv exec python -m venv .venv
source .venv/bin/activate
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export MODEL_API_ENV=local
export MODEL_API_SERVICE_TOKEN=replace-with-local-service-token
uvicorn model_api.app:create_app --factory --host 0.0.0.0 --port 8000
```

If you do not use `pyenv`, make sure `python -V` prints `3.13.11` before creating `.venv`.

Health/readiness checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl -H "authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" http://127.0.0.1:8000/model-info
```

## Training Runtime

Use the dedicated TensorFlow training environment:

```bash
deactivate 2>/dev/null || true
PYENV_VERSION=3.13.11 pyenv exec python -m venv training/.tf-venv-3.13
source training/.tf-venv-3.13/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
python -m pip install ipykernel jupyterlab
python -m ipykernel install --user --name bisakerja-model-tf313 --display-name "Bisakerja Model TF 3.13"
```

Open notebooks from repo root and select kernel `Bisakerja Model TF 3.13`.

## Targeted Verification

```bash
python -m unittest tests.model_api.test_phase_26_layout
python -m unittest tests.test_phase_29_model_api_hardening
python -m unittest tests.test_phase_31_release_gate
```

Run release-gate scripts when artifacts or reports change:

```bash
python scripts/verify_phase_27_1_27_2_release_gate.py --write
python scripts/verify_phase_27_3_27_4_release_evidence.py --write
python scripts/verify_phase_27_5_27_6_validation_expansion.py --write
python scripts/verify_phase_31_release_gate.py
```

## Backend Integration Testing

When Backend integration is in scope:

1. Clone or open <https://github.com/bisa-kerja/bisakerja-api> outside this repository.
2. Start Backend API with `MODEL_API_BASE_URL=http://localhost:8000` and a matching service token.
3. Run Backend-side health, contract, and route tests from the Backend repository.
4. Keep Backend source, env files, and database config outside this model repository.

## Troubleshooting

- If TensorFlow install fails, verify `python -V` is `3.13.x`, not `3.14`.
- If Model API readiness fails, inspect artifact paths, manifest hashes, TensorFlow loader, E5 backend, PDF parser, and `MODEL_API_SERVICE_TOKEN`.
- If Backend integration fails, verify Backend repo config, `MODEL_API_BASE_URL`, service-token match, and Model API reachability.
- If release evidence fails, regenerate reports through scripts instead of hand-editing generated JSON.
