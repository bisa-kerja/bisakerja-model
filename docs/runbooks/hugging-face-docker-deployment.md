# Hugging Face Docker Deployment Runbook

Deploy Bisakerja Model API to Hugging Face Spaces with Docker.

## Deployment Target

Use Hugging Face Spaces with `sdk: docker`.

Why Docker Spaces:

- FastAPI is supported because Docker Spaces can host arbitrary containers.
- The external public port defaults to `7860` and can be set with `app_port`.
- Space code is a git repository; every push rebuilds and restarts the container.
- Secrets and variables are configured from the Space Settings page, not committed.
- The container should run as user ID `1000`; `Dockerfile` creates `user` with UID `1000`.
- Disk writes are not durable across restarts; Model API must remain stateless.

## Files Added

| File                     | Purpose                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| `Dockerfile`             | Hugging Face Docker Space runtime for FastAPI/Uvicorn on port `7860`.                         |
| `.dockerignore`          | Keeps Docker build context focused on Model API source and Phase 25 runtime artifacts.        |
| `docker-compose.yml`     | Local Docker smoke runner using the same image.                                               |
| `README.md` front matter | Hugging Face Space metadata: `sdk: docker`, `app_port: 7860`, model preload, startup timeout. |

## Hugging Face Space Metadata

The root `README.md` must start with YAML front matter:

```yaml
---
title: Bisakerja Model API
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
suggested_hardware: cpu-upgrade
startup_duration_timeout: 1h
models:
  - intfloat/e5-base-v2
preload_from_hub:
  - intfloat/e5-base-v2
pinned: false
---
```

Notes:

- `sdk: docker` tells Spaces to build the root `Dockerfile`.
- `app_port: 7860` matches Uvicorn and `EXPOSE 7860`.
- `startup_duration_timeout: 1h` allows TensorFlow and E5 warmup on cold start.
- `preload_from_hub` downloads `intfloat/e5-base-v2` into the default Hugging Face cache during build.
- Do not override `HF_HOME` in Space Settings; Hugging Face preloading writes to the default Hub cache.

## Required Hugging Face Settings

Create a new Space:

1. Go to Hugging Face → Spaces → Create new Space.
2. Choose SDK: `Docker`.
3. Visibility: prefer `private` or `protected` for internal API.
4. Hardware: start with `CPU Upgrade`; use GPU only if latency is unacceptable.
5. Push this repository to the Space git remote.

Set these Space variables/secrets in Settings.

### Secrets

| Secret                    | Required | Notes                                                                                       |
| ------------------------- | -------- | ------------------------------------------------------------------------------------------- |
| `MODEL_API_SERVICE_TOKEN` | Yes      | Bearer token required for `/model-info`, `/inference/*`, and `/internal/model/cv-analysis`. |
| `HF_TOKEN`                | Optional | Useful if Hub downloads hit rate limits or gated dependencies are introduced later.         |
| `OPENROUTER_API_KEY`      | No       | Keep unset. GenAI wrapper is backend-owned and disabled here.                               |

### Variables

| Variable                                | Value                 |
| --------------------------------------- | --------------------- |
| `MODEL_API_ENV`                         | `staging`             |
| `MODEL_API_SERVICE_NAME`                | `bisakerja-model-api` |
| `MODEL_API_ALLOW_UNAUTHENTICATED_LOCAL` | `false`               |
| `MODEL_API_TIMEOUT_MS`                  | `30000`               |
| `MODEL_API_MAX_RECOMMENDATIONS`         | `5`                   |
| `MODEL_API_WARMUP_ON_STARTUP`           | `true`                |
| `MODEL_API_WARMUP_REQUIRED`             | `true`                |
| `MODEL_API_MAX_PDF_BYTES`               | `5000000`             |
| `MODEL_API_MAX_PDF_PAGES`               | `10`                  |
| `MODEL_API_ENABLE_GENAI_WRAPPER`        | `false`               |

The Docker image also sets safe defaults for these values, but Space Settings should own deployment-specific runtime config.

## Deploy From Full Repository Safely

Do not push this full repository directly to Hugging Face Spaces. The full repo contains training, legacy, and embedding files that are not needed by runtime and can exceed Hugging Face's normal git file size limit.

Use the deploy script to generate a minimal Space repository:

```bash
scripts/deploy_hf_space.sh
```

The script creates `../bisakerja-model-hf-space` by default and copies only:

- root Space files: `README.md`, `Dockerfile`, `.dockerignore`, `requirements.txt`
- `model_api/`
- `docs/`
- Phase 25 artifacts with `required_for_inference=true`
- `artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json`

The script intentionally excludes:

- `.env` and `.env.*`
- `legacy/`
- `training/`
- `references/`
- `reports/`
- `.venv/`
- non-runtime embeddings and TensorBoard output

Push manually after reviewing the generated deploy repo:

```bash
cd ../bisakerja-model-hf-space
git push --force-with-lease hf main
```

Or let the script push after confirmation:

```bash
scripts/deploy_hf_space.sh --push
```

If a required runtime model artifact becomes larger than 10 MiB, use Git LFS in the deploy repo:

```bash
scripts/deploy_hf_space.sh --push --with-lfs
```

`--push` rewrites the Hugging Face Space `main` branch. Use it only after confirming the target remote is the Space repository.

## Local Docker Smoke

Build and run locally:

```bash
docker compose up --build
```

Check health:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/ready
curl -H "authorization: Bearer local-dev-token" http://127.0.0.1:7860/model-info
```

Expected:

- `/health` returns HTTP 200 while the process is alive.
- `/ready.ready=true` requires artifacts verified and TensorFlow model loaded.
- In local compose, warmup is disabled to avoid long E5 cold start.

## Hugging Face Smoke

After Space builds, replace `<space-host>`:

```bash
export MODEL_API_URL="https://<space-host>.hf.space"
export MODEL_API_SERVICE_TOKEN="replace-with-space-secret"

curl "${MODEL_API_URL}/health"
curl "${MODEL_API_URL}/ready"
curl -H "authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" "${MODEL_API_URL}/model-info"
```

Protected/private Spaces may require Hugging Face access in addition to the service bearer token. Public Spaces expose `/health` and `/ready`, but inference endpoints still require `MODEL_API_SERVICE_TOKEN`.

## Backend Integration URL

Use the Space URL as the Backend internal Model API base URL:

```text
https://<space-host>.hf.space
```

Backend must call:

```text
POST /internal/model/cv-analysis
authorization: Bearer <MODEL_API_SERVICE_TOKEN>
content-type: multipart/form-data
```

Do not call Model API directly from Frontend.

## Runtime Notes

- Model API is stateless; do not write durable files inside the Space.
- Free CPU Spaces can sleep when idle; paid hardware avoids sleeping.
- Cold start can be slow because TensorFlow loads the Keras model and SentenceTransformers loads E5.
- If startup times out, increase `startup_duration_timeout`, use CPU Upgrade, or disable startup warmup only for non-production smoke.
- Do not commit real tokens or `.env` files.

## Troubleshooting

| Symptom                                | Likely Cause                             | Fix                                                                             |
| -------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------- |
| Build cannot find runtime artifacts    | Docker context missing Phase 25 files    | Ensure `artifacts/phase_25_tensorflow_training_delivery/**` exists before push. |
| App exits with service token error     | `MODEL_API_ENV=staging` without token    | Add `MODEL_API_SERVICE_TOKEN` as Space secret.                                  |
| `/ready` shows `warmupCompleted=false` | Warmup failed or disabled while required | Check Space logs; set `MODEL_API_WARMUP_ON_STARTUP=true`, verify E5 download.   |
| E5 download slow                       | Cache miss on cold build/start           | Keep `preload_from_hub: intfloat/e5-base-v2`; optionally add `HF_TOKEN`.        |
| 401 on inference                       | Missing/wrong bearer token               | Send `authorization: Bearer <MODEL_API_SERVICE_TOKEN>`.                         |
| Space sleeps                           | Free hardware lifecycle                  | Upgrade hardware if service must stay warm.                                     |
