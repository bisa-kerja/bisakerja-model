# VPS Docker Deployment Runbook

Runbook for deploying Bisakerja Model API to a self-managed VPS with Docker Compose and GitHub Actions.

This runbook is for the model-serving repository only. Backend API deployment remains owned by <https://github.com/bisa-kerja/bisakerja-api>.

## Target Runtime

Recommended minimum VPS shape:

- 4 vCPU
- 12 GB RAM
- 80-100 GB disk
- Docker Engine with Docker Compose v2
- Optional reverse proxy such as Caddy or Nginx

The production compose file binds Model API to `127.0.0.1:3004` by default while the container still listens on `7860`. Public traffic should enter through Nginx HTTPS, then proxy to this local port. Keep `MODEL_API_SERVICE_TOKEN` strong because protected endpoints require bearer auth.

## Files

| File                                     | Purpose                                                       |
| ---------------------------------------- | ------------------------------------------------------------- |
| `.github/workflows/deploy-model-api.yml` | Builds Docker image, pushes to GHCR, deploys to VPS over SSH. |
| `docker-compose.production.yml`          | Production Docker Compose service for Model API.              |
| `Dockerfile`                             | Container image definition for FastAPI + TensorFlow runtime.  |
| `model_api/.env.example`                 | Runtime env template. Do not commit real production values.   |

## VPS Preparation

Install Docker and create swap if the VPS has no swap:

```bash
sudo apt update
sudo apt install -y ca-certificates curl git ufw
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Open public ingress for HTTP/HTTPS only when Nginx handles the public domain:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

Do not expose Docker port `3004` publicly unless Nginx is not used. Public requests to protected endpoints must include `Authorization: Bearer <MODEL_API_SERVICE_TOKEN>`.

## GitHub Secrets

Create these repository or environment secrets:

| Secret                     | Purpose                                                   |
| -------------------------- | --------------------------------------------------------- |
| `DEPLOY_VPS_HOST`          | VPS host/IP.                                              |
| `DEPLOY_VPS_PORT`          | SSH port, usually `22`.                                   |
| `DEPLOY_VPS_USERNAME`      | SSH user.                                                 |
| `DEPLOY_VPS_KEY`           | Private SSH key allowed to deploy.                        |
| `DEPLOY_REMOTE_PATH`       | Remote directory, for example `/opt/bisakerja-model-api`. |
| `DEPLOY_ENV_FILE`          | Full production `.env.production` content.                |
| `GHCR_READ_PACKAGES_TOKEN` | GitHub token with read access to GHCR package.            |
| `GH_USERNAME`              | GitHub username used for GHCR login on VPS.               |

Required workflow permissions are already set for image publishing:

```yaml
permissions:
  contents: read
  packages: write
```

## Production Env Template

Use `DEPLOY_ENV_FILE` with values similar to:

```env
MODEL_API_ENV=production
MODEL_API_SERVICE_NAME=bisakerja-model-api
MODEL_API_SERVICE_TOKEN=replace-with-strong-internal-token
MODEL_API_ALLOW_UNAUTHENTICATED_LOCAL=false
MODEL_API_TIMEOUT_MS=60000
MODEL_API_MAX_RECOMMENDATIONS=5
MODEL_API_WARMUP_ON_STARTUP=true
MODEL_API_WARMUP_REQUIRED=true
MODEL_API_MAX_PDF_BYTES=5000000
MODEL_API_MAX_PDF_PAGES=10
MODEL_API_ARTIFACT_ROOT=artifacts/phase_46_calibration_model_card_manifest_handoff_refresh
MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/multilingual-e5-small
MODEL_API_ENABLE_GENAI_WRAPPER=false
SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers
```

Do not include Backend DB credentials or public user credentials in Model API env.

To rollback to the Phase 25 E5-base package, switch the artifact root and expected embedding together:

```env
MODEL_API_ARTIFACT_ROOT=artifacts/phase_25_tensorflow_training_delivery
MODEL_API_EXPECTED_EMBEDDING_MODEL=intfloat/e5-base-v2
```

If explicit artifact paths are set, update model path, TensorFlow feature config, feature config, calibration, model card, and manifest as one unit. Startup fails when env, feature config, model card, manifest, or runtime backend disagree on embedding model or prefix policy.

## Deployment Flow

Trigger options:

- Push to `main` or `develop`.
- Manual `workflow_dispatch` with `deploy_branch` set to `main` or `develop`.

CD toggle:

- Push events deploy by default unless repository variable `MODEL_API_CD_ENABLED=false` is set.
- Manual runs can set `deploy_enabled=false` to build and push the image without deploying to VPS.

Workflow steps:

1. Build Docker image from `Dockerfile`.
2. Push tags to GHCR:
   - `<branch>`
   - `sha-<commit>`
3. Skip VPS deployment when CD is disabled.
4. Upload `docker-compose.production.yml` to `DEPLOY_REMOTE_PATH`.
5. Write `.env.production` from `DEPLOY_ENV_FILE`.
6. Pull image on VPS.
7. Stop existing `model-api` before pull when `STOP_MODEL_API_BEFORE_PULL=true` to free disk on small VPS volumes.
8. Prune unused Docker containers, images, and build cache.
9. Restart `model-api` with Docker Compose.
10. Poll `http://127.0.0.1:3004/live` as the deploy success gate.
11. Wait for `/ready`; if not ready yet, leave service running and print diagnostics because model loading continues in the background.
12. Print compose diagnostics on failure.

## Nginx Reverse Proxy

Use `docs/runbooks/nginx-bisakerja-model-api.conf` as the Nginx/aaPanel config for:

```text
https://bisakerja-model.salmanabdurrahman.my.id -> http://127.0.0.1:3004
```

Install/check flow on VPS:

```bash
sudo cp docs/runbooks/nginx-bisakerja-model-api.conf /www/server/panel/vhost/nginx/bisakerja-model.salmanabdurrahman.my.id.conf
sudo nginx -t
sudo systemctl reload nginx
```

Keep the Model API service token enabled. Nginx forwards the `Authorization` header to the app.

## Manual VPS Commands

Useful remote checks:

```bash
cd /opt/bisakerja-model-api

docker compose -f docker-compose.production.yml --env-file .env.production ps

docker compose -f docker-compose.production.yml --env-file .env.production logs --tail=150 model-api

curl -fsS http://127.0.0.1:3004/live
curl -fsS http://127.0.0.1:3004/health
curl -fsS http://127.0.0.1:3004/ready
```

Manual restart:

```bash
cd /opt/bisakerja-model-api
MODEL_API_IMAGE=ghcr.io/bisa-kerja/bisakerja-model:main \
MODEL_API_ENV_FILE=.env.production \
COMPOSE_PROJECT_NAME=bisakerja-model-api \
  docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api
```

## Compose Overrides

`docker-compose.production.yml` supports these optional env overrides:

| Env var                              | Default                                   | Purpose                                                                              |
| ------------------------------------ | ----------------------------------------- | ------------------------------------------------------------------------------------ |
| `MODEL_API_IMAGE`                    | `ghcr.io/bisa-kerja/bisakerja-model:main` | Image to run.                                                                        |
| `MODEL_API_BIND_ADDRESS`             | `127.0.0.1`                               | Local host bind address for Nginx upstream.                                          |
| `MODEL_API_PORT`                     | `3004`                                    | Host port mapped to container port `7860`.                                           |
| `MODEL_API_ENV_FILE`                 | `.env.production`                         | Compose env file path.                                                               |
| `MODEL_API_MEM_LIMIT`                | `8g`                                      | Container memory limit.                                                              |
| `SENTENCE_TRANSFORMERS_HOME`         | `/home/user/.cache/sentence-transformers` | Image-baked E5 cache path; do not mount over it or build-time predownload is hidden. |
| `MODEL_API_ARTIFACT_ROOT`            | app default                               | Versioned artifact package root for model/config switch.                             |
| `MODEL_API_EXPECTED_EMBEDDING_MODEL` | unset                                     | Optional startup assertion for deployed embedding model.                             |
| `COMPOSE_PROJECT_NAME`               | `bisakerja-model-api`                     | Compose project name.                                                                |

For a 4 vCPU / 12 GB VPS, keep defaults first. Increase `MODEL_API_TIMEOUT_MS` before raising memory limits. The compose file intentionally does not set Docker CPU quota because some VPS kernels/cgroup drivers reject `cpu.cfs_quota_us` writes.

## Rollback

Deploy a previous SHA tag manually:

```bash
cd /opt/bisakerja-model-api
MODEL_API_IMAGE=ghcr.io/bisa-kerja/bisakerja-model:sha-<previous-sha> \
MODEL_API_ENV_FILE=.env.production \
COMPOSE_PROJECT_NAME=bisakerja-model-api \
  docker compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans model-api
```

Then verify:

```bash
curl -fsS http://127.0.0.1:3004/live
curl -fsS http://127.0.0.1:3004/health
curl -fsS http://127.0.0.1:3004/ready
curl -fsS -H "Authorization: Bearer ${MODEL_API_SERVICE_TOKEN}" http://127.0.0.1:3004/model-info
```

For artifact rollback without changing image tag, switch `MODEL_API_ARTIFACT_ROOT` and `MODEL_API_EXPECTED_EMBEDDING_MODEL` back to the rollback package, restart, then confirm `/ready.embeddingModel`, `/ready.artifactPhase`, `/model-info.model.version`, and `/model-info.model.artifact.sha256`.

## Troubleshooting

- `MODEL_API_ENV mismatch for deploy`: update `DEPLOY_ENV_FILE` so it contains `MODEL_API_ENV=production`, or change `EXPECTED_MODEL_API_ENV` in the workflow for a non-production target. GitHub may mask the found value as `***` when it matches a secret.
- Build fails on dependency install: check Python/TensorFlow wheel compatibility for image platform. The Dockerfile installs CPU-only `torch` before `sentence-transformers` to avoid CUDA-sized layers.
- `no space left on device` during image pull/extract: run `docker system df`, then `docker image prune -af` and `docker builder prune -af`. The workflow defaults `STOP_MODEL_API_BEFORE_PULL=true`, causing short deploy downtime so the old image can be removed before pulling the new image.
- `cpu.cfs_quota_us: invalid argument`: remove Docker CPU quota. Current compose intentionally does not set `cpus` for VPS compatibility.
- Container exits during startup: inspect `docker compose logs --tail=150 model-api`.
- `/live` fails: process/proxy is down; inspect container logs and Nginx logs.
- `/health` works but `/ready` is false: wait for TensorFlow and E5 model load; inference should retry on `503 MODEL_NOT_READY`.
- Out-of-memory symptoms: add swap, reduce concurrency from Backend, keep `MODEL_API_MEM_LIMIT=8g`, inspect `docker stats`.
- Public timeout: keep Backend request timeout above `MODEL_API_TIMEOUT_MS`; if adding reverse proxy later, keep public listener on port `3004`.
