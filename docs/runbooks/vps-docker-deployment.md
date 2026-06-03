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

The production compose file binds Model API publicly to `0.0.0.0:3004` by default while the container still listens on `7860`. Keep `MODEL_API_SERVICE_TOKEN` strong because port `3004` is intended for public ingress.

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

Open public ingress only for port `3004`:

```bash
sudo ufw allow 3004/tcp
```

If a cloud firewall is present, allow TCP `3004` there too. Public requests must include `Authorization: Bearer <MODEL_API_SERVICE_TOKEN>` for protected endpoints.

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
MODEL_API_WARMUP_REQUIRED=false
MODEL_API_MAX_PDF_BYTES=5000000
MODEL_API_MAX_PDF_PAGES=10
MODEL_API_ENABLE_GENAI_WRAPPER=false
SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers
```

Do not include Backend DB credentials or public user credentials in Model API env.

## Deployment Flow

Trigger options:

- Push to `main` or `develop`.
- Manual `workflow_dispatch` with `deploy_branch` set to `main` or `develop`.

Workflow steps:

1. Build Docker image from `Dockerfile`.
2. Push tags to GHCR:
   - `<branch>`
   - `sha-<commit>`
3. Upload `docker-compose.production.yml` to `DEPLOY_REMOTE_PATH`.
4. Write `.env.production` from `DEPLOY_ENV_FILE`.
5. Pull image on VPS.
6. Restart `model-api` with Docker Compose.
7. Poll `http://127.0.0.1:3004/health`.
8. Print compose diagnostics on failure.

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

| Env var                  | Default                                   | Purpose                                           |
| ------------------------ | ----------------------------------------- | ------------------------------------------------- |
| `MODEL_API_IMAGE`        | `ghcr.io/bisa-kerja/bisakerja-model:main` | Image to run.                                     |
| `MODEL_API_BIND_ADDRESS` | `0.0.0.0`                                 | Public host bind address.                         |
| `MODEL_API_PORT`         | `3004`                                    | Public host port mapped to container port `7860`. |
| `MODEL_API_ENV_FILE`     | `.env.production`                         | Compose env file path.                            |
| `MODEL_API_MEM_LIMIT`    | `8g`                                      | Container memory limit.                           |
| `MODEL_API_CPUS`         | `3.0`                                     | Container CPU quota.                              |
| `COMPOSE_PROJECT_NAME`   | `bisakerja-model-api`                     | Compose project name.                             |

For a 4 vCPU / 12 GB VPS, keep defaults first. Increase `MODEL_API_TIMEOUT_MS` before raising CPU/memory limits.

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
curl -fsS http://127.0.0.1:3004/health
```

## Troubleshooting

- Build fails on dependency install: check Python/TensorFlow wheel compatibility for image platform.
- Container exits during startup: inspect `docker compose logs --tail=150 model-api`.
- Health fails but container runs: wait for TensorFlow and E5 model load, then check `/ready`.
- Out-of-memory symptoms: add swap, reduce concurrency from Backend, keep `MODEL_API_MEM_LIMIT=8g`, inspect `docker stats`.
- Public timeout: keep Backend request timeout above `MODEL_API_TIMEOUT_MS`; if adding reverse proxy later, keep public listener on port `3004`.
