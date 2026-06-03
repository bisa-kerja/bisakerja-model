#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/deploy_hf_space.sh [options]

Prepare a minimal Hugging Face Docker Space deployment tree from this repo.
By default this script only writes the deploy directory. It does not push.

Options:
  --deploy-dir PATH   Deployment worktree directory (default: ../bisakerja-model-hf-space)
  --remote URL        Hugging Face Space git URL (default: current repo remote "hf")
  --push              Commit and push deploy tree to Hugging Face remote main
  --with-lfs          Track files larger than 10 MiB with Git LFS inside deploy repo
  --yes               Skip push confirmation prompt (only valid with --push)
  -h, --help          Show this help

Examples:
  scripts/deploy_hf_space.sh
  scripts/deploy_hf_space.sh --push
  scripts/deploy_hf_space.sh --push --with-lfs
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="${HF_DEPLOY_DIR:-$REPO_ROOT/../bisakerja-model-hf-space}"
REMOTE_URL="${HF_SPACE_REMOTE:-}"
PUSH=false
WITH_LFS=false
YES=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    --remote)
      REMOTE_URL="$2"
      shift 2
      ;;
    --push)
      PUSH=true
      shift
      ;;
    --with-lfs)
      WITH_LFS=true
      shift
      ;;
    --yes)
      YES=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$REMOTE_URL" ]]; then
  REMOTE_URL="$(git -C "$REPO_ROOT" remote get-url hf 2>/dev/null || true)"
fi

if [[ "$PUSH" == true && -z "$REMOTE_URL" ]]; then
  echo "Missing Hugging Face remote. Pass --remote or set HF_SPACE_REMOTE." >&2
  exit 1
fi

if [[ "$YES" == true && "$PUSH" != true ]]; then
  echo "--yes is only valid with --push." >&2
  exit 2
fi

command -v python >/dev/null || { echo "python not found" >&2; exit 1; }
command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
command -v rsync >/dev/null || { echo "rsync not found" >&2; exit 1; }

MARKER=".hf-space-deploy-dir"
if [[ -e "$DEPLOY_DIR" && ! -f "$DEPLOY_DIR/$MARKER" ]]; then
  echo "Refusing to use existing unmarked deploy dir: $DEPLOY_DIR" >&2
  echo "Choose another --deploy-dir or remove/create marker intentionally." >&2
  exit 1
fi

mkdir -p "$DEPLOY_DIR"
touch "$DEPLOY_DIR/$MARKER"

# Clean generated deploy tree while preserving its git metadata.
find "$DEPLOY_DIR" -mindepth 1 \
  ! -name .git \
  ! -path "$DEPLOY_DIR/.git/*" \
  ! -name "$MARKER" \
  -exec rm -rf {} +

echo "Preparing Hugging Face deploy tree: $DEPLOY_DIR"

rsync -a \
  "$REPO_ROOT/README.md" \
  "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT/.dockerignore" \
  "$REPO_ROOT/requirements.txt" \
  "$DEPLOY_DIR/"

rsync -a --exclude '.env' --exclude '.env.*' "$REPO_ROOT/model_api" "$DEPLOY_DIR/"
rsync -a "$REPO_ROOT/docs" "$DEPLOY_DIR/"

python - "$REPO_ROOT" "$DEPLOY_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

repo = Path(sys.argv[1])
deploy = Path(sys.argv[2])
manifest_path = repo / "artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

paths: set[str] = {"artifacts/phase_25_tensorflow_training_delivery/artifact_manifest.json"}
for item in manifest.get("artifacts", []):
    if item.get("required_for_inference") is True:
        paths.add(str(item["path"]))

missing: list[str] = []
for rel in sorted(paths):
    src = repo / rel
    dst = deploy / rel
    if not src.exists():
        missing.append(rel)
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

if missing:
    raise SystemExit("Missing required runtime artifacts:\n" + "\n".join(f"- {p}" for p in missing))

print("Runtime artifacts copied:")
for rel in sorted(paths):
    print(f"- {rel}")
PY

# Safety checks: deployment subset must not contain local secrets or non-runtime workspaces.
if find "$DEPLOY_DIR" -type f \( -name '.env' -o -name '.env.*' \) | grep -q .; then
  echo "Refusing deploy: .env file copied into deploy dir" >&2
  exit 1
fi

for forbidden in legacy training references .venv reports; do
  if [[ -e "$DEPLOY_DIR/$forbidden" ]]; then
    echo "Refusing deploy: forbidden workspace copied: $forbidden" >&2
    exit 1
  fi
done

LARGE_FILE_LIST="$(find "$DEPLOY_DIR" -type f -size +10M ! -path '*/.git/*' | sort)"
if [[ -n "$LARGE_FILE_LIST" ]]; then
  echo "Files larger than 10 MiB in deploy tree:"
  printf '%s\n' "$LARGE_FILE_LIST" | sed 's/^/ - /'
  if [[ "$WITH_LFS" != true ]]; then
    echo "Hugging Face rejects normal git files >10 MiB. Re-run with --with-lfs if these files are required." >&2
    exit 1
  fi
  command -v git-lfs >/dev/null 2>&1 || {
    echo "git-lfs not found. Install Git LFS or reduce runtime artifacts." >&2
    exit 1
  }
fi

if [[ ! -d "$DEPLOY_DIR/.git" ]]; then
  git -C "$DEPLOY_DIR" init -b main >/dev/null
fi

if [[ "$WITH_LFS" == true && -n "$LARGE_FILE_LIST" ]]; then
  git -C "$DEPLOY_DIR" lfs install --local >/dev/null
  while IFS= read -r file; do
    [[ -n "$file" ]] || continue
    rel="${file#$DEPLOY_DIR/}"
    git -C "$DEPLOY_DIR" lfs track "$rel" >/dev/null
  done <<< "$LARGE_FILE_LIST"
fi

if [[ -n "$REMOTE_URL" ]]; then
  if git -C "$DEPLOY_DIR" remote get-url hf >/dev/null 2>&1; then
    git -C "$DEPLOY_DIR" remote set-url hf "$REMOTE_URL"
  else
    git -C "$DEPLOY_DIR" remote add hf "$REMOTE_URL"
  fi
fi

git -C "$DEPLOY_DIR" add .
if git -C "$DEPLOY_DIR" diff --cached --quiet; then
  echo "No deploy changes to commit."
else
  git -C "$DEPLOY_DIR" commit -m "Deploy Bisakerja Model API Docker Space" >/dev/null
  echo "Deploy commit created: $(git -C "$DEPLOY_DIR" rev-parse --short HEAD)"
fi

echo "Deploy tree ready: $DEPLOY_DIR"

if [[ "$PUSH" != true ]]; then
  cat <<EOF

Next step when ready:
  cd "$DEPLOY_DIR"
  git push --force-with-lease hf main

Or run:
  scripts/deploy_hf_space.sh --push
EOF
  exit 0
fi

if [[ "$YES" != true ]]; then
  echo "About to push deploy subset to Hugging Face remote main: $REMOTE_URL"
  echo "This rewrites the Space repository main branch. Type 'push' to continue:"
  read -r confirmation
  if [[ "$confirmation" != "push" ]]; then
    echo "Push cancelled."
    exit 1
  fi
fi

# Update lease info when remote branch exists; ignore first-push fetch failures.
git -C "$DEPLOY_DIR" fetch hf main:refs/remotes/hf/main >/dev/null 2>&1 || true

git -C "$DEPLOY_DIR" push --force-with-lease hf main
