#!/usr/bin/env bash
# Deploy one service to a new image, wait until healthy, smoke-check it, and roll back automatically on failure.
#
#   ops/deploy.sh model-service ghcr.io/<owner>/dragonfly-model-service:<40-char sha>
#   ops/deploy.sh backend       ghcr.io/<owner>/dragonfly-backend:<sha>
#   ops/deploy.sh ui            ghcr.io/<owner>/dragonfly-ui:<sha>
#
# Only full commit-SHA tags are accepted, so every deploy is traceable to a commit. Run from the repo root on the host.
set -euo pipefail

SERVICE="${1:?usage: ops/deploy.sh <model-service|backend|ui> <image:sha>}"
IMAGE="${2:?usage: ops/deploy.sh <service> <image:sha>}"
COMPOSE=(docker compose -f docker-compose.prod.yml --env-file "docker/${ENV_FILE_NAME:-.env}" --project-name dragonfly)

case "$SERVICE" in
  model-service) VAR=MODEL_IMAGE;   HEALTH="python -c \"import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())\"" ;;
  backend)       VAR=BACKEND_IMAGE; HEALTH="wget -qO- http://127.0.0.1:3000/api/health" ;;
  ui)            VAR=UI_IMAGE;      HEALTH="wget -qO- http://127.0.0.1:80/ >/dev/null && echo '\"status\":\"ok\"'" ;;
  *) echo "unknown service: $SERVICE" >&2; exit 2 ;;
esac

if ! [[ "$IMAGE" =~ :[0-9a-f]{40}$ ]]; then
  echo "refusing $IMAGE: deploy images tagged with a full 40-character commit SHA" >&2
  exit 2
fi

# the image currently running, to roll back to
PREVIOUS="$(docker inspect --format '{{.Config.Image}}' "dragonfly-$SERVICE" 2>/dev/null || true)"
echo "deploying $SERVICE: ${PREVIOUS:-<none>} -> $IMAGE"

docker pull "$IMAGE"
export MODEL_IMAGE="${MODEL_IMAGE:-$(docker inspect --format '{{.Config.Image}}' dragonfly-model-service 2>/dev/null || echo "$IMAGE")}"
export BACKEND_IMAGE="${BACKEND_IMAGE:-$(docker inspect --format '{{.Config.Image}}' dragonfly-backend 2>/dev/null || echo "$IMAGE")}"
export UI_IMAGE="${UI_IMAGE:-$(docker inspect --format '{{.Config.Image}}' dragonfly-ui 2>/dev/null || echo "$IMAGE")}"
export "$VAR=$IMAGE"

if [[ "$SERVICE" == backend ]]; then
  echo "backing up postgres before the backend runs its migrations"
  mkdir -p backups
  docker exec dragonfly-postgres sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' | gzip > "backups/pg-$(date +%Y%m%d-%H%M%S).sql.gz"
fi

up() { "${COMPOSE[@]}" up -d --no-deps "$SERVICE"; }

wait_healthy() {
  for _ in $(seq 1 60); do
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' "dragonfly-$SERVICE" 2>/dev/null || echo missing)"
    [[ "$status" == healthy || ( "$SERVICE" == ui && "$status" == running ) ]] && return 0
    sleep 5
  done
  return 1
}

up
if wait_healthy && docker exec "dragonfly-$SERVICE" sh -c "$HEALTH" | grep -q '"status":"ok"'; then
  echo "deployed $SERVICE $IMAGE"
  exit 0
fi

echo "deploy failed: rolling back $SERVICE to ${PREVIOUS:-nothing}" >&2
if [[ -n "$PREVIOUS" ]]; then
  export "$VAR=$PREVIOUS"
  up
  wait_healthy || echo "rollback is not healthy either: investigate now" >&2
fi
exit 1
