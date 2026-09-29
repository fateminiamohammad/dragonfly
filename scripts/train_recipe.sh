#!/usr/bin/env bash
# The full Dragonfly training recipe, one step at a time (docs/TRAINING.md). Every step runs in the Docker `trainer`
# service, so it survives terminal and session restarts. Stop serving first: training needs the GPU
#   docker stop dragonfly-model-service dragonfly-perception dragonfly-trainer-worker
#
#   scripts/train_recipe.sh data     # fetch + convert + build the mix (CPU)
#   scripts/train_recipe.sh m        # tier M: continue the 4B model on the new mix (~3.5 h on an RTX 3090 Ti)
#   scripts/train_recipe.sh distill  # write M's calibrated probabilities as soft targets for tier S
#   scripts/train_recipe.sh s        # tier S: train on the distilled targets (~35 min base, ~1.5 h large)
#   scripts/train_recipe.sh cascade  # pick the S -> M threshold on calibration, report it once on test
#   scripts/train_recipe.sh eval     # both test suites through the serving stack (start serving first)
#
# Settings (environment): MIX (mix-v2), M_INIT (runs/dragonfly-m4b), M_OUT (runs/dragonfly-m4b-v2),
# S_BACKBONE (answerdotai/ModernBERT-base), S_OUT (runs/dragonfly-s3), EPOCHS_M (1), EPOCHS_S (3).
set -euo pipefail
cd "$(dirname "$0")/.."

MIX=${MIX:-mix-v2}
M_INIT=${M_INIT:-runs/dragonfly-m4b}
M_OUT=${M_OUT:-runs/dragonfly-m4b-v2}
S_BACKBONE=${S_BACKBONE:-answerdotai/ModernBERT-base}
S_OUT=${S_OUT:-runs/dragonfly-s3}
EPOCHS_M=${EPOCHS_M:-1}
EPOCHS_S=${EPOCHS_S:-3}
PY=${PY:-model-service/.venv/Scripts/python}
[ -x "$PY" ] || PY=python
SCRIPTS="$(pwd -W 2>/dev/null || pwd)/scripts"
COMPOSE=(docker compose -f docker-compose.local.yml --env-file docker/.env --profile train)
export MSYS_NO_PATHCONV=1  # Git Bash on Windows: keep /data and /runs as container paths

run() {  # run <name> <args...>: a detached trainer container named dragonfly-<name>, then follow its log
  local name=$1; shift
  docker rm -f "dragonfly-$name" >/dev/null 2>&1 || true
  "${COMPOSE[@]}" run -d --name "dragonfly-$name" "$@"
  echo "started dragonfly-$name; follow with: docker logs -f dragonfly-$name"
}

case "${1:-}" in
  data)
    "$PY" scripts/fetch_data.py
    "$PY" scripts/convert_typed_decisions.py
    "$PY" scripts/convert_public.py
    "$PY" scripts/build_mix.py --version "${MIX#mix-}"
    ;;
  m)
    run trainer-m-recipe trainer dragonfly-train --tier M --init "/$M_INIT" --grad-checkpointing --batch-size 8 \
      --epochs "$EPOCHS_M" --lr 5e-5 --head-lr 1.5e-4 --train "/data/$MIX/train.jsonl" \
      --calibration "/data/$MIX/calibration.jsonl" --test /data/decision-v2/test.jsonl --out "/$M_OUT" --log-every 200
    ;;
  distill)
    run trainer-distill-recipe -v "$SCRIPTS:/scripts" --entrypoint python trainer /scripts/distill_targets.py --teacher "/$M_OUT" \
      --data "/data/$MIX/train.jsonl" --out "/data/$MIX/train.distilled.jsonl" --mix 0.5
    ;;
  s)
    run trainer-s-recipe trainer dragonfly-train --backbone "$S_BACKBONE" --grad-checkpointing --epochs "$EPOCHS_S" \
      --train "/data/$MIX/train.distilled.jsonl" --calibration "/data/$MIX/calibration.jsonl" \
      --test /data/decision-v2/test.jsonl --out "/$S_OUT" --log-every 500
    ;;
  cascade)
    "${COMPOSE[@]}" run --rm -v "$SCRIPTS:/scripts" --entrypoint python trainer \
      /scripts/tune_cascade.py --small "/$S_OUT" --large "/$M_OUT" --calibration "/data/$MIX/calibration.jsonl" \
      --test /data/decision-v2/test.jsonl --out "/$S_OUT/cascade.json"
    ;;
  eval)
    : "${DRAGONFLY_API_KEY:?set DRAGONFLY_API_KEY to a key the model-service accepts}"
    for suite in decision-v2 typed-decisions; do
      "$PY" bench/compare_systemone.py --data "data/$suite/test.jsonl" --n 2000 \
        --system dragonfly=http://127.0.0.1:8000@DRAGONFLY_API_KEY --out "runs/recipe-$suite.json"
    done
    ;;
  *)
    sed -n '2,17p' "$0"; exit 1
    ;;
esac
