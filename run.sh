#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/11726517}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiments}"
EXPERIMENT="${EXPERIMENT:-main}"
YEARS="${YEARS:-2019}"
FLOORS="${FLOORS:-2 3 4 5 6 7}"
MODELS="${MODELS:-lafboost}"
EPOCHS="${EPOCHS:-20}"
STRIDE="${STRIDE:-30}"

read -r -a YEAR_ARGS <<< "$YEARS"
read -r -a FLOOR_ARGS <<< "$FLOORS"
read -r -a MODEL_ARGS <<< "$MODELS"

python -m lafboost \
  --experiment "$EXPERIMENT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --years "${YEAR_ARGS[@]}" \
  --floors "${FLOOR_ARGS[@]}" \
  --models "${MODEL_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --stride "$STRIDE" \
  "$@"
