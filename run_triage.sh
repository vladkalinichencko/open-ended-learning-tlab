#!/usr/bin/env bash
# Отсев идей учителя: короткий бюджет, один сид, смотрим не на solve rate, а на буфер.
#
# Solve rate на коротком прогоне обманывает — измерено: на 10% бюджета впереди
# оказывается простейший DR. Что не обманывает, так это чем score коррелирует и какие
# уровни попадают под переигрывание, и это видно с первых чекпойнтов.
#
#   ./run_triage.sh learnability
#   UPDATES=3000 ./run_triage.sh learnability_pvl
set -euo pipefail
cd "$(dirname "$0")"

SCORE="${1:?укажи score-функцию}"
UPDATES="${UPDATES:-2500}"

.venv/bin/python -u baseline.py --method mine --seed 0 \
  --run-name "triage_${SCORE}" \
  --score_function "$SCORE" \
  --num_updates "$UPDATES" \
  --eval_freq 500 \
  --eval_num_attempts 30 \
  --checkpoint_save_interval 1
