#!/usr/bin/env bash
# Отсев идей учителя: короткий бюджет, один сид, смотрим не на solve rate, а на буфер.
#
# Solve rate на коротком прогоне обманывает — измерено: на 10% бюджета впереди
# оказывается простейший DR. Что не обманывает, так это чем score коррелирует и какие
# уровни попадают под переигрывание, и это видно с первых чекпойнтов.
#
#   ./run_triage.sh sfl                  # отдельный цикл поиска, как в статье
#   ./run_triage.sh learnability_struct  # своя score-функция внутри PLR
set -euo pipefail
cd "$(dirname "$0")"

NAME="${1:?укажи sfl или имя score-функции}"
UPDATES="${UPDATES:-2500}"

if [ "$NAME" = "sfl" ]; then
  EXTRA=(--method sfl --sfl_num_levels 512 --sfl_attempts 8 --sfl_period 250)
else
  EXTRA=(--method mine --score_function "$NAME")
fi

.venv/bin/python -u baseline.py --seed 0 --run-name "triage_${NAME}" \
  --num_updates "$UPDATES" \
  --eval_freq 250 \
  --eval_num_attempts 30 \
  --checkpoint_save_interval 1 \
  "${EXTRA[@]}"
