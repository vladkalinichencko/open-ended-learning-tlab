#!/usr/bin/env bash
# Полный бюджет из условия: 30000 апдейтов ≈ 245 млн шагов среды. Это те числа,
# с которыми сравнивается свой teacher, короткие прогоны для них не годятся.
#
# Отличия от дефолтов jaxued — только измерение, не student:
#   --eval_num_attempts 30  разрешение ±0.09 вместо ±0.29 при десяти попытках
#   --eval_freq 500         чтобы оценка не съедала больше времени, чем при дефолте
#
#   ./run_full.sh accel     ./run_full.sh plr     ./run_full.sh dr
set -euo pipefail
cd "$(dirname "$0")"

METHOD="${1:?укажи метод: dr | plr | accel}"
SEED="${2:-0}"

.venv/bin/python -u baseline.py --method "$METHOD" --seed "$SEED" \
  --run-name "full_${METHOD}_s${SEED}" \
  --eval_freq 500 \
  --eval_num_attempts 30
