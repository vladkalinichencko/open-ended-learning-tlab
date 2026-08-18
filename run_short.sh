#!/usr/bin/env bash
# Короткий протокол для отбраковки идей: ~2 часа на CPU вместо 12.
#
# Смысл: быстро понять, делает ли идея вообще что-нибудь. НЕ для финальных чисел —
# в условии прямо сказано, что на коротких прогонах порядок методов другой.
#
# Два отличия от полного прогона:
#   --num_updates 5000      бюджет 1/6 (на A100 короткий протокол не нужен, там
#                           полные 30000 идут за час)
#   --eval_num_attempts 100 оценка стоит 0.5% от обучения, а разрешение с ±0.29
#                           улучшается до ±0.09. Это параметр измерения, а не
#                           student'а, так что рамки задания не нарушены.
#
#   ./run_short.sh accel     ./run_short.sh dr     ./run_short.sh plr
set -euo pipefail
cd "$(dirname "$0")"

METHOD="${1:?укажи метод: dr | plr | accel}"
SEED="${2:-0}"
UPDATES="${UPDATES:-5000}"
ATTEMPTS="${ATTEMPTS:-100}"

.venv/bin/python -u baseline.py --method "$METHOD" --seed "$SEED" \
  --run-name "short_${METHOD}_s${SEED}" \
  --num_updates "$UPDATES" \
  --eval_freq 500 \
  --eval_num_attempts "$ATTEMPTS" \
  --checkpoint_save_interval 2
