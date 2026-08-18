#!/usr/bin/env bash
# Часовые прогоны с разными настройками teacher'а — посмотреть, что вообще шевелит метрику.
# Меняются ТОЛЬКО флаги teacher-стороны; student, PPO и бюджет не трогаются.
#
#   ./run_teacher_variants.sh
set -uo pipefail
cd "$(dirname "$0")"
UPDATES="${UPDATES:-2500}"      # ~1 час на CPU
ATTEMPTS="${ATTEMPTS:-100}"     # разрешение +-0.09 вместо +-0.29

run () {
  local name="$1"; shift
  [ -d "results/var_$name" ] && { echo "=== $name: уже есть, пропускаю"; return; }
  echo "=== $name: $* ==="
  .venv/bin/python -u baseline.py --method accel --seed 0 --run-name "var_$name" \
    --num_updates "$UPDATES" --eval_freq 250 --eval_num_attempts "$ATTEMPTS" \
    --checkpoint_save_interval 4 "$@" > "tmp/var_${name}.log" 2>&1
  echo "  exit=$?"
}

# базовая точка отсчёта при том же коротком бюджете
run base
# 1. другая score-функция (обе уже реализованы в jaxued)
run pvl              --score_function pvl
# 2. реже переигрывать из буфера, чаще брать свежие уровни
run replay50         --replay_prob 0.5
# 3. острее приоритизация: температура ниже -> сильнее перекос на топовые уровни
run temp01           --temperature 0.1
# 4. больше мутаций за раз: ACCEL меняет 5 клеток, пробуем 15
run edits15          --num_edits 15
# 5. буфер меньше -> быстрее вытесняются старые оценки
run buf500           --level_buffer_capacity 500
