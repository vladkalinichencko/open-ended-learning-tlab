# Open-Ended Learning

Учитель отбирает уровни лабиринта по предсказанной обучаемости, а не по приближению сожаления.

Отчёт — [report.md](report.md). Диагностика — [DIAGNOSTICS.html](DIAGNOSTICS.html).
Чекпойнты для проверки — `checkpoints/submission/`, родной формат JaxUED.

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m oel.training           # обучение ученика с выбранной оценкой полезности
python -m oel.evaluate_heldout   # оценка на человеческих отложенных картах
python -m oel.summary_report     # пересборка DIAGNOSTICS.html
python export_checkpoint.py      # экспорт политики в формат JaxUED
```

## Где что лежит

| из отчёта | в коде |
|---|---|
| MaxMC, приближение сожаления | `scores/maxmc.py` |
| ошибка предсказания ценности | `scores/pvl.py` |
| обучаемость \(q(1-q)\) | `scores/learnability.py` |
| предикторы: признаки, свёртка, ResNet | `oel/methods.py` |
| обучение ученика и цикл учителя | `oel/training.py` |
| оценка на отложенных картах | `oel/evaluate_heldout.py` |
| кластеризация неудач | `oel/hierarchy.py`, `oel/diagnostics.py` |
| сборка страницы диагностики | `oel/summary_report.py` |
| бейзлайны DR, PLR, ACCEL | `teacher.py` |
| поиск SFL | `sfl.py` |
| общий контур PPO | `rl_core.py` |
| отложенные уровни задания | `levels.py` |
