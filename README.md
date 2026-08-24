# Teacher-side curricula для zero-shot переноса в лабиринтах

Тестовое задание T-LAB, направление open-ended learning. Условие — в [NOTES.md](NOTES.md),
отчёт — в [REPORT.md](REPORT.md).

## Результаты

| метод | validation solve rate | held-out solve rate |
|---|---:|---:|
| MaxMC | 0.994 | 0.100 |
| фиксированные признаки | 0.942 | 0.150 |
| SFL | 0.975 | 0.125 |
| CNN-предиктор | 0.952 | 0.375 |

Порядок методов на двух наборах разный: CNN не лучший на сгенерированной validation, но
даёт в два с половиной раза больше ближайшего на человеческих held-out картах. Один сид,
5000 updates, пять попыток на уровень.

## Чекпойнты для проверки

`checkpoints/submission/` в родном layout JaxUED:

- `accel_cnn_predictor_a100_triage_seed0/0/` — выбранный метод;
- `accel_maxmc_a100_triage_seed0/0/` — собственный прогон ACCEL.

Архитектура политики не менялась, поэтому чекпойнты грузятся штатным eval.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

```bash
python -m oel.training            # обучение student с выбранным teacher-score
python -m oel.evaluate_heldout    # оценка на человеческих held-out картах
python -m oel.summary_report      # -> DIAGNOSTICS.html
python make_figures.py            # рисунок отчёта
python export_checkpoint.py       # экспорт политики в layout JaxUED
```

Бейзлайны DR, PLR и ACCEL живут в `teacher.py`, поиск SFL — в `sfl.py`; общий PPO-контур
у них один, он в `rl_core.py`.

## Раскладка кода

| путь | что там |
|---|---|
| `oel/` | реализация метода: `training`, `methods`, `config`, `diagnostics`, `hierarchy` |
| `scores/` | по файлу на score-функцию teacher-а: `maxmc`, `pvl`, `learnability` |
| `rl_core.py` | общий PPO-контур JaxUED: rollout, GAE, обновление актёра и критика |
| `teacher.py`, `sfl.py` | ACCEL-подобное обучение и отдельный поиск SFL |
| `levels.py`, `baseline.py`, `diag_buffer.py` | held-out уровни, бейзлайн и разбор буфера |

Интерактивная диагностика: [DIAGNOSTICS.html](DIAGNOSTICS.html).
