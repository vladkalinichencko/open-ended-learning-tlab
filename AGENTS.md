# AGENTS.md

Тестовое задание T-LAB, направление **Open-Ended Learning (UED)**.
Постановка целиком — в [NOTES.md](NOTES.md). Прочитай её перед любой работой.

## Раскладка

| путь | что там | в git |
|---|---|---|
| `NOTES.md` | задание, сетап, лог, идеи, результаты | да |
| `baseline.py` | запуск DR / PLR$^\perp$ / ACCEL с нетронутым конфигом student'а | да |
| `solve_rate.py` | агрегация `results/` в таблицу solve rate по сидам | да |
| `maze_ours.py` | наш teacher (копия `maze_plr.py` с правками teacher-стороны) | да |
| `report.md` | финальный отчёт | да |
| `checkpoints/` | чекпойнты обучения, `<run_name>/<seed>/models/` | нет |
| `results/` | `results.npz` из `--mode eval` | нет |
| `datasets/` | свой валидационный набор уровней (dev-набор трогать нельзя) | нет |
| `runs/` | графики, сводные таблицы | нет |
| `tmp/` | всё, что нагенерил агент | нет |
| `ext/jaxued` | клон кодовой базы, **не изменять** | нет |

## Правила

- `ext/jaxued` не патчим. Свой метод — отдельный файл `maze_ours.py` в корне,
  сделанный копией `ext/jaxued/examples/maze_plr.py` (так задуман upstream:
  «simply copy one of the files, and start modifying»).
- **Student неприкосновенен**: класс `ActorCritic`, гиперпараметры PPO, `--num_updates 30000`.
  Меняется только teacher-сторона. Если правка задевает архитектуру политики —
  чекпойнт не загрузится в их eval и сдача не будет оценена.
- **Dev-уровни (8 штук) не используются** ни в обучении, ни в подборе
  гиперпараметров, ни как шаблоны для генерации. Нужен валидационный набор —
  свой, в `datasets/`, с описанием в отчёте.
- Сравнения только на полном бюджете и минимум на 3 сидах: на коротких прогонах
  порядок методов другой.
- Всё черновое — в `tmp/`.
- Каждый прогон — строка в NOTES → «Лог».

## Команды

```bash
source .venv/bin/activate
export WANDB_MODE=offline          # или wandb login
python baseline.py --method accel --seed 0
python baseline.py --method accel --seed 0 --mode eval
python solve_rate.py results/dr results/plr results/accel --plot
```

## Что сдаём

- [ ] GitHub-репозиторий: код + инструкция запуска
- [ ] Отчёт с **анализом**, а не только числами (самый важный пункт)
- [ ] Свой метод против DR / PLR$^\perp$ / ACCEL на dev-наборе, несколько сидов
- [ ] Чекпойнты своего метода и своего ACCEL, для каждого сида (`checkpoints/` в
      .gitignore — прикладываем архивом или `git add -f` только финальные)
