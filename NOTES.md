# Open-Ended Learning — свой teacher для UED

## Задание

### О чём это

Нас интересуют агенты, которые не просто решают фиксированный набор задач, а сами
порождают себе новые задачи по мере обучения — и за счёт этого выучивают то, чего
не было ни в обучающих данных, ни в исходной политике.

Удобная экспериментальная постановка — **Unsupervised Environment Design (UED)** [1].
В ней есть *student* — обычный RL-агент, и *teacher* — механизм, решающий, на каких
уровнях student'у сейчас тренироваться:

- **DR** (domain randomization) — teacher генерирует случайные уровни;
- **PLR** [2] — запоминает сыгранные уровни и чаще переигрывает те, на которых
  student'у предположительно есть чему учиться (это оценивает score-функция,
  например через value loss);
- **ACCEL** [3] — вдобавок мутирует уровни с высоким score, так что сложность
  растёт постепенно.

Слабое место у всех этих методов одно и то же — **score-функция**. В идеале она
должна находить уровни на границе способностей student'а: уже не тривиальные, но
ещё решаемые. На практике вместо этого используются грубые прокси regret'а
(positive value loss, MaxMC), которые измеряют не совсем то и на длинных
лабиринтах систематически промахиваются. Подробный разбор проблемы — в [4], с него
хорошо начинать.

Вот эту проблему и предлагается поисследовать: как дёшево понять, на каких уровнях
student'у есть чему учиться, и как вести генерацию уровней вдоль этой границы.

### Задача

Дана кодовая база JaxUED [5] с готовыми реализациями DR / PLR$^{\perp}$ / ACCEL и
зафиксированным student'ом (PPO+LSTM) в домене Minigrid-лабиринтов. Качество меряется
zero-shot: обученный student проверяется на уровнях, придуманных человеком, которых
не было в обучении.

**Придумать и реализовать свой teacher** — score-функцию, стратегию генерации и
мутаций уровней, логику replay или их комбинацию — так, чтобы student после обучения
решал held-out уровни лучше, чем после обучения с ACCEL и PLR$^{\perp}$, при том же
бюджете. Цель — побить оба бейзлайна.

### Рамки

- **Конфигурацию student'а не меняем.** Сам student, конечно, обучается, но его
  архитектура, гиперпараметры PPO и бюджет шагов среды
  ($30\,000$ апдейтов $\approx 245$ млн шагов) зафиксированы. Менять можно только
  teacher-сторону: какие уровни показывать и когда. Так сравнение остаётся
  сравнением курикулумов, а не подбором гиперпараметров RL.
- **Eval-уровни не трогаем.** Dev-набор — 8 нарисованных руками уровней, встроенных
  в JaxUED: `SixteenRooms`, `SixteenRooms2`, `Labyrinth`, `LabyrinthFlipped`,
  `Labyrinth2`, `StandardMaze`, `StandardMaze2`, `StandardMaze3` (дефолт
  `--eval_levels`). На нём мы сравниваемся с бейзлайнами и строим графики для отчёта,
  но использовать его в обучении, при подборе гиперпараметров или как шаблоны для
  генерации **нельзя**. Финальные числа посчитают сами: прогонят чекпойнты на
  секретном наборе уровней того же формата. Нужен валидационный набор — соберите
  свой и опишите его в отчёте.
- **Внутри teacher'а можно что угодно:** другие score-функции на сигналах student'а,
  вспомогательные обучаемые модели, эволюция популяции уровней и т.д. Опираться на
  идеи из статей можно и нужно, просто явно указывайте, что откуда взято.

### Код и бейзлайны

Код — публичный <https://github.com/DramaCow/jaxued>, не изменён. Версии
зависимостей в репозитории не запинены, поэтому ставить так:

```bash
git clone https://github.com/DramaCow/jaxued.git && cd jaxued
pip install "jax[cuda12]==0.4.30" flax==0.8.5 chex==0.1.86 optax==0.2.3 \
  distrax==0.1.5 gymnax==0.0.8 orbax-checkpoint==0.5.3 "numpy<2" \
  wandb==0.17.5 pillow imageio
pip install --no-deps -e .
```

Конфигурация student'а — прямо в `examples/maze_plr.py`: класс `ActorCritic` и
дефолты argparse; эти флаги не переопределяем.

Бейзлайны (`--checkpoint_save_interval 17` сохраняет чекпойнты, которые надо
приложить к сдаче):

| метод | команда |
|---|---|
| DR | `python examples/maze_dr.py` |
| PLR$^{\perp}$ | `python examples/maze_plr.py` |
| ACCEL | `python examples/maze_plr.py --use_accel` |

Полный прогон — от часа на A100 до нескольких часов на T4; бесплатных Colab или
Kaggle хватает. Финальные сравнения — только на полном бюджете: на коротких
прогонах порядок методов другой.

> Прежде чем строить своё, воспроизведите бейзлайн и сверьтесь с их числами
> (3 сида, solve rate на dev-наборе).
> *(таблица с их числами в присланном условии не скопировалась — запросить.)*

### Что прислать

1. Код решения — чистый, воспроизводимый, с инструкцией запуска.
2. Отчёт — честно говоря, самое важное.
3. Результаты: свой метод против DR / PLR$^{\perp}$ / ACCEL на dev-наборе, с
   несколькими сидами.
4. Чекпойнты финальных политик — своего метода и своего прогона ACCEL, для каждого
   сида: каталоги `checkpoints/<run_name>/<seed>` из обучения с
   `--checkpoint_save_interval 17` (несколько МБ на сид). По ним посчитают числа на
   секретном наборе. **Архитектура политики должна остаться исходной** — иначе
   чекпойнт не загрузится в их eval и сдача не будет оценена.

На финальном созвоне попросят коротко рассказать про решение и порассуждать вслух —
почему сделали так, что будет, если поменять. Важно самому понимать, что происходит.

### Правила

- **Проанализируйте полученные результаты.** Это самый важный пункт: хочется увидеть
  не только числа с метриками. Как объясняется увиденное поведение? Почему ваш score
  находит то, на чём student'у есть чему учиться, а не шум? На каких уровнях выиграли,
  на каких нет и почему? Что получилось, что нет?
- Нет правильного способа решить задачу — оценивают исследовательские способности.
  Отрицательный результат с честным разбором тоже ценен.
- Убедитесь, что результатам можно доверять: сиды, разброс, одинаковый бюджет у всех
  методов.
- Можно пользоваться любыми AI-агентами и ассистентами — важен результат и то,
  насколько вы понимаете своё решение.
- Код должен быть чистым и понятным: грязно оформленное решение могут отклонить.

### Ссылки

1. Dennis et al., 2020 — *Emergent Complexity and Zero-shot Transfer via Unsupervised
   Environment Design*, [arXiv:2012.02096](https://arxiv.org/pdf/2012.02096). Постановка
   UED и minimax regret.
2. Jiang et al., 2021 — *Replay-Guided Adversarial Environment Design*,
   [arXiv:2110.02439](https://arxiv.org/pdf/2110.02439). PLR$^{\perp}$, первый бейзлайн.
3. Parker-Holder et al., 2022 — *Evolving Curricula with Regret-Based Environment Design*,
   [arXiv:2203.01302](https://arxiv.org/pdf/2203.01302). ACCEL, главный бейзлайн.
4. Rutherford et al., 2024 — *No Regrets: Investigating and Improving Regret
   Approximations in UED*, [arXiv:2408.15099](https://arxiv.org/pdf/2408.15099). Критика
   score-функций и метод SFL. **Главная статья.**
5. Coward et al., 2024 — *JaxUED*, [arXiv:2403.13091](https://arxiv.org/pdf/2403.13091).

Свежее (необязательно, но идеи использовать можно — со ссылкой):
TRACED [arXiv:2506.19997](https://arxiv.org/pdf/2506.19997),
DEGen [arXiv:2601.14957](https://arxiv.org/pdf/2601.14957).

---

## Сетап

| поле | значение |
|---|---|
| среда | Maze 13×13, view 5, 25 стен |
| student | PPO + LSTM из JaxUED |
| бюджет | 30 000 updates |
| dev | 8 уровней из задания |
| локальная validation | 64 случайных уровня генератора, seed 10000; одинаковые для всех методов |
| финальное сравнение | 3 seeds, одинаковый бюджет |
| короткий отсев | 2500 updates, seed 0 |

## Эксперименты и идеи

| эксперимент | код | статус | запуск | результат | диагностика |
|---|---|---|---|---|---|
| 1. DR baseline | [baseline.py](baseline.py) | частичный | 8000 / 30 000, seed 0, лог | solve 0.288 на update 8000 | [ранний HTML](DIAGNOSTICS.html) |
| 2. Robust PLR с MaxMC | [baseline.py](baseline.py) | частичный | 10 500 / 30 000, seed 0, лог | solve 0.383 на update 10 500 | [ранний HTML](DIAGNOSTICS.html) |
| 3. ACCEL с MaxMC | [baseline.py](baseline.py) | полный бюджет, один seed | 30 000 updates, seed 0, команда из README; train, eval | train-final 0.300; сохранённый checkpoint 0.320. По уровням: SixteenRooms 0.86, SixteenRooms2 0.16, Labyrinth 0.82, LabyrinthFlipped 0.24, Labyrinth2 0.48, StandardMaze/2/3 0.00 | [ранний HTML](DIAGNOSTICS.html), eval-массивы |
| ACCEL с MaxMC, контроль общего training path | tmp/oel | предварительный Mac | 500 updates, 4.096 млн interactions, seed 0 | validation solve 0.361 на update 500 | [JSON и HTML](DIAGNOSTICS.html) |
| 4. ACCEL с MaxMC и buffer 500 | run_teacher_variants.sh | короткий прогон завершён | 2500 updates, seed 0, лог | solve 0.128 | — |
| 5. ACCEL с MaxMC и replay probability 0.5 | run_teacher_variants.sh | короткий прогон завершён | 2500 updates, seed 0, лог | solve 0.101 | — |
| 6. ACCEL с MaxMC и temperature 0.1 | run_teacher_variants.sh | короткий прогон завершён | 2500 updates, seed 0, лог | solve 0.041 | — |
| 7. ACCEL с MaxMC и 15 edits | run_teacher_variants.sh | короткий прогон завершён | 2500 updates, seed 0, лог | solve 0.044 | — |
| 8. ACCEL с PVL | run_teacher_variants.sh | короткий прогон завершён | 2500 updates, seed 0, лог | solve 0.037 | — |
| 9. ACCEL с learnability-score | [teacher.py](teacher.py) | короткий прогон завершён | 2500 updates, seed 0, лог | solve 0.029 против 0.175 у ACCEL с MaxMC на том же update | buffer |
| 10. SFL-like search с learnability-score и последующим MaxMC replay | [sfl.py](sfl.py) | некорректная проверка SFL | 2500 updates, seed 0, лог | — | buffer |
| 11. SFL по Algorithms 1–2: \(N=128\), \(K=64\), \(T=50\), \(\rho=0.5\), 5 эпизодов на уровень | tmp/oel | предварительный Mac | 500 updates, 4.096 млн training + 1.6 млн search interactions, seed 0 | validation solve 0.392 на update 500 | [JSON и HTML](DIAGNOSTICS.html) |
| 13. ACCEL с предиктором успеха по фиксированным признакам конфигурации лабиринта | methods.py | предварительный Mac | 500 updates, 4.096 млн interactions, seed 0 | validation solve: final 0.305, peak 0.345 | [JSON и HTML](DIAGNOSTICS.html) |
| 14. ACCEL с CNN-предиктором успеха по карте уровня | methods.py | предварительный Mac | 500 updates, 4.096 млн interactions, seed 0 | validation solve: final и peak 0.280 | [JSON и HTML](DIAGNOSTICS.html) |
| 15. ACCEL с замороженным предобученным ResNet-18 и обучаемым предиктором успеха | methods.py | предварительный Mac | 500 updates, 4.096 млн interactions, seed 0; ResNet на CPU | validation solve: final 0.325, peak 0.336 | [JSON и HTML](DIAGNOSTICS.html) |
| 16. ACCEL со score TRACED: PVL + ошибка предсказания следующего наблюдения | methods.py | предварительный Mac | 500 updates, 4.096 млн interactions, seed 0 | validation solve: final и peak 0.325 | [JSON и HTML](DIAGNOSTICS.html) |
| 17. ACCEL со score TRACED: PVL + ошибка предсказания следующего наблюдения + co-learnability | methods.py | предварительный Mac | 500 updates, 4.096 млн interactions, seed 0 | validation solve: final 0.303, peak 0.367 | [JSON и HTML](DIAGNOSTICS.html) |
| 18. A100-отсев: ACCEL с MaxMC, ACCEL с фиксированными признаками, ACCEL с CNN и SFL | tmp/oel | завершён, один seed | 5000 updates, seed 0; пять попыток на каждом held-out уровне | validation: 0.994 / 0.942 / 0.952 / 0.975; held-out mean: 0.100 / 0.150 / 0.375 / 0.125 | [динамика](DIAGNOSTICS.html), [held-out](DIAGNOSTICS.html), JSON, rollout NPZ и checkpoints |

### Почему SFL устроен сложнее, чем формула \(p(1-p)\)

[SFL, Algorithms 1–2](https://arxiv.org/pdf/2408.15099) сначала оценивает \(N\)
случайных уровней и полностью пересобирает buffer из top-\(K\) по \(p(1-p)\).
Большое \(N\) нужно потому, что по мере обучения случайный генератор всё реже попадает
на границу способностей. Buffer сохраняет уже найденные уровни, иначе дорогой поиск
пришлось бы повторять перед каждым PPO-update. В следующих \(T\) циклах доля \(\rho\)
уровней берётся равномерно из buffer, остальные уровни генерируются заново. Новые уровни
не дают curriculum замкнуться на старой границе способностей, а пересборка buffer после
\(T\) циклов удаляет уровни, которые student уже освоил. В статье \(N=5000\),
\(K=1000\), \(T=50\), \(\rho=0.5\); локальный прогон использует \(N=128\),
\(K=64\) и пять эпизодов для оценки \(p\), поэтому это проверка механизма, а не
буквальная репликация чисел статьи.

### Неочевидные решения

- Предиктор успеха обучается по свежим результатам текущего student: описание уровня
  подаётся на вход, доля успешных эпизодов служит целью, а score равен \(q(1-q)\).
  На каждом rollout выполняется один update предиктора без replay старых меток, потому что
  после обновлений student тот же уровень может решать иначе. Фиксированный вариант подаёт
  семь признаков из [levels.py](levels.py) в слой из 32 нейронов. CNN получает три канала
  13×13: стены, старт и цель; два слоя 3×3 на 16 и 32 каналов сохраняют расположение клеток.
  ResNet-18 предобучен на ImageNet и заморожен; обучается только линейная голова вероятности
  успеха. Score вычисляется до того, как предиктор увидит метку текущего rollout; затем
  выполняется один update с learning rate \(10^{-3}\). Во всех трёх вариантах student,
  PPO, buffer и mutations ACCEL одинаковы.
- TRACED не меняет генератор или мутации ACCEL. Отдельная teacher-модель предсказывает
  следующее наблюдение по текущему наблюдению и действию; score равен PVL плюс средней
  L1-ошибке этого предсказания. Co-learnability-вариант добавляет текущее среднее снижение
  difficulty к score уровней, на которых student обучался во время предыдущего replay.
  Оба коэффициента равны 1.0, как в конфигурации MiniGrid авторов. Источники:
  [статья, Sections 3.1–3.3](https://arxiv.org/pdf/2506.19997),
  [официальный код](https://github.com/Cho-Geonwoo/TRACED).

## Диагностика

- [DIAGNOSTICS.html](DIAGNOSTICS.html): единый отчёт по механике методов, A100-обучению,
  held-out уровням, zero-memory ablation и failure-context hierarchy.
- [DIAGNOSTICS.html](DIAGNOSTICS.html): цикл UED, уровни buffer и replay.
- [DIAGNOSTICS.html](DIAGNOSTICS.html): ранняя динамика DR, PLR и ACCEL.
- [DIAGNOSTICS.html](DIAGNOSTICS.html): общая динамика семи
  предварительных методов на одном validation-наборе.
- [DIAGNOSTICS.html](DIAGNOSTICS.html):
  A100-динамика четырёх методов на одном validation-наборе.
- [DIAGNOSTICS.html](DIAGNOSTICS.html):
  пять попыток на каждом held-out уровне; рядом лежат JSON и полные rollout NPZ.
- [DIAGNOSTICS.html](DIAGNOSTICS.html):
  деревья transition-embedding для окон 10–80 кадров с шагом 10, без разрезания на кластеры.
- Фиксированные признаки: сохранять семь признаков из [levels.py](levels.py), карты и
  результаты student; показывать диапазоны признаков и разные карты с одинаковыми или
  близкими векторами. Это проверяет, какие варианты карт такое представление различает.
- ACCEL с MaxMC: параллельно обучать transition predictor только для диагностики. Он не
  влияет на student, PPO, score, buffer, replay или mutations. Сохранять observations,
  actions и их вероятности, rewards, dones, позиции, hidden/cell student, состояние
  predictor, истинное и предсказанное следующее observation и ошибку. В HTML группировать
  отдельные состояния и повторяющиеся последовательности разной длины, связывая их с
  неудачными эпизодами и zero-memory ablation. Алгоритм группировки и длины
  последовательностей согласовать перед реализацией.
