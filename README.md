# UED: свой teacher для JaxUED-лабиринтов

Тестовое задание T-LAB, Open-Ended Learning. Условие — [NOTES.md](NOTES.md),
конвенции репозитория — [AGENTS.md](AGENTS.md).

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

На GPU-машине jax ставится отдельно: `pip install "jax[cuda12]==0.4.30"`.

## Baselines

```bash
python baseline.py --method dr    --seed 0
python baseline.py --method plr   --seed 0
python baseline.py --method accel --seed 0
```

Метрики пишутся в локальный MLflow (`sqlite:///mlflow.db`) через `wandb_shim/` —
wandb не нужен. Смотреть: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Чекпойнты — `checkpoints/<method>/<seed>/`, полный прогон ≈ час на A100.

## Eval

```bash
python baseline.py --method accel --seed 0 --mode eval
python solve_rate.py results/dr results/plr results/accel --plot
```

Черновые методы из `tmp/oel` сохраняют только policy. Перед штатным JaxUED eval
экспортируйте её в ожидаемую структуру:

```bash
python export_checkpoint.py runs/<run> checkpoints/<run>/<seed>
python baseline.py --method accel --run-name <run> --seed <seed> --mode eval
```
