"""Запуск фиксированных экспериментов без CLI-конфигурации."""

from .config import (
    ACCEL_CNN_SMOKE,
    ACCEL_FIXED_SMOKE,
    ACCEL_MAXMC_SMOKE,
    ACCEL_TRACED_COLEARN_SMOKE,
    ACCEL_TRACED_SMOKE,
    SFL_MAC,
    SFL_SMOKE,
)
from .training import run_accel, run_sfl


def smoke() -> None:
    run_sfl(SFL_SMOKE)
    run_accel(ACCEL_MAXMC_SMOKE)
    run_accel(ACCEL_FIXED_SMOKE)
    run_accel(ACCEL_CNN_SMOKE)
    run_accel(ACCEL_TRACED_SMOKE)
    run_accel(ACCEL_TRACED_COLEARN_SMOKE)


def mac() -> None:
    run_sfl(SFL_MAC)


if __name__ == "__main__":
    smoke()
