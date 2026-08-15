"""Подменный `wandb`, который пишет в локальный MLflow.

jaxued зашит на wandb и патчить его репозиторий не хочется. Поэтому этот модуль
кладётся первым в `PYTHONPATH` (это делает baseline.py): upstream-код продолжает
звать `wandb.init/log/Image/Video`, а всё уезжает в `./mlruns`.

Побочный плюс: `wandb.Video` в 0.17.5 требует `moviepy.editor`, которого нет в
moviepy 2.x, и падает *до* сохранения чекпойнта. Здесь видео пишется через imageio
и любая ошибка логирования не роняет обучение.

Смотреть результаты:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os
import tempfile

import mlflow
import numpy as np

__all__ = ["init", "log", "config", "define_metric", "login", "finish", "Image", "Video"]

_ARTIFACTS = os.environ.get("SHIM_LOG_ARTIFACTS", "1") == "1"


class _Config(dict):
    """wandb.config, каким его использует jaxued: словарь + .as_dict()."""

    def as_dict(self):
        return dict(self)


config = _Config()


class Image:
    def __init__(self, data, caption=None):
        self.data, self.caption = np.asarray(data), caption


class Video:
    def __init__(self, frames, fps=4):
        self.frames, self.fps = np.asarray(frames), fps


def login(*args, **kwargs):
    return True


def define_metric(*args, **kwargs):
    return None


def init(config=None, project=None, group=None, tags=None, **kwargs):
    globals()["config"] = _Config(config or {})
    # файловый бэкенд в свежем mlflow в maintenance mode, поэтому локальный sqlite
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(project or "jaxued")
    run = mlflow.start_run(run_name=group)
    if config:
        mlflow.log_params({k: str(v)[:250] for k, v in config.items()})
    if tags:
        mlflow.set_tags({str(t): "1" for t in tags})
    return run


def finish(*args, **kwargs):
    mlflow.end_run()


def _to_uint8(a):
    if a.dtype == np.uint8:
        return a
    a = np.asarray(a, dtype=np.float32)
    if a.max() <= 1.0:
        a = a * 255
    return a.clip(0, 255).astype(np.uint8)


def _frames_to_hwc(frames):
    """(T, C, H, W) -> (T, H, W, C); (T, H, W, C) оставляем как есть."""
    if frames.ndim == 4 and frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
        return frames.transpose(0, 2, 3, 1)
    return frames


def _log_artifacts(items, step):
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as tmp:
        for key, value in items:
            name = key.replace("/", "_")
            for i, item in enumerate(value if isinstance(value, list) else [value]):
                suffix = f"_{i}" if isinstance(value, list) else ""
                if isinstance(item, Image):
                    path = os.path.join(tmp, f"{name}{suffix}_{step}.png")
                    imageio.imwrite(path, _to_uint8(item.data))
                elif isinstance(item, Video):
                    path = os.path.join(tmp, f"{name}{suffix}_{step}.gif")
                    frames = _to_uint8(_frames_to_hwc(item.frames))
                    imageio.mimsave(path, list(frames), duration=1 / max(item.fps, 1))
                else:
                    continue
                mlflow.log_artifact(path, artifact_path=key.split("/")[0])


def _as_float(value):
    """Скаляр -> float. Работает и для jax/numpy-массивов нулевой размерности."""
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if hasattr(value, "ndim") and hasattr(value, "item"):
        try:
            return float(value) if value.ndim == 0 else None
        except (TypeError, ValueError):
            return None
    return None


def log(data, step=None, **kwargs):
    if step is None:
        step = int(_as_float(data.get("num_updates", 0)) or 0)

    metrics, artifacts = {}, []
    for key, value in data.items():
        scalar = _as_float(value)
        if scalar is not None:
            metrics[key] = scalar
        elif isinstance(value, (Image, Video)) or (
            isinstance(value, list) and value and isinstance(value[0], (Image, Video))
        ):
            artifacts.append((key, value))

    if metrics:
        mlflow.log_metrics(metrics, step=step)

    if artifacts and _ARTIFACTS:
        try:
            _log_artifacts(artifacts, step)
        except Exception as exc:  # логирование не должно ронять обучение
            print(f"[wandb_shim] артефакты не записались: {exc}", flush=True)
