from types import SimpleNamespace

from . import SETUP, STEP

def setup_teacher(config, run_dir, save_json):
    return SETUP[config["score"]](config, run_dir, save_json)

def teacher_step(config, ctx):
    return STEP[config["score"]](ctx)

def make_ctx(**kwargs):
    return SimpleNamespace(**kwargs)
