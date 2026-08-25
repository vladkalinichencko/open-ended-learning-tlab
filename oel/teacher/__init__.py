from . import learnability, maxmc, resnet, traced

SETUP = {
    "fixed": learnability.setup,
    "cnn": learnability.setup,
    "resnet": resnet.setup,
    "maxmc": maxmc.setup,
    "traced": traced.setup,
    "traced_colearn": traced.setup,
}

STEP = {
    "fixed": learnability.step,
    "cnn": learnability.step,
    "resnet": resnet.step,
    "maxmc": maxmc.step,
    "traced": traced.step,
    "traced_colearn": traced.step,
}
