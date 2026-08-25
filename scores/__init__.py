"""Один файл на score-функцию teacher-а; teacher.compute_score только выбирает нужную."""

from scores import learnability, learnability_pvl, maxmc, pvl

SCORES = {
    "MaxMC": maxmc.score,
    "pvl": pvl.score,
    "learnability": learnability.score,
    "learnability_pvl": learnability_pvl.score,
}
