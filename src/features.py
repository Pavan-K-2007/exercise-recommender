"""Context vector for LinUCB: shared user context + exercise features +
simple match features (PRD 14: exercise as candidate action)."""
from __future__ import annotations

import numpy as np

from . import config as C
from .data_loader import Exercise, UserProfile


def _onehot(value, vocab):
    return [1.0 if value == v else 0.0 for v in vocab]


def feature_vector(profile: UserProfile, ex: Exercise) -> np.ndarray:
    gap = C.FITNESS_LEVELS.index(ex.difficulty) - C.FITNESS_LEVELS.index(profile.fitness_level)
    x = (
        [1.0]  # bias
        + _onehot(profile.goal, C.GOALS)
        + _onehot(profile.fitness_level, C.FITNESS_LEVELS)
        + _onehot(ex.difficulty, C.FITNESS_LEVELS)
        + _onehot(ex.exercise_type, C.EXERCISE_TYPES)
        + [1.0 if profile.goal in ex.goal_tags else 0.0, float(gap == 0), float(gap > 0)]
    )
    return np.asarray(x, dtype=float)


FEATURE_DIM = 1 + len(C.GOALS) + 2 * len(C.FITNESS_LEVELS) + len(C.EXERCISE_TYPES) + 3
