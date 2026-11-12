"""Transparent content-based score (PRD 5.5). Operates on eligible exercises only."""
from __future__ import annotations

from . import config as C
from .data_loader import Exercise, UserProfile


def score_components(profile: UserProfile, ex: Exercise) -> dict:
    goal = 1.0 if profile.goal in ex.goal_tags else 0.0
    gap = C.FITNESS_LEVELS.index(ex.difficulty) - C.FITNESS_LEVELS.index(profile.fitness_level)
    # Matching level = 1; easier = 0.5; harder than the user's level = 0.
    diff = 1.0 if gap == 0 else (0.5 if gap == -1 else (0.25 if gap < -1 else 0.0))
    return {"goal_match": goal, "difficulty_match": diff}


def content_score(profile: UserProfile, ex: Exercise, weights: dict = None) -> tuple[float, dict]:
    w = weights or C.SCORE_WEIGHTS
    comps = score_components(profile, ex)
    total = sum(w[k] * v for k, v in comps.items()) / sum(w.values())  # in [0, 1]
    return total, comps


def rank_by_content(profile: UserProfile, eligible: list[Exercise]) -> list[tuple]:
    """Return [(exercise, score, components)] sorted by score desc, id asc (deterministic)."""
    scored = [(ex, *content_score(profile, ex)) for ex in eligible]
    return sorted(scored, key=lambda t: (-t[1], t[0].exercise_id))
