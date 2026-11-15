"""Transparent content-based score (PRD 5.5). Operates on eligible exercises only."""
from __future__ import annotations

from typing import Any

from . import config as C
from .data_loader import Exercise, UserProfile


def score_components(profile: UserProfile, ex: Exercise, persona: Any = None) -> dict:
    goal = 1.0 if profile.goal in ex.goal_tags else 0.0
    gap = C.FITNESS_LEVELS.index(ex.difficulty) - C.FITNESS_LEVELS.index(profile.fitness_level)
    # Matching level = 1; easier = 0.5; harder than the user's level = 0.
    diff = 1.0 if gap == 0 else (0.5 if gap == -1 else (0.25 if gap < -1 else 0.0))

    persona_score = 0.0
    if persona is not None:
        p_goal = 1.0 if getattr(persona, "dominant_goal", None) in ex.goal_tags else 0.0
        p_lvl = 1.0 if getattr(persona, "dominant_level", None) == ex.difficulty else 0.0
        persona_score = 0.5 * p_goal + 0.5 * p_lvl

    return {"goal_match": goal, "difficulty_match": diff, "persona_match": persona_score}


def content_score(
    profile: UserProfile,
    ex: Exercise,
    weights: dict = None,
    persona: Any = None,
) -> tuple[float, dict]:
    w = dict(weights or C.SCORE_WEIGHTS)
    comps = score_components(profile, ex, persona=persona)
    if persona is None:
        w.pop("persona_match", None)
    denom = sum(w.values()) or 1.0
    total = sum(w[k] * comps[k] for k in w if k in comps) / denom  # in [0, 1]
    return total, comps


def rank_by_content(
    profile: UserProfile,
    eligible: list[Exercise],
    persona: Any = None,
) -> list[tuple]:
    """Return [(exercise, score, components)] sorted by score desc, id asc (deterministic)."""
    scored = [(ex, *content_score(profile, ex, persona=persona)) for ex in eligible]
    return sorted(scored, key=lambda t: (-t[1], t[0].exercise_id))
