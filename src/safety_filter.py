"""Hard safety / feasibility gate (PRD 5.3).

This module only implements the MECHANISM. It contains no clinical rules: the
contraindication tags come from the (team-reviewed) catalogue. Exercises that
are not reviewed, or lack contraindication metadata, are never eligible.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config as C
from .data_loader import REVIEWED, Exercise, UserProfile

EQUIPMENT_UNAVAILABLE = "EQUIPMENT_UNAVAILABLE"
CONTRAINDICATION_CONFLICT = "CONTRAINDICATION_CONFLICT"
SAFETY_METADATA_MISSING = "SAFETY_METADATA_MISSING"
NOT_REVIEWED = "NOT_REVIEWED"
UNCERTAIN_RESTRICTION = "UNCERTAIN_RESTRICTION"


@dataclass(frozen=True)
class Exclusion:
    exercise_id: str
    reason_code: str
    detail: str = ""


def check_exercise(profile: UserProfile, ex: Exercise) -> list[Exclusion]:
    """All reasons an exercise is ineligible (empty list == eligible)."""
    reasons = []
    if ex.review_status != REVIEWED:
        reasons.append(Exclusion(ex.exercise_id, NOT_REVIEWED, ex.review_status))
    if not ex.contraindication_tags:
        reasons.append(Exclusion(ex.exercise_id, SAFETY_METADATA_MISSING))
    available = set(profile.equipment) | set(C.ALWAYS_AVAILABLE_EQUIPMENT)
    missing = sorted(set(ex.equipment) - available)
    if missing:
        reasons.append(Exclusion(ex.exercise_id, EQUIPMENT_UNAVAILABLE, ",".join(missing)))
    tags = set(ex.contraindication_tags) - {C.NO_RESTRICTION}
    if C.PREFER_NOT_TO_SAY in profile.restrictions and tags:
        # Conservative: unknown restrictions -> avoid anything that carries a tag.
        reasons.append(Exclusion(ex.exercise_id, UNCERTAIN_RESTRICTION, ",".join(sorted(tags))))
    conflict = sorted(tags & set(profile.restrictions))
    if conflict:
        reasons.append(Exclusion(ex.exercise_id, CONTRAINDICATION_CONFLICT, ",".join(conflict)))
    return reasons


def filter_exercises(profile: UserProfile, exercises) -> tuple[list[Exercise], list[Exclusion]]:
    eligible, exclusions = [], []
    for ex in exercises:
        reasons = check_exercise(profile, ex)
        if reasons:
            exclusions.extend(reasons)
        else:
            eligible.append(ex)
    return eligible, exclusions


def violations(profile: UserProfile, recommended) -> list[Exclusion]:
    """Invariant check on final recommendations (also used for the
    constraint-violation-rate metric)."""
    out = []
    for ex in recommended:
        out.extend(check_exercise(profile, ex))
    return out


def assert_no_violations(profile: UserProfile, recommended) -> None:
    bad = violations(profile, recommended)
    if bad:
        raise AssertionError(f"Safety invariant violated: {bad}")
