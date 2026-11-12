"""Dataclasses, validation and CSV loading for profiles and the catalogue."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from . import config as C

REVIEWED = "reviewed"


class CatalogueError(ValueError):
    pass


@dataclass(frozen=True)
class UserProfile:
    age: int
    bmi: float
    goal: str
    fitness_level: str
    equipment: tuple = ()
    restrictions: tuple = ()


@dataclass(frozen=True)
class Exercise:
    exercise_id: str
    name: str
    muscle_groups: tuple
    equipment: tuple
    difficulty: str
    exercise_type: str
    goal_tags: tuple = ()
    # Empty tuple == MISSING safety metadata (ineligible). Reviewed exercises
    # with no restrictions must carry the explicit marker ("none",).
    contraindication_tags: tuple = ()
    instructions: str = ""
    review_status: str = "draft_unreviewed"


def validate_profile(p: dict) -> list[str]:
    """Return a list of 'field: message' errors (empty list == valid)."""
    errs = []
    age = p.get("age")
    if not isinstance(age, int) or isinstance(age, bool):
        errs.append("age: must be a whole number")
    elif not C.AGE_BOUNDS[0] <= age <= C.AGE_BOUNDS[1]:
        errs.append(f"age: must be between {C.AGE_BOUNDS[0]} and {C.AGE_BOUNDS[1]}")
    bmi = p.get("bmi")
    if not isinstance(bmi, (int, float)) or isinstance(bmi, bool):
        errs.append("bmi: must be a number")
    elif not C.BMI_BOUNDS[0] <= bmi <= C.BMI_BOUNDS[1]:
        errs.append(f"bmi: must be between {C.BMI_BOUNDS[0]} and {C.BMI_BOUNDS[1]}")
    if p.get("goal") not in C.GOALS:
        errs.append(f"goal: must be one of {list(C.GOALS)}")
    if p.get("fitness_level") not in C.FITNESS_LEVELS:
        errs.append(f"fitness_level: must be one of {list(C.FITNESS_LEVELS)}")
    equip = p.get("equipment")
    if not equip:
        errs.append("equipment: select at least one option (e.g. bodyweight)")
    else:
        bad = [e for e in equip if e not in C.EQUIPMENT]
        if bad:
            errs.append(f"equipment: unknown values {bad}")
    restr = p.get("restrictions")
    if restr is None:
        errs.append("restrictions: required (select 'none' or the areas to avoid)")
    else:
        allowed = set(C.RESTRICTION_TAGS) | {C.PREFER_NOT_TO_SAY, C.NO_RESTRICTION}
        bad = [r for r in restr if r not in allowed]
        if bad:
            errs.append(f"restrictions: unknown values {bad}")
        if C.NO_RESTRICTION in restr and len(restr) > 1:
            errs.append("restrictions: 'none' cannot be combined with other values")
    return errs


def make_profile(p: dict) -> UserProfile:
    errs = validate_profile(p)
    if errs:
        raise ValueError("; ".join(errs))
    restr = tuple(r for r in p["restrictions"] if r != C.NO_RESTRICTION)
    return UserProfile(p["age"], float(p["bmi"]), p["goal"], p["fitness_level"],
                       tuple(p["equipment"]), restr)


def _split(value) -> tuple:
    return tuple(t.strip() for t in (value or "").split("|") if t.strip())


def validate_exercise_row(row: dict) -> list[str]:
    errs = []
    eid = row.get("exercise_id", "").strip()
    if not eid:
        errs.append("exercise_id missing")
    if not row.get("name", "").strip():
        errs.append(f"{eid}: name missing")
    for col, vocab in (("muscle_groups", C.MUSCLE_GROUPS), ("equipment", C.EQUIPMENT),
                       ("goal_tags", C.GOALS)):
        vals = _split(row.get(col))
        if col != "goal_tags" and not vals:
            errs.append(f"{eid}: {col} missing")
        bad = [v for v in vals if v not in vocab]
        if bad:
            errs.append(f"{eid}: unknown {col} {bad}")
    if row.get("difficulty") not in C.FITNESS_LEVELS:
        errs.append(f"{eid}: difficulty must be one of {list(C.FITNESS_LEVELS)}")
    if row.get("exercise_type") not in C.EXERCISE_TYPES:
        errs.append(f"{eid}: exercise_type must be one of {list(C.EXERCISE_TYPES)}")
    # Missing contraindication metadata is NOT an error here (it is excluded
    # by the safety filter), but unknown tags are.
    allowed = set(C.RESTRICTION_TAGS) | {C.NO_RESTRICTION}
    bad = [t for t in _split(row.get("contraindication_tags")) if t not in allowed]
    if bad:
        errs.append(f"{eid}: unknown contraindication_tags {bad}")
    return errs


def load_catalogue(path: Path = C.CATALOGUE_PATH) -> list[Exercise]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    errs, seen, out = [], set(), []
    for row in rows:
        errs += validate_exercise_row(row)
        eid = row.get("exercise_id", "").strip()
        if eid in seen:
            errs.append(f"{eid}: duplicate exercise_id")
        seen.add(eid)
    if errs:
        raise CatalogueError("Catalogue validation failed:\n" + "\n".join(errs))
    for row in rows:
        out.append(Exercise(
            exercise_id=row["exercise_id"].strip(), name=row["name"].strip(),
            muscle_groups=_split(row["muscle_groups"]), equipment=_split(row["equipment"]),
            difficulty=row["difficulty"], exercise_type=row["exercise_type"],
            goal_tags=_split(row.get("goal_tags")),
            contraindication_tags=_split(row.get("contraindication_tags")),
            instructions=row.get("instructions", "").strip(),
            review_status=row.get("review_status", "").strip() or "draft_unreviewed"))
    return out
