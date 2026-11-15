"""Evidence-grounded explanation generation (PRD 5.7).

Explanations are assembled strictly from verifiable profile matches, safety
filtering constraints, and logged bandit adaptations. No clinical or unsubstantiated
claims (e.g. 'prevents injury') are ever produced.
"""
from __future__ import annotations

from typing import Any

from . import config as C
from .data_loader import Exercise, UserProfile


def generate_explanation(
    profile: UserProfile,
    exercise: Exercise,
    content_components: dict,
    bandit_score: float | None = None,
    feedback_history: list[dict] | None = None,
    persona: Any = None,
) -> list[str]:
    reasons = []

    # 1. Goal Alignment
    if profile.goal in exercise.goal_tags:
        goal_display = profile.goal.replace("_", " ").title()
        reasons.append(f"Directly supports your stated **{goal_display}** goal.")

    # 2. Difficulty & Fitness Level match
    if exercise.difficulty == profile.fitness_level:
        reasons.append(f"Calibrated for your **{profile.fitness_level}** fitness level.")
    elif (
        C.FITNESS_LEVELS.index(exercise.difficulty)
        < C.FITNESS_LEVELS.index(profile.fitness_level)
    ):
        reasons.append(
            f"Provides a manageable {exercise.difficulty}-level foundation for your current conditioning."
        )

    # 3. Equipment Feasibility
    if exercise.equipment == ("bodyweight",):
        reasons.append("Requires **no special equipment** (bodyweight only).")
    else:
        eq_list = ", ".join(e.replace("_", " ") for e in exercise.equipment)
        reasons.append(f"Uses your confirmed equipment: **{eq_list}**.")

    # 4. Injury and Safe Area Verification
    if profile.restrictions:
        safe_from = ", ".join(r.replace("_", " ") for r in profile.restrictions)
        reasons.append(f"Cleared safety check: no stress flagged for restricted areas ({safe_from}).")

    # 5. Persona Cohort Alignment
    if persona is not None and content_components.get("persona_match", 0.0) >= 0.5:
        reasons.append(f"Aligns with your **{persona.name}** cohort profile.")

    # 5. Bandit Adaptation History (if interacted)
    if feedback_history:
        relevant_events = [
            ev for ev in feedback_history if ev.get("exercise_id") == exercise.exercise_id
        ]
        if relevant_events:
            last_event = relevant_events[-1]
            act = last_event.get("action")
            if act == "completion":
                reasons.append("Ranked higher based on your previous **completed workout** feedback.")
            elif act == "like":
                reasons.append("Prioritised because you previously **liked** this exercise.")
            elif act == "skip":
                reasons.append("Down-weighted due to previous skip feedback.")

    if not reasons:
        reasons.append("Eligible match based on overall profile and safety clearance.")

    return reasons
