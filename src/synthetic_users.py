"""Seeded synthetic user population (PRD 8.1). Synthetic data is NOT real-world
evidence. Run:  python -m src.synthetic_users"""
from __future__ import annotations

import csv
import json

import numpy as np

from . import config as C

# Documented assumptions (illustrative, not fitted to real data).
LEVEL_PROBS = [0.40, 0.40, 0.20]
GOAL_PROBS = {  # P(goal | level), order = C.GOALS
    "beginner": [0.10, 0.15, 0.20, 0.20, 0.35],
    "intermediate": [0.25, 0.25, 0.20, 0.10, 0.20],
    "advanced": [0.35, 0.30, 0.20, 0.05, 0.10],
}
EQUIP_PROB = {  # P(owns item | level)
    "beginner": 0.15, "intermediate": 0.40, "advanced": 0.65}
P_ANY_RESTRICTION = 0.25


def generate(n: int = 500, seed: int = C.SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    owned_items = [e for e in C.EQUIPMENT if e != "bodyweight"]
    for uid in range(n):
        level = C.FITNESS_LEVELS[rng.choice(3, p=LEVEL_PROBS)]
        age = int(np.clip(rng.normal(30, 10), *C.AGE_BOUNDS))
        bmi = round(float(np.clip(rng.normal(24.5, 4.0), *C.BMI_BOUNDS)), 1)
        goal = C.GOALS[rng.choice(len(C.GOALS), p=GOAL_PROBS[level])]
        equip = ["bodyweight"] + [e for e in owned_items if rng.random() < EQUIP_PROB[level]]
        restr = []
        if rng.random() < P_ANY_RESTRICTION:
            k = int(rng.integers(1, 3))
            restr = [str(r) for r in rng.choice(C.RESTRICTION_TAGS, size=k, replace=False)]
        rows.append({"user_id": f"u{uid:04d}", "age": age, "bmi": bmi, "goal": goal,
                     "fitness_level": level, "equipment": "|".join(equip),
                     "restrictions": "|".join(restr) or C.NO_RESTRICTION})
    return rows


def write(n: int = 500, seed: int = C.SEED) -> None:
    rows = generate(n, seed)
    C.DATA_DIR.mkdir(exist_ok=True)
    with open(C.USERS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (C.DATA_DIR / "simulated_users.config.json").write_text(json.dumps({
        "n": n, "seed": seed, "level_probs": LEVEL_PROBS, "goal_probs": GOAL_PROBS,
        "equip_prob": EQUIP_PROB, "p_any_restriction": P_ANY_RESTRICTION,
        "note": "Synthetic; illustrative assumptions only."}, indent=2))


if __name__ == "__main__":
    write()
    print(f"Wrote {C.USERS_PATH}")
