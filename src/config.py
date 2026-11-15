"""Central configuration. All values here are PROVISIONAL defaults from PRD
Section 14 and must be signed off by the team (Phase 0). Nothing here is
clinically validated."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CATALOGUE_PATH = DATA_DIR / "exercises.csv"
USERS_PATH = DATA_DIR / "simulated_users.csv"
BANDIT_STATE_PATH = DATA_DIR / "bandit_state.json"
FEEDBACK_LOG_PATH = DATA_DIR / "feedback_log.jsonl"
EXPERIMENT_LOG_PATH = DATA_DIR / "experiment_runs.jsonl"

SEED = 42
EVAL_SEEDS = (42, 101, 202, 303, 404)

# --- Controlled vocabularies (PRD 5.1, 5.2, 14) ---
GOALS = ("strength", "muscle_gain", "endurance", "flexibility_mobility", "general_fitness")
FITNESS_LEVELS = ("beginner", "intermediate", "advanced")  # ordinal order
EQUIPMENT = ("bodyweight", "dumbbells", "barbell", "resistance_band",
             "pull_up_bar", "kettlebell", "bench", "yoga_mat")
EXERCISE_TYPES = ("strength", "cardio", "mobility", "flexibility", "balance")
MUSCLE_GROUPS = ("chest", "back", "shoulders", "biceps", "triceps", "core",
                 "quadriceps", "hamstrings", "glutes", "calves", "full_body")
# Body-area restriction tags. PROVISIONAL: needs review by a qualified source.
RESTRICTION_TAGS = ("knee", "shoulder", "lower_back", "wrist", "ankle", "neck", "hip", "elbow")
PREFER_NOT_TO_SAY = "prefer_not_to_say"
NO_RESTRICTION = "none"  # explicit "reviewed: no restriction tags" marker in the catalogue

# Bodyweight needs no equipment, so it is always implicitly available.
ALWAYS_AVAILABLE_EQUIPMENT = ("bodyweight",)

# --- Profile validation bounds (configurable) ---
AGE_BOUNDS = (14, 90)
BMI_BOUNDS = (12.0, 60.0)

# --- Content score weights (must sum to 1; PRD 5.5) ---
SCORE_WEIGHTS = {"goal_match": 0.55, "difficulty_match": 0.35, "persona_match": 0.10}

# --- Reward mapping: a design hypothesis, NOT from the abstract (PRD 5.6) ---
REWARDS = {"completion": 1.0, "like": 0.5, "skip": -0.25}

# --- LinUCB ---
LINUCB_ALPHA = 0.5
LINUCB_RIDGE = 1.0

# --- Clustering ---
K_RANGE = range(2, 9)
MAX_CLUSTERS = max(K_RANGE)
