"""Evaluation module: Baselines, Ablations, and Metric Benchmark Runner (PRD 8)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import config as C
from .content_ranker import content_score, rank_by_content
from .data_loader import Exercise, UserProfile, load_catalogue, make_profile
from .features import FEATURE_DIM, feature_vector
from .linucb import LinUCB
from .metrics import constraint_violation_rate, cumulative_reward, ndcg_at_k, precision_at_k
from .safety_filter import filter_exercises, violations


def synthetic_relevance(profile: UserProfile, exercise: Exercise) -> float:
    """Documented synthetic relevance function for offline evaluation (PRD 8.2).
    
    Returns a graded score in [0.0, 1.0]:
    - Primary signal: goal match (+0.6)
    - Secondary signal: level calibration (+0.4)
    """
    rel = 0.0
    if profile.goal in exercise.goal_tags:
        rel += 0.6
    diff_idx_ex = C.FITNESS_LEVELS.index(exercise.difficulty)
    diff_idx_usr = C.FITNESS_LEVELS.index(profile.fitness_level)
    gap = abs(diff_idx_ex - diff_idx_usr)
    if gap == 0:
        rel += 0.4
    elif gap == 1:
        rel += 0.2
    return rel


@dataclass
class EvalSummary:
    method: str
    precision_at_5: float
    ndcg_at_5: float
    violation_rate: float
    avg_cumulative_reward: float


class Evaluator:
    def __init__(self, catalogue: list[Exercise] | None = None, seed: int = C.SEED):
        self.catalogue = catalogue or load_catalogue()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def evaluate_policy(
        self,
        test_profiles: list[UserProfile],
        policy_name: str,
        k: int = 5,
        n_rounds: int = 10,
    ) -> tuple[dict, list[float]]:
        """Run policy across test profiles over n_rounds of simulated feedback interactions."""
        p_at_k_scores = []
        ndcg_scores = []
        violation_counts = 0
        total_recs = 0
        rewards_by_round = np.zeros(n_rounds)

        for prof in test_profiles:
            # 1. Mandatory Safety & Feasibility Gate
            eligible, _ = filter_exercises(prof, self.catalogue)
            if not eligible:
                continue

            # Ground truth relevance
            rel_map = {ex.exercise_id: synthetic_relevance(prof, ex) for ex in eligible}
            rel_threshold = 0.5
            relevant_ids = {eid for eid, r in rel_map.items() if r >= rel_threshold}

            # Initialize LinUCB agent for this user session if adaptive
            agent = LinUCB(FEATURE_DIM, alpha=C.LINUCB_ALPHA)
            user_rewards = []

            for round_idx in range(n_rounds):
                # Rank candidates according to method
                if policy_name == "random":
                    shuffled = list(eligible)
                    self.rng.shuffle(shuffled)
                    ranked = [(ex, 0.0) for ex in shuffled]

                elif policy_name == "popularity":
                    # Synthetic popularity: frequency of exercise_id modulo/seeded
                    sorted_by_pop = sorted(
                        eligible,
                        key=lambda ex: (len(ex.goal_tags) * 10 - len(ex.contraindication_tags)),
                        reverse=True,
                    )
                    ranked = [(ex, 0.0) for ex in sorted_by_pop]

                elif policy_name == "no_adaptation":
                    # Content-based ranking only
                    content_ranked = rank_by_content(prof, eligible)
                    ranked = [(ex, score) for ex, score, _ in content_ranked]

                elif policy_name in ("hybrid", "no_clustering"):
                    # LinUCB adaptation with content score prior
                    content_ranked = {
                        ex.exercise_id: content_score(prof, ex)[0] for ex in eligible
                    }
                    vectors = [feature_vector(prof, ex) for ex in eligible]
                    priors = [content_ranked[ex.exercise_id] for ex in eligible]
                    ranked = agent.rank(eligible, vectors, priors=priors)
                else:
                    raise ValueError(f"Unknown policy: {policy_name}")

                top_k = [ex for ex, _ in ranked[:k]]
                top_k_ids = [ex.exercise_id for ex in top_k]

                # Safety violation check
                bad = violations(prof, top_k)
                violation_counts += len(bad)
                total_recs += len(top_k)

                # Metrics at this step
                p_at_k_scores.append(precision_at_k(top_k_ids, relevant_ids, k=k))
                ndcg_scores.append(ndcg_at_k(top_k_ids, rel_map, k=k))

                # Simulate user feedback on the #1 ranked recommendation
                if top_k:
                    chosen = top_k[0]
                    chosen_rel = rel_map[chosen.exercise_id]
                    # Probabilistic feedback model based on synthetic relevance
                    if chosen_rel >= 0.8:
                        action = "completion"
                        reward = C.REWARDS["completion"]
                    elif chosen_rel >= 0.5:
                        action = "like"
                        reward = C.REWARDS["like"]
                    else:
                        action = "skip"
                        reward = C.REWARDS["skip"]

                    user_rewards.append(reward)

                    # Update bandit if policy uses adaptive learning
                    if policy_name in ("hybrid", "no_clustering"):
                        vec = feature_vector(prof, chosen)
                        agent.update(vec, reward)

            if len(user_rewards) == n_rounds:
                rewards_by_round += np.array(user_rewards)

        n_users = max(len(test_profiles), 1)
        mean_rewards_per_round = (rewards_by_round / n_users).tolist()
        cum_reward_series = cumulative_reward(mean_rewards_per_round)

        results = {
            "method": policy_name,
            "precision@k": float(np.mean(p_at_k_scores)) if p_at_k_scores else 0.0,
            "ndcg@k": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "constraint_violation_rate": constraint_violation_rate(violation_counts, total_recs),
            "final_cumulative_reward": cum_reward_series[-1] if cum_reward_series else 0.0,
        }
        return results, cum_reward_series

    def run_benchmark(
        self,
        test_profiles: list[UserProfile],
        k: int = 5,
        n_rounds: int = 10,
    ) -> tuple[pd.DataFrame, dict[str, list[float]]]:
        """Runs the standard evaluation suite across baselines and ablations (PRD 8.3, 8.4)."""
        methods = ["random", "popularity", "no_adaptation", "hybrid"]
        rows = []
        reward_curves = {}

        for m in methods:
            metrics, cum_curve = self.evaluate_policy(test_profiles, m, k=k, n_rounds=n_rounds)
            rows.append(metrics)
            reward_curves[m] = cum_curve

        df_results = pd.DataFrame(rows)
        return df_results, reward_curves
