"""Evaluation module: Baselines, Ablations, and Multi-Seed Benchmark Runner (PRD 8)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from . import config as C
from .clustering import PersonaClusterer
from .content_ranker import content_score, rank_by_content
from .data_loader import Exercise, UserProfile, load_catalogue, make_profile
from .features import FEATURE_DIM, feature_vector
from .linucb import LinUCB
from .metrics import constraint_violation_rate, cumulative_reward, ndcg_at_k, precision_at_k
from .persistence import log_experiment_run
from .safety_filter import filter_exercises, violations
from .synthetic_users import generate as generate_users


def synthetic_relevance(profile: UserProfile, exercise: Exercise) -> float:
    """Documented synthetic relevance function for offline evaluation (PRD 8.2).

    Note on circularity: offline synthetic relevance relies on goal alignment (+0.6)
    and difficulty calibration (+0.4), which overlaps with the content scorer's
    features. Disclosed in the Evaluation UI per PRD 8.2.
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


def _profile_to_dict(prof: UserProfile) -> dict:
    return {
        "age": prof.age,
        "bmi": prof.bmi,
        "goal": prof.goal,
        "fitness_level": prof.fitness_level,
        "equipment": list(prof.equipment),
        "restrictions": list(prof.restrictions) if prof.restrictions else ["none"],
    }


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
        clusterer: PersonaClusterer | None = None,
    ) -> tuple[dict, list[float]]:
        """Run policy across test profiles over n_rounds of simulated feedback interactions."""
        p_at_k_scores = []
        ndcg_scores = []
        violation_counts = 0
        total_recs = 0
        rewards_by_round = np.zeros(n_rounds)
        valid_users = 0

        use_clustering = policy_name in ("hybrid", "no_adaptation") and clusterer is not None

        for prof in test_profiles:
            # 1. Mandatory Safety & Feasibility Gate
            eligible, _ = filter_exercises(prof, self.catalogue)
            if not eligible:
                continue

            cluster_id = None
            persona = None
            if use_clustering:
                cluster_id = clusterer.predict_cluster(_profile_to_dict(prof))
                persona = clusterer.personas.get(cluster_id)

            # Ground truth relevance
            rel_map = {ex.exercise_id: synthetic_relevance(prof, ex) for ex in eligible}
            relevant_ids = {eid for eid, r in rel_map.items() if r >= 0.5}

            # Initialize LinUCB agent for this user session if adaptive
            agent = LinUCB(FEATURE_DIM, alpha=C.LINUCB_ALPHA)
            user_rewards = []

            for _ in range(n_rounds):
                if policy_name == "random":
                    shuffled = list(eligible)
                    self.rng.shuffle(shuffled)
                    ranked = [(ex, 0.0) for ex in shuffled]

                elif policy_name == "popularity":
                    sorted_by_pop = sorted(
                        eligible,
                        key=lambda ex: (len(ex.goal_tags) * 10 - len(ex.contraindication_tags)),
                        reverse=True,
                    )
                    ranked = [(ex, 0.0) for ex in sorted_by_pop]

                elif policy_name == "no_adaptation":
                    content_ranked = rank_by_content(prof, eligible, persona=persona)
                    ranked = [(ex, score) for ex, score, _ in content_ranked]

                elif policy_name == "no_clustering":
                    content_ranked = {
                        ex.exercise_id: content_score(prof, ex, persona=None)[0]
                        for ex in eligible
                    }
                    vectors = [feature_vector(prof, ex, cluster_id=None) for ex in eligible]
                    priors = [content_ranked[ex.exercise_id] for ex in eligible]
                    ranked = agent.rank(eligible, vectors, priors=priors)

                elif policy_name == "hybrid":
                    content_ranked = {
                        ex.exercise_id: content_score(prof, ex, persona=persona)[0]
                        for ex in eligible
                    }
                    vectors = [feature_vector(prof, ex, cluster_id=cluster_id) for ex in eligible]
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
                    if chosen_rel >= 0.8:
                        reward = C.REWARDS["completion"]
                    elif chosen_rel >= 0.5:
                        reward = C.REWARDS["like"]
                    else:
                        reward = C.REWARDS["skip"]

                    user_rewards.append(reward)

                    if policy_name in ("hybrid", "no_clustering"):
                        cid = cluster_id if policy_name == "hybrid" else None
                        vec = feature_vector(prof, chosen, cluster_id=cid)
                        agent.update(vec, reward)

            if len(user_rewards) == n_rounds:
                rewards_by_round += np.array(user_rewards)
                valid_users += 1

        n_users = max(valid_users, 1)
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
        clusterer: PersonaClusterer | None = None,
    ) -> tuple[pd.DataFrame, dict[str, list[float]]]:
        """Runs baselines and ablations on a single profile set (PRD 8.3, 8.4)."""
        methods = ["random", "popularity", "no_adaptation", "no_clustering", "hybrid"]
        rows = []
        reward_curves = {}

        for m in methods:
            metrics, cum_curve = self.evaluate_policy(
                test_profiles, m, k=k, n_rounds=n_rounds, clusterer=clusterer
            )
            rows.append(metrics)
            reward_curves[m] = cum_curve

        return pd.DataFrame(rows), reward_curves

    def run_multi_seed_benchmark(
        self,
        seeds: tuple[int, ...] | list[int] = C.EVAL_SEEDS,
        n_dev: int = 200,
        n_test: int = 30,
        k: int = 5,
        n_rounds: int = 10,
        log_run: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
        """Runs repeated-seed evaluation with dev/eval split and reports mean ± std (PRD 8.5)."""
        methods = ["random", "popularity", "no_adaptation", "no_clustering", "hybrid"]
        per_method_metrics: dict[str, list[dict]] = {m: [] for m in methods}
        per_method_curves: dict[str, list[list[float]]] = {m: [] for m in methods}

        last_k = 0
        last_sil = 0.0

        for s in seeds:
            dev_df = pd.DataFrame(generate_users(n=n_dev, seed=s))
            clusterer = PersonaClusterer(seed=s).fit(dev_df)
            last_k = clusterer.best_k
            last_sil = clusterer.best_silhouette

            # Separate seed offset for held-out test profiles
            test_dicts = generate_users(n=n_test, seed=s + 1000)
            test_profiles = [
                make_profile({
                    **d,
                    "equipment": d["equipment"].split("|"),
                    "restrictions": d["restrictions"].split("|"),
                })
                for d in test_dicts
            ]

            ev = Evaluator(catalogue=self.catalogue, seed=s)
            for m in methods:
                res, curve = ev.evaluate_policy(
                    test_profiles, m, k=k, n_rounds=n_rounds, clusterer=clusterer
                )
                per_method_metrics[m].append(res)
                per_method_curves[m].append(curve)

        summary_rows = []
        curve_stats: dict[str, dict[str, list[float]]] = {}

        for m in methods:
            df_m = pd.DataFrame(per_method_metrics[m])
            row = {"method": m}
            for col in ("precision@k", "ndcg@k", "constraint_violation_rate", "final_cumulative_reward"):
                row[f"{col}_mean"] = float(df_m[col].mean())
                row[f"{col}_std"] = float(df_m[col].std(ddof=0))
            summary_rows.append(row)

            arr = np.array(per_method_curves[m])
            curve_stats[m] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0, ddof=0).tolist(),
            }

        if log_run:
            log_experiment_run(list(seeds), last_k, last_sil, summary_rows)

        return pd.DataFrame(summary_rows), curve_stats
