"""Evaluation metrics (PRD 8.2). Conventions are stated in each docstring."""
from __future__ import annotations

import math


def precision_at_k(ranked_ids, relevant_ids, k: int) -> float:
    """Relevant in top-k divided by k. If fewer than k items are returned the
    denominator is still k (empty lists score 0)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant_ids)
    return sum(1 for i in list(ranked_ids)[:k] if i in rel) / k


def ndcg_at_k(ranked_ids, relevance: dict, k: int) -> float:
    """Graded relevance dict id->gain (binary: 0/1). Returns 0.0 if ideal DCG is 0."""
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = sum(relevance.get(i, 0.0) / math.log2(r + 2) for r, i in enumerate(list(ranked_ids)[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def constraint_violation_rate(n_violating: int, n_recommended: int) -> float:
    """Violating items / recommended items; defined as 0.0 when nothing was recommended
    (report the empty-result rate separately)."""
    return n_violating / n_recommended if n_recommended else 0.0


def cumulative_reward(rewards) -> list:
    out, total = [], 0.0
    for r in rewards:
        total += r
        out.append(total)
    return out
