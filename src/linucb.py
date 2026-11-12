"""Shared-parameter LinUCB over (user, exercise) feature vectors (PRD 5.6).

It only ever ranks candidates handed to it, i.e. exercises that already passed
the safety filter; it cannot add items."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C


class LinUCB:
    def __init__(self, dim: int, alpha: float = C.LINUCB_ALPHA, ridge: float = C.LINUCB_RIDGE):
        self.dim, self.alpha, self.ridge = dim, alpha, ridge
        self.reset()

    def reset(self) -> None:
        self.A = self.ridge * np.eye(self.dim)
        self.b = np.zeros(self.dim)
        self.n_updates = 0

    def theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def ucb(self, x: np.ndarray) -> float:
        theta = self.theta()
        a_inv_x = np.linalg.solve(self.A, x)
        return float(theta @ x + self.alpha * np.sqrt(max(x @ a_inv_x, 0.0)))

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n_updates += 1

    def rank(self, candidates: list, vectors: list, priors: list = None) -> list:
        """Reorder `candidates` (never adds/removes). `priors` (content scores)
        are added to the UCB so a cold-start bandit reproduces content ranking."""
        priors = priors if priors is not None else [0.0] * len(candidates)
        scores = [p + self.ucb(x) for p, x in zip(priors, vectors)]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [(candidates[i], scores[i]) for i in order]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            "dim": self.dim, "alpha": self.alpha, "ridge": self.ridge,
            "A": self.A.tolist(), "b": self.b.tolist(), "n_updates": self.n_updates}))

    @classmethod
    def load(cls, path: Path) -> "LinUCB":
        d = json.loads(Path(path).read_text())
        m = cls(d["dim"], d["alpha"], d["ridge"])
        m.A, m.b, m.n_updates = np.array(d["A"]), np.array(d["b"]), d["n_updates"]
        return m
