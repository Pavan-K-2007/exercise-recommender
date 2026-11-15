"""Local persistence and structured event logging (PRD 5.6, 7.4)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from . import config as C
from .linucb import LinUCB


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_or_init_bandit(
    dim: int,
    alpha: float = C.LINUCB_ALPHA,
    path: Path = C.BANDIT_STATE_PATH,
) -> LinUCB:
    if path.exists():
        try:
            loaded = LinUCB.load(path)
            if loaded.dim == dim:
                return loaded
        except Exception:
            pass
    return LinUCB(dim, alpha=alpha)


def log_feedback_event(
    exercise_id: str,
    exercise_name: str,
    action: str,
    reward: float,
    session_id: str = "local_session",
    cluster_id: int | None = None,
    path: Path = C.FEEDBACK_LOG_PATH,
) -> dict[str, Any]:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "exercise_id": exercise_id,
        "exercise_name": exercise_name,
        "action": action,
        "reward": float(reward),
        "cluster_id": cluster_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return event


def load_feedback_log(path: Path = C.FEEDBACK_LOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def log_experiment_run(
    seeds: list[int],
    best_k: int,
    silhouette: float,
    summary_rows: list[dict[str, Any]],
    path: Path = C.EXPERIMENT_LOG_PATH,
) -> dict[str, Any]:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "catalogue_hash": file_sha256(C.CATALOGUE_PATH),
        "users_hash": file_sha256(C.USERS_PATH),
        "seeds": list(seeds),
        "k_range": [C.K_RANGE.start, C.K_RANGE.stop - 1],
        "selected_k": int(best_k),
        "silhouette_score": float(silhouette),
        "score_weights": dict(C.SCORE_WEIGHTS),
        "rewards": dict(C.REWARDS),
        "linucb_alpha": C.LINUCB_ALPHA,
        "linucb_ridge": C.LINUCB_RIDGE,
        "results": summary_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def clear_local_state(
    bandit_path: Path = C.BANDIT_STATE_PATH,
    feedback_path: Path = C.FEEDBACK_LOG_PATH,
) -> None:
    for p in (bandit_path, feedback_path):
        if p.exists():
            p.unlink()
