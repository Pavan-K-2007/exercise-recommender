"""User clustering and persona generation using K-Means and PCA (PRD 5.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from . import config as C
from .preprocessing import UserPreprocessor


@dataclass
class Persona:
    cluster_id: int
    name: str
    dominant_goal: str
    dominant_level: str
    avg_age: float
    avg_bmi: float
    common_equipment: list[str]
    description: str


class PersonaClusterer:
    def __init__(self, seed: int = C.SEED):
        self.seed = seed
        self.preprocessor = UserPreprocessor()
        self.k_range = list(C.K_RANGE)
        self.best_k: int = 4
        self.best_silhouette: float = 0.0
        self.silhouette_scores: dict[int, float] = {}
        self.kmeans: KMeans | None = None
        self.pca: PCA | None = None
        self.personas: dict[int, Persona] = {}
        self.fitted_data_2d: np.ndarray | None = None
        self.fitted_labels: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "PersonaClusterer":
        X = self.preprocessor.fit_transform(df)

        best_score = -1.0
        best_k = self.k_range[0]
        scores = {}

        for k in self.k_range:
            if k >= len(df):
                continue
            km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            labels = km.fit_predict(X)
            score = float(silhouette_score(X, labels))
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        self.best_k = best_k
        self.best_silhouette = best_score
        self.silhouette_scores = scores

        self.kmeans = KMeans(n_clusters=self.best_k, random_state=self.seed, n_init=10)
        self.fitted_labels = self.kmeans.fit_predict(X)

        self.pca = PCA(n_components=2, random_state=self.seed)
        self.fitted_data_2d = self.pca.fit_transform(X)

        # Build persona summaries
        self._build_personas(df, self.fitted_labels)
        return self

    def _build_personas(self, df: pd.DataFrame, labels: np.ndarray) -> None:
        df_copy = df.copy()
        df_copy["cluster"] = labels
        self.personas = {}

        for cid in range(self.best_k):
            sub = df_copy[df_copy["cluster"] == cid]
            if len(sub) == 0:
                continue
            dom_goal = sub["goal"].mode().iloc[0] if not sub["goal"].empty else "general_fitness"
            dom_level = sub["fitness_level"].mode().iloc[0] if not sub["fitness_level"].empty else "beginner"
            avg_age = round(float(sub["age"].mean()), 1)
            avg_bmi = round(float(sub["bmi"].mean()), 1)

            # Common equipment
            all_eq = [eq for eq_str in sub["equipment"] for eq in str(eq_str).split("|")]
            eq_series = pd.Series(all_eq).value_counts()
            top_eq = eq_series.head(3).index.tolist()

            # Readable, respectful persona label
            name = f"{dom_level.capitalize()} {dom_goal.replace('_', ' ').capitalize()} Cohort"
            desc = (
                f"Cluster {cid}: Typically {dom_level}s focused on {dom_goal.replace('_', ' ')}. "
                f"Average age ~{avg_age}, average BMI ~{avg_bmi}. Commonly use: {', '.join(top_eq)}."
            )

            self.personas[cid] = Persona(
                cluster_id=cid,
                name=name,
                dominant_goal=dom_goal,
                dominant_level=dom_level,
                avg_age=avg_age,
                avg_bmi=avg_bmi,
                common_equipment=top_eq,
                description=desc,
            )

    def predict_cluster(self, profile_dict: dict) -> int:
        if self.kmeans is None:
            raise RuntimeError("Clusterer not fitted")
        x = self.preprocessor.transform_single(profile_dict).reshape(1, -1)
        return int(self.kmeans.predict(x)[0])

    def transform_pca(self, profile_dict: dict) -> tuple[float, float]:
        if self.pca is None:
            raise RuntimeError("Clusterer not fitted")
        x = self.preprocessor.transform_single(profile_dict).reshape(1, -1)
        coords = self.pca.transform(x)[0]
        return float(coords[0]), float(coords[1])
