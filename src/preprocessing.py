"""Preprocessing pipeline for user clustering and feature representations (PRD 5.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


class UserPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_cols = ["age", "bmi"]
        self.cat_cols = ["goal", "fitness_level"]
        self.encoder = OneHotEncoder(
            categories=[list(C.GOALS), list(C.FITNESS_LEVELS)],
            sparse_output=False,
            handle_unknown="ignore",
        )
        self.fitted = False
        self.feature_names: list[str] = []

    def _prepare_df(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Numeric features
        num_vals = df[self.num_cols].values.astype(float)
        
        # Categorical features
        cat_vals = df[self.cat_cols].values
        
        # Equipment multi-hot
        equip_multi = np.zeros((len(df), len(C.EQUIPMENT)), dtype=float)
        for i, eq_str in enumerate(df["equipment"]):
            tokens = set(str(eq_str).split("|"))
            for j, eq in enumerate(C.EQUIPMENT):
                if eq in tokens:
                    equip_multi[i, j] = 1.0

        return num_vals, cat_vals, equip_multi

    def fit(self, df: pd.DataFrame) -> "UserPreprocessor":
        num_vals, cat_vals, equip_multi = self._prepare_df(df)
        self.scaler.fit(num_vals)
        self.encoder.fit(cat_vals)
        
        cat_feat_names = list(self.encoder.get_feature_names_out(self.cat_cols))
        equip_feat_names = [f"equip_{e}" for e in C.EQUIPMENT]
        self.feature_names = self.num_cols + cat_feat_names + equip_feat_names
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        num_vals, cat_vals, equip_multi = self._prepare_df(df)
        scaled_num = self.scaler.transform(num_vals)
        encoded_cat = self.encoder.transform(cat_vals)
        return np.hstack([scaled_num, encoded_cat, equip_multi])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_single(self, profile_dict: dict) -> np.ndarray:
        df = pd.DataFrame([{
            "age": profile_dict["age"],
            "bmi": profile_dict["bmi"],
            "goal": profile_dict["goal"],
            "fitness_level": profile_dict["fitness_level"],
            "equipment": "|".join(profile_dict.get("equipment", ["bodyweight"])),
            "restrictions": "|".join(profile_dict.get("restrictions", ["none"])),
        }])
        return self.transform(df)[0]
