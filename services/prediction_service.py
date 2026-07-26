"""
services/prediction_service.py
===============================
Prediction Service for Paper Grade AI Decision Intelligence Layer.
Responsible exclusively for model artifact loading, feature alignment,
and predicting Basis Weight off-spec risk probability.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
CATEGORICAL_COLUMNS: List[str] = ["grade_from", "grade_to"]


class PredictionService:
    """Service responsible only for predicting Basis Weight off-spec risk."""

    def __init__(self, artifacts_dir: Union[str, Path] = ARTIFACTS_DIR) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model: Any = None
        self.encoder: Any = None
        self.feature_order: List[str] = []
        self.threshold: float = 0.5
        self.feature_count: int = 0
        self.load_artifacts()

    def load_artifacts(self) -> None:
        model_path = self.artifacts_dir / "offspec_model.pkl"
        encoder_path = self.artifacts_dir / "encoders.pkl"
        feature_order_path = self.artifacts_dir / "feature_order.json"
        model_meta_path = self.artifacts_dir / "model_meta.json"

        if model_path.is_file():
            self.model = joblib.load(model_path)
        if encoder_path.is_file():
            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)
        if feature_order_path.is_file():
            with open(feature_order_path, "r", encoding="utf-8") as f:
                self.feature_order = json.load(f)
        if model_meta_path.is_file():
            with open(model_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.threshold = float(meta.get("optimal_threshold", 0.5))
                self.feature_count = int(meta.get("feature_count", len(self.feature_order)))

    def preprocess_features(self, df: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(df, dict):
            processed_df = pd.DataFrame([df])
        elif isinstance(df, list):
            processed_df = pd.DataFrame(df)
        elif isinstance(df, pd.DataFrame):
            processed_df = df.copy()
        elif hasattr(df, "to_dict"):
            processed_df = pd.DataFrame([df.to_dict()])
        else:
            processed_df = pd.DataFrame([dict(df)])

        if self.encoder is not None:
            cat_cols = [c for c in CATEGORICAL_COLUMNS if c in processed_df.columns]
            if len(cat_cols) == len(CATEGORICAL_COLUMNS):
                processed_df[cat_cols] = self.encoder.transform(processed_df[cat_cols].astype(str))
            else:
                for col in cat_cols:
                    if col in processed_df.columns:
                        processed_df[col] = 0

        if self.feature_order:
            missing_cols = [col for col in self.feature_order if col not in processed_df.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0.0, index=processed_df.index, columns=missing_cols)
                processed_df = pd.concat([processed_df, missing_df], axis=1)
            processed_df = processed_df[self.feature_order].copy()

        return processed_df

    def predict(self, df: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(df, dict):
            df_input = pd.DataFrame([df])
        elif isinstance(df, pd.DataFrame):
            df_input = df
            if len(df_input) != 1:
                return self.predict_batch(df_input)
        elif hasattr(df, "to_dict"):
            df_input = pd.DataFrame([df.to_dict()])
        else:
            df_input = pd.DataFrame([dict(df)])

        batch_result = self.predict_batch(df_input)
        prob = batch_result["probabilities"][0]
        horizon_minutes = max(2, int(round((1.0 - prob) * 20.0))) if prob >= self.threshold else 0

        return {
            "prediction": batch_result["predictions"][0],
            "risk_probability": prob,
            "threshold": round(self.threshold, 4),
            "confidence": batch_result["confidences"][0],
            "risk_level": batch_result["risk_levels"][0],
            "prediction_horizon_minutes": horizon_minutes,
        }

    def predict_batch(self, df: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        processed_df = self.preprocess_features(df)
        if self.model is None:
            n = len(processed_df)
            return {
                "probabilities": [0.5] * n,
                "predictions": ["SAFE"] * n,
                "confidences": ["LOW"] * n,
                "risk_levels": ["LOW"] * n,
                "threshold": self.threshold,
            }

        probabilities: np.ndarray = self.model.predict_proba(processed_df)[:, 1]
        is_offspec = probabilities >= self.threshold
        predictions = np.where(is_offspec, "BW_OFFSPEC_RISK", "BW_WITHIN_SPEC").tolist()

        conf_conds = [(probabilities < 0.40), (probabilities >= 0.40) & (probabilities < 0.70), (probabilities >= 0.70)]
        conf_choices = ["LOW", "MEDIUM", "HIGH"]
        confidences = np.select(conf_conds, conf_choices, default="LOW").tolist()

        risk_conds = [(probabilities < 0.40), (probabilities >= 0.40) & (probabilities < 0.70), (probabilities >= 0.70)]
        risk_choices = ["LOW", "MEDIUM", "HIGH"]
        risk_levels = np.select(risk_conds, risk_choices, default="LOW").tolist()

        return {
            "probabilities": np.round(probabilities.astype(float), 4).tolist(),
            "predictions": predictions,
            "confidences": confidences,
            "risk_levels": risk_levels,
            "threshold": self.threshold,
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model is not None,
            "encoder_loaded": self.encoder is not None,
            "feature_count": len(self.feature_order),
            "threshold": self.threshold,
        }
