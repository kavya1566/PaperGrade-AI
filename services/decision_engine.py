"""
services/decision_engine.py
============================
Central Decision Engine Orchestrator for Paper Grade AI.
Acts as the core intelligence coordinator:
  1. Receives current process telemetry.
  2. Invokes PredictionService to estimate Basis Weight off-spec risk.
  3. Invokes ExplanationService to determine why the risk exists.
  4. Invokes RecommendationService to generate corrective actions.
  5. Invokes DigitalTwin to validate proposed actions via closed-loop simulation.
  6. Invokes HistoricalIntelligenceService to retrieve similar transitions & evidence.
  7. Aggregates all into a unified DecisionContext for UI rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from simulator.engine import DigitalTwin
from services.prediction_service import PredictionService
from services.explanation_service import ExplanationService
from services.recommendation_service import RecommendationService
from services.history_service import HistoricalIntelligenceService


@dataclass
class DecisionContext:
    """Unified data package carrying complete decision intelligence outputs."""
    telemetry: Dict[str, Any] = field(default_factory=dict)
    prediction: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)
    recommendation: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    historical_evidence: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision package to dictionary representation for backward compatibility."""
        res = {}
        res.update(self.prediction)
        res.update(self.explanation)
        res.update(self.recommendation)
        res["validation"] = self.validation
        res["historical_evidence"] = self.historical_evidence
        return res


class DecisionEngine:
    """Central Decision Coordinator for Industrial AI Decision Support Platform."""

    def __init__(
        self,
        prediction_service: Optional[PredictionService] = None,
        explanation_service: Optional[ExplanationService] = None,
        recommendation_service: Optional[RecommendationService] = None,
        history_service: Optional[HistoricalIntelligenceService] = None,
    ) -> None:
        self.prediction_service = prediction_service or PredictionService()
        self.explanation_service = explanation_service or ExplanationService()
        self.recommendation_service = recommendation_service or RecommendationService()
        self.history_service = history_service or HistoricalIntelligenceService()
        self.digital_twin = DigitalTwin()

    def evaluate_decision_package(
        self,
        df_row: Union[pd.DataFrame, Dict[str, Any]],
        grade_from: str = "G20",
        grade_to: str = "G70",
        failure_type: str = "NONE",
        peak_dev: float = 0.0,
        baseline_overrides: Optional[Dict[str, float]] = None,
        tuned_overrides: Optional[Dict[str, float]] = None,
    ) -> DecisionContext:
        """Execute complete multi-service intelligence workflow and assemble unified DecisionContext."""
        # Extract telemetry dictionary
        if isinstance(df_row, dict):
            telemetry = dict(df_row)
        elif isinstance(df_row, pd.DataFrame) and not df_row.empty:
            row = df_row.iloc[0]
            telemetry = dict(row) if hasattr(row, "to_dict") else dict(row)
        else:
            telemetry = dict(df_row) if hasattr(df_row, "to_dict") else {}

        current_mvs = {
            "stock_flow": float(telemetry.get("stock_flow_mean", telemetry.get("stock_flow", 600.0))),
            "machine_speed": float(telemetry.get("machine_speed_mean", telemetry.get("machine_speed", 910.0))),
            "steam_pressure": float(telemetry.get("steam_pressure_mean", telemetry.get("steam_pressure", 4.2))),
            "filler_flow": float(telemetry.get("filler_flow_mean", telemetry.get("filler_flow", 34.0))),
        }

        # 1. Prediction
        pred = self.prediction_service.predict(df_row)

        # 2. Explanation
        exp = self.explanation_service.explain_instance(df_row, self.prediction_service)

        # 3. Recommendation
        rec = self.recommendation_service.generate_recommendation(
            grade_from=grade_from,
            grade_to=grade_to,
            current_mv=current_mvs,
            failure_type=failure_type,
            peak_dev=peak_dev,
            risk_probability=pred.get("risk_probability", 0.5),
            threshold=pred.get("threshold", 0.5),
        )

        # 4. Digital Twin validation
        overrides = tuned_overrides or rec.get("actuator_deltas", {})
        val_df, val_metrics = self.digital_twin.simulate(
            grade_from=grade_from,
            grade_to=grade_to,
            setpoint_overrides=overrides,
            failure_type=failure_type,
        )
        validation_results = {
            "validation_df": val_df,
            "validation_metrics": val_metrics,
            "tuned_overrides": overrides,
        }

        # 5. Historical evidence
        hist_evidence = self.history_service.search_similar_transitions(
            grade_from=grade_from,
            grade_to=grade_to,
            failure_type=failure_type,
            top_k=5,
        )

        return DecisionContext(
            telemetry=telemetry,
            prediction=pred,
            explanation=exp,
            recommendation=rec,
            validation=validation_results,
            historical_evidence=hist_evidence,
        )

    def get_operator_feedback_analytics(self) -> Dict[str, Any]:
        """Compute human-in-the-loop metrics and traceability metadata."""
        from pathlib import Path
        import json
        feedback_path = Path(__file__).resolve().parents[1] / "simulator" / "operator_feedback.json"
        records = []
        if feedback_path.exists():
            try:
                with open(feedback_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = []
        accepted = sum(1 for r in records if r.get("action_taken") in ("ACCEPT", "APPLY_DCS"))
        rejected = sum(1 for r in records if r.get("action_taken") in ("REJECT", "DISCARD", "CANCEL"))
        total = accepted + rejected
        rate = (accepted / total * 100.0) if total > 0 else 100.0
        return {
            "total_accepted": accepted,
            "total_rejected": rejected,
            "total_feedback_count": total,
            "acceptance_rate_pct": round(rate, 1),
            "traceability_badge": "LINKED TO DCS RETRAINING PIPELINE",
        }
