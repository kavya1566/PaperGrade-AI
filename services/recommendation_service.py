"""
services/recommendation_service.py
==================================
Recommendation Service for Paper Grade AI Decision Intelligence Layer.
Responsible exclusively for generating quantitative setpoint deltas,
stabilization time estimates, operator action reasoning, and origin source tag traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from simulator.engine import compute_quantitative_recommendations, estimate_stabilization_metrics

PROCESS_RECIPE_LIMITS = {
    "stock_flow": (450.0, 850.0),
    "machine_speed": (850.0, 960.0),
    "steam_pressure": (3.5, 5.0),
    "filler_flow": (25.0, 50.0),
}

SOURCE_TAGS = {
    "HISTORICAL_DATA": "Historical Data",
    "RECIPE_CONSTRAINT": "Recipe Constraint",
    "DIGITAL_TWIN": "Digital Twin Simulation",
    "LAG_CORRELATION": "Lag Correlation Model",
}


@dataclass
class RecommendationResult:
    """Dataclass holding structured recommendation output with explicit source tag metadata."""
    actuator_deltas: Dict[str, float]
    recommendation_reason: str
    source_traceability: str
    source_tag: str
    baseline_stabilization_minutes: float | int
    optimized_stabilization_minutes: float | int
    time_saved_minutes: float | int
    actuator_reasons: Dict[str, str] = field(default_factory=dict)
    structured_recommendations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for dual API compatibility."""
        return {
            "actuator_deltas": self.actuator_deltas,
            "recommendation_reason": self.recommendation_reason,
            "source_traceability": self.source_traceability,
            "source_tag": self.source_tag,
            "baseline_stabilization_minutes": self.baseline_stabilization_minutes,
            "optimized_stabilization_minutes": self.optimized_stabilization_minutes,
            "time_saved_minutes": self.time_saved_minutes,
            "actuator_reasons": self.actuator_reasons,
            "structured_recommendations": self.structured_recommendations,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()


ACTUATOR_SPECIFIC_RATIONALES = {
    "stock_flow": "Reduces stock flow slope to match target basis weight ramp curve.",
    "machine_speed": "Extends sheet residence time in dryer section to halt moisture carryover.",
    "steam_pressure": "Accelerates thermal drying response to eliminate moisture lag.",
    "filler_flow": "Adjusts filler flow to stabilize sheet ash content and density.",
}

EXPECTED_IMPROVEMENTS = {
    "stock_flow": "Reduced basis weight tracking error",
    "steam_pressure": "Earlier moisture stabilization",
    "machine_speed": "Thermal lag mitigation",
    "filler_flow": "Ash & density stabilization",
}

ACTUATOR_SOURCE_TAGS = {
    "steam_pressure": "Lag Correlation Model",
    "stock_flow": "Digital Twin Simulation",
    "machine_speed": "Recipe Constraint",
    "filler_flow": "Historical Data",
}


class RecommendationService:
    """Service responsible for generating operator recommendations, setpoint adjustments, and origin tagging."""

    def generate_recommendation(
        self,
        grade_from: str = "G20",
        grade_to: str = "G70",
        current_mv: Dict[str, float] | None = None,
        failure_type: str = "NONE",
        peak_dev: float = 0.0,
        risk_probability: float = 0.5,
        threshold: float = 0.5,
        is_digital_twin_resim: bool = False,
    ) -> RecommendationResult | Dict[str, Any]:
        """Generate quantitative setpoint deltas, recipe limit adherence, and stabilization metrics."""
        mvs = current_mv or {"stock_flow": 600.0, "machine_speed": 910.0, "steam_pressure": 4.2, "filler_flow": 34.0}
        rec_info = compute_quantitative_recommendations(grade_from, grade_to, mvs, failure_type, peak_dev)

        # Apply strict process recipe limit constraints on setpoint deltas
        deltas = dict(rec_info["actuator_deltas"])
        is_clamped_by_recipe = False
        clamped_status = {}
        for param, (min_lim, max_lim) in PROCESS_RECIPE_LIMITS.items():
            if param in deltas:
                curr = mvs.get(param, (min_lim + max_lim) / 2.0)
                proposed_sp = curr + deltas[param]
                if proposed_sp < min_lim:
                    deltas[param] = round(min_lim - curr, 2)
                    is_clamped_by_recipe = True
                    clamped_status[param] = "RECIPE_CLAMPED"
                elif proposed_sp > max_lim:
                    deltas[param] = round(max_lim - curr, 2)
                    is_clamped_by_recipe = True
                    clamped_status[param] = "RECIPE_CLAMPED"
                else:
                    clamped_status[param] = "VALID"

        # Assign explicit source tag based on setpoint origin
        traceability = rec_info.get("source_traceability", "RECIPE_SPEC")
        if is_clamped_by_recipe:
            source_tag = SOURCE_TAGS["RECIPE_CONSTRAINT"]
        elif is_digital_twin_resim or traceability == "DIGITAL_TWIN":
            source_tag = SOURCE_TAGS["DIGITAL_TWIN"]
        elif traceability == "HISTORICAL_LIBRARY":
            source_tag = SOURCE_TAGS["HISTORICAL_DATA"]
        elif failure_type in ("STEAM_VALVE_LAG", "STOCK_PUMP_OSCILLATION") or traceability == "PHYSICS_MODEL":
            source_tag = SOURCE_TAGS["LAG_CORRELATION"]
        else:
            source_tag = SOURCE_TAGS["RECIPE_CONSTRAINT"]

        stab_info = estimate_stabilization_metrics(
            grade_from, grade_to, off_spec_minutes=12 if risk_probability >= threshold else 0, has_intervention=True
        )

        reason = rec_info.get("recommendation_reason", "")
        # Prevent rendering fallback text whenever Risk >= Threshold
        if risk_probability >= threshold or failure_type != "NONE":
            if not reason or "within ±2.5% specification" in reason:
                top_param = max(deltas.items(), key=lambda x: abs(x[1]))[0] if deltas else "steam_pressure"
                reason = ACTUATOR_SPECIFIC_RATIONALES.get(top_param, "Accelerates thermal drying response to eliminate moisture lag.")

        actuator_reasons = {
            param: ACTUATOR_SPECIFIC_RATIONALES.get(param, reason)
            for param in deltas
        }

        structured = {}
        for param, d_val in deltas.items():
            curr_val = mvs.get(param, 0.0)
            rec_val = round(curr_val + d_val, 3)
            param_tag = SOURCE_TAGS["RECIPE_CONSTRAINT"] if clamped_status.get(param) == "RECIPE_CLAMPED" else ACTUATOR_SOURCE_TAGS.get(param, source_tag)
            structured[param] = {
                "actuator": param.replace("_", " ").title(),
                "current_value": curr_val,
                "recommended_value": rec_val,
                "delta": d_val,
                "delta_str": f"{d_val:+.3g}",
                "expected_improvement": EXPECTED_IMPROVEMENTS.get(param, "Earlier moisture stabilization"),
                "confidence": "HIGH" if abs(d_val) > 0.1 else "MEDIUM",
                "recipe_constraint_status": clamped_status.get(param, "VALID"),
                "source_tag": param_tag,
            }

        res = RecommendationResult(
            actuator_deltas=deltas,
            recommendation_reason=reason,
            source_traceability=traceability,
            source_tag=source_tag,
            baseline_stabilization_minutes=stab_info["baseline_stabilization_minutes"],
            optimized_stabilization_minutes=stab_info["optimized_stabilization_minutes"],
            time_saved_minutes=stab_info["time_saved_minutes"],
            actuator_reasons=actuator_reasons,
            structured_recommendations=structured,
        )
        return res


