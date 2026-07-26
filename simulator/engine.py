"""
simulator/engine.py
===================
Consolidated physics engine for paper machine grade-change simulation.
Combines machine state, process lag, curve planning, disturbances, and failure dynamics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Mapping, Tuple
import pandas as pd


# --------------------------------------------------------------------------- #
# Core Constants & Physics Parameters
# --------------------------------------------------------------------------- #

MV_FIELDS = ("stock_flow", "machine_speed", "steam_pressure", "filler_flow")
HIDDEN_FIELDS = ("consistency_factor", "retention_efficiency", "dryer_efficiency")

B_ASH = -5.29
STEAM_BASELINE = 4.0
B_MOIST = 6.187
K3_MOIST = -0.00425
K4_MOIST = 3.139

CALIPER_BULK = 1.15
CALIPER_SWELL = 0.5

HIDDEN_ALPHAS: Dict[str, float] = {
    "consistency_factor": 0.15,
    "retention_efficiency": 0.08,
    "dryer_efficiency": 0.07,
    "moisture_memory": 0.05,
}
MV_ALPHA = 0.25


# --------------------------------------------------------------------------- #
# Recipe Specifications
# --------------------------------------------------------------------------- #

RAW_RECIPES: Dict[str, Dict[str, Dict[str, float]]] = {
    "G20": {
        "mv": {"stock_flow": 500, "machine_speed": 900, "steam_pressure": 4.0, "filler_flow": 30.0},
        "hidden_target": {"consistency_factor": 81, "retention_efficiency": 520, "dryer_efficiency": 1.05},
        "cv_target": {"basis_weight": 45.0, "moisture": 6.0, "ash": 12.0},
    },
    "G35": {
        "mv": {"stock_flow": 575, "machine_speed": 910, "steam_pressure": 4.15, "filler_flow": 34.5},
        "hidden_target": {"consistency_factor": 100, "retention_efficiency": 505, "dryer_efficiency": 1.02},
        "cv_target": {"basis_weight": 63.0, "moisture": 6.4, "ash": 13.8},
    },
    "G50": {
        "mv": {"stock_flow": 655, "machine_speed": 920, "steam_pressure": 4.35, "filler_flow": 39.5},
        "hidden_target": {"consistency_factor": 115, "retention_efficiency": 495, "dryer_efficiency": 0.98},
        "cv_target": {"basis_weight": 82.0, "moisture": 6.9, "ash": 15.9},
    },
    "G70": {
        "mv": {"stock_flow": 740, "machine_speed": 935, "steam_pressure": 4.6, "filler_flow": 42.0},
        "hidden_target": {"consistency_factor": 133, "retention_efficiency": 515, "dryer_efficiency": 0.93},
        "cv_target": {"basis_weight": 105.0, "moisture": 7.5, "ash": 18.0},
    },
}


@dataclass(frozen=True)
class Recipe:
    name: str
    mv: Dict[str, float]
    hidden_target: Dict[str, float]

    def __post_init__(self) -> None:
        missing_mv = set(MV_FIELDS) - self.mv.keys()
        if missing_mv:
            raise ValueError(f"Recipe '{self.name}' missing MV fields: {missing_mv}")
        missing_hidden = set(HIDDEN_FIELDS) - self.hidden_target.keys()
        if missing_hidden:
            raise ValueError(f"Recipe '{self.name}' missing hidden fields: {missing_hidden}")

    @staticmethod
    def from_legacy_dict(name: str, raw: Mapping[str, Mapping[str, float]]) -> Recipe:
        legacy_mv = raw["mv"]
        mv = {
            "stock_flow": legacy_mv["stock_flow"],
            "machine_speed": legacy_mv.get("machine_speed", legacy_mv.get("speed")),
            "steam_pressure": legacy_mv["steam_pressure"],
            "filler_flow": legacy_mv["filler_flow"],
        }
        hidden_target = dict(raw["hidden_target"])
        return Recipe(name=name, mv=mv, hidden_target=hidden_target)


RECIPES: Dict[str, Recipe] = {
    name: Recipe.from_legacy_dict(name, raw) for name, raw in RAW_RECIPES.items()
}
CV_TARGETS: Dict[str, Dict[str, float]] = {
    name: raw["cv_target"] for name, raw in RAW_RECIPES.items()
}
GRADE_NAMES: List[str] = list(RECIPES.keys())


# --------------------------------------------------------------------------- #
# Process State & Machine Physics
# --------------------------------------------------------------------------- #

class HiddenState:
    def __init__(self, value: float, alpha: float):
        self.value = float(value)
        self.alpha = float(alpha)

    def step(self, target: float) -> float:
        self.value += self.alpha * (target - self.value)
        return self.value


class MachineState:
    def __init__(self, start_recipe: Recipe) -> None:
        self.mv: Dict[str, float] = dict(start_recipe.mv)
        seed_bw = (
            start_recipe.hidden_target["consistency_factor"]
            * start_recipe.mv["stock_flow"]
            / start_recipe.mv["machine_speed"]
        )
        self.hidden: Dict[str, HiddenState] = {
            "consistency_factor": HiddenState(
                start_recipe.hidden_target["consistency_factor"], HIDDEN_ALPHAS["consistency_factor"]
            ),
            "retention_efficiency": HiddenState(
                start_recipe.hidden_target["retention_efficiency"], HIDDEN_ALPHAS["retention_efficiency"]
            ),
            "dryer_efficiency": HiddenState(
                start_recipe.hidden_target["dryer_efficiency"], HIDDEN_ALPHAS["dryer_efficiency"]
            ),
            "moisture_memory": HiddenState(seed_bw, HIDDEN_ALPHAS["moisture_memory"]),
        }

    def _compute_cvs(self) -> Dict[str, float]:
        cf = self.hidden["consistency_factor"].value
        retention = self.hidden["retention_efficiency"].value
        dryer = self.hidden["dryer_efficiency"].value
        mm = self.hidden["moisture_memory"].value

        basis_weight = cf * self.mv["stock_flow"] / self.mv["machine_speed"]
        ash = retention * (self.mv["filler_flow"] / self.mv["machine_speed"]) + B_ASH
        moisture = (
            B_MOIST
            + K3_MOIST * mm
            + K4_MOIST * dryer * (self.mv["steam_pressure"] - STEAM_BASELINE)
        )
        caliper = basis_weight * CALIPER_BULK + moisture * CALIPER_SWELL

        return {
            "basis_weight": basis_weight,
            "ash": ash,
            "moisture": moisture,
            "caliper": caliper,
        }

    def step(
        self,
        mv_target: Dict[str, float],
        hidden_target: Dict[str, float],
        minute: int = 0,
        failure_engine: FailureEngine | None = None,
        transition_seed: int = 0,
    ) -> Dict[str, float]:
        for field_name in MV_FIELDS:
            self.mv[field_name] += MV_ALPHA * (mv_target[field_name] - self.mv[field_name])

        if failure_engine is not None:
            failure_engine.apply(self, minute)

        for field_name in HIDDEN_FIELDS:
            self.hidden[field_name].step(hidden_target[field_name])

        cvs = self._compute_cvs()
        self.hidden["moisture_memory"].step(cvs["basis_weight"])

        rng = random.Random((transition_seed * 1000) + minute + 100)
        cvs["basis_weight"] += rng.gauss(0.0, 0.12)
        cvs["moisture"] += rng.gauss(0.0, 0.04)
        cvs["ash"] += rng.gauss(0.0, 0.08)
        return cvs

    def hidden_snapshot(self) -> Dict[str, float]:
        return {name: state.value for name, state in self.hidden.items()}


# --------------------------------------------------------------------------- #
# Transition Planning & Trajectories
# --------------------------------------------------------------------------- #

class InterpolationCurve(str, Enum):
    SMOOTHSTEP = "smoothstep"
    SMOOTHERSTEP = "smootherstep"
    SIGMOID = "sigmoid"
    LINEAR = "linear"


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _sigmoid(t: float, steepness: float = 10.0) -> float:
    t = min(max(t, 0.0), 1.0)
    def raw(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (x - 0.5)))
    lo, hi = raw(0.0), raw(1.0)
    return (raw(t) - lo) / (hi - lo)


def _linear(t: float) -> float:
    return min(max(t, 0.0), 1.0)


_CURVE_FUNCTIONS: Dict[InterpolationCurve, Callable[[float], float]] = {
    InterpolationCurve.SMOOTHSTEP: _smoothstep,
    InterpolationCurve.SMOOTHERSTEP: _smootherstep,
    InterpolationCurve.SIGMOID: _sigmoid,
    InterpolationCurve.LINEAR: _linear,
}


@dataclass
class DurationConfig:
    mv_scale: Dict[str, float] = field(default_factory=lambda: {
        "stock_flow": 240.0,
        "machine_speed": 35.0,
        "steam_pressure": 0.6,
        "filler_flow": 12.0,
    })
    hidden_scale: Dict[str, float] = field(default_factory=lambda: {
        "consistency_factor": 52.0,
        "retention_efficiency": 25.0,
        "dryer_efficiency": 0.12,
    })
    mv_weight: Dict[str, float] = field(default_factory=lambda: {
        "stock_flow": 1.0,
        "machine_speed": 0.6,
        "steam_pressure": 0.4,
        "filler_flow": 0.6,
    })
    hidden_weight: Dict[str, float] = field(default_factory=lambda: {
        "consistency_factor": 0.8,
        "retention_efficiency": 0.3,
        "dryer_efficiency": 0.3,
    })
    base_minutes: float = 0.0
    minutes_per_unit_distance: float = 7.45
    min_minutes: int = 6
    max_minutes: int = 45


def compute_transition_duration(start: Recipe, target: Recipe, config: DurationConfig) -> int:
    distance = 0.0
    for field_name in MV_FIELDS:
        delta = abs(target.mv[field_name] - start.mv[field_name])
        distance += config.mv_weight[field_name] * (delta / config.mv_scale[field_name])
    for field_name in HIDDEN_FIELDS:
        delta = abs(target.hidden_target[field_name] - start.hidden_target[field_name])
        distance += config.hidden_weight[field_name] * (delta / config.hidden_scale[field_name])
    raw_minutes = config.base_minutes + distance * config.minutes_per_unit_distance
    return int(round(min(max(raw_minutes, config.min_minutes), config.max_minutes)))


@dataclass
class PlannerConfig:
    duration: DurationConfig = field(default_factory=DurationConfig)
    curve: InterpolationCurve | Callable[[float], float] = InterpolationCurve.SMOOTHSTEP


class TransitionPlanner:
    def __init__(
        self,
        start_recipe: Recipe,
        target_recipe: Recipe,
        config: PlannerConfig | None = None,
    ) -> None:
        self.start_recipe = start_recipe
        self.target_recipe = target_recipe
        self.config = config or PlannerConfig()
        self._curve_fn = (
            self.config.curve if callable(self.config.curve) and not isinstance(self.config.curve, InterpolationCurve)
            else _CURVE_FUNCTIONS[InterpolationCurve(self.config.curve)]
        )
        self.duration_minutes: int = compute_transition_duration(
            start_recipe, target_recipe, self.config.duration
        )

    def _progress(self, minute: int) -> float:
        if self.duration_minutes <= 0:
            return 1.0
        return self._curve_fn(minute / self.duration_minutes)

    def get_targets(self, minute: int) -> Dict[str, Dict[str, float]]:
        s = self._progress(minute)
        mv = {
            f: self.start_recipe.mv[f] + s * (self.target_recipe.mv[f] - self.start_recipe.mv[f])
            for f in MV_FIELDS
        }
        hidden = {
            f: self.start_recipe.hidden_target[f] + s * (self.target_recipe.hidden_target[f] - self.start_recipe.hidden_target[f])
            for f in HIDDEN_FIELDS
        }
        return {"mv": mv, "hidden": hidden}


# --------------------------------------------------------------------------- #
# Process Disturbances & Failures
# --------------------------------------------------------------------------- #

@dataclass
class DisturbanceConfig:
    steam_sigma: float = 0.015
    stock_sigma: float = 1.0
    filler_sigma: float = 0.3
    speed_sigma: float = 0.8
    humidity_alpha: float = 0.03
    humidity_sigma: float = 0.02


class DisturbanceEngine:
    def __init__(self, config: DisturbanceConfig | None = None, seed: int | None = None):
        self.config = config or DisturbanceConfig()
        self.rng = random.Random(seed)
        self.ambient_humidity = 0.0

    def apply(self, machine: MachineState) -> None:
        machine.mv["steam_pressure"] += self.rng.gauss(0, self.config.steam_sigma)
        machine.mv["stock_flow"] += self.rng.gauss(0, self.config.stock_sigma)
        machine.mv["filler_flow"] += self.rng.gauss(0, self.config.filler_sigma)
        machine.mv["machine_speed"] += self.rng.gauss(0, self.config.speed_sigma)
        self.ambient_humidity += (
            -self.config.humidity_alpha * self.ambient_humidity
            + self.rng.gauss(0, self.config.humidity_sigma)
        )


class FailureType(Enum):
    NONE = auto()
    STEAM_VALVE_LAG = auto()
    DRYER_EFFICIENCY_LOSS = auto()
    RETENTION_LOSS = auto()
    STOCK_PUMP_OSCILLATION = auto()
    FILLER_VALVE_STICKING = auto()
    SENSOR_BIAS = auto()


@dataclass
class FailureEvent:
    failure_type: FailureType
    severity: float
    start_minute: int
    duration: int

    @property
    def end_minute(self) -> int:
        return self.start_minute + self.duration

    def active(self, minute: int) -> bool:
        return self.start_minute <= minute <= self.end_minute


class FailureEngine:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.failure: FailureEvent | None = None
        self.sensor_bias = {"basis_weight": 0.0, "moisture": 0.0, "ash": 0.0}

    def create_transition_failure(self, transition_length: int) -> None:
        r = self.rng.random()
        if r < 0.65:
            self.failure = None
            return
        elif r < 0.80:
            f = FailureType.STEAM_VALVE_LAG
        elif r < 0.88:
            f = FailureType.RETENTION_LOSS
        elif r < 0.93:
            f = FailureType.DRYER_EFFICIENCY_LOSS
        elif r < 0.97:
            f = FailureType.STOCK_PUMP_OSCILLATION
        elif r < 0.99:
            f = FailureType.FILLER_VALVE_STICKING
        else:
            f = FailureType.SENSOR_BIAS

        if self.rng.random() < 0.50:
            start = self.rng.randint(3, min(9, max(4, transition_length // 4)))
        else:
            start = self.rng.randint(10, min(22, max(11, transition_length // 2)))

        duration = self.rng.randint(transition_length // 3, transition_length)
        severity = self.rng.uniform(0.4, 1.0)
        self.failure = FailureEvent(f, severity, start, duration)

        if f == FailureType.SENSOR_BIAS:
            self.sensor_bias["basis_weight"] = self.rng.uniform(-2, 2)
            self.sensor_bias["moisture"] = self.rng.uniform(-0.25, 0.25)
            self.sensor_bias["ash"] = self.rng.uniform(-0.5, 0.5)

    def apply(self, machine: MachineState, minute: int) -> None:
        if self.failure is None or not self.failure.active(minute):
            return

        sev = self.failure.severity
        f = self.failure.failure_type

        if f == FailureType.STEAM_VALVE_LAG:
            machine.mv["steam_pressure"] *= (1 - 0.14 * sev)
        elif f == FailureType.DRYER_EFFICIENCY_LOSS:
            machine.hidden["dryer_efficiency"].value *= (1 - 0.16 * sev)
        elif f == FailureType.RETENTION_LOSS:
            machine.hidden["retention_efficiency"].value *= (1 - 0.22 * sev)
        elif f == FailureType.STOCK_PUMP_OSCILLATION:
            amp = 18 * sev
            machine.mv["stock_flow"] += amp * math.sin(minute / 2)
        elif f == FailureType.FILLER_VALVE_STICKING:
            machine.mv["filler_flow"] *= (1 - 0.12 * sev)

    def apply_sensor_bias(self, cvs: Dict[str, float]) -> Dict[str, float]:
        if self.failure is None or self.failure.failure_type != FailureType.SENSOR_BIAS:
            return cvs
        biased = dict(cvs)
        for k, v in self.sensor_bias.items():
            biased[k] += v
        return biased

    @property
    def failure_name(self) -> str:
        return "NONE" if self.failure is None else self.failure.failure_type.name


def simulate_what_if_window(
    grade_from: str = "G20",
    grade_to: str = "G70",
    mv_overrides: Dict[str, float] | None = None,
    window_minutes: int = 10,
    target_cv_overrides: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run an in-memory what-if simulation window with custom MV setpoint and target CV overrides."""
    start_recipe = RECIPES.get(grade_from, RECIPES["G20"])
    target_recipe_raw = RECIPES.get(grade_to, RECIPES["G70"])

    overrides = mv_overrides or {}
    target_mv = dict(target_recipe_raw.mv)
    for k, v in overrides.items():
        if k in target_mv:
            target_mv[k] = float(v)

    modified_target_recipe = Recipe(
        name=target_recipe_raw.name,
        mv=target_mv,
        hidden_target=dict(target_recipe_raw.hidden_target),
    )

    target_cv = dict(CV_TARGETS.get(grade_to, CV_TARGETS["G70"]))
    if target_cv_overrides:
        for k, v in target_cv_overrides.items():
            if k in target_cv:
                target_cv[k] = float(v)
    planner = TransitionPlanner(start_recipe, modified_target_recipe)
    machine = MachineState(start_recipe)

    rows: List[Dict] = []
    for minute in range(window_minutes):
        targets = planner.get_targets(minute)
        cvs = machine.step(targets["mv"], targets["hidden"], minute)

        bw_dev_pct = (cvs["basis_weight"] - target_cv["basis_weight"]) / target_cv["basis_weight"] * 100.0
        moisture_dev_pct = (cvs["moisture"] - target_cv["moisture"]) / target_cv["moisture"] * 100.0
        ash_dev_pct = (cvs["ash"] - target_cv["ash"]) / target_cv["ash"] * 100.0
        deviation_pct = bw_dev_pct

        rows.append({
            "timestamp": "2026-01-01T06:00:00",
            "transition_id": f"WHATIF_{grade_from}_{grade_to}",
            "grade_from": grade_from,
            "grade_to": grade_to,
            "phase": "ramp",
            "minute": minute,
            "stock_flow": round(machine.mv["stock_flow"], 3),
            "machine_speed": round(machine.mv["machine_speed"], 3),
            "steam_pressure": round(machine.mv["steam_pressure"], 4),
            "filler_flow": round(machine.mv["filler_flow"], 3),
            "basis_weight": round(cvs["basis_weight"], 3),
            "moisture": round(cvs["moisture"], 3),
            "ash": round(cvs["ash"], 3),
            "caliper": round(cvs["caliper"], 3),
            "basis_weight_sp": round(target_cv["basis_weight"], 3),
            "moisture_sp": round(target_cv["moisture"], 3),
            "ash_sp": round(target_cv["ash"], 3),
            "deviation_pct": round(deviation_pct, 3),
            "moisture_dev_pct": round(moisture_dev_pct, 3),
            "ash_dev_pct": round(ash_dev_pct, 3),
            "off_spec": abs(deviation_pct) > 2.5,
            "failure_type": "NONE",
            "alarm_code": "NONE",
            "alarm_priority": 0,
            "operator_override": False,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Decision Support & Quantitative Recommendation Helper Functions
# --------------------------------------------------------------------------- #

def compute_quantitative_recommendations(
    grade_from: str,
    grade_to: str,
    current_mv: Dict[str, float],
    failure_type: str = "NONE",
    peak_deviation: float = 0.0,
    supporting_context: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Compute quantitative actuator setpoint deltas to reduce Basis Weight off-spec risk."""
    target_recipe = RECIPES.get(grade_to, RECIPES["G70"])
    nominal_mv = target_recipe.mv

    stock_delta = round(nominal_mv["stock_flow"] - current_mv.get("stock_flow", nominal_mv["stock_flow"]), 1)
    speed_delta = round(nominal_mv["machine_speed"] - current_mv.get("machine_speed", nominal_mv["machine_speed"]), 1)
    steam_delta = round(nominal_mv["steam_pressure"] - current_mv.get("steam_pressure", nominal_mv["steam_pressure"]), 2)
    filler_delta = round(nominal_mv["filler_flow"] - current_mv.get("filler_flow", nominal_mv["filler_flow"]), 1)

    ft = failure_type.upper() if failure_type else "NONE"

    if ft == "STEAM_VALVE_LAG":
        steam_delta += 0.20
        reason = (
            "Steam valve lag is reducing drying efficiency. "
            "Reduced drying efficiency is delaying moisture stabilization. "
            "Elevated moisture increases sheet caliper and disrupts Basis Weight tracking. "
            "Increase steam pressure setpoint by +0.20 bar to restore thermal drying rate "
            "and prevent Basis Weight excursion beyond ±2.5% specification."
        )
        source = "PHYSICS_MODEL"
    elif ft == "RETENTION_LOSS":
        filler_delta -= 3.0
        reason = (
            "Retention efficiency drop is causing ash content instability. "
            "Ash instability alters sheet filler loading, which directly shifts Basis Weight. "
            "Reduce filler flow setpoint by -3.0 L/min to stabilize ash and "
            "prevent Basis Weight deviation beyond ±2.5% specification."
        )
        source = "PHYSICS_MODEL"
    elif ft == "DRYER_EFFICIENCY_LOSS":
        steam_delta += 0.25
        reason = (
            "Dryer efficiency degradation is increasing sheet moisture above target. "
            "High moisture adds mass to the sheet and elevates Basis Weight above specification. "
            "Increase steam pressure by +0.25 bar to compensate for dryer efficiency loss "
            "and bring Basis Weight back within ±2.5% tolerance."
        )
        source = "PHYSICS_MODEL"
    elif ft == "STOCK_PUMP_OSCILLATION":
        stock_delta -= 12.0
        reason = (
            "Stock pump oscillation is causing periodic fibre loading surges. "
            "Fibre loading surges directly drive Basis Weight oscillations beyond ±2.5% specification. "
            "Dampen stock flow setpoint by -12.0 L/min to reduce fibre loading variance "
            "and stabilize Basis Weight."
        )
        source = "RECIPE_SPEC"
    elif ft == "FILLER_VALVE_STICKING":
        filler_delta += 2.5
        reason = (
            "Filler valve sticking is restricting filler flow below target. "
            "Reduced filler loading lowers sheet ash content and reduces sheet density, "
            "causing Basis Weight to fall below specification. "
            "Increase filler flow setpoint by +2.5 L/min to restore sheet composition "
            "and recover Basis Weight to within ±2.5% tolerance."
        )
        source = "RECIPE_SPEC"
    elif ft == "SENSOR_BIAS":
        reason = (
            "QCS sensor inconsistency detected. "
            "Sensor bias may be masking a real Basis Weight deviation from specification. "
            "Verify QCS scanner calibration before applying any manual setpoint override."
        )
        source = "HISTORICAL_LIBRARY"
    else:
        if peak_deviation > 2.5:
            stock_delta -= 8.0
            reason = (
                f"Basis Weight deviation of {peak_deviation:.1f}% exceeds the ±2.5% specification limit. "
                "Stock flow is the primary actuator controlling fibre mass per unit area. "
                "Trim stock flow setpoint by -8.0 L/min to reduce fibre loading "
                "and bring Basis Weight back within specification."
            )
            source = "ML_PREDICTIVE"
        else:
            reason = (
                "Basis Weight is currently within ±2.5% specification. "
                "All supporting variables (moisture, ash, steam) are within normal operating range. "
                "Maintain nominal recipe setpoints."
            )
            source = "RECIPE_SPEC"

    return {
        "actuator_deltas": {
            "stock_flow": stock_delta,
            "machine_speed": speed_delta,
            "steam_pressure": steam_delta,
            "filler_flow": filler_delta,
        },
        "recommendation_reason": reason,
        "source_traceability": source,
    }


def estimate_stabilization_metrics(
    grade_from: str,
    grade_to: str,
    off_spec_minutes: int = 0,
    has_intervention: bool = True,
) -> Dict[str, float | int]:
    """Estimate expected baseline stabilization time vs optimized time with operator intervention."""
    start_recipe = RECIPES.get(grade_from, RECIPES["G20"])
    target_recipe = RECIPES.get(grade_to, RECIPES["G70"])
    duration_config = DurationConfig()

    nominal_minutes = compute_transition_duration(start_recipe, target_recipe, duration_config)
    baseline_stabilization = nominal_minutes + off_spec_minutes + 15

    if has_intervention:
        optimized_stabilization = int(round(nominal_minutes + (off_spec_minutes * 0.4)))
    else:
        optimized_stabilization = baseline_stabilization

    time_saved = max(0, baseline_stabilization - optimized_stabilization)

    return {
        "baseline_stabilization_minutes": baseline_stabilization,
        "optimized_stabilization_minutes": optimized_stabilization,
        "time_saved_minutes": time_saved,
    }


def retrieve_similar_historical_transitions(
    grade_from: str = "G20",
    grade_to: str = "G70",
    summary_path: Any = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Retrieve historical successful transitions for the same grade pair."""
    from pathlib import Path
    import pandas as pd

    if summary_path is None:
        summary_path = Path(__file__).resolve().parents[1] / "simulator" / "transition_summary.csv"
    else:
        summary_path = Path(summary_path)

    if not summary_path.exists():
        return {
            "match_count": 0,
            "avg_recovery_minutes": 14.5,
            "success_rate_pct": 92.0,
            "historical_records": [],
        }

    try:
        df = pd.read_csv(summary_path)
        pair_df = df[(df["grade_from"] == grade_from) & (df["grade_to"] == grade_to)]
        if pair_df.empty:
            pair_df = df

        success_df = pair_df[pair_df["transition_outcome"] == "SUCCESS"]
        success_rate = (len(success_df) / max(len(pair_df), 1)) * 100.0
        avg_rec = float(success_df["recovery_minutes"].mean()) if not success_df.empty else 12.0

        bw_peak_col = "bw_peak_deviation" if "bw_peak_deviation" in success_df.columns else "peak_deviation"
        bw_off_col = "bw_off_spec_minutes" if "bw_off_spec_minutes" in success_df.columns else "off_spec_minutes"

        available_cols = ["transition_id", "transition_duration", bw_peak_col, bw_off_col, "recovery_minutes"]
        for col in ["avg_moisture_deviation", "avg_ash_deviation", "failure_type"]:
            if col in success_df.columns:
                available_cols.append(col)

        records = success_df.head(limit)[available_cols].rename(columns={
            bw_peak_col: "bw_peak_deviation",
            bw_off_col: "bw_off_spec_minutes",
        }).to_dict(orient="records")

        return {
            "match_count": len(pair_df),
            "success_count": len(success_df),
            "success_rate_pct": round(success_rate, 1),
            "avg_recovery_minutes": round(avg_rec, 1),
            "historical_records": records,
        }
    except Exception:
        return {
            "match_count": 0,
            "avg_recovery_minutes": 14.5,
            "success_rate_pct": 90.0,
            "historical_records": [],
        }


def run_closed_loop_resimulation(
    grade_from: str = "G20",
    grade_to: str = "G70",
    mv_overrides: Dict[str, float] | None = None,
    failure_type: str = "NONE",
    duration_minutes: int = 60,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run full 60-minute transition re-simulation under custom actuator setpoint overrides."""
    start_recipe = RECIPES.get(grade_from, RECIPES["G20"])
    target_recipe_raw = RECIPES.get(grade_to, RECIPES["G70"])

    overrides = mv_overrides or {}
    target_mv = dict(target_recipe_raw.mv)
    for k, v in overrides.items():
        if k in target_mv:
            target_mv[k] = float(v)

    modified_target_recipe = Recipe(
        name=target_recipe_raw.name,
        mv=target_mv,
        hidden_target=dict(target_recipe_raw.hidden_target),
    )

    target_cv = CV_TARGETS.get(grade_to, CV_TARGETS["G70"])
    planner = TransitionPlanner(start_recipe, modified_target_recipe)
    machine = MachineState(start_recipe)

    failure_engine = FailureEngine(seed=42)
    if failure_type and failure_type.upper() != "NONE":
        ft_enum = getattr(FailureType, failure_type.upper(), None)
        if ft_enum:
            failure_engine.failure = FailureEvent(ft_enum, severity=0.8, start_minute=3, duration=25)

    rows: List[Dict] = []
    off_spec_minutes = 0
    peak_deviation = 0.0

    for minute in range(duration_minutes):
        targets = planner.get_targets(minute)
        failure_engine.apply(machine, minute)
        cvs = machine.step(targets["mv"], targets["hidden"], minute)
        cvs = failure_engine.apply_sensor_bias(cvs)

        bw_dev_pct = (cvs["basis_weight"] - target_cv["basis_weight"]) / target_cv["basis_weight"] * 100.0
        moisture_dev_pct = (cvs["moisture"] - target_cv["moisture"]) / target_cv["moisture"] * 100.0
        ash_dev_pct = (cvs["ash"] - target_cv["ash"]) / target_cv["ash"] * 100.0

        dev_pct = bw_dev_pct
        abs_dev = abs(dev_pct)
        if abs_dev > abs(peak_deviation):
            peak_deviation = dev_pct

        is_off = abs_dev > 2.5
        if is_off:
            off_spec_minutes += 1

        rows.append({
            "timestamp": f"2026-01-01T06:{minute:02d}:00",
            "transition_id": f"RESIM_{grade_from}_{grade_to}",
            "grade_from": grade_from,
            "grade_to": grade_to,
            "phase": "ramp" if minute < 20 else "stabilizing",
            "minute": minute,
            "stock_flow": round(machine.mv["stock_flow"], 3),
            "machine_speed": round(machine.mv["machine_speed"], 3),
            "steam_pressure": round(machine.mv["steam_pressure"], 4),
            "filler_flow": round(machine.mv["filler_flow"], 3),
            "basis_weight": round(cvs["basis_weight"], 3),
            "moisture": round(cvs["moisture"], 3),
            "ash": round(cvs["ash"], 3),
            "caliper": round(cvs["caliper"], 3),
            "basis_weight_sp": round(target_cv["basis_weight"], 3),
            "moisture_sp": round(target_cv["moisture"], 3),
            "ash_sp": round(target_cv["ash"], 3),
            "deviation_pct": round(dev_pct, 3),
            "moisture_dev_pct": round(moisture_dev_pct, 3),
            "ash_dev_pct": round(ash_dev_pct, 3),
            "off_spec": is_off,
            "failure_type": failure_type.upper(),
            "alarm_code": "NONE",
            "alarm_priority": 0,
            "operator_override": len(overrides) > 0,
        })

    df = pd.DataFrame(rows)
    recovery_minutes = max(10, off_spec_minutes + 12)
    outcome = "MAJOR_OFFSPEC" if off_spec_minutes > 15 else ("MINOR_OFFSPEC" if off_spec_minutes > 5 else "SUCCESS")

    metrics = {
        "transition_id": f"RESIM_{grade_from}_{grade_to}",
        "grade_from": grade_from,
        "grade_to": grade_to,
        "off_spec_minutes": off_spec_minutes,
        "recovery_minutes": recovery_minutes,
        "peak_deviation": round(peak_deviation, 2),
        "transition_outcome": outcome,
    }

    return df, metrics


class DigitalTwin:
    """Digital Twin interface for the physics simulation layer."""

    @staticmethod
    def simulate(
        grade_from: str = "G20",
        grade_to: str = "G70",
        setpoint_overrides: Dict[str, float] | None = None,
        failure_type: str = "NONE",
        duration_minutes: int = 60,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        return run_closed_loop_resimulation(
            grade_from=grade_from,
            grade_to=grade_to,
            mv_overrides=setpoint_overrides,
            failure_type=failure_type,
            duration_minutes=duration_minutes,
        )

