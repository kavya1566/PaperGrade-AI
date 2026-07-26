"""
simulator/generator.py
======================
Generates historian.csv (time-series telemetry) and transition_summary.csv
for 10,000 simulated paper machine grade transitions.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv
import os
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from simulator.engine import (
    CV_TARGETS,
    GRADE_NAMES,
    MachineState,
    PlannerConfig,
    RECIPES,
    TransitionPlanner,
    FailureEngine,
)
from simulator.assessment import (
    OFF_SPEC_THRESHOLD_PCT,
    compute_failure_severity,
    determine_operator_action_for_row,
    summarize_transition,
)

STEADY_PRE_MINUTES = 5
STABILIZE_MINUTES = 25

HISTORIAN_COLUMNS = [
    "timestamp", "transition_id", "grade_from", "grade_to", "phase", "minute",
    "stock_flow", "machine_speed", "steam_pressure", "filler_flow",
    "basis_weight", "moisture", "ash", "caliper",
    "basis_weight_sp", "moisture_sp", "ash_sp",
    "deviation_pct", "off_spec", "failure_type",
    "alarm_code", "alarm_priority", "operator_override",
    "operator_action", "failure_severity", "peak_deviation",
    "recovery_minutes", "transition_outcome", "recommendation_reason",
]

SUMMARY_COLUMNS = [
    "transition_id", "grade_from", "grade_to", "grade_distance",
    "transition_duration", "failure_type",
    "bw_off_spec_minutes", "bw_peak_deviation", "bw_avg_deviation", "recovery_minutes",
    "max_moisture", "avg_moisture_deviation", "max_ash", "avg_ash_deviation",
    "max_basis_weight", "min_basis_weight",
    "transition_outcome", "recommended_operator_action", "recommendation_reason",
]

HISTORIAN_PATH = os.path.join(os.path.dirname(__file__), "historian.csv")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "transition_summary.csv")


def stable_transition_seed(transition_id: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(transition_id))


@dataclass
class TransitionResult:
    historian_rows: List[Dict]
    next_timestamp: datetime


def simulate_transition(
    transition_id: str,
    grade_from: str,
    grade_to: str,
    start_timestamp: datetime,
    planner_config: PlannerConfig | None = None,
) -> TransitionResult:
    start_recipe = RECIPES[grade_from]
    target_recipe = RECIPES[grade_to]
    start_cv = CV_TARGETS[grade_from]
    target_cv = CV_TARGETS[grade_to]

    planner = TransitionPlanner(start_recipe, target_recipe, config=planner_config)
    machine = MachineState(start_recipe)

    t_seed = stable_transition_seed(transition_id)
    failure_engine = FailureEngine(seed=t_seed)
    failure_engine.create_transition_failure(planner.duration_minutes + STABILIZE_MINUTES)

    historian_rows: List[Dict] = []
    timestamp = start_timestamp
    minute_offset = 0

    def record(phase: str, mv_target: Dict[str, float], hidden_target: Dict[str, float], sp: Dict[str, float]) -> None:
        nonlocal timestamp, minute_offset
        cvs = machine.step(mv_target, hidden_target, minute_offset, failure_engine, transition_seed=t_seed)
        cvs = failure_engine.apply_sensor_bias(cvs)

        bw_deviation_pct = (cvs["basis_weight"] - sp["basis_weight"]) / sp["basis_weight"] * 100.0
        deviation_pct = bw_deviation_pct
        alarm_code = "NONE"
        alarm_priority = 0

        off_spec = False if phase == "ramp" else abs(bw_deviation_pct) > OFF_SPEC_THRESHOLD_PCT
        if off_spec:
            alarm_code = "BW_OFF_SPEC"
            alarm_priority = 3
        elif cvs["moisture"] > 7.8:
            alarm_code = "HIGH_MOISTURE"
            alarm_priority = 2
        elif cvs["ash"] > 18.5:
            alarm_code = "HIGH_ASH"
            alarm_priority = 2
        elif machine.mv["steam_pressure"] > 4.7:
            alarm_code = "HIGH_STEAM"
            alarm_priority = 1

        operator_override = alarm_priority >= 2 and random.random() < 0.35

        historian_rows.append({
            "timestamp": timestamp.isoformat(),
            "transition_id": transition_id,
            "grade_from": grade_from,
            "grade_to": grade_to,
            "phase": phase,
            "minute": minute_offset,
            "stock_flow": round(machine.mv["stock_flow"], 3),
            "machine_speed": round(machine.mv["machine_speed"], 3),
            "steam_pressure": round(machine.mv["steam_pressure"], 4),
            "filler_flow": round(machine.mv["filler_flow"], 3),
            "basis_weight": round(cvs["basis_weight"], 3),
            "moisture": round(cvs["moisture"], 3),
            "ash": round(cvs["ash"], 3),
            "caliper": round(cvs["caliper"], 3),
            "basis_weight_sp": round(sp["basis_weight"], 3),
            "moisture_sp": round(sp["moisture"], 3),
            "ash_sp": round(sp["ash"], 3),
            "deviation_pct": round(deviation_pct, 3),
            "off_spec": off_spec,
            "failure_type": failure_engine.failure_name,
            "alarm_code": alarm_code,
            "alarm_priority": alarm_priority,
            "operator_override": operator_override,
        })

        timestamp += timedelta(minutes=1)
        minute_offset += 1

    for _ in range(STEADY_PRE_MINUTES):
        record("steady_pre", start_recipe.mv, start_recipe.hidden_target, start_cv)

    for m in range(planner.duration_minutes + 1):
        targets = planner.get_targets(m)
        record("ramp", targets["mv"], targets["hidden"], target_cv)

    for _ in range(STABILIZE_MINUTES):
        record("stabilizing", target_recipe.mv, target_recipe.hidden_target, target_cv)

    summary = summarize_transition(historian_rows)
    failure_severity = compute_failure_severity(failure_engine.failure_name, summary["bw_off_spec_minutes"]) if summary else "LOW"

    for r in historian_rows:
        action, reason = determine_operator_action_for_row(r)
        r["operator_action"] = action
        r["failure_severity"] = failure_severity
        r["peak_deviation"] = summary["bw_peak_deviation"]
        r["recovery_minutes"] = summary["recovery_minutes"]
        r["transition_outcome"] = summary["transition_outcome"]
        r["recommendation_reason"] = reason

    return TransitionResult(historian_rows, timestamp)


def _build_transition_plan(n_transitions: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    all_pairs = [(a, b) for a in GRADE_NAMES for b in GRADE_NAMES if a != b]
    plan: List[Tuple[str, str]] = []
    current_grade = GRADE_NAMES[0]
    pairs_to_cover = all_pairs.copy()
    rng.shuffle(pairs_to_cover)

    while len(plan) < n_transitions:
        if pairs_to_cover:
            match_idx = next((i for i, (a, _) in enumerate(pairs_to_cover) if a == current_grade), None)
            if match_idx is not None:
                grade_from, grade_to = pairs_to_cover.pop(match_idx)
            else:
                grade_from = current_grade
                grade_to = rng.choice([g for g in GRADE_NAMES if g != current_grade])
        else:
            grade_from = current_grade
            grade_to = rng.choice([g for g in GRADE_NAMES if g != current_grade])

        plan.append((grade_from, grade_to))
        current_grade = grade_to

    return plan[:n_transitions]


def generate_dataset(
    n_transitions: int = 10000,
    seed: int = 42,
    start_timestamp: datetime | None = None,
    planner_config: PlannerConfig | None = None,
) -> List[Dict]:
    plan = _build_transition_plan(n_transitions, seed)
    timestamp = start_timestamp or datetime(2026, 1, 1, 6, 0, 0)
    all_historian_rows: List[Dict] = []

    for i, (grade_from, grade_to) in enumerate(plan):
        transition_id = f"T{i:04d}_{grade_from}_{grade_to}"
        result = simulate_transition(transition_id, grade_from, grade_to, timestamp, planner_config)
        all_historian_rows.extend(result.historian_rows)
        timestamp = result.next_timestamp

    return all_historian_rows


def write_csv(rows: List[Dict], columns: List[str], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate_transition_summary(input_path: str = HISTORIAN_PATH, output_path: str = SUMMARY_PATH) -> List[Dict]:
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    transitions = defaultdict(list)
    for r in rows:
        transitions[r["transition_id"]].append(r)

    summaries = []
    for tid, trows in sorted(transitions.items()):
        first = trows[0]
        grade_from = first.get("grade_from", "")
        grade_to = first.get("grade_to", "")

        m_from = re.search(r"(\d+)", grade_from or "")
        m_to = re.search(r"(\d+)", grade_to or "")
        grade_distance = abs(int(m_to.group(1)) - int(m_from.group(1))) if m_from and m_to else 0

        minutes = [int(r["minute"]) for r in trows]
        duration = max(minutes) - min(minutes)
        failure_types = [r.get("failure_type", "NONE") for r in trows if r.get("failure_type")]
        failure_type = failure_types[0] if failure_types else "NONE"

        bw_off_spec_minutes = sum(1 for r in trows if r.get("off_spec") in ("True", "true", True, 1, "1"))

        post_ramp_trows = [r for r in trows if r.get("phase") != "ramp"]
        eval_trows = post_ramp_trows if post_ramp_trows else trows
        bw_deviations = [abs(float(r.get("deviation_pct", 0) or 0)) for r in eval_trows]
        bw_peak_dev = max(bw_deviations) if bw_deviations else 0.0
        bw_avg_dev = statistics.mean(bw_deviations) if bw_deviations else 0.0

        recovery_vals = [int(r.get("recovery_minutes")) for r in trows if r.get("recovery_minutes") is not None]
        recovery_minutes = recovery_vals[0] if recovery_vals else 0

        basis = [float(r.get("basis_weight", 0) or 0) for r in trows]
        moisture_vals = [float(r.get("moisture", 0) or 0) for r in trows]
        moisture_sp_vals = [float(r.get("moisture_sp", 0) or 0) for r in trows]
        ash_vals = [float(r.get("ash", 0) or 0) for r in trows]
        ash_sp_vals = [float(r.get("ash_sp", 0) or 0) for r in trows]

        moisture_devs = [abs(m - sp) / sp * 100.0 if sp > 0 else 0.0 for m, sp in zip(moisture_vals, moisture_sp_vals)]
        ash_devs = [abs(a - sp) / sp * 100.0 if sp > 0 else 0.0 for a, sp in zip(ash_vals, ash_sp_vals)]

        if bw_off_spec_minutes <= 6 and bw_peak_dev <= 7.5:
            outcome = "SUCCESS"
        elif bw_off_spec_minutes <= 8 and bw_peak_dev <= 9.5:
            outcome = "MINOR_OFFSPEC"
        else:
            outcome = "MAJOR_OFFSPEC"

        action_map = {
            "STEAM_VALVE_LAG": ("INCREASE_STEAM", "Steam valve lag delayed moisture stabilization."),
            "RETENTION_LOSS": ("REDUCE_FILLER", "Retention efficiency dropped causing ash instability."),
            "DRYER_EFFICIENCY_LOSS": ("INCREASE_STEAM", "Dryer efficiency loss increased moisture — increase steam."),
            "STOCK_PUMP_OSCILLATION": ("REDUCE_STOCK_FLOW", "Stock pump oscillation caused basis-weight oscillations."),
            "FILLER_VALVE_STICKING": ("INCREASE_FILLER", "Filler valve sticking restricted filler flow and reduced ash response."),
            "SENSOR_BIAS": ("VERIFY_SENSOR", "Sensor bias detected. Verify instrumentation before process adjustment."),
            "NONE": ("NONE", "No corrective action required."),
        }
        action, reason = action_map.get(failure_type, ("NONE", f"Observed failure '{failure_type}'. No automated action mapped."))

        summaries.append({
            "transition_id": tid,
            "grade_from": grade_from,
            "grade_to": grade_to,
            "grade_distance": grade_distance,
            "transition_duration": duration,
            "failure_type": failure_type,
            "bw_off_spec_minutes": bw_off_spec_minutes,
            "bw_peak_deviation": round(bw_peak_dev, 3),
            "bw_avg_deviation": round(bw_avg_dev, 3),
            "recovery_minutes": recovery_minutes,
            "max_moisture": round(max(moisture_vals) if moisture_vals else 0.0, 3),
            "avg_moisture_deviation": round(statistics.mean(moisture_devs), 3) if moisture_devs else 0.0,
            "max_ash": round(max(ash_vals) if ash_vals else 0.0, 3),
            "avg_ash_deviation": round(statistics.mean(ash_devs), 3) if ash_devs else 0.0,
            "max_basis_weight": round(max(basis) if basis else 0.0, 3),
            "min_basis_weight": round(min(basis) if basis else 0.0, 3),
            "transition_outcome": outcome,
            "recommended_operator_action": action,
            "recommendation_reason": reason,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summaries)

    return summaries


if __name__ == "__main__":
    historian_rows = generate_dataset(n_transitions=10000, seed=42)
    write_csv(historian_rows, HISTORIAN_COLUMNS, HISTORIAN_PATH)
    generate_transition_summary(HISTORIAN_PATH, SUMMARY_PATH)
    print("Dataset generation complete.")
