"""
simulator/assessment.py
========================
Process Assessment layer for Paper Machine transition quality evaluation.
Responsible exclusively for evaluating process quality, deviation metrics,
off-spec durations, failure severity, and outcome classifications (SUCCESS, MINOR_OFFSPEC, MAJOR_OFFSPEC).
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Tuple

OFF_SPEC_THRESHOLD_PCT: float = 2.5


def compute_failure_severity(failure_type: str, bw_off_spec_minutes: int) -> str:
    """Evaluate severity level based on failure dynamics and off-spec duration."""
    if not failure_type or failure_type.upper() == "NONE":
        return "NONE"
    if bw_off_spec_minutes > 15:
        return "CRITICAL"
    if bw_off_spec_minutes > 8:
        return "HIGH"
    if bw_off_spec_minutes > 3:
        return "MEDIUM"
    return "LOW"


def determine_operator_action_for_row(r: Dict[str, Any]) -> Tuple[str, str]:
    """Determine immediate corrective action and reasoning for a telemetry timestamp."""
    off_spec = bool(r.get("off_spec"))
    alarm_priority = int(r.get("alarm_priority") or 0)
    alarm_code = r.get("alarm_code") or "NONE"

    if not (off_spec or alarm_priority > 0 or (alarm_code and alarm_code != "NONE")):
        return "NONE", "System operating normally. No corrective action required."

    failure_type = (r.get("failure_type") or "NONE").upper()

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(r.get(key) or default)
        except Exception:
            return float(default)

    moisture = _f("moisture")
    moisture_sp = _f("moisture_sp")
    ash = _f("ash")
    ash_sp = _f("ash_sp")
    deviation_pct = abs(_f("deviation_pct"))

    if failure_type == "STEAM_VALVE_LAG":
        if moisture > moisture_sp:
            return "INCREASE_STEAM", "Moisture exceeds target during steam valve lag."
        return "NONE", "Steam valve lag detected but moisture within target. No action required."

    if failure_type == "RETENTION_LOSS":
        if ash < ash_sp * 0.95:
            return "REDUCE_FILLER", "Retention loss detected — ash below target. Reduce filler flow to stabilize retention."
        return "INCREASE_RETENTION_AID", "Retention loss detected. Increase retention aid chemical dosing."

    if failure_type == "DRYER_EFFICIENCY_LOSS":
        return "INCREASE_STEAM", "Dryer thermal efficiency dropped. Increase steam pressure setpoint."

    if failure_type == "STOCK_PUMP_OSCILLATION":
        return "REDUCE_STOCK_FLOW", "Stock pump delivery oscillating. Trim stock flow setpoint to damp basis weight variation."

    if failure_type == "FILLER_VALVE_STICKING":
        return "INCREASE_FILLER", "Filler control valve sticking below target. Increase filler flow setpoint."

    if failure_type == "SENSOR_BIAS":
        return "VERIFY_SENSOR", "Scannability bias detected on quality sensor. Cross-check with manual lab sample."

    if deviation_pct > OFF_SPEC_THRESHOLD_PCT:
        return "TRIM_RECIPE", f"Basis Weight deviation ({deviation_pct:.1f}%) exceeds specification threshold (±2.5%)."

    return "NONE", "Minor process variation detected. Automatic control loop responding."


def classify_transition_outcome(
    bw_off_spec_minutes: int,
    bw_peak_dev: float,
    recovery_minutes: int = 0
) -> str:
    """Classify overall transition outcome exclusively based on Basis Weight quality criteria."""
    if bw_off_spec_minutes <= 6 and bw_peak_dev <= 7.5:
        return "SUCCESS"
    elif bw_off_spec_minutes <= 8 and bw_peak_dev <= 9.5:
        return "MINOR_OFFSPEC"
    else:
        return "MAJOR_OFFSPEC"


def summarize_transition(historian_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary assessment metrics from a sequence of transition telemetry rows."""
    if not historian_rows:
        return {}

    first = historian_rows[0]
    grade_from = first.get("grade_from", "")
    grade_to = first.get("grade_to", "")

    m_from = re.search(r"(\d+)", grade_from or "")
    m_to = re.search(r"(\d+)", grade_to or "")
    gfrom_num = int(m_from.group(1)) if m_from else 0
    gto_num = int(m_to.group(1)) if m_to else 0
    grade_distance = abs(gto_num - gfrom_num)

    minutes = [int(r["minute"]) for r in historian_rows]
    duration = max(minutes) - min(minutes) if minutes else 0

    failure_types = [r.get("failure_type", "NONE") for r in historian_rows if r.get("failure_type")]
    failure_type = failure_types[0] if failure_types else "NONE"

    bw_off_spec_minutes = sum(1 for r in historian_rows if r.get("off_spec") in ("True", "true", True, 1, "1"))

    post_ramp_rows = [r for r in historian_rows if r.get("phase") != "ramp"]
    eval_rows = post_ramp_rows if post_ramp_rows else historian_rows
    bw_deviations = [abs(float(r.get("deviation_pct", 0) or 0)) for r in eval_rows]
    bw_peak_dev = max(bw_deviations) if bw_deviations else 0.0
    bw_avg_dev = statistics.mean(bw_deviations) if bw_deviations else 0.0

    recovery_vals = [int(r.get("recovery_minutes")) for r in historian_rows if r.get("recovery_minutes") is not None]
    recovery_minutes = recovery_vals[0] if recovery_vals else 0

    moisture_vals = [float(r.get("moisture", 0) or 0) for r in historian_rows]
    moisture_sp_vals = [float(r.get("moisture_sp", 0) or 0) for r in historian_rows]
    ash_vals = [float(r.get("ash", 0) or 0) for r in historian_rows]
    ash_sp_vals = [float(r.get("ash_sp", 0) or 0) for r in historian_rows]

    moisture_devs = [
        abs(m - sp) / sp * 100.0 if sp > 0 else 0.0
        for m, sp in zip(moisture_vals, moisture_sp_vals)
    ]
    ash_devs = [
        abs(a - sp) / sp * 100.0 if sp > 0 else 0.0
        for a, sp in zip(ash_vals, ash_sp_vals)
    ]
    avg_moisture_dev = round(statistics.mean(moisture_devs), 3) if moisture_devs else 0.0
    avg_ash_dev = round(statistics.mean(ash_devs), 3) if ash_devs else 0.0

    outcome = classify_transition_outcome(bw_off_spec_minutes, bw_peak_dev, recovery_minutes)

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

    failure_severity = compute_failure_severity(failure_type, bw_off_spec_minutes)

    return {
        "transition_id": first.get("transition_id", ""),
        "grade_from": grade_from,
        "grade_to": grade_to,
        "grade_distance": grade_distance,
        "transition_duration": duration,
        "failure_type": failure_type,
        "failure_severity": failure_severity,
        "bw_off_spec_minutes": bw_off_spec_minutes,
        "bw_peak_deviation": round(bw_peak_dev, 3),
        "bw_avg_deviation": round(bw_avg_dev, 3),
        "recovery_minutes": recovery_minutes,
        "avg_moisture_deviation": avg_moisture_dev,
        "avg_ash_deviation": avg_ash_dev,
        "max_moisture": round(max(moisture_vals), 2) if moisture_vals else 0.0,
        "max_ash": round(max(ash_vals), 2) if ash_vals else 0.0,
        "max_basis_weight": round(max([float(r.get("basis_weight", 0) or 0) for r in historian_rows]), 2) if historian_rows else 0.0,
        "min_basis_weight": round(min([float(r.get("basis_weight", 0) or 0) for r in historian_rows]), 2) if historian_rows else 0.0,
        "transition_outcome": outcome,
        "recommended_operator_action": action,
        "recommendation_reason": reason,
    }
