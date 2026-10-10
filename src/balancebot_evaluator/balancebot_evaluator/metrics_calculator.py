"""Quantitative Performance Metrics Calculator for BalanceBot.

Computes integrated absolute error metrics, settling times, clearance,
and landing recovery durations according to authoritative acceptance criteria.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple


class MetricsCalculator:
    """Calculates quantitative performance metrics from simulation history."""

    @staticmethod
    def calculate_integrated_errors(
        history: List[Dict[str, float]],
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute integrated absolute velocity error J_v and pitch error J_theta."""
        j_v = 0.0
        j_theta = 0.0
        t_prev = None

        for rec in history:
            t = rec["time"]
            if t < t_start:
                continue
            if t_end is not None and t > t_end:
                break
            if t_prev is None:
                t_prev = t
                continue
            dt = t - t_prev
            t_prev = t
            j_v += abs(rec["v"] - rec["v_ref"]) * dt
            j_theta += abs(rec["theta"] - rec["theta_ref"]) * dt

        return float(j_v), float(j_theta)

    @staticmethod
    def calculate_settling_time(
        history: List[Dict[str, float]],
        v_target: float,
        theta_target: float = 0.0,
        v_band: float = 0.10,
        theta_band_rad: float = math.radians(2.0),
    ) -> float:
        """Compute settling time T_s where state permanently remains within tolerance."""
        if not history:
            return 0.0

        t_settled = history[-1]["time"]
        for i in range(len(history) - 1, -1, -1):
            rec = history[i]
            v_err = abs(rec["v"] - v_target)
            theta_err = abs(rec["theta"] - theta_target)
            if v_err > v_band or theta_err > theta_band_rad:
                t_settled = history[i]["time"]
                break

        t_start = history[0]["time"]
        return float(max(0.0, t_settled - t_start))

    @staticmethod
    def verify_quantitative_criteria(
        history: List[Dict[str, float]],
        max_steady_pitch_deg: float = 5.0,
        max_j_v: float = 1.5,
        max_j_theta: float = 0.15,
        max_t_s: float = 2.0,
        min_clearance: float = 0.0,
        max_recovery_time: float = 2.5,
    ) -> Dict[str, Any]:
        """Verify the 5 authoritative quantitative acceptance criteria."""
        j_v, j_theta = MetricsCalculator.calculate_integrated_errors(history)

        # Steady-state pitch check (last 30% of run)
        n = len(history)
        steady_history = history[int(0.7 * n):] if n > 0 else history
        steady_pitches = [abs(math.degrees(r["theta"])) for r in steady_history]
        max_pitch_deg = max(steady_pitches) if steady_pitches else 0.0

        # Settling time
        v_ref = history[-1]["v_ref"] if history else 0.0
        theta_ref = history[-1]["theta_ref"] if history else 0.0
        t_s = MetricsCalculator.calculate_settling_time(history, v_ref, theta_ref)

        # Minimum clearance
        min_clear = min(r.get("clearance", 0.1) for r in history) if history else 0.0

        # Landing recovery
        touchdown_idx = next(
            (i for i, r in enumerate(history) if r.get("fsm_state") == 3),  # 3: TOUCHDOWN_ABSORPTION
            None,
        )
        recovery_time = 0.0
        if touchdown_idx is not None:
            t_touch = history[touchdown_idx]["time"]
            for r in history[touchdown_idx:]:
                if abs(r["theta"]) < math.radians(5.0):
                    recovery_time = r["time"] - t_touch
                    break

        steady_pitch_pass = bool(max_pitch_deg < max_steady_pitch_deg)
        j_v_pass = bool(j_v <= max_j_v)
        j_theta_pass = bool(j_theta <= max_j_theta)
        settling_time_pass = bool(t_s < max_t_s)
        clearance_pass = bool(min_clear > min_clearance)
        recovery_pass = bool(recovery_time <= max_recovery_time)

        all_passed = bool(
            steady_pitch_pass
            and j_v_pass
            and j_theta_pass
            and settling_time_pass
            and clearance_pass
            and recovery_pass
        )

        return {
            "steady_pitch_deg": float(max_pitch_deg),
            "steady_pitch_pass": steady_pitch_pass,
            "j_v": float(j_v),
            "j_v_pass": j_v_pass,
            "j_theta": float(j_theta),
            "j_theta_pass": j_theta_pass,
            "settling_time": float(t_s),
            "settling_time_pass": settling_time_pass,
            "min_clearance": float(min_clear),
            "clearance_pass": clearance_pass,
            "recovery_time": float(recovery_time),
            "recovery_pass": recovery_pass,
            "all_passed": all_passed,
        }

    @staticmethod
    def save_scorecard_json(results: Dict[str, Any], filepath: str) -> None:
        """Write scorecard results to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "PASSED" if results.get("all_passed") else "FAILED",
            "metrics": results,
            "thresholds": {
                "steady_pitch_deg": "< 5.0 deg",
                "j_v": "<= 1.5 m/s*s",
                "j_theta": "<= 0.15 rad*s",
                "settling_time": "< 2.0 s",
                "min_clearance": "> 0.0 m",
                "recovery_time": "<= 2.5 s",
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def save_scorecard_csv(results: Dict[str, Any], filepath: str) -> None:
        """Write scorecard metrics table to CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        rows = [
            ("Metric", "Measured Value", "Threshold", "Status"),
            (
                "Steady-State Pitch Balance",
                f"{results.get('steady_pitch_deg', 0.0):.3f} deg",
                "< 5.0 deg",
                "PASS" if results.get("steady_pitch_pass") else "FAIL",
            ),
            (
                "Integrated Velocity Error (J_v)",
                f"{results.get('j_v', 0.0):.4f} m/s*s",
                "<= 1.5 m/s*s",
                "PASS" if results.get("j_v_pass") else "FAIL",
            ),
            (
                "Integrated Pitch Error (J_theta)",
                f"{results.get('j_theta', 0.0):.4f} rad*s",
                "<= 0.15 rad*s",
                "PASS" if results.get("j_theta_pass") else "FAIL",
            ),
            (
                "Settling Time (T_s)",
                f"{results.get('settling_time', 0.0):.3f} s",
                "< 2.0 s",
                "PASS" if results.get("settling_time_pass") else "FAIL",
            ),
            (
                "Chassis Ground Clearance (z_min)",
                f"{results.get('min_clearance', 0.0):.4f} m",
                "> 0.0 m",
                "PASS" if results.get("clearance_pass") else "FAIL",
            ),
            (
                "Landing Recovery Time (T_recover)",
                f"{results.get('recovery_time', 0.0):.3f} s",
                "<= 2.5 s",
                "PASS" if results.get("recovery_pass") else "FAIL",
            ),
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    @staticmethod
    def format_markdown_report(
        results: Dict[str, Any], run_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format scorecard results into a clean markdown report."""
        status = "**PASSED**" if results.get("all_passed") else "**FAILED**"
        info = run_info or {}
        scenario = info.get("scenario", "Standard Verification Suite")
        duration = info.get("duration", 0.0)

        lines = [
            "# RL-Robust-BalanceBot Quantitative Evaluation Report",
            "",
            f"- **Overall Status**: {status}",
            f"- **Scenario**: {scenario}",
            f"- **Duration**: {duration:.2f} s",
            f"- **Timestamp**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            "",
            "## Acceptance Criteria Scorecard",
            "",
            "| Metric | Formulation | Measured Value | Threshold | Result |",
            "|---|---|:---:|:---:|:---:|",
            f"| Steady-State Pitch Balance | $\\max |\\theta|$ | {results.get('steady_pitch_deg', 0.0):.3f}° | < 5.0° | {'PASS' if results.get('steady_pitch_pass') else 'FAIL'} |",
            f"| Integrated Velocity Error | $J_v = \\int |v - v_{{ref}}| dt$ | {results.get('j_v', 0.0):.4f} m/s·s | ≤ 1.5 m/s·s | {'PASS' if results.get('j_v_pass') else 'FAIL'} |",
            f"| Integrated Pitch Error | $J_\\theta = \\int |\\theta - \\theta_{{ref}}| dt$ | {results.get('j_theta', 0.0):.4f} rad·s | ≤ 0.15 rad·s | {'PASS' if results.get('j_theta_pass') else 'FAIL'} |",
            f"| Settling Time | $T_s$ (5% velocity, 2° pitch) | {results.get('settling_time', 0.0):.3f} s | < 2.0 s | {'PASS' if results.get('settling_time_pass') else 'FAIL'} |",
            f"| Chassis Ground Clearance | $z_{{min}} = \\min (z - z_{{g}})$ | {results.get('min_clearance', 0.0):.4f} m | > 0.0 m | {'PASS' if results.get('clearance_pass') else 'FAIL'} |",
            f"| Landing Recovery Time | $T_{{recover}} = t_{{bal}} - t_{{td}}$ | {results.get('recovery_time', 0.0):.3f} s | ≤ 2.5 s | {'PASS' if results.get('recovery_pass') else 'FAIL'} |",
            "",
        ]
        return "\n".join(lines)
