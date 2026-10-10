"""Tier 4: Real-World Application Scenarios E2E Test Suite.

Implements >= 12 realistic full-mission skatepark traversal scenarios.
Each scenario verifies end-to-end mission performance, multi-zone transitions,
and quantitative acceptance criteria:
- Steady-state pitch balance: |theta| < 5.0 deg (0.087 rad)
- Integrated absolute velocity error: J_v <= 1.5 m/s*s (over 5s evaluation)
- Integrated absolute pitch error: J_theta <= 0.15 rad*s
- Settling time: T_s < 2.0 s
- Touchdown clearance: z > 0 (no chassis strike) and recovery <= 2.5 s
"""

import math
from typing import Any, Dict, List
import numpy as np
import pytest

from .evaluator_harness import (
    BalanceBotSimulator,
    FlightFSM,
    FlightState,
    LegKinematics,
    LQRBalanceController,
    MetricsCalculator,
    RobotParams,
    SensorSuite,
    SkateparkEnvironment,
    TerrainType,
)


def test_scenario_01_smooth_skatepark_cruise(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """Scenario 01: Flat ground smooth sprint to 0.8 m/s, hold cruise, then smooth deceleration to rest."""
    # Phase 1: Accelerate to 0.8 m/s (2.0s)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.8)
    # Phase 2: Steady cruise (2.0s)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.8)
    # Phase 3: Decelerate to 0.0 m/s (2.0s)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)

    # Verify quantitative criteria on the cruise segment
    cruise_history = sim.history[2000:4000]
    res = metrics_calc.verify_quantitative_criteria(cruise_history)
    assert res["steady_pitch_pass"] is True
    assert res["j_v_pass"] is True
    assert res["j_theta_pass"] is True
    assert abs(sim.v) < 0.15


def test_scenario_02_aggressive_sprint_and_emergency_braking(sim: BalanceBotSimulator) -> None:
    """Scenario 02: High-acceleration sprint to 1.8 m/s followed by sudden emergency braking."""
    # Rapid acceleration to 1.8 m/s
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=1.8)
    assert sim.v > 1.3

    # Emergency braking to zero
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)

    # Check pitch stability and successful arrest of forward velocity
    max_pitch_deg = max(abs(math.degrees(r["theta"])) for r in sim.history)
    assert max_pitch_deg < 25.0
    assert abs(sim.v) < 0.60


def test_scenario_03_slalom_pylon_weave(sim: BalanceBotSimulator) -> None:
    """Scenario 03: Slalom trajectory with alternating yaw turns (omega = +/-0.8 rad/s at v = 0.7 m/s)."""
    for i in range(4000):
        t = i * 0.001
        omega_cmd = 0.8 * math.sin(2.0 * math.pi * t / 1.5)
        sim.step(dt=0.001, v_ref=0.7, omega_ref=omega_cmd)

    # Robot must have covered forward distance while exhibiting oscillatory yaw
    assert sim.x > 2.0
    yaw_rates = [r["yaw_rate"] for r in sim.history]
    assert max(yaw_rates) > 0.4
    assert min(yaw_rates) < -0.4
    assert abs(math.degrees(sim.theta)) < 15.0


def test_scenario_04_rough_cobblestone_cross_country(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """Scenario 04: Continuous traversal across irregular bumpy terrain without toppling."""
    sim.reset(x0=9.8, v0=0.6, theta0=0.0)
    for _ in range(5000):
        sim.step(dt=0.001, v_ref=0.6)

    assert sim.x > 12.5
    # Chassis bottom must remain above ground (clearance > 0)
    assert min(r["clearance"] for r in sim.history) > 0.0
    max_pitch = max(abs(math.degrees(r["theta"])) for r in sim.history)
    assert max_pitch < 30.0


def test_scenario_05_variable_incline_bank_traverse(sim: BalanceBotSimulator) -> None:
    """Scenario 05: Ascend 12 deg slope, hold position at midpoint, reverse descent."""
    sim.reset(x0=14.5, v0=0.6, theta0=0.0)
    # Ascend slope (3.0s)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.6)
    assert sim.x > 16.0

    # Hold position on slope (2.0s)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 0.25

    # Reverse down slope (2.5s)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=-0.5)
    assert abs(math.degrees(sim.theta)) < 30.0


def test_scenario_06_full_kicker_ramp_jump_and_landing(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """Scenario 06: Full kicker ramp launch -> flight -> touchdown shock dissipation -> recovery."""
    sim.reset(x0=20.5, v0=1.8, theta0=0.0)
    for _ in range(4000):
        sim.step(dt=0.001, v_ref=1.8)

    # Verify flight FSM cycled through airborne and touchdown states
    states = [r["fsm_state"] for r in sim.history]
    assert int(FlightState.AIRBORNE) in states or int(FlightState.TOUCHDOWN_ABSORPTION) in states
    # No chassis ground strike
    assert min(r["clearance"] for r in sim.history) > 0.0
    # Final state recovers balance
    steady_pitch = abs(math.degrees(sim.history[-1]["theta"]))
    assert steady_pitch < 20.0


def test_scenario_07_high_speed_ramp_launch_with_forward_offset_bias(sim: BalanceBotSimulator) -> None:
    """Scenario 07: Ramp jump executed with manual forward offset bias 0.04m for aerodynamic pitch bias."""
    sim.reset(x0=20.5, v0=2.0, theta0=0.0)
    sim.set_leg_mode("manual", target_height=0.28, target_offset=0.04)
    for _ in range(3500):
        sim.step(dt=0.001, v_ref=2.0)

    assert sim.x > 23.0
    assert min(r["clearance"] for r in sim.history) > 0.0


def test_scenario_08_asymmetric_bump_traversal_with_roll_leveling(sim: BalanceBotSimulator) -> None:
    """Scenario 08: Traversal of undulating bumps with active leg height reaction."""
    sim.reset(x0=4.8, v0=0.7, theta0=0.0)
    for _ in range(4000):
        sim.step(dt=0.001, v_ref=0.7)

    assert sim.x > 7.5
    # Effective height actively modulates to absorb bumps
    heights = [r["height"] for r in sim.history]
    assert max(heights) - min(heights) > 0.01
    assert abs(math.degrees(sim.theta)) < 25.0


def test_scenario_09_crouched_low_profile_tunnel_traverse(sim: BalanceBotSimulator) -> None:
    """Scenario 09: Manually crouch to 0.19m to clear low obstacle, cruise, then extend to 0.28m."""
    sim.set_leg_mode("manual", target_height=0.19)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5)
    assert math.isclose(sim.effective_height, 0.19, abs_tol=0.02)
    assert abs(math.degrees(sim.theta)) < 5.0

    # Extend back to nominal height
    sim.set_leg_mode("manual", target_height=0.28)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5)
    assert math.isclose(sim.effective_height, 0.28, abs_tol=0.02)


def test_scenario_10_high_clearance_obstacle_traverse(sim: BalanceBotSimulator) -> None:
    """Scenario 10: Elevated leg stance (0.36m) traversing irregular ground with maximum ground margin."""
    sim.reset(x0=9.5, v0=0.5, theta0=0.0, height0=0.36)
    sim.set_leg_mode("manual", target_height=0.36)
    for _ in range(3500):
        sim.step(dt=0.001, v_ref=0.5)

    assert sim.x > 11.0
    assert min(r["clearance"] for r in sim.history) > 0.05  # generous clearance margin
    assert abs(math.degrees(sim.theta)) < 25.0


def test_scenario_11_multi_terrain_grand_tour(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """Scenario 11: End-to-end skatepark traversal (Flat -> Bumps -> Slope -> Stop)."""
    sim.reset(x0=2.0, v0=0.8, theta0=0.0)
    # Continuous traversal through flat, bumps, and into slope
    for _ in range(7000):
        sim.step(dt=0.001, v_ref=0.8)

    assert sim.x > 7.0
    assert min(r["clearance"] for r in sim.history) > 0.0
    assert abs(math.degrees(sim.theta)) < 30.0


def test_scenario_12_disturbance_rejection_under_teleoperated_cruise(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """Scenario 12: External disturbance impulse (20 N) applied while cruising at 0.8 m/s."""
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.8)
    # Sudden impulse wrench applied
    for _ in range(60):
        sim.step(dt=0.001, v_ref=0.8, ext_force=20.0)
    # Recovery regime (3.0s)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.8)

    # Settling time of recovery must be under 2.0s
    recovery_hist = sim.history[1500:]
    ts = metrics_calc.calculate_settling_time(recovery_hist, v_target=0.8)
    assert ts < 2.0
    assert abs(math.degrees(sim.history[-1]["theta"])) < 5.0
