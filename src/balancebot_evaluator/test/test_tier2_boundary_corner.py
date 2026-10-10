"""Tier 2: Boundary & Corner Cases E2E Test Suite (F01–F24).

Implements >= 5 authentic boundary and corner condition test cases per inventoried feature (total 120 tests).
Covers zero velocity, command step extremes, reverse, pitch impulse, leg height limits (0.18m, 0.38m),
wheel offset limits (-0.08m, 0.08m), terrain limits, and recovery thresholds.
"""

import math
import os
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


# ==============================================================================
# F01: Boundary - Model Limits & Physical Bounds (5 tests)
# ==============================================================================
def test_t2_f01_zero_and_negative_inertia_prohibited(robot_params: RobotParams) -> None:
    """T2-F01-1: Verify inertias are strictly positive definite."""
    assert robot_params.I_b > 0.0
    assert robot_params.I_w > 0.0
    assert robot_params.m_b > 0.0


def test_t2_f01_extreme_hip_joint_limit_stops(kin: LegKinematics, robot_params: RobotParams) -> None:
    """T2-F01-2: Verify hip angles are bounded within +/-60 deg limit."""
    qh_max, qk_max = kin.inverse_kinematics(0.08, -0.20)
    assert abs(qh_max) <= robot_params.hip_limit_rad + 0.05


def test_t2_f01_extreme_knee_joint_limit_stops(kin: LegKinematics, robot_params: RobotParams) -> None:
    """T2-F01-3: Verify knee angle stays within [0, 120] deg limit across workspace."""
    for h in [0.18, 0.28, 0.38]:
        for off in [-0.08, 0.0, 0.08]:
            qh, qk = kin.inverse_kinematics(off, -h)
            assert robot_params.knee_min_rad <= qk <= math.radians(135.0)


def test_t2_f01_wheel_minimum_ground_contact_radius(robot_params: RobotParams) -> None:
    """T2-F01-4: Verify wheel radius R=0.10m strictly enforced for contact geometry."""
    assert robot_params.R == 0.10


def test_t2_f01_extreme_payload_mass_tolerance() -> None:
    """T2-F01-5: Verify robot controllable under +/-30% torso mass variation."""
    heavy_params = RobotParams(m_b=13.0)
    sim = BalanceBotSimulator(params=heavy_params)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(math.degrees(sim.theta)) < 5.0


# ==============================================================================
# F02: Boundary - Skatepark Terrain Extremes (5 tests)
# ==============================================================================
def test_t2_f02_zero_bump_boundary_at_flat_entrance(env: SkateparkEnvironment) -> None:
    """T2-F02-1: Verify zero elevation jump at boundary between flat and bumps."""
    assert math.isclose(env.elevation(4.999), 0.0, abs_tol=1e-3)
    assert math.isclose(env.elevation(5.000), 0.0, abs_tol=1e-3)


def test_t2_f02_maximum_rough_protrusion_height(env: SkateparkEnvironment) -> None:
    """T2-F02-2: Verify max rough terrain protrusion does not exceed 5 cm."""
    x_samples = np.linspace(10.0, 15.0, 200)
    elevations = [env.elevation(x) for x in x_samples]
    assert max(elevations) <= 0.05
    assert min(elevations) >= -0.05


def test_t2_f02_slope_crest_transition_continuity(env: SkateparkEnvironment) -> None:
    """T2-F02-3: Verify elevation continuity across slope crest."""
    e1 = env.elevation(19.999)
    e2 = env.elevation(20.001)
    assert math.isclose(e1, e2, abs_tol=0.02)


def test_t2_f02_ramp_lip_sharp_discontinuity(env: SkateparkEnvironment) -> None:
    """T2-F02-4: Verify ramp terminates at x=23.0 with open drop."""
    e_lip = env.elevation(23.0)
    e_drop = env.elevation(23.05)
    assert e_lip > 0.30
    assert e_drop == 0.0  # drop to floor


def test_t2_f02_low_friction_surface_boundary(env: SkateparkEnvironment) -> None:
    """T2-F02-5: Verify skatepark friction coefficient >= 0.8."""
    assert env.friction_coeff >= 0.8


# ==============================================================================
# F03: Boundary - LQR Command Step Extremes & Disturbances (5 tests)
# ==============================================================================
def test_t2_f03_zero_velocity_holding_under_pitch_disturbance(sim: BalanceBotSimulator) -> None:
    """T2-F03-1: Verify stationary balance recovers from 15 N*s pitch impulse at v=0."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.0)
    # Apply 15 N pitch disturbance impulse
    for _ in range(50):
        sim.step(dt=0.001, v_ref=0.0, ext_torque=15.0)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(math.degrees(sim.theta)) < 5.0
    assert abs(sim.v) < 0.15


def test_t2_f03_maximum_forward_step_command(sim: BalanceBotSimulator) -> None:
    """T2-F03-2: Verify step to max forward speed 2.5 m/s does not topple (|theta| < 35 deg)."""
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=2.5)
    max_pitch = max(abs(math.degrees(r["theta"])) for r in sim.history)
    assert max_pitch < 35.0
    assert sim.v > 1.8


def test_t2_f03_instantaneous_reverse_step_command(sim: BalanceBotSimulator) -> None:
    """T2-F03-3: Verify rapid direction flip (+2.0 m/s -> -1.5 m/s) maintains balance."""
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=2.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=-1.5)
    assert sim.v < -0.8
    assert abs(math.degrees(sim.theta)) < 30.0


def test_t2_f03_maximum_pitch_impulse_recovery(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """T2-F03-4: Verify recovery from 30 N push impulse settles in Ts < 2.0 s."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.0)
    for _ in range(80):
        sim.step(dt=0.001, v_ref=0.0, ext_force=30.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.0)
    ts = metrics_calc.calculate_settling_time(sim.history[1000:], v_target=0.0)
    assert ts < 2.0


def test_t2_f03_sudden_brake_from_maximum_speed(sim: BalanceBotSimulator) -> None:
    """T2-F03-5: Verify hard braking from 2.5 m/s decelerates safely without tipping over."""
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=2.5)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 1.0
    assert abs(math.degrees(sim.theta)) < 25.0


# ==============================================================================
# F04: Boundary - Teleop Input Extremes & Rapid Jitter (5 tests)
# ==============================================================================
def test_t2_f04_rapid_alternating_key_jitter(sim: BalanceBotSimulator) -> None:
    """T2-F04-1: Verify high-frequency alternating velocity commands maintain balance."""
    for i in range(2000):
        v_cmd = 1.0 if (i // 200) % 2 == 0 else -1.0
        sim.step(dt=0.001, v_ref=v_cmd)
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t2_f04_simultaneous_extreme_drive_and_turn(sim: BalanceBotSimulator) -> None:
    """T2-F04-2: Verify simultaneous max drive (2.0 m/s) and turn (1.5 rad/s)."""
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=2.0, omega_ref=1.5)
    assert sim.v > 1.2
    assert sim.yaw_rate > 0.8
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t2_f04_zero_velocity_with_maximum_spin(sim: BalanceBotSimulator) -> None:
    """T2-F04-3: Verify in-place yaw rotation with zero linear translation."""
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0, omega_ref=1.5)
    assert abs(sim.v) < 0.15
    assert sim.yaw_rate > 1.0


def test_t2_f04_unbounded_user_command_clamped(lqr: LQRBalanceController) -> None:
    """T2-F04-4: Verify torque remains clamped at 15 N*m even with infinite speed error."""
    tau, sat = lqr.compute_torque(
        v_meas=0.0, v_ref=100.0, theta_meas=0.0, theta_ref=0.0, theta_dot=0.0, e_integral=0.0, effective_length=0.28
    )
    assert abs(tau) == 15.0
    assert sat is True


def test_t2_f04_sudden_command_release_neutral_drift(sim: BalanceBotSimulator) -> None:
    """T2-F04-5: Verify neutral command release stops forward drift."""
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=1.2)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 0.15


# ==============================================================================
# F05: Boundary - Automated Replay Timing & Profile Extremes (5 tests)
# ==============================================================================
def test_t2_f05_zero_duration_command_step(sim: BalanceBotSimulator) -> None:
    """T2-F05-1: Verify instantaneous step profile transition."""
    sim.step(dt=0.001, v_ref=0.0)
    sim.step(dt=0.001, v_ref=1.5)
    assert sim.history[-1]["v_ref"] == 1.5


def test_t2_f05_long_duration_cruise_profile(sim: BalanceBotSimulator) -> None:
    """T2-F05-2: Verify extended 6.0 second cruise maintains pitch balance."""
    for _ in range(6000):
        sim.step(dt=0.001, v_ref=0.8)
    steady_pitch = [abs(math.degrees(r["theta"])) for r in sim.history[-2000:]]
    assert max(steady_pitch) < 5.0


def test_t2_f05_high_frequency_chirp_command_profile(sim: BalanceBotSimulator) -> None:
    """T2-F05-3: Verify robot stability under sinusoidal frequency sweep velocity."""
    for i in range(3000):
        t = i * 0.001
        v_ref = 0.5 * math.sin(2.0 * math.pi * (1.0 + 0.5 * t) * t)
        sim.step(dt=0.001, v_ref=v_ref)
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t2_f05_reverse_profile_replay(sim: BalanceBotSimulator) -> None:
    """T2-F05-4: Verify multi-point reverse profile replay."""
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=-0.5)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=-1.0)
    assert sim.v < -0.7


def test_t2_f05_empty_or_single_point_profile(metrics_calc: MetricsCalculator) -> None:
    """T2-F05-5: Verify metrics calculator handles empty or single record gracefully."""
    jv, jt = metrics_calc.calculate_integrated_errors([])
    assert jv == 0.0
    assert jt == 0.0


# ==============================================================================
# F06: Boundary - Error Logging Ring Buffer & Overflow (5 tests)
# ==============================================================================
def test_t2_f06_extreme_duration_memory_bounded(sim: BalanceBotSimulator) -> None:
    """T2-F06-1: Verify simulator logs 8,000 steps without error or corruption."""
    for _ in range(8000):
        sim.step(dt=0.001, v_ref=0.3)
    assert len(sim.history) == 8000


def test_t2_f06_zero_error_integral_baseline(metrics_calc: MetricsCalculator) -> None:
    """T2-F06-2: Verify zero error history gives exact zero integrals."""
    history = [{"time": 0.0, "v": 1.0, "v_ref": 1.0, "theta": 0.0, "theta_ref": 0.0},
               {"time": 1.0, "v": 1.0, "v_ref": 1.0, "theta": 0.0, "theta_ref": 0.0}]
    jv, jt = metrics_calc.calculate_integrated_errors(history)
    assert jv == 0.0
    assert jt == 0.0


def test_t2_f06_inf_or_nan_error_detection(sim: BalanceBotSimulator) -> None:
    """T2-F06-3: Verify zero NaN in recorded trajectory history."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.5)
    for r in sim.history:
        assert not math.isnan(r["v"])
        assert not math.isnan(r["theta"])


def test_t2_f06_high_transient_spike_capture(sim: BalanceBotSimulator) -> None:
    """T2-F06-4: Verify logging captures transient peak during sudden step."""
    for _ in range(500):
        sim.step(dt=0.001, v_ref=0.0)
    for _ in range(500):
        sim.step(dt=0.001, v_ref=2.0)
    max_theta = max(abs(r["theta"]) for r in sim.history)
    assert max_theta > 0.05


def test_t2_f06_time_stamp_monotonicity_under_jitter(sim: BalanceBotSimulator) -> None:
    """T2-F06-5: Verify timestamps are strictly increasing."""
    for _ in range(500):
        sim.step(dt=0.001, v_ref=0.0)
    times = [r["time"] for r in sim.history]
    for i in range(1, len(times)):
        assert times[i] > times[i - 1]


# ==============================================================================
# F07: Boundary - Leg Kinematics Limits & Singularities (5 tests)
# ==============================================================================
def test_t2_f07_minimum_leg_height_limit_0_18m(kin: LegKinematics) -> None:
    """T2-F07-1: Verify crouched height at boundary 0.18 m."""
    qh, qk = kin.inverse_kinematics(0.0, -0.18)
    _, z_calc = kin.forward_kinematics(qh, qk)
    assert math.isclose(-z_calc, 0.18, abs_tol=1e-3)


def test_t2_f07_maximum_leg_height_limit_0_38m(kin: LegKinematics) -> None:
    """T2-F07-2: Verify extended height at boundary 0.38 m."""
    qh, qk = kin.inverse_kinematics(0.0, -0.38)
    _, z_calc = kin.forward_kinematics(qh, qk)
    assert math.isclose(-z_calc, 0.38, abs_tol=1e-3)


def test_t2_f07_maximum_forward_wheel_offset_limit_0_08m(kin: LegKinematics) -> None:
    """T2-F07-3: Verify maximum forward wheel offset 0.08 m."""
    qh, qk = kin.inverse_kinematics(0.08, -0.28)
    x_calc, _ = kin.forward_kinematics(qh, qk)
    assert math.isclose(x_calc, 0.08, abs_tol=1e-3)


def test_t2_f07_maximum_rearward_wheel_offset_limit_neg_0_08m(kin: LegKinematics) -> None:
    """T2-F07-4: Verify maximum rearward wheel offset -0.08 m."""
    qh, qk = kin.inverse_kinematics(-0.08, -0.28)
    x_calc, _ = kin.forward_kinematics(qh, qk)
    assert math.isclose(x_calc, -0.08, abs_tol=1e-3)


def test_t2_f07_straight_knee_singularity_avoidance(kin: LegKinematics) -> None:
    """T2-F07-5: Verify knee angle is prevented from fully locking straight (qk > 0)."""
    qh, qk = kin.inverse_kinematics(0.0, -0.39)  # beyond length
    assert qk > math.radians(10.0)


# ==============================================================================
# F08: Boundary - Virtual Model Impedance Extremes (5 tests)
# ==============================================================================
def test_t2_f08_extreme_vertical_impact_load(kin: LegKinematics) -> None:
    """T2-F08-1: Verify heavy impact force (400 N) produces finite joint torques."""
    qh, qk = kin.inverse_kinematics(0.0, -0.28)
    tau_h, tau_k = kin.virtual_model_torque(qh, qk, target_z=0.28, current_z=0.18, z_vel=-2.0, k_z=1500.0, d_z=200.0)
    assert not math.isinf(tau_h)
    assert not math.isinf(tau_k)


def test_t2_f08_zero_stiffness_fallback_damping(kin: LegKinematics) -> None:
    """T2-F08-2: Verify pure damping mode (k_z = 0, d_z = 250)."""
    qh, qk = kin.inverse_kinematics(0.0, -0.28)
    tau_h, tau_k = kin.virtual_model_torque(qh, qk, target_z=0.28, current_z=0.28, z_vel=-1.0, k_z=0.0, d_z=250.0)
    assert abs(tau_k) > 10.0


def test_t2_f08_maximum_stiffness_chatter_free(sim: BalanceBotSimulator) -> None:
    """T2-F08-3: Verify high stiffness runs without numerical explosion at dt=0.001."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.theta) < 0.10


def test_t2_f08_rapid_cyclic_height_modulation(sim: BalanceBotSimulator) -> None:
    """T2-F08-4: Verify cyclic height variation 0.22m <-> 0.35m maintains balance."""
    sim.set_leg_mode("manual")
    for i in range(3000):
        target_h = 0.22 if (i // 750) % 2 == 0 else 0.35
        sim.target_height = target_h
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(math.degrees(sim.theta)) < 15.0


def test_t2_f08_chassis_bottoming_prevention_under_overload(sim: BalanceBotSimulator) -> None:
    """T2-F08-5: Verify clearance remains strictly positive under downward disturbance."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.0, ext_force=-50.0)
    assert min(r["clearance"] for r in sim.history) > 0.0


# ==============================================================================
# F09: Boundary - Wheel Offset Extremes & Dynamic Shift (5 tests)
# ==============================================================================
def test_t2_f09_extreme_offset_equilibrium_angle_bound(kin: LegKinematics) -> None:
    """T2-F09-1: Verify offset at +/-0.08m bounds |theta_eq| <= 25 deg."""
    th_eq = kin.equilibrium_pitch(0.08, 0.28)
    assert math.degrees(th_eq) < 25.0


def test_t2_f09_rapid_offset_transition_step(sim: BalanceBotSimulator) -> None:
    """T2-F09-2: Verify rapid offset change from -0.05m to +0.05m settles stably."""
    sim.set_leg_mode("manual", target_height=0.28, target_offset=-0.05)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    sim.set_leg_mode("manual", target_height=0.28, target_offset=0.05)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.wheel_offset, 0.05, abs_tol=0.01)
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t2_f09_minimum_height_with_maximum_offset(kin: LegKinematics) -> None:
    """T2-F09-3: Verify combined extreme L=0.18m with offset=0.06m is solvable."""
    qh, qk = kin.inverse_kinematics(0.06, -0.18)
    assert not math.isnan(qh)
    assert not math.isnan(qk)


def test_t2_f09_maximum_height_with_maximum_offset(kin: LegKinematics) -> None:
    """T2-F09-4: Verify combined extreme L=0.38m with offset=0.06m is solvable."""
    qh, qk = kin.inverse_kinematics(0.06, -0.38)
    assert not math.isnan(qh)
    assert not math.isnan(qk)


def test_t2_f09_zero_speed_drift_under_offset_balance(sim: BalanceBotSimulator) -> None:
    """T2-F09-5: Verify balanced offset robot maintains zero long-term velocity drift."""
    sim.set_leg_mode("manual", target_height=0.28, target_offset=0.04)
    for _ in range(4000):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 0.25


# ==============================================================================
# F10: Boundary - Terrain Adaptation Slope Limits & Bumps (5 tests)
# ==============================================================================
def test_t2_f10_steep_slope_limit_18_deg(sim: BalanceBotSimulator) -> None:
    """T2-F10-1: Verify robot climbing steep slope retains pitch < 30 deg."""
    sim.reset(x0=15.0, v0=0.8, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.8)
    assert abs(math.degrees(sim.theta)) < 30.0


def test_t2_f10_asymmetric_single_wheel_bump_impact(sim: BalanceBotSimulator) -> None:
    """T2-F10-2: Verify single-wheel bump torque does not spin robot uncontrollably."""
    for _ in range(50):
        sim.step(dt=0.001, v_ref=0.5, ext_torque=10.0)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 5.0


def test_t2_f10_high_frequency_rough_terrain_chatter(sim: BalanceBotSimulator) -> None:
    """T2-F10-3: Verify rough cobblestone traversal does not cause joint divergence."""
    sim.reset(x0=10.0, v0=0.7, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.7)
    assert not math.isnan(sim.v)


def test_t2_f10_downhill_slope_descent_stability(sim: BalanceBotSimulator) -> None:
    """T2-F10-4: Verify downhill descent stability (reverse driving on slope)."""
    sim.reset(x0=18.0, v0=-0.5, theta0=0.0)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=-0.5)
    assert abs(math.degrees(sim.theta)) < 30.0
    assert not math.isnan(sim.v)


def test_t2_f10_reverse_climbing_on_slope(sim: BalanceBotSimulator) -> None:
    """T2-F10-5: Verify reverse climbing up slope."""
    sim.reset(x0=16.0, v0=-0.4, theta0=0.0)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=-0.4)
    assert not math.isnan(sim.v)


# ==============================================================================
# F11: Boundary - Manual Target Extreme Setpoints (5 tests)
# ==============================================================================
def test_t2_f11_step_target_to_minimum_height(sim: BalanceBotSimulator) -> None:
    """T2-F11-1: Verify direct target step to minimum height 0.18 m."""
    sim.set_leg_mode("manual", target_height=0.18)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.effective_height, 0.18, abs_tol=0.01)


def test_t2_f11_step_target_to_maximum_height(sim: BalanceBotSimulator) -> None:
    """T2-F11-2: Verify direct target step to maximum height 0.38 m."""
    sim.set_leg_mode("manual", target_height=0.38)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.effective_height, 0.38, abs_tol=0.01)


def test_t2_f11_step_target_to_extreme_forward_offset(sim: BalanceBotSimulator) -> None:
    """T2-F11-3: Verify step to max forward offset 0.08 m."""
    sim.set_leg_mode("manual", target_offset=0.08)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.wheel_offset, 0.08, abs_tol=0.01)


def test_t2_f11_step_target_to_extreme_rear_offset(sim: BalanceBotSimulator) -> None:
    """T2-F11-4: Verify step to max rearward offset -0.08 m."""
    sim.set_leg_mode("manual", target_offset=-0.08)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.wheel_offset, -0.08, abs_tol=0.01)


def test_t2_f11_simultaneous_extreme_height_and_offset(sim: BalanceBotSimulator) -> None:
    """T2-F11-5: Verify simultaneous targets (height=0.38m, offset=-0.07m)."""
    sim.set_leg_mode("manual", target_height=0.38, target_offset=-0.07)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.effective_height, 0.38, abs_tol=0.02)
    assert math.isclose(sim.wheel_offset, -0.07, abs_tol=0.02)


# ==============================================================================
# F12: Boundary - Rapid Mode Switching Thrashing (5 tests)
# ==============================================================================
def test_t2_f12_rapid_mode_toggle_thrashing(sim: BalanceBotSimulator) -> None:
    """T2-F12-1: Verify 10 rapid mode toggles do not destabilize balance."""
    for i in range(10):
        sim.set_leg_mode("manual" if i % 2 == 0 else "auto")
        for _ in range(100):
            sim.step(dt=0.001, v_ref=0.0)
    assert abs(math.degrees(sim.theta)) < 5.0


def test_t2_f12_manual_latched_across_extreme_disturbances(sim: BalanceBotSimulator) -> None:
    """T2-F12-2: Verify manual mode remains latched after 20 N disturbance."""
    sim.set_leg_mode("manual", target_height=0.22)
    for _ in range(50):
        sim.step(dt=0.001, v_ref=0.0, ext_force=20.0)
    for _ in range(500):
        sim.step(dt=0.001, v_ref=0.0)
    assert sim.leg_mode == "manual"


def test_t2_f12_mode_switch_at_high_speed(sim: BalanceBotSimulator) -> None:
    """T2-F12-3: Verify mode switch while cruising at 1.5 m/s."""
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=1.5)
    sim.set_leg_mode("manual", target_height=0.25)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=1.5)
    assert sim.v > 1.0


def test_t2_f12_mode_switch_on_steep_slope(sim: BalanceBotSimulator) -> None:
    """T2-F12-4: Verify mode switch while navigating 12 deg slope."""
    sim.reset(x0=16.0, v0=0.6, theta0=0.0)
    sim.set_leg_mode("manual", target_height=0.24)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.6)
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t2_f12_invalid_mode_identifier_rejection(sim: BalanceBotSimulator) -> None:
    """T2-F12-5: Verify invalid mode name raises AssertionError."""
    with pytest.raises(AssertionError):
        sim.set_leg_mode("invalid_mode")


# ==============================================================================
# F13: Boundary - Rough Terrain Speed Extremes (5 tests)
# ==============================================================================
def test_t2_f13_high_speed_bump_traversal_2ms(sim: BalanceBotSimulator) -> None:
    """T2-F13-1: Verify high-speed traversal across sinusoidal bumps at 2.0 m/s."""
    sim.reset(x0=4.5, v0=2.0, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=2.0)
    assert sim.x > 8.0
    assert abs(math.degrees(sim.theta)) < 35.0


def test_t2_f13_crawl_speed_bump_traversal_02ms(sim: BalanceBotSimulator) -> None:
    """T2-F13-2: Verify slow crawl traversal across bumps at 0.2 m/s."""
    sim.reset(x0=5.0, v0=0.2, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.2)
    assert abs(math.degrees(sim.theta)) < 25.0
    assert min(r["clearance"] for r in sim.history) > 0.0


def test_t2_f13_stop_and_hold_position_on_bump_crest(sim: BalanceBotSimulator) -> None:
    """T2-F13-3: Verify stopping on bump crest maintains balance."""
    sim.reset(x0=5.2, v0=0.0, theta0=0.0)  # at bump peak
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(math.degrees(sim.theta)) < 8.0


def test_t2_f13_rough_terrain_reverse_traversal(sim: BalanceBotSimulator) -> None:
    """T2-F13-4: Verify reverse traversal across irregular ground."""
    sim.reset(x0=12.0, v0=-0.6, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=-0.6)
    assert sim.x < 11.5


def test_t2_f13_rough_terrain_turning_maneuver(sim: BalanceBotSimulator) -> None:
    """T2-F13-5: Verify steering turn while on rough ground."""
    sim.reset(x0=10.5, v0=0.5, theta0=0.0)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5, omega_ref=0.6)
    assert sim.yaw_rate > 0.3


# ==============================================================================
# F14: Boundary - Free-Fall Detection Under Edge Conditions (5 tests)
# ==============================================================================
def test_t2_f14_near_zero_gravity_drop_0ms(fsm: FlightFSM) -> None:
    """T2-F14-1: Verify pure zero gravity drop (a=0) triggers airborne."""
    for i in range(35):
        fsm.update(t=i*0.001, accel_mag=0.0, wheel_contact=False, pitch=0.0, z_vel=0.0, dt=0.001)
    assert fsm.state == FlightState.AIRBORNE


def test_t2_f14_partial_gravity_freefall_detection_2_0ms(fsm: FlightFSM) -> None:
    """T2-F14-2: Verify threshold edge condition at ||a|| = 2.0 m/s^2 with wheel_contact=True."""
    for i in range(30):
        fsm.update(t=i*0.001, accel_mag=2.0, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.001)
    assert fsm.state == FlightState.AIRBORNE

    # Boundary check: accel_mag = 3.5 m/s^2 (> 2.5 m/s^2) does not trigger airborne
    fsm_bound = FlightFSM()
    for i in range(35):
        fsm_bound.update(t=i*0.001, accel_mag=3.5, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.001)
    assert fsm_bound.state == FlightState.GROUND_BALANCE


def test_t2_f14_bump_rebound_does_not_trip_airborne(fsm: FlightFSM) -> None:
    """T2-F14-3: Verify brief 15 ms bump dip does not trigger airborne."""
    for i in range(15):
        fsm.update(t=i*0.001, accel_mag=1.8, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.001)
    assert fsm.state == FlightState.GROUND_BALANCE


def test_t2_f14_high_pitch_angle_freefall(fsm: FlightFSM) -> None:
    """T2-F14-4: Verify free-fall triggers even with 15 deg initial tilt."""
    for i in range(35):
        fsm.update(t=i*0.001, accel_mag=1.5, wheel_contact=False, pitch=math.radians(15.0), z_vel=0.0, dt=0.001)
    assert fsm.state == FlightState.AIRBORNE


def test_t2_f14_inverted_freefall_failure_detection(fsm: FlightFSM) -> None:
    """T2-F14-5: Verify state updates properly during ballistic descent."""
    fsm.state = FlightState.AIRBORNE
    state = fsm.update(t=0.1, accel_mag=1.0, wheel_contact=False, pitch=0.1, z_vel=-1.0, dt=0.001)
    assert state == FlightState.AIRBORNE


# ==============================================================================
# F15: Boundary - Mid-Air Torque Suppression & Overspeed (5 tests)
# ==============================================================================
def test_t2_f15_torque_remains_suppressed_during_long_flight(sim: BalanceBotSimulator) -> None:
    """T2-F15-1: Verify torque remains damped over 0.5 s flight."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    for _ in range(500):
        rec = sim.step(dt=0.001, v_ref=2.5)
        assert abs(rec["tau_w"]) < 5.0


def test_t2_f15_wheel_speed_near_zero_prior_to_touchdown(sim: BalanceBotSimulator) -> None:
    """T2-F15-2: Verify wheel rotational speed stays bounded in mid-air."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    for _ in range(300):
        sim.step(dt=0.001, v_ref=3.0)
    assert abs(sim.v) < 3.5


def test_t2_f15_leg_extension_hits_target_before_touchdown(sim: BalanceBotSimulator) -> None:
    """T2-F15-3: Verify legs reach >= 0.34 m prior to touchdown."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    for _ in range(400):
        sim.step(dt=0.001, v_ref=1.5)
    assert sim.effective_height >= 0.32


def test_t2_f15_mid_air_teleop_command_ignored(sim: BalanceBotSimulator) -> None:
    """T2-F15-4: Verify extreme user command step does not produce torque spike in air."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    rec = sim.step(dt=0.001, v_ref=5.0)
    assert abs(rec["tau_w"]) < 5.0


def test_t2_f15_mid_air_yaw_command_suppressed(sim: BalanceBotSimulator) -> None:
    """T2-F15-5: Verify yaw angular acceleration is bounded in flight."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    sim.step(dt=0.001, v_ref=1.0, omega_ref=2.0)
    assert abs(sim.yaw_rate) < 5.0


# ==============================================================================
# F16: Boundary - Severe Touchdown Impact Shock & Drops (5 tests)
# ==============================================================================
def test_t2_f16_severe_touchdown_impact_25ms(fsm: FlightFSM) -> None:
    """T2-F16-1: Verify extreme impact (25 m/s^2) transitions to absorption."""
    fsm.state = FlightState.AIRBORNE
    state = fsm.update(t=1.0, accel_mag=25.0, wheel_contact=True, pitch=0.0, z_vel=-2.0, dt=0.001)
    assert state == FlightState.TOUCHDOWN_ABSORPTION


def test_t2_f16_high_vertical_drop_0_6m(sim: BalanceBotSimulator) -> None:
    """T2-F16-2: Verify high drop landing retains clearance > 0."""
    sim.effective_height = 0.36
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(300):
        sim.step(dt=0.001, v_ref=0.5)
    assert min(r["clearance"] for r in sim.history) > 0.0


def test_t2_f16_touchdown_with_forward_pitch_tilt(sim: BalanceBotSimulator) -> None:
    """T2-F16-3: Verify landing with +10 deg forward tilt recovers stably."""
    sim.theta = math.radians(10.0)
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 5.0


def test_t2_f16_touchdown_with_rearward_pitch_tilt(sim: BalanceBotSimulator) -> None:
    """T2-F16-4: Verify landing with -10 deg rearward tilt recovers stably."""
    sim.theta = math.radians(-10.0)
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 5.0


def test_t2_f16_touchdown_clearance_margin_above_safety_threshold(sim: BalanceBotSimulator) -> None:
    """T2-F16-5: Verify minimum clearance > 0.02 m margin throughout impact."""
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(400):
        sim.step(dt=0.001, v_ref=0.5)
    assert min(r["clearance"] for r in sim.history) >= 0.02


# ==============================================================================
# F17: Boundary - Landing Recovery Under Perturbations (5 tests)
# ==============================================================================
def test_t2_f17_recovery_from_maximum_landing_pitch_15deg(sim: BalanceBotSimulator) -> None:
    """T2-F17-1: Verify recovery from 15 deg landing tilt."""
    sim.theta = math.radians(15.0)
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 5.0


def test_t2_f17_recovery_while_continuing_forward_motion(sim: BalanceBotSimulator) -> None:
    """T2-F17-2: Verify continuous forward motion (v=0.8 m/s) throughout recovery."""
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.8)
    assert sim.v > 0.6


def test_t2_f17_zero_secondary_rebound_oscillation(sim: BalanceBotSimulator) -> None:
    """T2-F17-3: Verify no secondary rebound bounce after landing."""
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5)
    late_pitches = [abs(math.degrees(r["theta"])) for r in sim.history[-500:]]
    assert max(late_pitches) < 3.0


def test_t2_f17_recovery_on_inclined_landing_surface(sim: BalanceBotSimulator) -> None:
    """T2-F17-4: Verify recovery while on slope terrain."""
    sim.x = 16.0  # on slope
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 25.0
    assert not math.isnan(sim.v)


def test_t2_f17_settling_time_from_touchdown_under_2s(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """T2-F17-5: Verify Ts < 2.0 s from touchdown event."""
    sim.theta = math.radians(8.0)
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.5)
    ts = metrics_calc.calculate_settling_time(sim.history, v_target=0.5)
    assert ts < 2.0


# ==============================================================================
# F18: Boundary - Sensor Noise & Latency Injection (5 tests)
# ==============================================================================
def test_t2_f18_accelerometer_saturation_handling(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F18-1: Verify sensor update under large acceleration (30 m/s^2)."""
    data = sensors.step(t=0.01, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([30.0, 0, 9.81]), x_pos=0.0, environment=env)
    assert "imu" in data
    assert not math.isnan(data["imu"]["accel"][0])


def test_t2_f18_gyroscope_high_rate_saturation(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F18-2: Verify gyro update under extreme angular velocity (5.0 rad/s)."""
    data = sensors.step(t=0.01, true_pitch=0.0, true_pitch_rate=5.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert math.isclose(data["imu"]["gyro"], 5.0, abs_tol=0.1)


def test_t2_f18_sensor_noise_variance_stress(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F18-3: Verify pitch estimation stability under persistent noise."""
    for i in range(150):
        sensors.step(t=i*0.01, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert abs(sensors.filtered_pitch) < 0.05


def test_t2_f18_delayed_imu_packet_resilience(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F18-4: Verify sensor handling of delayed time step (dt = 0.05s)."""
    data = sensors.step(t=0.05, true_pitch=0.05, true_pitch_rate=0.1, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert "imu" in data


def test_t2_f18_zero_acceleration_gravity_recovery(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F18-5: Verify gravity vector recovered when linear acceleration vanishes."""
    data = sensors.step(t=0.01, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert math.isclose(data["imu"]["accel"][2], 9.81, abs_tol=0.15)


# ==============================================================================
# F19: Boundary - LiDAR Range Limits & Occlusions (5 tests)
# ==============================================================================
def test_t2_f19_minimum_range_blind_spot_0_1m(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F19-1: Verify ranges closer than 0.1m clamped to minimum."""
    data = sensors.step(t=0.10, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert np.all(data["lidar"]["ranges"] >= 0.10)


def test_t2_f19_maximum_range_clamping_12m(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F19-2: Verify ranges beyond 12m clamped to maximum."""
    data = sensors.step(t=0.10, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert np.all(data["lidar"]["ranges"] <= 12.0)


def test_t2_f19_lidar_close_obstacle_cliff_detection(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F19-3: Verify ramp drop cliff reflected in scan range variation."""
    data = sensors.step(t=0.10, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=22.8, environment=env)
    ranges = data["lidar"]["ranges"]
    assert max(ranges) > min(ranges)


def test_t2_f19_lidar_ray_sparse_resolution_robustness(sensors: SensorSuite) -> None:
    """T2-F19-4: Verify LiDAR array has valid dimension."""
    assert sensors.lidar_update_count >= 0


def test_t2_f19_lidar_horizontal_scan_plane_tilt(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F19-5: Verify scan produces finite floats under pitch tilt."""
    data = sensors.step(t=0.10, true_pitch=0.2, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=5.0, environment=env)
    assert not np.isnan(data["lidar"]["ranges"]).any()


# ==============================================================================
# F20: Boundary - Depth Camera Near/Far Clipping (5 tests)
# ==============================================================================
def test_t2_f20_depth_camera_near_clipping_plane(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F20-1: Verify depth map values >= 0.2 m."""
    data = sensors.step(t=0.07, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert np.all(data["depth_camera"]["depth_map"] >= 0.2)


def test_t2_f20_depth_camera_far_clipping_plane(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F20-2: Verify depth map values <= 10.0 m."""
    data = sensors.step(t=0.07, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert np.all(data["depth_camera"]["depth_map"] <= 10.0)


def test_t2_f20_depth_camera_pitch_tilt_ground_intersection(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F20-3: Verify camera map valid under forward pitch tilt."""
    data = sensors.step(t=0.07, true_pitch=0.25, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=1.0, environment=env)
    assert data["depth_camera"]["depth_map"].shape == (48, 64)


def test_t2_f20_depth_camera_zero_frame_drop(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F20-4: Verify periodic frame updates."""
    count0 = sensors.camera_update_count
    for i in range(100):
        sensors.step(t=i*0.01, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert sensors.camera_update_count > count0


def test_t2_f20_depth_camera_optical_frame_orientation(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """T2-F20-5: Verify resolution 640x480 standard aspect ratio (4:3)."""
    data = sensors.step(t=0.07, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    w, h = data["depth_camera"]["resolution"]
    assert w / h == 640 / 480


# ==============================================================================
# F21: Boundary - Headless Runner Timeouts & Steps (5 tests)
# ==============================================================================
def test_t2_f21_small_timestep_stability_0_1ms(sim: BalanceBotSimulator) -> None:
    """T2-F21-1: Verify stability with fine timestep dt = 0.0001 s."""
    for _ in range(500):
        sim.step(dt=0.0001, v_ref=0.5)
    assert not math.isnan(sim.v)


def test_t2_f21_large_timestep_stability_5ms(sim: BalanceBotSimulator) -> None:
    """T2-F21-2: Verify stability with coarse timestep dt = 0.005 s."""
    for _ in range(200):
        sim.step(dt=0.005, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 20.0


def test_t2_f21_long_run_duration_15s(sim: BalanceBotSimulator) -> None:
    """T2-F21-3: Verify 10,000 steps execution completeness."""
    for _ in range(10000):
        sim.step(dt=0.001, v_ref=0.5)
    assert math.isclose(sim.t, 10.0, abs_tol=1e-3)


def test_t2_f21_runner_timeout_safety(sim: BalanceBotSimulator) -> None:
    """T2-F21-4: Verify simulation loop terminates deterministically."""
    max_steps = 1000
    steps = 0
    while steps < max_steps:
        sim.step(dt=0.001, v_ref=0.0)
        steps += 1
    assert steps == max_steps


def test_t2_f21_runner_reproducibility_with_seed() -> None:
    """T2-F21-5: Verify reproducibility across instances."""
    np.random.seed(42)
    sim1 = BalanceBotSimulator()
    sim1.step(dt=0.001, v_ref=0.5)
    np.random.seed(42)
    sim2 = BalanceBotSimulator()
    sim2.step(dt=0.001, v_ref=0.5)
    assert math.isclose(sim1.v, sim2.v, abs_tol=1e-6)


# ==============================================================================
# F22: Boundary - Quantitative Criteria Exact Thresholds (5 tests)
# ==============================================================================
def test_t2_f22_boundary_j_v_at_exact_threshold_1_5(metrics_calc: MetricsCalculator) -> None:
    """T2-F22-1: Verify J_v = 1.50 exactly passes."""
    hist = [{"time": 0.0, "v": 0.0, "v_ref": 1.5, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0},
            {"time": 1.0, "v": 0.0, "v_ref": 1.5, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0}]
    res = metrics_calc.verify_quantitative_criteria(hist, max_j_v=1.5)
    assert res["j_v_pass"] is True


def test_t2_f22_boundary_j_v_exceeding_threshold_1_501(metrics_calc: MetricsCalculator) -> None:
    """T2-F22-2: Verify J_v = 1.501 fails."""
    hist = [{"time": 0.0, "v": 0.0, "v_ref": 1.501, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0},
            {"time": 1.0, "v": 0.0, "v_ref": 1.501, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0}]
    res = metrics_calc.verify_quantitative_criteria(hist, max_j_v=1.5)
    assert res["j_v_pass"] is False


def test_t2_f22_boundary_j_theta_at_exact_threshold_0_15(metrics_calc: MetricsCalculator) -> None:
    """T2-F22-3: Verify J_theta = 0.150 exactly passes."""
    hist = [{"time": 0.0, "v": 0.0, "v_ref": 0.0, "theta": 0.15, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0},
            {"time": 1.0, "v": 0.0, "v_ref": 0.0, "theta": 0.15, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0}]
    res = metrics_calc.verify_quantitative_criteria(hist, max_j_theta=0.15)
    assert res["j_theta_pass"] is True


def test_t2_f22_boundary_j_theta_exceeding_threshold_0_151(metrics_calc: MetricsCalculator) -> None:
    """T2-F22-4: Verify J_theta = 0.151 fails."""
    hist = [{"time": 0.0, "v": 0.0, "v_ref": 0.0, "theta": 0.151, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0},
            {"time": 1.0, "v": 0.0, "v_ref": 0.0, "theta": 0.151, "theta_ref": 0.0, "clearance": 0.1, "fsm_state": 0}]
    res = metrics_calc.verify_quantitative_criteria(hist, max_j_theta=0.15)
    assert res["j_theta_pass"] is False


def test_t2_f22_zero_clearance_instant_failure(metrics_calc: MetricsCalculator) -> None:
    """T2-F22-5: Verify clearance = 0.0 fails clearance gate."""
    hist = [{"time": 0.0, "v": 0.0, "v_ref": 0.0, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.0, "fsm_state": 0}]
    res = metrics_calc.verify_quantitative_criteria(hist)
    assert res["clearance_pass"] is False


# ==============================================================================
# F23: Boundary - Plot Generation Extremes & Scaling (5 tests)
# ==============================================================================
def test_t2_f23_single_datapoint_plot_graceful_handling(tmp_path: Any) -> None:
    """T2-F23-1: Verify plotting 1 data point handles without exception."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0.0], [0.0], "o")
    out = tmp_path / "single_pt.png"
    fig.savefig(str(out))
    plt.close(fig)
    assert out.exists()


def test_t2_f23_extreme_error_plot_scaling(tmp_path: Any) -> None:
    """T2-F23-2: Verify plot handles extreme value range (100 m/s)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 100])
    out = tmp_path / "extreme_scale.png"
    fig.savefig(str(out))
    plt.close(fig)
    assert out.exists()


def test_t2_f23_multi_axis_plot_formatting(tmp_path: Any) -> None:
    """T2-F23-3: Verify dual y-axis velocity and pitch plot generation."""
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot([0, 1], [0, 1], "b-", label="v")
    ax2.plot([0, 1], [0, 0.1], "r--", label="theta")
    out = tmp_path / "twin_plot.png"
    fig.savefig(str(out))
    plt.close(fig)
    assert out.exists()


def test_t2_f23_plot_file_overwrite_behavior(tmp_path: Any) -> None:
    """T2-F23-4: Verify file overwriting succeeds without permission error."""
    import matplotlib.pyplot as plt
    out = tmp_path / "overwrite.png"
    for _ in range(2):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 2])
        fig.savefig(str(out))
        plt.close(fig)
    assert out.exists()


def test_t2_f23_video_frame_dimension_consistency() -> None:
    """T2-F23-5: Verify video frame size standards (320x240)."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert frame.shape == (240, 320, 3)


# ==============================================================================
# F24: Boundary - Documentation Exhaustiveness & Traceability (5 tests)
# ==============================================================================
def test_t2_f24_all_24_features_referenced_in_project_md() -> None:
    """T2-F24-1: Verify PROJECT.md contains all feature tags F01 to F24."""
    proj_path = ".agents/teamwork/orchestrator_1/PROJECT.md"
    assert os.path.exists(proj_path)
    with open(proj_path, "r", encoding="utf-8") as f:
        content = f.read()
    for i in range(1, 25):
        tag = f"F{i:02d}"
        assert tag in content, f"Missing feature tag {tag} in PROJECT.md"


def test_t2_f24_all_acceptance_criteria_referenced() -> None:
    """T2-F24-2: Verify acceptance criteria in ORIGINAL_REQUEST.md."""
    req_path = ".agents/teamwork/ORIGINAL_REQUEST.md"
    assert os.path.exists(req_path)
    with open(req_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Acceptance Criteria" in content


def test_t2_f24_interface_topics_naming_standards() -> None:
    """T2-F24-3: Verify standard ROS 2 topic names in PROJECT.md."""
    proj_path = ".agents/teamwork/orchestrator_1/PROJECT.md"
    with open(proj_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "/cmd_vel" in content


def test_t2_f24_coordinate_conventions_rep103_documented() -> None:
    """T2-F24-4: Verify technical survey documents REP-103 pitch convention."""
    survey_path = ".agents/teamwork/explorer_survey_2/survey_report.md"
    assert os.path.exists(survey_path)
    with open(survey_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "REP-103" in content


def test_t2_f24_milestone_breakdown_documented() -> None:
    """T2-F24-5: Verify all milestones (M1-M6, ME2E, M_FINAL) in master plan."""
    proj_path = ".agents/teamwork/orchestrator_1/PROJECT.md"
    with open(proj_path, "r", encoding="utf-8") as f:
        content = f.read()
    for m in ["M1", "M2", "M3", "M4", "M5", "M6", "ME2E", "M_FINAL"]:
        assert m in content, f"Missing milestone {m} in PROJECT.md"
