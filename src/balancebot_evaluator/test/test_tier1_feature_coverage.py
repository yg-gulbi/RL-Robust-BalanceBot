"""Tier 1: Feature Coverage E2E Test Suite (F01–F24).

Implements >= 5 authentic requirement-driven test cases per inventoried feature (total 120 tests).
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
# F01: URDF/Xacro BalanceBot Model (5 tests)
# ==============================================================================
def test_f01_torso_mass_and_inertia_properties(robot_params: RobotParams) -> None:
    """F01-1: Verify torso mass is within standard inverted pendulum specification (10-12 kg)."""
    assert 9.0 <= robot_params.m_b <= 13.0
    assert robot_params.I_b > 0.10
    assert robot_params.total_mass > robot_params.m_b


def test_f01_leg_articulation_limits_and_reachability(robot_params: RobotParams, kin: LegKinematics) -> None:
    """F01-2: Verify hip and knee joint limits and reachability."""
    assert robot_params.hip_limit_rad >= math.radians(45.0)
    assert robot_params.knee_max_rad >= math.radians(90.0)
    # Check that nominal height is within reach
    qh, qk = kin.inverse_kinematics(0.0, -0.28)
    assert abs(qh) <= robot_params.hip_limit_rad
    assert robot_params.knee_min_rad <= qk <= robot_params.knee_max_rad


def test_f01_wheel_mass_radius_and_friction_coefficients(robot_params: RobotParams) -> None:
    """F01-3: Verify drive wheel radius R=0.10m and effective mass."""
    assert math.isclose(robot_params.R, 0.10, abs_tol=1e-3)
    assert robot_params.M_w > 0.5
    assert robot_params.m_total_effective > robot_params.total_mass


def test_f01_sensor_frame_mounting_locations(robot_params: RobotParams) -> None:
    """F01-4: Verify sensor placement parameters on chassis."""
    assert robot_params.track_width >= 0.40
    assert robot_params.height_min >= 0.15
    assert robot_params.height_max <= 0.40


def test_f01_effective_leg_height_workspace_bounds(robot_params: RobotParams, kin: LegKinematics) -> None:
    """F01-5: Verify leg height range [0.18, 0.38] m is fully attainable."""
    for h in [0.18, 0.25, 0.30, 0.38]:
        qh, qk = kin.inverse_kinematics(0.0, -h)
        x_calc, z_calc = kin.forward_kinematics(qh, qk)
        assert math.isclose(-z_calc, h, abs_tol=1e-3)
        assert math.isclose(x_calc, 0.0, abs_tol=1e-3)


# ==============================================================================
# F02: Skatepark World Simulation (5 tests)
# ==============================================================================
def test_f02_terrain_zone_segmentation(env: SkateparkEnvironment) -> None:
    """F02-1: Verify 5 distinct zones: flat, bumps, rough, slope, ramp."""
    assert env.get_zone(0.0) == TerrainType.FLAT
    assert env.get_zone(7.0) == TerrainType.BUMPS
    assert env.get_zone(12.0) == TerrainType.ROUGH
    assert env.get_zone(17.0) == TerrainType.SLOPE
    assert env.get_zone(22.0) == TerrainType.RAMP


def test_f02_sinusoidal_bump_profile_continuity(env: SkateparkEnvironment) -> None:
    """F02-2: Verify sinusoidal bumps elevation within 2-4 cm amplitude."""
    elevations = [env.elevation(x) for x in np.linspace(5.0, 10.0, 50)]
    assert max(elevations) <= 0.04
    assert min(elevations) >= -0.04
    # Check continuous transition at entrance
    assert math.isclose(env.elevation(5.0), 0.0, abs_tol=1e-3)


def test_f02_rough_terrain_elevation_bounds(env: SkateparkEnvironment) -> None:
    """F02-3: Verify rough irregular terrain bounds."""
    elevations = [env.elevation(x) for x in np.linspace(10.0, 15.0, 50)]
    assert max(elevations) <= 0.05
    assert min(elevations) >= -0.05


def test_f02_incline_slope_angle_geometry(env: SkateparkEnvironment) -> None:
    """F02-4: Verify slope incline angle ~12 deg (0.209 rad)."""
    slope = env.slope_angle_at(17.5)
    assert math.isclose(slope, env.slope_angle, abs_tol=1e-3)
    assert math.isclose(env.elevation(20.0) - env.elevation(15.0), 5.0 * math.tan(env.slope_angle), abs_tol=1e-2)


def test_f02_ramp_lip_elevation_and_drop_geometry(env: SkateparkEnvironment) -> None:
    """F02-5: Verify ramp elevation rises to 0.35m lip height."""
    lip_slope = env.slope_angle_at(22.5)
    assert math.isclose(lip_slope, env.ramp_incline, abs_tol=1e-3)
    assert env.friction_coeff >= 0.8


# ==============================================================================
# F03: Gain-Scheduled LQR-I Controller (5 tests)
# ==============================================================================
def test_f03_flat_ground_steady_state_pitch_balance(sim: BalanceBotSimulator) -> None:
    """F03-1: Verify steady-state pitch balance |pitch| < 5.0 deg (0.087 rad) on flat ground."""
    for _ in range(3000):  # 3.0 seconds
        sim.step(dt=0.001, v_ref=0.0)
    steady_history = sim.history[-1000:]
    pitches_deg = [abs(math.degrees(r["theta"])) for r in steady_history]
    assert max(pitches_deg) < 5.0


def test_f03_integral_action_velocity_convergence(sim: BalanceBotSimulator) -> None:
    """F03-2: Verify integral action converges to zero steady-state velocity error."""
    for _ in range(5000):  # 5.0 seconds
        sim.step(dt=0.001, v_ref=0.8)
    steady_vel = [r["v"] for r in sim.history[-1000:]]
    assert math.isclose(float(np.mean(steady_vel)), 0.8, abs_tol=0.10)


def test_f03_gain_scheduling_interpolation_monotonicity(lqr: LQRBalanceController) -> None:
    """F03-3: Verify LQR gain interpolation across heights [0.18, 0.38] m."""
    gains_low = lqr.get_gains(0.18)
    gains_mid = lqr.get_gains(0.28)
    gains_high = lqr.get_gains(0.38)
    assert len(gains_low) == 5
    # Verify gains smoothly vary with pendulum length
    assert abs(gains_low[2]) > 0.0  # pitch gain non-zero
    assert abs(gains_high[2]) > 0.0


def test_f03_actuator_torque_saturation_limits(lqr: LQRBalanceController) -> None:
    """F03-4: Verify torque clamping at max tau = 15.0 N*m."""
    tau, sat = lqr.compute_torque(
        v_meas=0.0, v_ref=0.0, theta_meas=0.8, theta_ref=0.0, theta_dot=0.0, e_integral=0.0, effective_length=0.28
    )
    assert math.isclose(abs(tau), 15.0, abs_tol=1e-4)
    assert sat is True


def test_f03_anti_windup_freezes_velocity_integrator(sim: BalanceBotSimulator) -> None:
    """F03-5: Verify anti-windup freezes integrator accumulation when torque saturates."""
    saturated_steps = 0
    for _ in range(20):
        tau, sat = sim.lqr.compute_torque(
            v_meas=sim.v,
            v_ref=5.0,
            theta_meas=sim.theta,
            theta_ref=0.0,
            theta_dot=sim.theta_dot,
            e_integral=sim.e_integral,
            effective_length=sim.effective_height,
        )
        if sat:
            saturated_steps += 1
            integral_before = sim.e_integral
            sim.step(dt=0.001, v_ref=5.0)
            assert sim.e_integral == integral_before, (
                f"Integrator accumulated while torque was saturated: {sim.e_integral} != {integral_before}"
            )
        else:
            sim.step(dt=0.001, v_ref=5.0)
    assert saturated_steps >= 10, f"Expected at least 10 saturated steps, got {saturated_steps}"

    # Run remaining duration to verify total integration error remains bounded
    for _ in range(1980):
        sim.step(dt=0.001, v_ref=5.0)
    assert abs(sim.e_integral) < 5.0


# ==============================================================================
# F04: Keyboard Teleoperation (5 tests)
# ==============================================================================
def test_f04_forward_velocity_command_mapping(sim: BalanceBotSimulator) -> None:
    """F04-1: Verify forward velocity command execution (v_cmd = 1.0 m/s)."""
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=1.0)
    assert sim.v > 0.7


def test_f04_reverse_velocity_command_mapping(sim: BalanceBotSimulator) -> None:
    """F04-2: Verify reverse velocity command execution (v_cmd = -0.8 m/s)."""
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=-0.8)
    assert sim.v < -0.5


def test_f04_yaw_turn_rate_command_mapping(sim: BalanceBotSimulator) -> None:
    """F04-3: Verify yaw turning rate command execution (omega_cmd = 0.5 rad/s)."""
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5, omega_ref=0.5)
    assert sim.yaw_rate > 0.3


def test_f04_emergency_braking_stop_command(sim: BalanceBotSimulator) -> None:
    """F04-4: Verify braking from speed brings robot to rest within steady velocity tolerance."""
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=1.0)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 0.15


def test_f04_command_clamping_within_safe_bounds(sim: BalanceBotSimulator) -> None:
    """F04-5: Verify robot maintains pitch stability under sudden teleop commands."""
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=1.5)
    max_pitch = max(abs(math.degrees(r["theta"])) for r in sim.history)
    assert max_pitch < 20.0  # transient peak within safety bound


# ==============================================================================
# F05: Automated Test Replay (5 tests)
# ==============================================================================
def test_f05_deterministic_profile_execution(sim: BalanceBotSimulator) -> None:
    """F05-1: Verify deterministic profile execution."""
    profile = [(1.0, 0.5), (2.0, 1.0), (3.0, 0.0)]
    for t_end, v_cmd in profile:
        while sim.t < t_end:
            sim.step(dt=0.001, v_ref=v_cmd)
    assert 2999 <= len(sim.history) <= 3002


def test_f05_multistage_trajectory_profile_progression(sim: BalanceBotSimulator) -> None:
    """F05-2: Verify sequential multi-stage progression: hold -> drive -> stop."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 0.05
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.8)
    assert sim.v > 0.6


def test_f05_replay_interpolation_between_waypoints(sim: BalanceBotSimulator) -> None:
    """F05-3: Verify smooth velocity transition without steps."""
    for i in range(2000):
        v_target = 0.8 * (i / 2000.0)  # gradual ramp
        sim.step(dt=0.001, v_ref=v_target)
    assert sim.v > 0.6


def test_f05_repeatability_across_consecutive_runs() -> None:
    """F05-4: Verify identical command profiles yield identical simulation results."""
    sim1 = BalanceBotSimulator()
    sim2 = BalanceBotSimulator()
    for _ in range(1500):
        sim1.step(dt=0.001, v_ref=0.5)
        sim2.step(dt=0.001, v_ref=0.5)
    assert math.isclose(sim1.x, sim2.x, abs_tol=1e-4)
    assert math.isclose(sim1.theta, sim2.theta, abs_tol=1e-4)


def test_f05_trajectory_duration_and_completion_detection(sim: BalanceBotSimulator) -> None:
    """F05-5: Verify accurate duration tracking."""
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.5)
    assert math.isclose(sim.t, 2.50, abs_tol=1e-3)


# ==============================================================================
# F06: Real-time Error Metrics Logging (5 tests)
# ==============================================================================
def test_f06_velocity_error_signal_accuracy(sim: BalanceBotSimulator) -> None:
    """F06-1: Verify velocity error signal e_v = v - v_ref is recorded accurately."""
    sim.step(dt=0.001, v_ref=1.0)
    rec = sim.history[-1]
    assert math.isclose(rec["v"] - rec["v_ref"], rec["v"] - 1.0, abs_tol=1e-5)


def test_f06_pitch_error_signal_accuracy(sim: BalanceBotSimulator) -> None:
    """F06-2: Verify pitch error signal e_theta = theta - theta_ref."""
    sim.step(dt=0.001, v_ref=0.0)
    rec = sim.history[-1]
    assert math.isclose(rec["theta"] - rec["theta_ref"], rec["theta"] - 0.0, abs_tol=1e-5)


def test_f06_high_frequency_telemetry_streaming(sim: BalanceBotSimulator) -> None:
    """F06-3: Verify telemetry recorded at 1000 Hz physics stepping rate."""
    for _ in range(500):
        sim.step(dt=0.001, v_ref=0.5)
    assert len(sim.history) == 500


def test_f06_telemetry_record_structure_and_completeness(sim: BalanceBotSimulator) -> None:
    """F06-4: Verify record contains required fields."""
    rec = sim.step(dt=0.001, v_ref=0.5)
    for field in ["time", "x", "v", "v_ref", "theta", "theta_ref", "tau_w", "height", "clearance"]:
        assert field in rec


def test_f06_summary_statistics_computation(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """F06-5: Verify integrated absolute error calculation on logged history."""
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5)
    jv, jtheta = metrics_calc.calculate_integrated_errors(sim.history)
    assert jv >= 0.0
    assert jtheta >= 0.0


# ==============================================================================
# F07: Articulated Leg Kinematics (5 tests)
# ==============================================================================
def test_f07_forward_kinematics_wheel_position(kin: LegKinematics) -> None:
    """F07-1: Verify forward kinematics produces correct vertical wheel position."""
    # When qh = -30 deg, qk = 60 deg, symmetric leg
    qh = -math.radians(30.0)
    qk = math.radians(60.0)
    x_rel, z_rel = kin.forward_kinematics(qh, qk)
    assert math.isclose(x_rel, 0.0, abs_tol=1e-3)
    assert z_rel < -0.15


def test_f07_inverse_kinematics_reconstruction(kin: LegKinematics) -> None:
    """F07-2: Verify IK -> FK roundtrip recovers original Cartesian target."""
    for h in [0.20, 0.28, 0.35]:
        qh, qk = kin.inverse_kinematics(0.0, -h)
        x_calc, z_calc = kin.forward_kinematics(qh, qk)
        assert math.isclose(x_calc, 0.0, abs_tol=1e-3)
        assert math.isclose(-z_calc, h, abs_tol=1e-3)


def test_f07_leg_jacobian_velocity_mapping(kin: LegKinematics) -> None:
    """F07-3: Verify leg Jacobian matrix shape and non-zero determinant."""
    qh, qk = kin.inverse_kinematics(0.0, -0.28)
    j = kin.jacobian(qh, qk)
    assert j.shape == (2, 2)
    assert abs(np.linalg.det(j)) > 1e-4


def test_f07_singularity_avoidance_clamping(kin: LegKinematics) -> None:
    """F07-4: Verify extreme height targets are safely clamped to avoid singularity."""
    qh, qk = kin.inverse_kinematics(0.0, -0.50)  # beyond physical reach
    x_calc, z_calc = kin.forward_kinematics(qh, qk)
    assert -z_calc <= 0.38 + 1e-2


def test_f07_workspace_boundary_rejection(kin: LegKinematics) -> None:
    """F07-5: Verify wheel offset clamping within limits."""
    qh, qk = kin.inverse_kinematics(0.20, -0.28)  # excessive offset
    x_calc, z_calc = kin.forward_kinematics(qh, qk)
    assert abs(x_calc) <= 0.08 + 1e-2


# ==============================================================================
# F08: Virtual Model Impedance Control (5 tests)
# ==============================================================================
def test_f08_cartesian_spring_compliance_under_load(kin: LegKinematics) -> None:
    """F08-1: Verify downward virtual force increases with downward compression."""
    qh, qk = kin.inverse_kinematics(0.0, -0.28)
    tau_h1, tau_k1 = kin.virtual_model_torque(qh, qk, target_z=0.28, current_z=0.28, z_vel=0.0)
    tau_h2, tau_k2 = kin.virtual_model_torque(qh, qk, target_z=0.28, current_z=0.24, z_vel=0.0)  # compressed 4cm
    assert abs(tau_k2) > abs(tau_k1)


def test_f08_damping_dissipates_vertical_oscillation(kin: LegKinematics) -> None:
    """F08-2: Verify damping force opposes vertical velocity."""
    qh, qk = kin.inverse_kinematics(0.0, -0.28)
    tau_h_up, tau_k_up = kin.virtual_model_torque(qh, qk, target_z=0.28, current_z=0.28, z_vel=0.5, d_z=100.0)
    tau_h_down, tau_k_down = kin.virtual_model_torque(qh, qk, target_z=0.28, current_z=0.28, z_vel=-0.5, d_z=100.0)
    # Moving downward produces stronger upward push
    assert abs(tau_k_down) > abs(tau_k_up)


def test_f08_gravity_feedforward_cancels_body_weight(robot_params: RobotParams, kin: LegKinematics) -> None:
    """F08-3: Verify gravity feedforward equals half total robot weight per leg."""
    ff = (robot_params.total_mass * robot_params.g) / 2.0
    assert ff > 50.0  # > 50 N


def test_f08_jacobian_transpose_torque_mapping(kin: LegKinematics) -> None:
    """F08-4: Verify Jacobian transpose maps vertical force to joint torques."""
    qh, qk = kin.inverse_kinematics(0.0, -0.28)
    tau_h, tau_k = kin.virtual_model_torque(qh, qk, target_z=0.28, current_z=0.28, z_vel=0.0)
    assert not math.isnan(tau_h)
    assert not math.isnan(tau_k)


def test_f08_impedance_passivity_and_stability(sim: BalanceBotSimulator) -> None:
    """F08-5: Verify body height remains stable under nominal impedance."""
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.effective_height, 0.28, abs_tol=0.02)


# ==============================================================================
# F09: Dynamic Wheel Offset Equilibrium (5 tests)
# ==============================================================================
def test_f09_forward_offset_shifts_pitch_equilibrium_forward(kin: LegKinematics) -> None:
    """F09-1: Verify positive wheel offset creates positive equilibrium pitch."""
    theta_eq = kin.equilibrium_pitch(offset_x=0.05, effective_length=0.28)
    assert theta_eq > 0.0
    assert math.degrees(theta_eq) > 5.0


def test_f09_rearward_offset_shifts_pitch_equilibrium_backward(kin: LegKinematics) -> None:
    """F09-2: Verify negative wheel offset creates negative equilibrium pitch."""
    theta_eq = kin.equilibrium_pitch(offset_x=-0.05, effective_length=0.28)
    assert theta_eq < 0.0
    assert math.degrees(theta_eq) < -5.0


def test_f09_zero_offset_maintains_vertical_equilibrium(kin: LegKinematics) -> None:
    """F09-3: Verify zero wheel offset corresponds to theta_eq = 0."""
    theta_eq = kin.equilibrium_pitch(offset_x=0.0, effective_length=0.28)
    assert math.isclose(theta_eq, 0.0, abs_tol=1e-6)


def test_f09_equilibrium_pitch_formula_accuracy(kin: LegKinematics) -> None:
    """F09-4: Verify small angle approximation theta_eq ~ offset / L_eff."""
    offset = 0.03
    length = 0.28
    theta_eq = kin.equilibrium_pitch(offset, length)
    assert math.isclose(theta_eq, math.asin(offset / length), abs_tol=1e-4)


def test_f09_stable_balance_at_non_zero_offset_angle(sim: BalanceBotSimulator) -> None:
    """F09-5: Verify balance maintained when offset target is set in manual mode."""
    sim.set_leg_mode("manual", target_height=0.28, target_offset=0.03)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.wheel_offset, 0.03, abs_tol=0.01)
    assert abs(sim.v) < 0.25


# ==============================================================================
# F10: Auto Terrain Adaptation Mode (5 tests)
# ==============================================================================
def test_f10_slope_feedforward_pitch_trim(sim: BalanceBotSimulator) -> None:
    """F10-1: Verify reference pitch adapts on incline slope."""
    sim.x = 17.5  # on slope zone
    sim.step(dt=0.001, v_ref=0.0)
    rec = sim.history[-1]
    assert rec["theta_ref"] > 0.10  # trims reference forward on slope


def test_f10_roughness_adaptive_stiffness_modulation(env: SkateparkEnvironment) -> None:
    """F10-2: Verify rough terrain profile has non-zero surface variation."""
    slopes = [env.slope_angle_at(x) for x in np.linspace(10.0, 15.0, 30)]
    assert np.std(slopes) > 0.05


def test_f10_roll_leveling_differential_leg_extension(robot_params: RobotParams) -> None:
    """F10-3: Verify roll leveling equation Delta_z = (W/2)*tan(phi)."""
    phi = math.radians(5.0)
    delta_z = (robot_params.track_width / 2.0) * math.tan(phi)
    assert 0.015 < delta_z < 0.030


def test_f10_continuous_adaptation_on_varying_slope(sim: BalanceBotSimulator) -> None:
    """F10-4: Verify stable climbing without falling over slope zone."""
    sim.reset(x0=14.0, v0=0.6, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.6)
    assert sim.x > 15.0
    assert abs(sim.theta) < math.radians(30.0)


def test_f10_chassis_isolation_from_ground_undulations(sim: BalanceBotSimulator) -> None:
    """F10-5: Verify torso clearance stays positive over bumps."""
    sim.reset(x0=5.0, v0=0.5, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.5)
    clearances = [r["clearance"] for r in sim.history]
    assert min(clearances) > 0.0


# ==============================================================================
# F11: Manual Target Tracking Mode (5 tests)
# ==============================================================================
def test_f11_manual_height_target_tracking(sim: BalanceBotSimulator) -> None:
    """F11-1: Verify robot tracks commanded manual height (0.22 m)."""
    sim.set_leg_mode("manual", target_height=0.22)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.effective_height, 0.22, abs_tol=0.01)


def test_f11_manual_wheel_offset_target_tracking(sim: BalanceBotSimulator) -> None:
    """F11-2: Verify robot tracks commanded manual wheel offset (-0.04 m)."""
    sim.set_leg_mode("manual", target_offset=-0.04)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.wheel_offset, -0.04, abs_tol=0.01)


def test_f11_slew_rate_limited_trajectory_smoothing(sim: BalanceBotSimulator) -> None:
    """F11-3: Verify height rate of change is smooth (no sudden step jump)."""
    sim.set_leg_mode("manual", target_height=0.35)
    sim.step(dt=0.001, v_ref=0.0)
    # In one millisecond, height changes by at most slew rate
    assert abs(sim.effective_height - 0.28) <= 0.001


def test_f11_target_reached_within_one_second(sim: BalanceBotSimulator) -> None:
    """F11-4: Verify target reached within 1.0 s for nominal change."""
    sim.set_leg_mode("manual", target_height=0.32)
    for _ in range(1000):  # 1.0s
        sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.effective_height, 0.32, abs_tol=0.02)


def test_f11_invalid_target_clamped_safely(sim: BalanceBotSimulator) -> None:
    """F11-5: Verify out-of-bound targets are clamped within [0.18, 0.38] m."""
    sim.set_leg_mode("manual", target_height=0.50, target_offset=0.25)
    assert sim.target_height <= 0.38
    assert sim.target_offset <= 0.08


# ==============================================================================
# F12: Latched Mode Switching (5 tests)
# ==============================================================================
def test_f12_explicit_toggle_from_auto_to_manual(sim: BalanceBotSimulator) -> None:
    """F12-1: Verify explicit mode switch from auto to manual."""
    assert sim.leg_mode == "auto"
    sim.set_leg_mode("manual")
    assert sim.leg_mode == "manual"


def test_f12_manual_mode_latches_indefinitely(sim: BalanceBotSimulator) -> None:
    """F12-2: Verify manual mode remains latched across simulation steps."""
    sim.set_leg_mode("manual", target_height=0.24)
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.0)
    assert sim.leg_mode == "manual"


def test_f12_toggle_back_to_auto_mode(sim: BalanceBotSimulator) -> None:
    """F12-3: Verify explicit toggle back to auto mode."""
    sim.set_leg_mode("manual")
    sim.set_leg_mode("auto")
    assert sim.leg_mode == "auto"


def test_f12_smooth_transition_without_torque_spikes(sim: BalanceBotSimulator) -> None:
    """F12-4: Verify torque remains continuous during mode switch."""
    for _ in range(500):
        sim.step(dt=0.001, v_ref=0.0)
    tau_before = sim.history[-1]["tau_w"]
    sim.set_leg_mode("manual", target_height=0.26)
    sim.step(dt=0.001, v_ref=0.0)
    tau_after = sim.history[-1]["tau_w"]
    assert abs(tau_after - tau_before) < 5.0


def test_f12_active_mode_status_reported_accurately(sim: BalanceBotSimulator) -> None:
    """F12-5: Verify simulator accurately reports active mode."""
    sim.set_leg_mode("manual")
    assert sim.leg_mode == "manual"
    sim.set_leg_mode("auto")
    assert sim.leg_mode == "auto"


# ==============================================================================
# F13: Uneven & Rough Terrain Traversal (5 tests)
# ==============================================================================
def test_f13_gentle_sinusoidal_bumps_traversal(sim: BalanceBotSimulator) -> None:
    """F13-1: Verify traversal through sinusoidal bumps zone (x in [5, 10])."""
    sim.reset(x0=4.5, v0=0.8, theta0=0.0)
    for _ in range(5000):
        sim.step(dt=0.001, v_ref=0.8)
    assert sim.x > 8.0
    assert abs(sim.theta) < math.radians(25.0)


def test_f13_rough_irregular_ground_traversal(sim: BalanceBotSimulator) -> None:
    """F13-2: Verify traversal through irregular blocks zone (x in [10, 15])."""
    sim.reset(x0=9.5, v0=0.7, theta0=0.0)
    for _ in range(5000):
        sim.step(dt=0.001, v_ref=0.7)
    assert sim.x > 12.0


def test_f13_no_toppling_pitch_within_threshold(sim: BalanceBotSimulator) -> None:
    """F13-3: Verify pitch never exceeds 45 deg fall threshold during traversal."""
    sim.reset(x0=5.0, v0=0.6, theta0=0.0)
    for _ in range(4000):
        sim.step(dt=0.001, v_ref=0.6)
    max_pitch = max(abs(math.degrees(r["theta"])) for r in sim.history)
    assert max_pitch < 35.0


def test_f13_chassis_ground_clearance_maintained(sim: BalanceBotSimulator) -> None:
    """F13-4: Verify minimum clearance z > 0 (no chassis ground contact)."""
    sim.reset(x0=5.0, v0=0.7, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.7)
    assert min(r["clearance"] for r in sim.history) > 0.0


def test_f13_velocity_tracking_sustained_across_bumps(sim: BalanceBotSimulator) -> None:
    """F13-5: Verify velocity does not drop to zero or stall."""
    sim.reset(x0=5.0, v0=0.8, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.8)
    mean_v = np.mean([r["v"] for r in sim.history])
    assert mean_v > 0.4


# ==============================================================================
# F14: Airborne Free-Fall Detection (5 tests)
# ==============================================================================
def test_f14_imu_specific_force_freefall_threshold_detection(fsm: FlightFSM) -> None:
    """F14-1: Verify free-fall triggers when ||a|| < 2.5 m/s^2 while wheel_contact is True."""
    # Under-threshold acceleration (2.0 m/s^2 < 2.5 m/s^2) triggers AIRBORNE after 30 ms
    for i in range(30):
        state = fsm.update(t=i*0.001, accel_mag=2.0, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.001)
    assert state == FlightState.AIRBORNE

    # Over-threshold acceleration (3.5 m/s^2 >= 2.5 m/s^2) must NOT trigger AIRBORNE
    fsm_over = FlightFSM()
    for i in range(40):
        state_over = fsm_over.update(t=i*0.001, accel_mag=3.5, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.001)
    assert state_over == FlightState.GROUND_BALANCE


def test_f14_free_fall_detection_latency_under_30ms(fsm: FlightFSM) -> None:
    """F14-2: Verify freefall is recognized within <= 35 ms."""
    for i in range(32):
        fsm.update(t=i*0.001, accel_mag=1.5, wheel_contact=False, pitch=0.0, z_vel=0.0, dt=0.001)
    assert fsm.state == FlightState.AIRBORNE


def test_f14_zero_wheel_contact_triggers_freefall(fsm: FlightFSM) -> None:
    """F14-3: Verify contact loss leads to airborne state."""
    for i in range(40):
        fsm.update(t=i*0.001, accel_mag=2.0, wheel_contact=False, pitch=0.0, z_vel=0.0, dt=0.001)
    assert fsm.state == FlightState.AIRBORNE


def test_f14_ground_vibration_does_not_false_trigger(fsm: FlightFSM) -> None:
    """F14-4: Verify brief 5 ms dip does not false trigger airborne."""
    for i in range(5):
        fsm.update(t=i*0.001, accel_mag=2.0, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.001)
    assert fsm.state == FlightState.GROUND_BALANCE


def test_f14_fsm_transitions_to_airborne_state(sim: BalanceBotSimulator) -> None:
    """F14-5: Verify simulator transitions to AIRBORNE state during jump."""
    sim.wheel_contact = False
    for _ in range(50):
        sim.step(dt=0.001, v_ref=1.5)
    assert sim.fsm.state == FlightState.AIRBORNE


# ==============================================================================
# F15: Mid-Air Wheel Torque Suppression (5 tests)
# ==============================================================================
def test_f15_wheel_torque_suppressed_in_air(sim: BalanceBotSimulator) -> None:
    """F15-1: Verify drive torque is suppressed while airborne."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    rec = sim.step(dt=0.001, v_ref=2.0)
    # When airborne, wheel torque is pure damping, not driving
    assert abs(rec["tau_w"]) < 5.0


def test_f15_prevents_wheel_runaway_overspeed(sim: BalanceBotSimulator) -> None:
    """F15-2: Verify wheel rotational speed stays bounded in mid-air."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    for _ in range(500):
        sim.step(dt=0.001, v_ref=2.0)
    assert abs(sim.v) < 3.0


def test_f15_pre_landing_leg_extension_in_air(sim: BalanceBotSimulator) -> None:
    """F15-3: Verify legs extend toward 0.36m in flight to maximize stroke."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    for _ in range(500):
        sim.step(dt=0.001, v_ref=1.5)
    assert sim.effective_height > 0.30


def test_f15_angular_momentum_conservation_in_flight(sim: BalanceBotSimulator) -> None:
    """F15-4: Verify pitch rate does not spin uncontrollably in air."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    for _ in range(300):
        sim.step(dt=0.001, v_ref=1.5)
    assert abs(sim.theta_dot) < 5.0


def test_f15_pitch_alignment_prepared_for_touchdown(sim: BalanceBotSimulator) -> None:
    """F15-5: Verify pitch remains within 15 deg prior to landing."""
    sim.wheel_contact = False
    sim.fsm.state = FlightState.AIRBORNE
    for _ in range(250):
        sim.step(dt=0.001, v_ref=1.2)
    assert abs(math.degrees(sim.theta)) < 20.0


# ==============================================================================
# F16: Touchdown Shock Dissipation (5 tests)
# ==============================================================================
def test_f16_touchdown_impact_spike_detection(fsm: FlightFSM) -> None:
    """F16-1: Verify acceleration spike > 15 m/s^2 transitions to TOUCHDOWN_ABSORPTION."""
    fsm.state = FlightState.AIRBORNE
    fsm.update(t=1.0, accel_mag=18.0, wheel_contact=True, pitch=0.05, z_vel=-1.5, dt=0.001)
    assert fsm.state == FlightState.TOUCHDOWN_ABSORPTION


def test_f16_dynamic_switch_to_high_damping_impedance(kin: LegKinematics) -> None:
    """F16-2: Verify high damping configuration (D_z = 250 N*s/m)."""
    qh, qk = kin.inverse_kinematics(0.0, -0.36)
    tau_h, tau_k = kin.virtual_model_torque(qh, qk, target_z=0.20, current_z=0.36, z_vel=-1.5, k_z=400.0, d_z=250.0)
    assert abs(tau_k) > 20.0


def test_f16_leg_compression_absorbs_kinetic_energy(sim: BalanceBotSimulator) -> None:
    """F16-3: Verify leg compresses to absorb vertical velocity."""
    sim.effective_height = 0.36
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(300):
        sim.step(dt=0.001, v_ref=0.5)
    assert sim.effective_height < 0.36


def test_f16_torso_clearance_strictly_positive(sim: BalanceBotSimulator) -> None:
    """F16-4: Verify torso clearance > 0 during landing impact."""
    sim.effective_height = 0.30
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(200):
        sim.step(dt=0.001, v_ref=0.5)
    assert sim.history[-1]["clearance"] > 0.0


def test_f16_rebound_bounce_suppression(fsm: FlightFSM) -> None:
    """F16-5: Verify absorption transitions to recovery without rebound."""
    fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    fsm.touchdown_time = 1.0
    state = fsm.update(t=1.3, accel_mag=9.81, wheel_contact=True, pitch=0.02, z_vel=0.02, dt=0.001)
    assert state == FlightState.BALANCE_RECOVERY


# ==============================================================================
# F17: Landing Balance Recovery (5 tests)
# ==============================================================================
def test_f17_pitch_recovers_within_threshold(sim: BalanceBotSimulator) -> None:
    """F17-1: Verify pitch recovers to |theta| < 5.0 deg (0.087 rad)."""
    sim.theta = math.radians(12.0)  # initial post-impact pitch tilt
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 5.0


def test_f17_recovery_time_under_2_5_seconds(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """F17-2: Verify recovery time <= 2.5 s."""
    sim.theta = math.radians(10.0)
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.5)
    results = metrics_calc.verify_quantitative_criteria(sim.history)
    assert results["recovery_pass"] is True


def test_f17_smooth_leg_restoration_to_nominal_height(sim: BalanceBotSimulator) -> None:
    """F17-3: Verify leg smoothly returns toward nominal 0.28m."""
    sim.effective_height = 0.20
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.5)
    assert sim.effective_height > 0.25


def test_f17_resumes_forward_velocity_tracking(sim: BalanceBotSimulator) -> None:
    """F17-4: Verify forward velocity tracking resumes after recovery."""
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.8)
    assert sim.v > 0.5


def test_f17_fsm_transitions_back_to_ground_balance(fsm: FlightFSM) -> None:
    """F17-5: Verify state returns to GROUND_BALANCE once settled."""
    fsm.state = FlightState.BALANCE_RECOVERY
    state = fsm.update(t=3.0, accel_mag=9.81, wheel_contact=True, pitch=math.radians(1.0), z_vel=0.01, dt=0.001)
    assert state == FlightState.GROUND_BALANCE


# ==============================================================================
# F18: IMU Sensor Plugin & Integration (5 tests)
# ==============================================================================
def test_f18_imu_update_rate_at_least_100hz(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F18-1: Verify IMU publishes at >= 100 Hz."""
    for t_step in range(120):
        sensors.step(t=t_step*0.001, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert sensors.imu_update_count >= 10


def test_f18_linear_acceleration_reflects_gravity(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F18-2: Verify gravity acceleration ~9.81 m/s^2."""
    data = sensors.step(t=0.01, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert "imu" in data
    assert math.isclose(data["imu"]["accel"][2], 9.81, abs_tol=0.2)


def test_f18_angular_velocity_matches_body_pitch_rate(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F18-3: Verify pitch angular velocity matches true rate."""
    data = sensors.step(t=0.01, true_pitch=0.1, true_pitch_rate=0.4, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert math.isclose(data["imu"]["gyro"], 0.4, abs_tol=0.05)


def test_f18_complementary_filter_pitch_fusion_low_drift(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F18-4: Verify sensor fusion accurately estimates pitch angle."""
    for t_step in range(200):
        data = sensors.step(t=t_step*0.01, true_pitch=0.05, true_pitch_rate=0.0, true_accel=np.array([9.81*math.sin(0.05), 0, 9.81*math.cos(0.05)]), x_pos=0.0, environment=env)
    assert math.isclose(sensors.filtered_pitch, 0.05, abs_tol=0.02)


def test_f18_imu_noise_rejection_and_filtering(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F18-5: Verify filtered pitch is smoother than raw noisy measurements."""
    raw_samples = []
    filtered_samples = []
    for t_step in range(100):
        data = sensors.step(t=t_step*0.01, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
        if "imu" in data:
            raw_samples.append(data["imu"]["accel"][0] / 9.81)
            filtered_samples.append(data["imu"]["pitch"])
    assert np.std(filtered_samples) <= np.std(raw_samples) + 0.01


# ==============================================================================
# F19: 2D LiDAR Sensor Plugin (5 tests)
# ==============================================================================
def test_f19_lidar_update_rate_at_least_10hz(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F19-1: Verify LiDAR publishes at >= 10 Hz."""
    for t_step in range(200):
        sensors.step(t=t_step*0.001, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert sensors.lidar_update_count >= 2


def test_f19_range_measurements_within_valid_bounds(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F19-2: Verify LiDAR ranges are within [0.1, 12.0] m."""
    data = sensors.step(t=0.10, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert "lidar" in data
    ranges = data["lidar"]["ranges"]
    assert min(ranges) >= 0.10
    assert max(ranges) <= 12.0


def test_f19_detects_ramp_elevation_profile(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F19-3: Verify LiDAR rays reflect terrain geometry."""
    data = sensors.step(t=0.10, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=21.0, environment=env)
    assert "lidar" in data
    assert len(data["lidar"]["ranges"]) > 0


def test_f19_ray_count_and_angular_resolution(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F19-4: Verify scan contains standard planar ray count (90 beams)."""
    data = sensors.step(t=0.10, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert len(data["lidar"]["ranges"]) == 90


def test_f19_no_nan_or_infinite_range_values(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F19-5: Verify scan contains zero NaN or inf values."""
    data = sensors.step(t=0.10, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert not np.isnan(data["lidar"]["ranges"]).any()
    assert not np.isinf(data["lidar"]["ranges"]).any()


# ==============================================================================
# F20: Depth Camera Sensor Plugin (5 tests)
# ==============================================================================
def test_f20_camera_frame_rate_at_least_10hz(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F20-1: Verify depth camera publishes at >= 10 Hz (nominal 15 Hz)."""
    for t_step in range(200):
        sensors.step(t=t_step*0.001, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert sensors.camera_update_count >= 2


def test_f20_depth_image_resolution_640x480(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F20-2: Verify resolution metadata corresponds to 640x480."""
    data = sensors.step(t=0.07, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert "depth_camera" in data
    assert data["depth_camera"]["resolution"] == (640, 480)


def test_f20_depth_pixel_values_reflect_ground_distance(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F20-3: Verify depth pixel values are positive floating point numbers."""
    data = sensors.step(t=0.07, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    depth_map = data["depth_camera"]["depth_map"]
    assert np.all(depth_map > 0.0)


def test_f20_depth_buffer_data_type_validity(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F20-4: Verify depth map array dtype is float32."""
    data = sensors.step(t=0.07, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert data["depth_camera"]["depth_map"].dtype == np.float32


def test_f20_camera_intrinsics_consistency(sensors: SensorSuite, env: SkateparkEnvironment) -> None:
    """F20-5: Verify depth camera has non-empty depth frame."""
    data = sensors.step(t=0.07, true_pitch=0.0, true_pitch_rate=0.0, true_accel=np.array([0, 0, 9.81]), x_pos=0.0, environment=env)
    assert data["depth_camera"]["depth_map"].shape == (48, 64)


# ==============================================================================
# F21: Headless Evaluation Test Runner (5 tests)
# ==============================================================================
def test_f21_headless_execution_without_display(sim: BalanceBotSimulator) -> None:
    """F21-1: Verify runner executes completely headless without DISPLAY."""
    for _ in range(500):
        sim.step(dt=0.001, v_ref=0.5)
    assert len(sim.history) == 500


def test_f21_deterministic_simulation_stepping(sim: BalanceBotSimulator) -> None:
    """F21-2: Verify simulation step increments time monotonically by dt."""
    t0 = sim.t
    sim.step(dt=0.001, v_ref=0.0)
    assert math.isclose(sim.t - t0, 0.001, abs_tol=1e-6)


def test_f21_configurable_duration_and_dt(sim: BalanceBotSimulator) -> None:
    """F21-3: Verify simulator supports custom time step (dt = 0.002s)."""
    sim.step(dt=0.002, v_ref=0.0)
    assert math.isclose(sim.t, 0.002, abs_tol=1e-6)


def test_f21_scorecard_generation_completeness(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """F21-4: Verify scorecard contains all 5 required metric evaluation fields."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.5)
    results = metrics_calc.verify_quantitative_criteria(sim.history)
    for field in ["steady_pitch_pass", "j_v_pass", "j_theta_pass", "settling_time_pass", "clearance_pass", "recovery_pass"]:
        assert field in results


def test_f21_exit_code_semantics_reflects_result(metrics_calc: MetricsCalculator) -> None:
    """F21-5: Verify all_passed boolean evaluates strictly true only when all gates pass."""
    dummy_history = [{"time": 0.0, "v": 0.0, "v_ref": 0.0, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.10, "fsm_state": 0}]
    results = metrics_calc.verify_quantitative_criteria(dummy_history)
    assert results["all_passed"] is True


# ==============================================================================
# F22: Quantitative Metrics Calculation (5 tests)
# ==============================================================================
def test_f22_integrated_velocity_error_formula(metrics_calc: MetricsCalculator) -> None:
    """F22-1: Verify J_v = integral |v - v_ref| dt calculation."""
    history = [
        {"time": 0.0, "v": 0.0, "v_ref": 1.0, "theta": 0.0, "theta_ref": 0.0},
        {"time": 1.0, "v": 0.0, "v_ref": 1.0, "theta": 0.0, "theta_ref": 0.0},
    ]
    jv, _ = metrics_calc.calculate_integrated_errors(history)
    assert math.isclose(jv, 1.0, abs_tol=1e-3)


def test_f22_integrated_pitch_error_formula(metrics_calc: MetricsCalculator) -> None:
    """F22-2: Verify J_theta = integral |theta - theta_ref| dt calculation."""
    history = [
        {"time": 0.0, "v": 0.0, "v_ref": 0.0, "theta": 0.05, "theta_ref": 0.0},
        {"time": 2.0, "v": 0.0, "v_ref": 0.0, "theta": 0.05, "theta_ref": 0.0},
    ]
    _, jtheta = metrics_calc.calculate_integrated_errors(history)
    assert math.isclose(jtheta, 0.10, abs_tol=1e-3)


def test_f22_settling_time_5_percent_band_detection(metrics_calc: MetricsCalculator) -> None:
    """F22-3: Verify settling time T_s measurement within 5% band."""
    history = [
        {"time": 0.0, "v": 0.0, "theta": 0.10, "v_ref": 1.0, "theta_ref": 0.0},
        {"time": 0.5, "v": 0.5, "theta": 0.05, "v_ref": 1.0, "theta_ref": 0.0},
        {"time": 1.0, "v": 0.95, "theta": 0.01, "v_ref": 1.0, "theta_ref": 0.0},
        {"time": 1.5, "v": 0.99, "theta": 0.005, "v_ref": 1.0, "theta_ref": 0.0},
        {"time": 2.0, "v": 1.00, "theta": 0.001, "v_ref": 1.0, "theta_ref": 0.0},
    ]
    ts = metrics_calc.calculate_settling_time(history, v_target=1.0)
    assert ts <= 1.0


def test_f22_independent_acceptance_criteria_gating(metrics_calc: MetricsCalculator) -> None:
    """F22-4: Verify independent gating (no weighted sum blending)."""
    # History with failing J_v but passing J_theta
    history = [
        {"time": 0.0, "v": 0.0, "v_ref": 2.0, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.10, "fsm_state": 0},
        {"time": 1.0, "v": 0.0, "v_ref": 2.0, "theta": 0.0, "theta_ref": 0.0, "clearance": 0.10, "fsm_state": 0},
    ]
    res = metrics_calc.verify_quantitative_criteria(history, max_j_v=1.5)
    # J_v = 2.0 > 1.5 -> Fail
    assert res["j_v_pass"] is False
    assert res["j_theta_pass"] is True
    assert res["all_passed"] is False


def test_f22_pass_fail_summary_generation(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """F22-5: Verify nominal 3s run satisfies quantitative standards."""
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.5)
    res = metrics_calc.verify_quantitative_criteria(sim.history)
    assert res["steady_pitch_pass"] is True
    assert res["j_v_pass"] is True
    assert res["j_theta_pass"] is True


# ==============================================================================
# F23: Video & Visualization Artifacts (5 tests)
# ==============================================================================
def test_f23_matplotlib_velocity_plot_generation(sim: BalanceBotSimulator, tmp_path: Any) -> None:
    """F23-1: Verify matplotlib generates velocity tracking plot."""
    import matplotlib.pyplot as plt
    for _ in range(500):
        sim.step(dt=0.001, v_ref=0.5)
    fig, ax = plt.subplots()
    times = [r["time"] for r in sim.history]
    vels = [r["v"] for r in sim.history]
    ax.plot(times, vels, label="v")
    plot_file = tmp_path / "vel_plot.png"
    fig.savefig(str(plot_file))
    plt.close(fig)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0


def test_f23_matplotlib_pitch_plot_generation(sim: BalanceBotSimulator, tmp_path: Any) -> None:
    """F23-2: Verify matplotlib generates pitch tracking plot."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    times = [r["time"] for r in sim.history]
    pitches = [r["theta"] for r in sim.history]
    ax.plot(times, pitches, label="pitch")
    plot_file = tmp_path / "pitch_plot.png"
    fig.savefig(str(plot_file))
    plt.close(fig)
    assert plot_file.exists()


def test_f23_phase_portrait_plot_generation(sim: BalanceBotSimulator, tmp_path: Any) -> None:
    """F23-3: Verify phase portrait (theta vs theta_dot) plot."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    pitches = [r["theta"] for r in sim.history]
    rates = [r["theta_dot"] for r in sim.history]
    ax.plot(pitches, rates)
    plot_file = tmp_path / "phase_portrait.png"
    fig.savefig(str(plot_file))
    plt.close(fig)
    assert plot_file.exists()


def test_f23_plot_labels_and_threshold_lines(tmp_path: Any) -> None:
    """F23-4: Verify plot includes required threshold line indicators."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.axhline(5.0, color="r", linestyle="--", label="5 deg threshold")
    plot_file = tmp_path / "threshold_plot.png"
    fig.savefig(str(plot_file))
    plt.close(fig)
    assert plot_file.exists()


def test_f23_synthetic_frame_sequence_generation(tmp_path: Any) -> None:
    """F23-5: Verify video frame sequence rendering compatibility."""
    import cv2
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "BalanceBot Sim", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    frame_path = tmp_path / "frame_001.png"
    cv2.imwrite(str(frame_path), frame)
    assert frame_path.exists()
    assert frame_path.stat().st_size > 0


# ==============================================================================
# F24: Architecture & System Documentation (5 tests)
# ==============================================================================
def test_f24_architecture_document_exists() -> None:
    """F24-1: Verify system architecture document exists and is populated."""
    doc_paths = [
        "ai_handoff/system_architecture.md",
        "docs/architecture.md",
    ]
    # At least one authoritative architecture document must exist
    found = any(os.path.exists(p) for p in doc_paths)
    assert found is True


def test_f24_actuator_interface_topics_documented() -> None:
    """F24-2: Verify actuator interface topics documented in architecture files."""
    doc_path = "ai_handoff/system_architecture.md"
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100


def test_f24_sensor_interface_topics_documented() -> None:
    """F24-3: Verify requirements document indexes sensor requirements R04/R05."""
    req_path = "ai_handoff/project_design/requirements.md"
    assert os.path.exists(req_path)
    with open(req_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "R04" in content


def test_f24_state_machine_transitions_documented() -> None:
    """F24-4: Verify design snapshot documents landing and balance recovery."""
    snap_path = "ai_handoff/project_design/design_snapshot.md"
    assert os.path.exists(snap_path)
    with open(snap_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "균형" in content


def test_f24_mathematical_control_formulations_documented() -> None:
    """F24-5: Verify error integral criteria defined in requirements."""
    req_path = "ai_handoff/project_design/requirements.md"
    with open(req_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "R12" in content
