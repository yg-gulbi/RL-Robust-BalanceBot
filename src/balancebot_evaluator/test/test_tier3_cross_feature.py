"""Tier 3: Cross-Feature Combinations E2E Test Suite.

Implements pairwise and multi-feature interaction tests covering coupled dynamics:
speed + terrain, turn + leg adjustment, offset + landing, sensor streaming + maneuvers.
(Total 24 authentic cross-feature tests).
"""

import math
from typing import Any
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


def test_t3_01_high_speed_rough_terrain_traversal(sim: BalanceBotSimulator) -> None:
    """T3-01: High forward speed (1.5 m/s) + irregular terrain traversal (F03 + F13)."""
    sim.reset(x0=9.5, v0=1.5, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=1.5)
    assert sim.x > 12.5
    assert abs(math.degrees(sim.theta)) < 30.0


def test_t3_02_simultaneous_turning_and_leg_height_adjustment(sim: BalanceBotSimulator) -> None:
    """T3-02: Steering turn (0.8 rad/s) while adjusting height 0.22m -> 0.35m (F03 + F04 + F07 + F11)."""
    sim.set_leg_mode("manual", target_height=0.22)
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.6, omega_ref=0.8)
    sim.set_leg_mode("manual", target_height=0.35)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.6, omega_ref=0.8)
    assert sim.effective_height > 0.30
    assert sim.yaw_rate > 0.4
    assert abs(math.degrees(sim.theta)) < 20.0


def test_t3_03_slope_ascent_with_forward_wheel_offset_bias(sim: BalanceBotSimulator) -> None:
    """T3-03: Incline ascent on 12 deg slope with forward wheel offset 0.04m (F09 + F10)."""
    sim.reset(x0=14.5, v0=0.7, theta0=0.0)
    sim.set_leg_mode("manual", target_height=0.26, target_offset=0.04)
    for _ in range(3500):
        sim.step(dt=0.001, v_ref=0.7)
    assert sim.x > 16.0
    assert abs(math.degrees(sim.theta)) < 30.0


def test_t3_04_ramp_jump_airborne_suppression_and_shock_absorption(sim: BalanceBotSimulator) -> None:
    """T3-04: Full launch off ramp -> airborne suppression -> touchdown shock absorption (F13 + F14 + F15 + F16 + F17)."""
    sim.reset(x0=21.0, v0=2.0, theta0=0.0)
    for _ in range(3500):
        sim.step(dt=0.001, v_ref=2.0)
    states = [r["fsm_state"] for r in sim.history]
    assert int(FlightState.AIRBORNE) in states or int(FlightState.TOUCHDOWN_ABSORPTION) in states
    assert min(r["clearance"] for r in sim.history) > 0.0


def test_t3_05_manual_mode_latched_under_high_speed_turning(sim: BalanceBotSimulator) -> None:
    """T3-05: Manual mode latched during simultaneous high speed and steering turn (F04 + F11 + F12)."""
    sim.set_leg_mode("manual", target_height=0.24, target_offset=-0.03)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=1.2, omega_ref=1.0)
    assert sim.leg_mode == "manual"
    assert math.isclose(sim.wheel_offset, -0.03, abs_tol=0.01)
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t3_06_sensor_fusion_under_rough_terrain_vibration(sim: BalanceBotSimulator) -> None:
    """T3-06: Sensor streaming (IMU, LiDAR) during rough terrain traversal (F10 + F13 + F18 + F19)."""
    sim.reset(x0=10.0, v0=0.6, theta0=0.0)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.6)
    assert sim.sensors.imu_update_count >= 150
    assert sim.sensors.lidar_update_count >= 15
    assert not math.isnan(sim.sensors.filtered_pitch)


def test_t3_07_teleoperation_emergency_stop_on_incline_slope(sim: BalanceBotSimulator) -> None:
    """T3-07: Emergency braking to rest while driving on 12 deg incline slope (F03 + F04 + F10)."""
    sim.reset(x0=16.0, v0=0.8, theta0=0.0)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.8)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 0.25
    assert abs(math.degrees(sim.theta)) < 30.0


def test_t3_08_dynamic_wheel_offset_change_during_cruise(sim: BalanceBotSimulator) -> None:
    """T3-08: Shift offset -0.03m -> +0.03m while cruising at 1.0 m/s (F03 + F07 + F09)."""
    sim.set_leg_mode("manual", target_height=0.28, target_offset=-0.03)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=1.0)
    sim.set_leg_mode("manual", target_height=0.28, target_offset=0.03)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=1.0)
    assert math.isclose(sim.wheel_offset, 0.03, abs_tol=0.01)
    assert sim.v > 0.7


def test_t3_09_automated_replay_with_continuous_leg_modulation(sim: BalanceBotSimulator) -> None:
    """T3-09: Script replay with concurrent height modulation (F05 + F07 + F11)."""
    sim.set_leg_mode("manual")
    for i in range(3000):
        target_h = 0.24 + 0.08 * math.sin(2.0 * math.pi * (i * 0.001) / 2.0)
        sim.target_height = target_h
        sim.step(dt=0.001, v_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 15.0


def test_t3_10_touchdown_landing_with_yaw_rotation(sim: BalanceBotSimulator) -> None:
    """T3-10: Touchdown impact absorption with non-zero yaw turning (F04 + F16 + F17)."""
    sim.theta = math.radians(8.0)
    sim.yaw_rate = 0.5
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5, omega_ref=0.5)
    assert abs(math.degrees(sim.theta)) < 5.0
    assert min(r["clearance"] for r in sim.history) > 0.0


def test_t3_11_airborne_detection_with_crouched_stance(sim: BalanceBotSimulator) -> None:
    """T3-11: Freefall detection and extension initiated from crouched 0.20m stance (F07 + F14 + F15)."""
    sim.effective_height = 0.20
    sim.wheel_contact = False
    for _ in range(500):
        sim.step(dt=0.001, v_ref=1.5)
    assert sim.fsm.state == FlightState.AIRBORNE
    assert sim.effective_height > 0.28


def test_t3_12_rough_terrain_traversal_with_manual_elevated_height(sim: BalanceBotSimulator) -> None:
    """T3-12: Bumpy terrain traversal at maximum elevated height 0.36m (F11 + F12 + F13)."""
    sim.reset(x0=9.5, v0=0.6, theta0=0.0, height0=0.36)
    sim.set_leg_mode("manual", target_height=0.36)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=0.6)
    assert sim.x > 11.0
    assert abs(math.degrees(sim.theta)) < 30.0


def test_t3_13_depth_camera_awareness_during_ramp_ascent(sim: BalanceBotSimulator) -> None:
    """T3-13: Depth camera frame stream during ramp approach (F10 + F20)."""
    sim.reset(x0=19.5, v0=0.8, theta0=0.0)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.8)
    assert sim.sensors.camera_update_count >= 15


def test_t3_14_realtime_logging_and_metric_calculation_during_jump(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator) -> None:
    """T3-14: High-frequency logging during jump and recovery (F06 + F14 + F17 + F22)."""
    sim.reset(x0=21.0, v0=1.8, theta0=0.0)
    for _ in range(3500):
        sim.step(dt=0.001, v_ref=1.8)
    res = metrics_calc.verify_quantitative_criteria(sim.history)
    assert "j_v" in res
    assert "settling_time" in res
    assert res["clearance_pass"] is True


def test_t3_15_reverse_traversal_over_sinusoidal_bumps(sim: BalanceBotSimulator) -> None:
    """T3-15: Backward driving across sinusoidal bumps (F03 + F04 + F13)."""
    sim.reset(x0=8.0, v0=-0.6, theta0=0.0)
    for _ in range(3000):
        sim.step(dt=0.001, v_ref=-0.6)
    assert sim.x < 7.5
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t3_16_downhill_slope_acceleration_and_braking(sim: BalanceBotSimulator) -> None:
    """T3-16: Downhill acceleration then braking on 12 deg incline slope (F03 + F04 + F10)."""
    sim.reset(x0=18.0, v0=-0.4, theta0=0.0)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=-0.8)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.0)
    assert abs(sim.v) < 0.40


def test_t3_17_virtual_model_compliance_tuning_over_jagged_terrain(sim: BalanceBotSimulator) -> None:
    """T3-17: Active impedance compliance absorbing jagged ground shocks (F08 + F10 + F13)."""
    sim.reset(x0=10.0, v0=0.8, theta0=0.0)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=0.8)
    assert min(r["clearance"] for r in sim.history) > 0.0


def test_t3_18_mode_toggle_from_manual_to_auto_on_bump_zone(sim: BalanceBotSimulator) -> None:
    """T3-18: Mode toggle from manual to auto while crossing bumps (F10 + F12 + F13)."""
    sim.reset(x0=6.0, v0=0.6, theta0=0.0)
    sim.set_leg_mode("manual", target_height=0.24)
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.6)
    sim.set_leg_mode("auto")
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.6)
    assert sim.leg_mode == "auto"
    assert abs(math.degrees(sim.theta)) < 25.0


def test_t3_19_lidar_and_imu_concurrency_during_high_speed_slalom(sim: BalanceBotSimulator) -> None:
    """T3-19: Concurrent sensor streaming during slalom steering (F04 + F18 + F19)."""
    for i in range(2000):
        omega = 0.8 * math.sin(2.0 * math.pi * (i * 0.001) / 1.0)
        sim.step(dt=0.001, v_ref=1.0, omega_ref=omega)
    assert sim.sensors.imu_update_count >= 150
    assert sim.sensors.lidar_update_count >= 15


def test_t3_20_landing_shock_dissipation_with_rearward_offset(sim: BalanceBotSimulator) -> None:
    """T3-20: Landing impact absorbed when wheel offset is rearward -0.04m (F09 + F16 + F17)."""
    sim.set_leg_mode("manual", target_height=0.32, target_offset=-0.04)
    sim.fsm.state = FlightState.TOUCHDOWN_ABSORPTION
    sim.theta = math.radians(-5.0)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.5)
    assert min(r["clearance"] for r in sim.history) > 0.0
    assert abs(math.degrees(sim.theta)) < 15.0


def test_t3_21_automated_replay_scorecard_and_plot_export(sim: BalanceBotSimulator, metrics_calc: MetricsCalculator, tmp_path: Any) -> None:
    """T3-21: Replay execution generating scorecard and matplotlib plot (F05 + F21 + F22 + F23)."""
    import matplotlib.pyplot as plt
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.6)
    res = metrics_calc.verify_quantitative_criteria(sim.history)
    assert res["j_v_pass"] is True
    fig, ax = plt.subplots()
    ax.plot([r["time"] for r in sim.history], [r["v"] for r in sim.history])
    plot_file = tmp_path / "t3_plot.png"
    fig.savefig(str(plot_file))
    plt.close(fig)
    assert plot_file.exists()


def test_t3_22_impulse_recovery_during_simultaneous_turn(sim: BalanceBotSimulator) -> None:
    """T3-22: Lateral impulse perturbation while executing turn (F03 + F04 + F18)."""
    for _ in range(1000):
        sim.step(dt=0.001, v_ref=0.6, omega_ref=0.6)
    for _ in range(50):
        sim.step(dt=0.001, v_ref=0.6, omega_ref=0.6, ext_torque=12.0)
    for _ in range(2000):
        sim.step(dt=0.001, v_ref=0.6, omega_ref=0.6)
    assert abs(math.degrees(sim.theta)) < 10.0


def test_t3_23_full_skatepark_transition_flat_to_bumps_to_slope(sim: BalanceBotSimulator) -> None:
    """T3-23: Continuous driving across 3 distinct zones (F02 + F03 + F10 + F13)."""
    sim.reset(x0=3.0, v0=1.0, theta0=0.0)
    for _ in range(8000):
        sim.step(dt=0.001, v_ref=1.0)
    # Starts in flat (3m), crosses bumps (5-10m), reaches rough/slope (>10m)
    assert sim.x > 9.0
    assert abs(math.degrees(sim.theta)) < 30.0


def test_t3_24_landing_recovery_followed_by_immediate_cruise(sim: BalanceBotSimulator) -> None:
    """T3-24: Immediate acceleration command right after touchdown balance recovery (F03 + F16 + F17)."""
    sim.fsm.state = FlightState.BALANCE_RECOVERY
    for _ in range(1500):
        sim.step(dt=0.001, v_ref=0.3)
    for _ in range(2500):
        sim.step(dt=0.001, v_ref=1.2)
    assert sim.v > 0.8
    assert abs(math.degrees(sim.theta)) < 5.0
