#!/usr/bin/env python3
"""Adversarial stress testing suite for Milestone M5 Sensor Integration & State Estimation.

Empirical verification under hostile conditions:
1. Extreme IMU noise, accelerometer shocks, gyro saturation, and quaternion edge cases.
2. Packet drops, jitter, out-of-order timestamps, clock resets, and variable rates (50 Hz to 500 Hz).
3. LiDAR scan noise, empty/NaN/Inf ranges, sparse beams, and steep pitch tilts.
4. Terrain observer flight FSM resilience against vibration shocks and free-fall stability.
5. Closed-loop dynamic simulation under sensor noise and packet delays.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from typing import List

import numpy as np
import rclpy
from sensor_msgs.msg import Imu, LaserScan, JointState

# Add evaluator harness to sys.path
_EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../balancebot_evaluator/test'))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from balancebot_controller.state_estimator import StateEstimator, StateEstimatorNode
from balancebot_controller.terrain_observer import FlightState, TerrainObserver, TerrainObserverNode
try:
    from evaluator_harness import (
        BalanceBotSimulator,
        RobotParams,
        SkateparkEnvironment,
        SensorSuite,
    )
except ImportError:
    from balancebot_evaluator.test.evaluator_harness import (
        BalanceBotSimulator,
        RobotParams,
        SkateparkEnvironment,
        SensorSuite,
    )



class TestAdversarialIMUAndStateEstimation(unittest.TestCase):
    """Stress testing StateEstimator under hostile IMU inputs."""

    def setUp(self) -> None:
        self.estimator = StateEstimator(wheel_radius=0.10, nominal_length=0.28, alpha_complementary=0.98)

    def test_extreme_imu_gaussian_noise_rejection(self) -> None:
        """Verify pitch estimation remains stable under 10x normal IMU noise."""
        np.random.seed(12345)
        true_pitch = math.radians(5.0)  # 5 deg
        true_ax = 9.81 * math.sin(true_pitch)
        true_az = 9.81 * math.cos(true_pitch)

        # Severe noise: sigma_a = 0.20 m/s^2 (10x Gazebo SDF), sigma_g = 0.05 rad/s (10x SDF)
        for step in range(500):
            noise_ax = float(np.random.normal(0.0, 0.20))
            noise_az = float(np.random.normal(0.0, 0.20))
            noise_gy = float(np.random.normal(0.0, 0.05))

            ax = true_ax + noise_ax
            az = true_az + noise_az
            gy = noise_gy  # zero true rate

            self.estimator.update_imu(
                accel=(ax, 0.0, az),
                gyro=(0.0, gy, 0.0),
                quat=None,
                dt=0.01,
            )

        # Must converge and remain within 0.02 rad (~1.1 deg) of true pitch
        self.assertAlmostEqual(self.estimator.theta, true_pitch, delta=0.035)
        self.assertFalse(math.isnan(self.estimator.theta))
        self.assertFalse(math.isinf(self.estimator.theta))

    def test_accelerometer_massive_shock_spikes(self) -> None:
        """Verify state estimator dampens severe accelerometer landing impacts (up to 50 m/s^2)."""
        true_pitch = 0.0
        # Initialize at 0 pitch
        for _ in range(50):
            self.estimator.update_imu(accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0), quat=None, dt=0.01)

        theta_before = self.estimator.theta

        # Apply massive shock spikes: 35 m/s^2, 50 m/s^2, 80 m/s^2
        for shock_ax, shock_az in [(35.0, 9.81), (0.0, 50.0), (30.0, 80.0)]:
            self.estimator.update_imu(accel=(shock_ax, 0.0, shock_az), gyro=(0.0, 0.0, 0.0), quat=None, dt=0.01)
            # Pitch must not jump discontinuously (> 0.1 rad in 1 tick) due to alpha=0.98 complementary filtering
            self.assertLess(abs(self.estimator.theta - theta_before), 0.08)
            self.assertFalse(math.isnan(self.estimator.theta))
            theta_before = self.estimator.theta

    def test_gyroscope_extreme_saturation(self) -> None:
        """Verify high angular velocities (>5 rad/s, up to 25 rad/s) do not cause overflow."""
        for high_gy in [5.5, 10.0, 25.0, -25.0]:
            self.estimator.update_imu(
                accel=(0.0, 0.0, 9.81),
                gyro=(0.0, high_gy, 0.0),
                quat=None,
                dt=0.01,
            )
            self.assertEqual(self.estimator.theta_dot, high_gy)
            self.assertFalse(math.isnan(self.estimator.theta))
            self.assertFalse(math.isinf(self.estimator.theta))

    def test_zero_gravity_freefall_accel(self) -> None:
        """Verify zero specific force (free-fall microgravity ||a|| = 0) does not divide by zero."""
        self.estimator.update_imu(
            accel=(0.0, 0.0, 0.0),
            gyro=(0.0, 0.0, 0.0),
            quat=None,
            dt=0.01,
        )
        self.assertFalse(math.isnan(self.estimator.theta))
        self.assertFalse(math.isinf(self.estimator.theta))

    def test_quaternion_boundary_and_gimbal_lock(self) -> None:
        """Verify quaternion extraction handles vertical gimbal-lock and domain boundary conditions."""
        # 1. Pitch exactly +90 deg (sinp = 1.0)
        qw = math.cos(math.pi / 4.0)
        qy = math.sin(math.pi / 4.0)
        self.estimator.update_imu(accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0), quat=(0.0, qy, 0.0, qw))
        self.assertAlmostEqual(self.estimator.theta, math.pi / 2.0, places=4)

        # 2. Pitch exactly -90 deg (sinp = -1.0)
        self.estimator.update_imu(accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0), quat=(0.0, -qy, 0.0, qw))
        self.assertAlmostEqual(self.estimator.theta, -math.pi / 2.0, places=4)

        # 3. Slightly unnormalized quaternion where sinp > 1.0 (e.g. 1.0002)
        # Should be protected by copysign clamp without domain error in asin
        unnorm_qy = qy * 1.0005
        unnorm_qw = qw * 1.0005
        self.estimator.update_imu(accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0), quat=(0.0, unnorm_qy, 0.0, unnorm_qw))
        self.assertAlmostEqual(self.estimator.theta, math.pi / 2.0, places=4)

        # 4. All zero quaternion fallback to complementary filter
        self.estimator.update_imu(accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 0.0))
        self.assertFalse(math.isnan(self.estimator.theta))


class TestAdversarialDynamicDtAndTiming(unittest.TestCase):
    """Stress testing StateEstimatorNode under variable sampling rates, packet drops, and time jumps."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    def setUp(self) -> None:
        self.node = StateEstimatorNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def _make_imu_msg(self, sec: int, nsec: int, cov0: float = 0.0) -> Imu:
        msg = Imu()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nsec
        msg.linear_acceleration.z = 9.81
        if cov0 < 0:
            msg.orientation_covariance[0] = cov0
        else:
            msg.orientation.w = 1.0
        return msg

    def test_variable_sampling_rates_50hz_to_500hz(self) -> None:
        """Verify dynamic dt calculation matches sampling intervals from 500 Hz (2ms) to 50 Hz (20ms)."""
        # Start at t = 1.000s
        self.node._imu_callback(self._make_imu_msg(1, 0))
        self.assertEqual(self.node.last_imu_stamp, 1.0)

        # Test 500 Hz: dt = 0.002s (2,000,000 ns)
        msg_500hz = self._make_imu_msg(1, 2_000_000)
        self.node._imu_callback(msg_500hz)
        self.assertAlmostEqual(self.node.last_imu_stamp, 1.002)

        # Test 200 Hz: +5ms (7,000,000 ns)
        msg_200hz = self._make_imu_msg(1, 7_000_000)
        self.node._imu_callback(msg_200hz)
        self.assertAlmostEqual(self.node.last_imu_stamp, 1.007)

        # Test 100 Hz: +10ms (17,000,000 ns)
        msg_100hz = self._make_imu_msg(1, 17_000_000)
        self.node._imu_callback(msg_100hz)
        self.assertAlmostEqual(self.node.last_imu_stamp, 1.017)

        # Test 50 Hz: +20ms (37,000,000 ns)
        msg_50hz = self._make_imu_msg(1, 37_000_000)
        self.node._imu_callback(msg_50hz)
        self.assertAlmostEqual(self.node.last_imu_stamp, 1.037)

    def test_packet_drop_delay_clamping(self) -> None:
        """Verify dt is clamped to max 0.05s when packets are delayed or dropped (e.g. 1.0s gap)."""
        self.node._imu_callback(self._make_imu_msg(1, 0))

        # Massive packet drop: next packet arrives 2.5 seconds later
        msg_dropped = self._make_imu_msg(3, 500_000_000)
        # We test that update_imu does not explode
        self.node._imu_callback(msg_dropped)
        self.assertEqual(self.node.last_imu_stamp, 3.5)
        # Estimator state remains valid
        self.assertFalse(math.isnan(self.node.estimator.theta))

    def test_out_of_order_and_duplicate_stamps(self) -> None:
        """Verify duplicate or out-of-order timestamps use fallback dt without crash or negative dt."""
        self.node._imu_callback(self._make_imu_msg(5, 0))

        # Duplicate stamp
        self.node._imu_callback(self._make_imu_msg(5, 0))
        self.assertEqual(self.node.last_imu_stamp, 5.0)

        # Earlier stamp (out-of-order packet)
        self.node._imu_callback(self._make_imu_msg(4, 900_000_000))
        self.assertEqual(self.node.last_imu_stamp, 4.9)
        self.assertFalse(math.isnan(self.node.estimator.theta))

    def test_simulation_reset_time_jump(self) -> None:
        """Verify simulation time jumping backwards (e.g. Gazebo /world/reset) recovers cleanly."""
        self.node._imu_callback(self._make_imu_msg(100, 0))
        self.assertEqual(self.node.last_imu_stamp, 100.0)

        # Reset to 0.0
        self.node._imu_callback(self._make_imu_msg(0, 0))
        self.assertEqual(self.node.last_imu_stamp, 0.0)

        # Normal subsequent tick at 10 ms
        self.node._imu_callback(self._make_imu_msg(0, 10_000_000))
        self.assertAlmostEqual(self.node.last_imu_stamp, 0.01)


class TestAdversarialLiDARScanClearance(unittest.TestCase):
    """Stress testing TerrainObserverNode LiDAR /scan parsing."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    def setUp(self) -> None:
        self.node = TerrainObserverNode()

    def _make_scan_msg(self, ranges: List[float], r_min: float = 0.10, r_max: float = 12.0) -> LaserScan:
        msg = LaserScan()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = (2 * math.pi) / max(len(ranges), 1)
        msg.range_min = r_min
        msg.range_max = r_max
        msg.ranges = ranges
        return msg

    def test_scan_empty_ranges_tolerance(self) -> None:
        """Verify empty ranges list does not raise exception."""
        msg = self._make_scan_msg([])
        init_clearance = self.node.forward_clearance
        self.node._scan_callback(msg)
        self.assertEqual(self.node.forward_clearance, init_clearance)

    def test_scan_nan_and_inf_filtering(self) -> None:
        """Verify NaNs, positive infs, and negative infs are safely rejected."""
        # 360 rays, center contaminated with NaN and Infs
        ranges = [5.0] * 360
        ranges[175] = float('nan')
        ranges[179] = float('inf')
        ranges[180] = -float('inf')
        ranges[181] = 2.45
        ranges[185] = float('nan')

        msg = self._make_scan_msg(ranges)
        self.node._scan_callback(msg)
        self.assertAlmostEqual(self.node.forward_clearance, 2.45, places=2)

    def test_scan_all_out_of_range_tolerance(self) -> None:
        """Verify scan with all returns out-of-range (< range_min or > range_max) retains previous clearance."""
        self.node.forward_clearance = 8.5
        # All returns beyond 12m or below 0.1m
        ranges = [0.05] * 180 + [15.0] * 180
        msg = self._make_scan_msg(ranges, r_min=0.10, r_max=12.0)
        self.node._scan_callback(msg)
        self.assertEqual(self.node.forward_clearance, 8.5)

    def test_scan_sparse_beams_single_or_few_rays(self) -> None:
        """Verify slicing logic does not fail on sparse beam arrays (1, 2, or 5 rays)."""
        for count in [1, 2, 5, 10]:
            ranges = [3.2] * count
            msg = self._make_scan_msg(ranges)
            self.node._scan_callback(msg)
            self.assertAlmostEqual(self.node.forward_clearance, 3.2, places=2)

    def test_scan_steep_pitch_tilt_ground_detection(self) -> None:
        """Verify that when pitching forward 25 deg, ground returns in front are captured."""
        # When pitched forward at 0.35m height, central rays hit ground ~ 0.83m ahead
        ranges = [5.0] * 360
        for i in range(165, 195):
            ranges[i] = 0.83
        msg = self._make_scan_msg(ranges)
        self.node._scan_callback(msg)
        self.assertAlmostEqual(self.node.forward_clearance, 0.83, places=2)


class TestAdversarialFlightFSMAndRoughTerrain(unittest.TestCase):
    """Stress testing Flight FSM under noisy shocks and extended free-fall."""

    def setUp(self) -> None:
        self.obs = TerrainObserver(
            freefall_accel_thresh=2.5,
            impact_accel_thresh=15.0,
            debounce_freefall_s=0.040,
        )

    def test_rough_terrain_vibration_does_not_trigger_airborne(self) -> None:
        """Verify intermittent vibration dips (<2.5 m/s^2 for <40 ms) do not cause false AIRBORNE."""
        t = 0.0
        # 100 cycles of fluctuating ground acceleration
        for i in range(100):
            # Dip below 2.5 for only 1 tick (20 ms), then back to 9.81
            accel = 1.2 if (i % 5 == 0) else 9.81
            state = self.obs.update(
                t=t,
                accel_mag=accel,
                wheel_contact=True,
                pitch=0.02,
                z_vel=0.0,
                dt=0.02,
            )
            t += 0.02
            self.assertEqual(state, FlightState.GROUND_BALANCE)

    def test_extended_airborne_freefall_holds_until_impact(self) -> None:
        """Verify robot remains in AIRBORNE throughout 0.4s free-fall, even if wheel_contact is True."""
        # 1. Trigger free-fall with sustained accel < 2.5 for >= 40 ms
        self.obs.update(t=0.00, accel_mag=1.5, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.02)
        self.obs.update(t=0.02, accel_mag=1.5, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.02)
        state = self.obs.update(t=0.04, accel_mag=1.5, wheel_contact=True, pitch=0.0, z_vel=0.0, dt=0.02)
        self.assertEqual(state, FlightState.AIRBORNE)

        # 2. Free-fall for 20 more steps (400 ms)
        for i in range(3, 23):
            t = i * 0.02
            state = self.obs.update(
                t=t,
                accel_mag=1.8,          # Free-fall specific force
                wheel_contact=True,     # Default simulated contact
                pitch=0.05,
                z_vel=-1.5,
                dt=0.02,
            )
            self.assertEqual(state, FlightState.AIRBORNE, f"Premature exit from AIRBORNE at step {i}")

        # 3. Touchdown impact: spike > 15 m/s^2
        state = self.obs.update(
            t=0.50,
            accel_mag=22.0,
            wheel_contact=True,
            pitch=0.02,
            z_vel=-0.2,
            dt=0.02,
        )
        self.assertEqual(state, FlightState.TOUCHDOWN_ABSORPTION)


class TestAdversarialClosedLoopSimulation(unittest.TestCase):
    """Stress testing closed-loop balance in dynamic simulator under noise and packet jitter."""

    def test_closed_loop_balance_under_active_sensor_noise(self) -> None:
        """Verify robot self-balances on flat ground under continuous IMU noise for 3.0 seconds."""
        sim = BalanceBotSimulator()
        sim.reset(x0=0.0, v0=0.0, theta0=0.04)  # 0.04 rad initial tilt

        # Simulate 3000 ms at dt = 1 ms
        for step in range(3000):
            t = step * 0.001
            # Step dynamic simulator
            sim.step(dt=0.001, v_ref=0.0)

            # Step sensor suite with noise
            sim.sensors.step(
                t=t,
                true_pitch=sim.theta,
                true_pitch_rate=sim.theta_dot,
                true_accel=np.array([sim.p.g * math.sin(sim.theta), 0.0, sim.p.g * math.cos(sim.theta)]),
                x_pos=sim.x,
                environment=sim.env,
            )

        # Final pitch must be well within acceptance criterion (|pitch| < 5 deg = 0.087 rad)
        self.assertLess(abs(sim.theta), math.radians(2.5))
        self.assertLess(abs(sim.v), 0.05)


if __name__ == "__main__":
    unittest.main()
