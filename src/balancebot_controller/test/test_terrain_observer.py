#!/usr/bin/env python3
"""Comprehensive unit tests for TerrainObserver, TerrainObserverNode, and M4 integration.

Verifies:
- 5-State discrete flight FSM transitions and thresholds
- Specific force freefall detection (||a|| < 2.5 m/s^2, 30 ms debounce)
- Touchdown impact spike detection (||a|| > 15.0 m/s^2)
- Touchdown absorption dissipation (|z_dot| < 0.05 m/s or dt > 0.25 s)
- Balance recovery stabilization (|theta| < 5.0 deg, |z_dot| < 0.03 m/s)
- Ramp ascent crouch and liftoff
- Ground slope estimation
- ROS 2 TerrainObserverNode publication and callbacks
- Torque suppression and integrator freeze in BalanceControllerNode
- Height and impedance modulation in LegKinematicsControllerNode
"""

import math
import unittest
from unittest.mock import MagicMock

from balancebot_controller.balance_controller import BalanceControllerNode
from balancebot_controller.leg_kinematics import LegKinematicsControllerNode
from balancebot_controller.terrain_observer import (
    FlightState,
    TerrainObserver,
    TerrainObserverNode,
)
import numpy as np
import rclpy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64, Float64MultiArray, Int32


class TestTerrainObserver(unittest.TestCase):
    """Unit test suite for the pure mathematical TerrainObserver class."""

    def setUp(self) -> None:
        self.obs = TerrainObserver(
            freefall_accel_thresh=2.5,
            impact_accel_thresh=15.0,
            debounce_freefall_s=0.030,
            absorption_settle_vel=0.05,
            absorption_timeout_s=0.25,
            recovery_pitch_thresh=0.08726,
            recovery_settle_vel=0.03,
            ramp_pitch_thresh=0.15,
            nominal_height=0.28,
            airborne_height=0.36,
            absorption_height=0.20,
            ramp_height=0.22,
        )

    def test_initial_state_is_ground_balance(self) -> None:
        """Verify initial FSM state is GROUND_BALANCE (0)."""
        self.assertEqual(self.obs.state, FlightState.GROUND_BALANCE)
        h, kz, dz, suppress = self.obs.get_control_targets()
        self.assertAlmostEqual(h, 0.28)
        self.assertAlmostEqual(kz, 1200.0)
        self.assertAlmostEqual(dz, 80.0)
        self.assertFalse(suppress)

    def test_freefall_detection_under_threshold(self) -> None:
        """Verify ||a|| < 2.5 m/s^2 persisting >= 30 ms triggers AIRBORNE."""
        for i in range(30):
            state = self.obs.update(
                t=i * 0.001,
                accel_mag=2.0,
                wheel_contact=True,
                pitch=0.0,
                z_vel=0.0,
                dt=0.001,
            )
        self.assertEqual(state, FlightState.AIRBORNE)
        self.assertEqual(self.obs.state, FlightState.AIRBORNE)

    def test_over_threshold_does_not_trigger_airborne(self) -> None:
        """Verify ||a|| >= 2.5 m/s^2 does not trigger AIRBORNE."""
        for i in range(50):
            state = self.obs.update(
                t=i * 0.001,
                accel_mag=3.5,
                wheel_contact=True,
                pitch=0.0,
                z_vel=0.0,
                dt=0.001,
            )
        self.assertEqual(state, FlightState.GROUND_BALANCE)

    def test_brief_vibration_dip_does_not_false_trigger(self) -> None:
        """Verify a brief 10 ms drop under 2.5 m/s^2 does not trigger AIRBORNE."""
        for i in range(10):
            self.obs.update(
                t=i * 0.001,
                accel_mag=1.8,
                wheel_contact=True,
                pitch=0.0,
                z_vel=0.0,
                dt=0.001,
            )
        self.assertEqual(self.obs.state, FlightState.GROUND_BALANCE)

        # Restoring acceleration resets freefall debounce timer
        self.obs.update(
            t=0.011,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=0.0,
            z_vel=0.0,
            dt=0.001,
        )
        self.assertAlmostEqual(self.obs.freefall_timer, 0.0)

    def test_contact_loss_triggers_airborne(self) -> None:
        """Verify wheel contact loss triggers AIRBORNE after debounce window."""
        for i in range(35):
            self.obs.update(
                t=i * 0.001,
                accel_mag=2.0,
                wheel_contact=False,
                pitch=0.0,
                z_vel=0.0,
                dt=0.001,
            )
        self.assertEqual(self.obs.state, FlightState.AIRBORNE)

    def test_airborne_control_targets(self) -> None:
        """Verify AIRBORNE targets 0.36m extension, Kz=600, Dz=40, and torque suppression."""
        self.obs.state = FlightState.AIRBORNE
        h, kz, dz, suppress = self.obs.get_control_targets()
        self.assertAlmostEqual(h, 0.36)
        self.assertAlmostEqual(kz, 600.0)
        self.assertAlmostEqual(dz, 40.0)
        self.assertTrue(suppress)

    def test_touchdown_impact_spike_detection(self) -> None:
        """Verify acceleration spike ||a|| > 15.0 m/s^2 transitions to TOUCHDOWN_ABSORPTION."""
        self.obs.state = FlightState.AIRBORNE
        state = self.obs.update(
            t=1.0,
            accel_mag=18.0,
            wheel_contact=True,
            pitch=0.05,
            z_vel=-1.8,
            dt=0.001,
        )
        self.assertEqual(state, FlightState.TOUCHDOWN_ABSORPTION)
        self.assertEqual(self.obs.touchdown_time, 1.0)

    def test_touchdown_absorption_targets(self) -> None:
        """Verify TOUCHDOWN_ABSORPTION targets 0.20m, Kz=400, Dz=250, and torque suppression."""
        self.obs.state = FlightState.TOUCHDOWN_ABSORPTION
        h, kz, dz, suppress = self.obs.get_control_targets()
        self.assertAlmostEqual(h, 0.20)
        self.assertAlmostEqual(kz, 400.0)
        self.assertAlmostEqual(dz, 250.0)
        self.assertTrue(suppress)

    def test_absorption_settling_by_vertical_velocity(self) -> None:
        """Verify settling |z_vel| < 0.05 m/s transitions to BALANCE_RECOVERY."""
        self.obs.state = FlightState.TOUCHDOWN_ABSORPTION
        self.obs.touchdown_time = 1.0
        state = self.obs.update(
            t=1.1,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=0.04,
            z_vel=0.02,
            dt=0.01,
        )
        self.assertEqual(state, FlightState.BALANCE_RECOVERY)

    def test_absorption_settling_by_timeout(self) -> None:
        """Verify absorption transitions to BALANCE_RECOVERY after 0.25 s timeout."""
        self.obs.state = FlightState.TOUCHDOWN_ABSORPTION
        self.obs.touchdown_time = 1.0
        # z_vel is still 0.08 m/s (> 0.05 m/s), but time elapsed is 0.26 s (> 0.25 s)
        state = self.obs.update(
            t=1.26,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=0.04,
            z_vel=0.08,
            dt=0.01,
        )
        self.assertEqual(state, FlightState.BALANCE_RECOVERY)

    def test_balance_recovery_settling_to_ground_balance(self) -> None:
        """Verify recovery transitions to GROUND_BALANCE when |pitch| < 5 deg and |z_vel| < 0.03 m/s."""
        self.obs.state = FlightState.BALANCE_RECOVERY
        # Unsettled pitch (7 deg)
        self.obs.update(
            t=2.0,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=math.radians(7.0),
            z_vel=0.01,
            dt=0.01,
        )
        self.assertEqual(self.obs.state, FlightState.BALANCE_RECOVERY)

        # Settled pitch (2 deg) and low vertical velocity
        state = self.obs.update(
            t=2.1,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=math.radians(2.0),
            z_vel=0.01,
            dt=0.01,
        )
        self.assertEqual(state, FlightState.GROUND_BALANCE)

    def test_ramp_ascent_and_lip_liftoff(self) -> None:
        """Verify transition to RAMP_ASCENT on pitch incline and liftoff into AIRBORNE."""
        # Forward speed + ramp pitch incline
        state = self.obs.update(
            t=0.5,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=0.20,
            z_vel=0.1,
            dt=0.02,
            v_meas=0.5,
        )
        self.assertEqual(state, FlightState.RAMP_ASCENT)
        h, kz, dz, suppress = self.obs.get_control_targets()
        self.assertAlmostEqual(h, 0.22)
        self.assertFalse(suppress)

        # Liftoff off ramp lip (freefall or contact lost)
        state = self.obs.update(
            t=0.8,
            accel_mag=1.5,
            wheel_contact=False,
            pitch=0.20,
            z_vel=0.5,
            dt=0.02,
            v_meas=1.5,
        )
        self.assertEqual(state, FlightState.AIRBORNE)

    def test_slope_estimation(self) -> None:
        """Verify ground slope estimation from pitch and accelerometer components."""
        # 12 deg slope incline (~0.2094 rad)
        slope_angle = math.radians(12.0)
        ax = 9.81 * math.sin(slope_angle)
        az = 9.81 * math.cos(slope_angle)
        slope = self.obs.estimate_slope(pitch=slope_angle, ax=ax, az=az)
        self.assertAlmostEqual(slope, slope_angle, places=3)


class TestTerrainObserverNode(unittest.TestCase):
    """ROS 2 Node level tests for TerrainObserverNode."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    def setUp(self) -> None:
        self.node = TerrainObserverNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_node_initialization(self) -> None:
        """Verify node parameter declaration and initial publisher state."""
        self.assertAlmostEqual(self.node.dt, 0.02)
        self.assertEqual(self.node.observer.state, FlightState.GROUND_BALANCE)

    def test_imu_callback_processing(self) -> None:
        """Verify IMU callback extracts acceleration magnitude and pitch correctly."""
        imu_msg = Imu()
        imu_msg.linear_acceleration.x = 0.0
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = 9.81
        imu_msg.angular_velocity.y = 0.05
        # Quaternion for pitch = +10 deg around Y: q_y = sin(5 deg), q_w = cos(5 deg)
        pitch_rad = math.radians(10.0)
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = math.sin(pitch_rad / 2.0)
        imu_msg.orientation.z = 0.0
        imu_msg.orientation.w = math.cos(pitch_rad / 2.0)

        self.node._imu_callback(imu_msg)
        self.assertAlmostEqual(self.node.accel_mag, 9.81, places=2)
        self.assertAlmostEqual(self.node.pitch_rate, 0.05, places=2)
        self.assertAlmostEqual(self.node.pitch, pitch_rad, places=3)

    def test_joint_callback_processing(self) -> None:
        """Verify joint callback computes leg height and forward velocity."""
        js_msg = JointState()
        js_msg.name = [
            'left_hip_joint', 'left_knee_joint',
            'right_hip_joint', 'right_knee_joint',
            'left_wheel_joint', 'right_wheel_joint',
        ]
        # Hip = -30 deg, Knee = +60 deg -> height ~ 0.346m
        qh = -math.radians(30.0)
        qk = math.radians(60.0)
        js_msg.position = [qh, qk, qh, qk, 0.0, 0.0]
        js_msg.velocity = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0]  # 10 rad/s * 0.10m = 1.0 m/s

        self.node._joint_callback(js_msg)
        expected_h = 2.0 * 0.20 * math.cos(math.radians(30.0))
        self.assertAlmostEqual(self.node.eff_height, expected_h, places=3)
        self.assertAlmostEqual(self.node.v_meas, 1.0, places=3)

    def test_control_loop_publications(self) -> None:
        """Verify control loop publishes Int32 flight state and Float64 slope."""
        pub_state_mock = MagicMock()
        pub_slope_mock = MagicMock()
        self.node.pub_flight_state.publish = pub_state_mock
        self.node.pub_terrain_slope.publish = pub_slope_mock

        self.node._control_loop()

        pub_state_mock.assert_called_once()
        state_msg = pub_state_mock.call_args[0][0]
        self.assertIsInstance(state_msg, Int32)
        self.assertEqual(state_msg.data, int(FlightState.GROUND_BALANCE))

        pub_slope_mock.assert_called_once()
        slope_msg = pub_slope_mock.call_args[0][0]
        self.assertIsInstance(slope_msg, Float64)


class TestControllerIntegration(unittest.TestCase):
    """Integration verification between controllers and FlightState modulation."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    def test_balance_controller_airborne_torque_suppression(self) -> None:
        """Verify BalanceControllerNode suppresses torque and freezes integrator in AIRBORNE."""
        node = BalanceControllerNode()
        try:
            # Set initial velocity error to produce non-zero torque under ground balance
            node.v_ref = 1.0
            node.estimator.v = 0.0
            node.flight_state = FlightState.GROUND_BALANCE

            pub_torque_mock = MagicMock()
            node.pub_torque.publish = pub_torque_mock

            node._control_loop()
            tau_ground = pub_torque_mock.call_args[0][0].data[0]
            self.assertNotEqual(tau_ground, 0.0)

            # Switch to AIRBORNE state
            msg = Int32()
            msg.data = int(FlightState.AIRBORNE)
            node._flight_state_callback(msg)
            self.assertEqual(node.flight_state, FlightState.AIRBORNE)

            e_int_before = node.e_integral
            node._control_loop()
            tau_airborne_l = pub_torque_mock.call_args[0][0].data[0]
            tau_airborne_r = pub_torque_mock.call_args[0][0].data[1]

            # Drive wheel balance torque is zero (differential yaw is 0 since omega_ref=0, yaw_rate=0)
            self.assertAlmostEqual(tau_airborne_l, 0.0)
            self.assertAlmostEqual(tau_airborne_r, 0.0)
            # Integrator must be frozen (no accumulation)
            self.assertEqual(node.e_integral, e_int_before)

            # Switch to TOUCHDOWN_ABSORPTION state
            msg.data = int(FlightState.TOUCHDOWN_ABSORPTION)
            node._flight_state_callback(msg)
            node._control_loop()
            tau_touch_l = pub_torque_mock.call_args[0][0].data[0]
            self.assertAlmostEqual(tau_touch_l, 0.0)
            self.assertEqual(node.e_integral, e_int_before)
        finally:
            node.destroy_node()

    def test_leg_kinematics_flight_state_modulation(self) -> None:
        """Verify LegKinematicsControllerNode adapts height and impedance by flight state."""
        node = LegKinematicsControllerNode()
        try:
            node.leg_mode = 0  # AUTO_ADAPTATION

            # 1. RAMP_ASCENT: targets crouched 0.22m
            msg = Int32()
            msg.data = int(FlightState.RAMP_ASCENT)
            node._flight_state_callback(msg)
            for _ in range(50):
                node._control_loop()
            self.assertAlmostEqual(node.current_height, 0.22, delta=0.01)

            # 2. AIRBORNE: extends to 0.36m
            msg.data = int(FlightState.AIRBORNE)
            node._flight_state_callback(msg)
            for _ in range(50):
                node._control_loop()
            self.assertAlmostEqual(node.current_height, 0.36, delta=0.01)

            # 3. TOUCHDOWN_ABSORPTION: compresses to 0.20m with Kz=400, Dz=250
            msg.data = int(FlightState.TOUCHDOWN_ABSORPTION)
            node._flight_state_callback(msg)
            for _ in range(50):
                node._control_loop()
            self.assertAlmostEqual(node.current_height, 0.20, delta=0.01)

            # 4. BALANCE_RECOVERY: smooth restoration back toward 0.28m
            msg.data = int(FlightState.BALANCE_RECOVERY)
            node._flight_state_callback(msg)
            h_start = node.current_height
            # In 10 steps (0.2s), rate is <= 0.20 m/s -> max change <= 0.04m
            for _ in range(10):
                node._control_loop()
            self.assertLessEqual(node.current_height - h_start, 0.04 + 1e-4)

            # In 100 steps (2.0s), fully recovers to nominal 0.28m
            for _ in range(100):
                node._control_loop()
            self.assertAlmostEqual(node.current_height, 0.28, delta=0.01)
        finally:
            node.destroy_node()


if __name__ == '__main__':
    unittest.main()
