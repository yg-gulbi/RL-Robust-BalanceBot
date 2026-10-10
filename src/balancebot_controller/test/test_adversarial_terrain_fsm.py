#!/usr/bin/env python3
"""Milestone M5 Adversarial Challenge 2: Terrain Adaptation, Slope Feedforward & Flight FSM.

Adversarial empirical stress-test suite verifying:
1. AIRBORNE state persistence during ballistic free-fall under Gaussian sensor noise,
   rejection of in-flight acceleration spikes below impact threshold, and reliable
   transition to TOUCHDOWN_ABSORPTION upon hard and soft touchdown.
2. Chatter-free absorption dissipation, absorption timeout safety guarantee, and strict
   two-variable convergence criteria (|theta| <= 5 deg, |z_dot| <= 0.03 m/s) in BALANCE_RECOVERY.
3. Ground slope feedforward tilt behavior on steep ramps (up to +/- 30 deg), low-pass filter
   attenuation of rough terrain chatter, exact steady-state gravity torque cancellation,
   and saturation bounded torque safety under extreme incline angles.
4. Robust forward clearance calculation from /scan under pathological inputs (empty scans,
   all-NaN, all-Inf, negative values, out-of-range beams, single-beam and 20,000-beam scans),
   central sector angular cone isolation, and nearest hazard tracking.
5. ROS 2 node-level end-to-end integration and topic publishing integrity.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock

from balancebot_controller.balance_controller import BalanceControllerNode
from balancebot_controller.terrain_observer import (
    FlightState,
    TerrainObserver,
    TerrainObserverNode,
)
import numpy as np
import rclpy
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float64, Float64MultiArray, Int32


class TestAdversarialFlightFSM(unittest.TestCase):
    """Stress tests for ballistic free-fall persistence, shock absorption, and landing recovery."""

    def setUp(self) -> None:
        self.obs = TerrainObserver(
            freefall_accel_thresh=2.5,
            impact_accel_thresh=15.0,
            debounce_freefall_s=0.030,
            absorption_settle_vel=0.05,
            absorption_timeout_s=0.25,
            recovery_pitch_thresh=math.radians(5.0),
            recovery_settle_vel=0.03,
            ramp_pitch_thresh=0.15,
            nominal_height=0.28,
            airborne_height=0.36,
            absorption_height=0.20,
            ramp_height=0.22,
        )

    def test_airborne_persistence_ballistic_freefall_sensor_noise(self) -> None:
        """Verify AIRBORNE persists through 1.0s ballistic free-fall with Gaussian sensor noise."""
        # Enter AIRBORNE via free-fall
        for i in range(2):
            self.obs.update(t=i * 0.02, accel_mag=0.1, wheel_contact=False, dt=0.02)
        self.assertEqual(self.obs.state, FlightState.AIRBORNE)

        # 50 ticks (1.0 second) of ballistic freefall with Gaussian noise (sigma=0.5 m/s^2)
        np.random.seed(42)
        for i in range(50):
            noise_accel = np.random.normal(0.0, 0.5, size=3)
            accel_mag = float(np.linalg.norm(noise_accel))
            state = self.obs.update(
                t=0.04 + i * 0.02,
                accel_mag=accel_mag,
                wheel_contact=False,
                pitch=0.02 * math.sin(i * 0.2),
                z_vel=2.0 - 9.81 * (i * 0.02),
                dt=0.02,
            )
            self.assertEqual(state, FlightState.AIRBORNE)

        # Check target modulation during flight
        h, kz, dz, suppress = self.obs.get_control_targets()
        self.assertAlmostEqual(h, 0.36)
        self.assertAlmostEqual(kz, 600.0)
        self.assertAlmostEqual(dz, 40.0)
        self.assertTrue(suppress)

    def test_airborne_rejection_of_sub_impact_spikes(self) -> None:
        """Verify acceleration spikes below 15.0 m/s^2 during flight do not abort AIRBORNE."""
        self.obs.state = FlightState.AIRBORNE
        sub_impact_spikes = [5.0, 8.5, 12.0, 14.5, 14.95]
        for idx, spike in enumerate(sub_impact_spikes):
            state = self.obs.update(
                t=0.1 + idx * 0.02,
                accel_mag=spike,
                wheel_contact=False,
                pitch=0.0,
                z_vel=0.5,
                dt=0.02,
            )
            self.assertEqual(
                state,
                FlightState.AIRBORNE,
                f"Sub-impact spike {spike} m/s^2 falsely triggered touchdown transition",
            )

    def test_touchdown_impact_spike_triggering(self) -> None:
        """Verify hard landing impact spike (> 15.0 m/s^2) triggers TOUCHDOWN_ABSORPTION."""
        self.obs.state = FlightState.AIRBORNE
        # Impact spike at t=0.5s with 28.5 m/s^2
        state = self.obs.update(
            t=0.50,
            accel_mag=28.5,
            wheel_contact=True,
            pitch=0.04,
            z_vel=-2.1,
            dt=0.02,
        )
        self.assertEqual(state, FlightState.TOUCHDOWN_ABSORPTION)
        self.assertEqual(self.obs.touchdown_time, 0.50)

        # In absorption: leg crouch (0.20m), high damping (Dz=250), torque suppressed
        h, kz, dz, suppress = self.obs.get_control_targets()
        self.assertAlmostEqual(h, 0.20)
        self.assertAlmostEqual(kz, 400.0)
        self.assertAlmostEqual(dz, 250.0)
        self.assertTrue(suppress)

    def test_soft_touchdown_restored_contact_triggering(self) -> None:
        """Verify gentle touchdown with normal gravity (9.81 m/s^2) transitions to absorption."""
        self.obs.state = FlightState.AIRBORNE
        state = self.obs.update(
            t=0.60,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=0.01,
            z_vel=-0.2,
            dt=0.02,
        )
        self.assertEqual(state, FlightState.TOUCHDOWN_ABSORPTION)

    def test_post_impact_bounce_vibration_chatter_free(self) -> None:
        """Verify high post-impact oscillations (5 to 35 m/s^2) do not flap FSM state."""
        self.obs.state = FlightState.TOUCHDOWN_ABSORPTION
        self.obs.touchdown_time = 1.0

        # Simulate 5 ticks (100 ms) of high landing vibrations while z_vel is still settling
        vibration_accels = [32.0, 18.0, 8.0, 24.0, 14.0]
        for idx, acc in enumerate(vibration_accels):
            state = self.obs.update(
                t=1.0 + (idx + 1) * 0.02,
                accel_mag=acc,
                wheel_contact=True,
                pitch=0.05,
                z_vel=-0.15 + idx * 0.02,  # still > 0.05 m/s
                dt=0.02,
            )
            self.assertEqual(
                state,
                FlightState.TOUCHDOWN_ABSORPTION,
                f"Post-impact vibration {acc} m/s^2 caused improper state flapping",
            )

    def test_absorption_timeout_safety_guarantee(self) -> None:
        """Verify absorption transitions to BALANCE_RECOVERY after 0.25s even if z_vel > 0.05 m/s."""
        self.obs.state = FlightState.TOUCHDOWN_ABSORPTION
        self.obs.touchdown_time = 2.00

        # At t=2.24s (240 ms elapsed < 250 ms), remains in absorption if z_vel is high
        state_pre = self.obs.update(
            t=2.24,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=0.03,
            z_vel=0.09,
            dt=0.02,
        )
        self.assertEqual(state_pre, FlightState.TOUCHDOWN_ABSORPTION)

        # At t=2.26s (260 ms elapsed > 250 ms), timeout forces transition to BALANCE_RECOVERY
        state_post = self.obs.update(
            t=2.26,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=0.03,
            z_vel=0.09,
            dt=0.02,
        )
        self.assertEqual(state_post, FlightState.BALANCE_RECOVERY)

    def test_balance_recovery_two_variable_convergence(self) -> None:
        """Verify BALANCE_RECOVERY strictly requires BOTH |theta| < 5 deg AND |z_dot| < 0.03 m/s."""
        self.obs.state = FlightState.BALANCE_RECOVERY

        # Case 1: z_vel settled (0.01 m/s), but pitch too high (6.5 deg = 0.113 rad)
        state_pitch_unsettled = self.obs.update(
            t=3.0,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=math.radians(6.5),
            z_vel=0.01,
            dt=0.02,
        )
        self.assertEqual(state_pitch_unsettled, FlightState.BALANCE_RECOVERY)

        # Case 2: pitch settled (2.0 deg = 0.035 rad), but z_vel too high (0.045 m/s)
        state_z_unsettled = self.obs.update(
            t=3.02,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=math.radians(2.0),
            z_vel=0.045,
            dt=0.02,
        )
        self.assertEqual(state_z_unsettled, FlightState.BALANCE_RECOVERY)

        # Case 3: BOTH pitch settled (3.0 deg = 0.052 rad) and z_vel settled (0.02 m/s)
        state_converged = self.obs.update(
            t=3.04,
            accel_mag=9.81,
            wheel_contact=True,
            pitch=math.radians(3.0),
            z_vel=0.02,
            dt=0.02,
        )
        self.assertEqual(state_converged, FlightState.GROUND_BALANCE)


class TestAdversarialSlopeFeedforward(unittest.TestCase):
    """Stress tests for ground slope feedforward tilt, chatter filtering, and torque compensation."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    def setUp(self) -> None:
        self.obs = TerrainObserver()
        self.ctrl_node = BalanceControllerNode()

    def tearDown(self) -> None:
        self.ctrl_node.destroy_node()

    def test_slope_estimation_accuracy_across_steep_angles(self) -> None:
        """Verify ground slope estimation accuracy across [-30 deg, +30 deg] steep inclines."""
        angles_deg = [-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0]
        for deg in angles_deg:
            rad = math.radians(deg)
            ax = 9.81 * math.sin(rad)
            az = 9.81 * math.cos(rad)
            estimated_slope = self.obs.estimate_slope(pitch=rad, ax=ax, az=az)
            self.assertAlmostEqual(
                estimated_slope,
                rad,
                places=3,
                msg=f"Slope estimation mismatch at {deg} degrees",
            )

    def test_slope_filter_chatter_attenuation_on_rough_terrain(self) -> None:
        """Verify alpha=0.95 exponential filter attenuates high-frequency roughness chatter."""
        # Baseline ground slope is 0.0
        self.obs.state = FlightState.GROUND_BALANCE
        self.obs.terrain_slope = 0.0

        # Inject 10 Hz high-frequency slope fluctuation (+/- 0.15 rad) over 10 ticks (200 ms)
        filtered_slopes = []
        for i in range(10):
            t = i * 0.02
            chatter_slope = 0.15 * math.sin(2.0 * math.pi * 10.0 * t)
            ax = 9.81 * math.sin(chatter_slope)
            az = 9.81 * math.cos(chatter_slope)
            self.obs.update(
                t=t,
                accel_mag=9.81,
                wheel_contact=True,
                pitch=chatter_slope,
                z_vel=0.0,
                dt=0.02,
                ax=ax,
                az=az,
            )
            filtered_slopes.append(self.obs.terrain_slope)

        # Maximum filtered slope amplitude should be significantly attenuated compared to 0.15 rad
        max_amplitude = max(abs(s) for s in filtered_slopes)
        self.assertLess(
            max_amplitude,
            0.08,
            f"Slope filter failed to attenuate rough terrain chatter: max amp {max_amplitude} rad",
        )

    def test_balance_controller_slope_superposition(self) -> None:
        """Verify reference pitch includes theta_cmd + theta_eq + terrain_slope superposition."""
        self.ctrl_node.theta_cmd = 0.05
        self.ctrl_node.theta_eq = -0.02

        # Send slope message: +12 deg (+0.2094 rad)
        slope_msg = Float64()
        slope_msg.data = 0.2094
        self.ctrl_node._slope_callback(slope_msg)
        self.assertAlmostEqual(self.ctrl_node.terrain_slope, 0.2094)

        self.ctrl_node._control_loop()
        expected_theta_ref = 0.05 + (-0.02) + 0.2094
        self.assertAlmostEqual(self.ctrl_node.theta_ref, expected_theta_ref, places=4)

    def test_exact_steady_state_gravity_torque_cancellation_on_slope(self) -> None:
        """Verify on a slope of angle alpha, tilting by theta=alpha eliminates balance effort."""
        slope_angle = math.radians(15.0)  # ~0.2618 rad
        self.ctrl_node.theta_cmd = 0.0
        self.ctrl_node.theta_eq = 0.0
        self.ctrl_node.terrain_slope = slope_angle
        self.ctrl_node.v_ref = 0.0
        self.ctrl_node.e_integral = 0.0

        # Robot has converged to slope tilt: pitch = slope_angle, velocity = 0, pitch_rate = 0
        self.ctrl_node.estimator.v = 0.0
        self.ctrl_node.estimator.theta = slope_angle
        self.ctrl_node.estimator.theta_dot = 0.0
        self.ctrl_node.estimator.effective_length = 0.28
        self.ctrl_node.estimator.yaw_rate = 0.0

        pub_mock = MagicMock()
        self.ctrl_node.pub_torque.publish = pub_mock

        self.ctrl_node._control_loop()
        pub_mock.assert_called_once()
        torques = pub_mock.call_args[0][0].data
        tau_l, tau_r = torques[0], torques[1]

        # Steady-state torque to drive wheels must be exactly 0.0 N*m!
        self.assertAlmostEqual(
            tau_l,
            0.0,
            places=5,
            msg=f"Residual torque {tau_l} Nm remaining on 15 deg slope despite slope feedforward tilt",
        )
        self.assertAlmostEqual(tau_r, 0.0, places=5)

    def test_extreme_slope_torque_saturation_and_antiwindup(self) -> None:
        """Verify extreme 60 deg slope saturates torque cleanly to tau_max without NaN or windup."""
        slope_angle = math.radians(60.0)  # ~1.0472 rad (raw LQR torque ~ 29.25 Nm > tau_max=25.0 Nm)
        self.ctrl_node.terrain_slope = slope_angle
        self.ctrl_node.estimator.theta = 0.0  # Robot still level, generating large pitch error
        self.ctrl_node.estimator.v = 0.0
        self.ctrl_node.v_ref = 0.0
        self.ctrl_node.e_integral = 0.0

        pub_mock = MagicMock()
        self.ctrl_node.pub_torque.publish = pub_mock

        self.ctrl_node._control_loop()
        torques = pub_mock.call_args[0][0].data
        tau_l = torques[0]

        # Torque must clamp to exactly tau_max (25.0 Nm), not blow up or produce NaN
        self.assertAlmostEqual(abs(tau_l), 25.0, places=2)
        self.assertFalse(math.isnan(tau_l))
        self.assertFalse(math.isinf(tau_l))


class TestAdversarialLiDARClearance(unittest.TestCase):
    """Stress tests for /scan forward clearance calculation under malformed and boundary inputs."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    def setUp(self) -> None:
        self.node = TerrainObserverNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_empty_scan_ranges_handled_safely(self) -> None:
        """Verify empty scan ranges list does not raise exception or corrupt clearance."""
        scan = LaserScan()
        scan.ranges = []
        initial_clearance = self.node.forward_clearance
        self.node._scan_callback(scan)
        self.assertEqual(self.node.forward_clearance, initial_clearance)

    def test_pathological_float_scans_nan_and_inf(self) -> None:
        """Verify all-NaN, all-+Inf, and all--Inf scans do not cause NaN or crashes."""
        scan = LaserScan()
        scan.range_min = 0.10
        scan.range_max = 12.0

        # All-NaN
        scan.ranges = [float('nan')] * 360
        self.node._scan_callback(scan)
        self.assertFalse(math.isnan(self.node.forward_clearance))

        # All-+Inf
        scan.ranges = [float('inf')] * 360
        self.node._scan_callback(scan)
        self.assertFalse(math.isinf(self.node.forward_clearance))

        # All--Inf
        scan.ranges = [-float('inf')] * 360
        self.node._scan_callback(scan)
        self.assertFalse(math.isinf(self.node.forward_clearance))

    def test_mixed_pathological_and_valid_ranges(self) -> None:
        """Verify mixed corruptions with one valid obstacle correctly extracts the obstacle."""
        scan = LaserScan()
        scan.range_min = 0.10
        scan.range_max = 12.0
        # 360 beams, center is index 180
        ranges = [float('nan')] * 360
        ranges[180] = 3.45  # valid beam in center
        ranges[185] = float('inf')
        ranges[175] = -1.0
        scan.ranges = ranges

        self.node._scan_callback(scan)
        self.assertAlmostEqual(self.node.forward_clearance, 3.45, places=2)

    def test_out_of_range_beams_filtered(self) -> None:
        """Verify beams < range_min (self-chassis) and > range_max (sky) are filtered."""
        scan = LaserScan()
        scan.range_min = 0.15
        scan.range_max = 10.0
        scan.ranges = [10.0] * 360

        # Inject chassis self-reflection (0.05m < 0.15m) in center
        scan.ranges[180] = 0.05
        scan.ranges[181] = 4.20
        self.node._scan_callback(scan)

        # 0.05 must be rejected, valid minimum is 4.20
        self.assertAlmostEqual(self.node.forward_clearance, 4.20, places=2)

        # Inject beam > range_max (15.0m > 10.0m)
        scan.ranges = [15.0] * 360
        scan.ranges[180] = 6.80
        self.node._scan_callback(scan)
        self.assertAlmostEqual(self.node.forward_clearance, 6.80, places=2)

    def test_central_cone_isolation_ignores_side_obstacles(self) -> None:
        """Verify forward cone (+/- 15 deg) ignores close obstacles on the sides or behind."""
        scan = LaserScan()
        scan.range_min = 0.10
        scan.range_max = 12.0
        scan.ranges = [8.0] * 360  # all beams 8.0m

        # Obstacle at index 45 (+45 deg, outside +/- 15 deg center) at 0.30m
        scan.ranges[45] = 0.30
        # Obstacle at index 90 (+90 deg, lateral side) at 0.20m
        scan.ranges[90] = 0.20
        # Obstacle at index 270 (-90 deg, lateral side) at 0.25m
        scan.ranges[270] = 0.25

        self.node._scan_callback(scan)
        # Forward clearance must ignore side obstacles and report central 8.0m
        self.assertAlmostEqual(self.node.forward_clearance, 8.0, places=2)

        # Now place obstacle directly in front (index 180, 0 deg center) at 1.85m
        scan.ranges[180] = 1.85
        self.node._scan_callback(scan)
        self.assertAlmostEqual(self.node.forward_clearance, 1.85, places=2)

    def test_closest_hazard_selected_among_multiple_forward_beams(self) -> None:
        """Verify closest obstacle is picked when multiple obstacles are in the center cone."""
        scan = LaserScan()
        scan.range_min = 0.10
        scan.range_max = 12.0
        scan.ranges = [10.0] * 360
        scan.ranges[178] = 4.5
        scan.ranges[180] = 1.75
        scan.ranges[182] = 2.90

        self.node._scan_callback(scan)
        self.assertAlmostEqual(self.node.forward_clearance, 1.75, places=2)

    def test_arbitrary_beam_count_resolutions(self) -> None:
        """Verify scanner resolutions from 1 beam to 20,000 beams execute safely."""
        resolutions = [1, 2, 7, 100, 360, 720, 1080, 2048, 20000]
        for num_beams in resolutions:
            scan = LaserScan()
            scan.range_min = 0.10
            scan.range_max = 12.0
            scan.ranges = [5.0] * num_beams
            mid = num_beams // 2
            scan.ranges[mid] = 2.15

            self.node._scan_callback(scan)
            self.assertAlmostEqual(
                self.node.forward_clearance,
                2.15,
                places=2,
                msg=f"Resolution with {num_beams} beams failed to track obstacle",
            )


class TestFullJumpAndLandingNodeIntegration(unittest.TestCase):
    """End-to-end integration test of TerrainObserverNode across a full jump trajectory."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    def setUp(self) -> None:
        self.node = TerrainObserverNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_full_5_state_jump_flight_trajectory(self) -> None:
        """Verify complete flight trajectory: GROUND -> RAMP -> AIRBORNE -> TOUCHDOWN -> RECOVERY -> GROUND."""
        fsm_history: list[FlightState] = []
        imu = Imu()
        imu.orientation.w = 1.0

        # Phase 1: Ground Driving (flat ground, v=0.5 m/s, az=9.81 m/s^2)
        imu.linear_acceleration.z = 9.81
        imu.linear_acceleration.x = 0.0
        self.node.v_meas = 0.5
        for _ in range(5):
            self.node._imu_callback(imu)
            self.node._control_loop()
            fsm_history.append(FlightState(self.node.observer.state))
        self.assertEqual(fsm_history[-1], FlightState.GROUND_BALANCE)

        # Phase 2: Ramp Ascent (pitch=12 deg = 0.209 rad, v=0.8 m/s)
        pitch_rad = math.radians(12.0)
        imu.orientation.x = 0.0
        imu.orientation.y = math.sin(pitch_rad / 2.0)
        imu.orientation.z = 0.0
        imu.orientation.w = math.cos(pitch_rad / 2.0)
        imu.linear_acceleration.x = 9.81 * math.sin(pitch_rad)
        imu.linear_acceleration.z = 9.81 * math.cos(pitch_rad)
        self.node.v_meas = 0.8
        for _ in range(5):
            self.node._imu_callback(imu)
            self.node._control_loop()
            fsm_history.append(FlightState(self.node.observer.state))
        self.assertEqual(fsm_history[-1], FlightState.RAMP_ASCENT)

        # Phase 3: Airborne Free-Fall (liftoff, az ~ 0, amag < 2.5 m/s^2 for > 30ms)
        imu.linear_acceleration.x = 0.0
        imu.linear_acceleration.y = 0.0
        imu.linear_acceleration.z = 0.05
        for _ in range(10):  # 200 ms in air
            self.node._imu_callback(imu)
            self.node._control_loop()
            fsm_history.append(FlightState(self.node.observer.state))
        self.assertEqual(fsm_history[-1], FlightState.AIRBORNE)

        # Phase 4: Touchdown Impact (spike = 26 m/s^2)
        imu.linear_acceleration.z = 26.0
        self.node._imu_callback(imu)
        self.node._control_loop()
        fsm_history.append(FlightState(self.node.observer.state))
        self.assertEqual(fsm_history[-1], FlightState.TOUCHDOWN_ABSORPTION)

        # Phase 5: Absorption Damping (vertical velocity settles below 0.05 m/s)
        imu.linear_acceleration.z = 9.81
        self.node.z_vel = 0.02
        self.node._imu_callback(imu)
        self.node._control_loop()
        fsm_history.append(FlightState(self.node.observer.state))
        self.assertEqual(fsm_history[-1], FlightState.BALANCE_RECOVERY)

        # Phase 6: Balance Recovery Convergence (|pitch| < 5 deg, |z_vel| < 0.03 m/s)
        imu.orientation.y = 0.0
        imu.orientation.w = 1.0
        self.node.z_vel = 0.01
        self.node._imu_callback(imu)
        self.node._control_loop()
        fsm_history.append(FlightState(self.node.observer.state))
        self.assertEqual(fsm_history[-1], FlightState.GROUND_BALANCE)

        # Verify that all 5 states were visited in proper physical sequence
        unique_visited_states = []
        for s in fsm_history:
            if not unique_visited_states or unique_visited_states[-1] != s:
                unique_visited_states.append(s)

        expected_sequence = [
            FlightState.GROUND_BALANCE,
            FlightState.RAMP_ASCENT,
            FlightState.AIRBORNE,
            FlightState.TOUCHDOWN_ABSORPTION,
            FlightState.BALANCE_RECOVERY,
            FlightState.GROUND_BALANCE,
        ]
        self.assertEqual(unique_visited_states, expected_sequence)


if __name__ == '__main__':
    unittest.main()
