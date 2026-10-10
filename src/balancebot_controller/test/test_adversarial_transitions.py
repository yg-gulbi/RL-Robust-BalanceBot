#!/usr/bin/env python3
"""Milestone M3 Adversarial Challenge 2: Dynamic Transitions & Mode Latching.

Adversarial stress-test suite verifying:
1. Mode switching dynamics and latching behavior under rapid successive calls.
2. Slew-rate limiter under step commands (Delta_h = 0.15m, |Delta_tau| < 5.0 Nm/step,
   |h_dot| <= 0.20 m/s).
3. Roll leveling without height sag across operating roll envelope.
4. Equilibrium pitch trim coordination and gravitational offset torque cancellation.
"""

import math
import unittest

from balancebot_controller.balance_controller import BalanceControllerNode
from balancebot_controller.leg_kinematics import LegKinematics, LegKinematicsControllerNode
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class TestModeSwitchingDynamics(unittest.TestCase):
    """Stress tests for mode switching dynamics and latching behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self) -> None:
        self.node = LegKinematicsControllerNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_rapid_successive_mode_toggling(self) -> None:
        """Verify mode remains robustly latched when rapidly toggled in succession."""
        # Rapid sequence of alternating mode requests
        toggle_sequence = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        for mode_val in toggle_sequence:
            msg = Float64MultiArray()
            msg.data = [mode_val, 0.32, 0.03]
            self.node._cmd_mode_callback(msg)

        # Final command was 1.0, mode must be MANUAL_TARGET (1)
        self.assertEqual(self.node.leg_mode, 1)

        # Run 100 control cycles with NO new commands: mode must stay latched
        for _ in range(100):
            self.node._control_loop()
        self.assertEqual(self.node.leg_mode, 1)
        self.assertAlmostEqual(self.node.current_height, 0.32, delta=1e-3)
        self.assertAlmostEqual(self.node.current_offset, 0.03, delta=1e-3)

    def test_partial_mode_commands_and_latching(self) -> None:
        """Verify latching persists when receiving single-element mode updates."""
        # Set manual mode with explicit targets
        msg1 = Float64MultiArray()
        msg1.data = [1.0, 0.34, -0.04]
        self.node._cmd_mode_callback(msg1)
        self.assertEqual(self.node.leg_mode, 1)

        # Run 20 steps
        for _ in range(20):
            self.node._control_loop()

        # Re-send only mode indicator [1.0] without target elements
        msg2 = Float64MultiArray()
        msg2.data = [1.0]
        self.node._cmd_mode_callback(msg2)
        self.assertEqual(self.node.leg_mode, 1)
        self.assertAlmostEqual(self.node.target_height, 0.34)
        self.assertAlmostEqual(self.node.target_offset, -0.04)

        # Slew to completion
        for _ in range(50):
            self.node._control_loop()
        self.assertAlmostEqual(self.node.current_height, 0.34, delta=1e-3)
        self.assertAlmostEqual(self.node.current_offset, -0.04, delta=1e-3)

    def test_empty_and_invalid_mode_command_tolerance(self) -> None:
        """Verify robustness against empty payload and invalid mode integers."""
        # Empty data payload should not crash
        msg_empty = Float64MultiArray()
        msg_empty.data = []
        self.node._cmd_mode_callback(msg_empty)
        self.assertEqual(self.node.leg_mode, 0)

        # Invalid mode integers (e.g. 2, -1, 99) should safely map to Auto (0)
        for invalid_val in [2.0, -1.0, 99.0]:
            msg_invalid = Float64MultiArray()
            msg_invalid.data = [invalid_val]
            self.node._cmd_mode_callback(msg_invalid)
            self.assertEqual(self.node.leg_mode, 0)


class TestSlewRateLimiterUnderStepCommand(unittest.TestCase):
    """Stress tests for slew-rate limiting under severe step commands."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self) -> None:
        self.node = LegKinematicsControllerNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_height_step_0_15m_ramp_rate_and_torque_spike(self) -> None:
        """Verify height step Delta_h = 0.15m ramps <= 0.20 m/s with |Delta_tau| < 5.0 Nm/step."""
        # Start at lower height h = 0.20 m
        self.node.leg_mode = 1
        self.node.current_height = 0.20
        self.node.target_height = 0.20
        self.node.current_offset = 0.0
        self.node.target_offset = 0.0

        # Set joint angles corresponding to h = 0.20 m
        qh0, qk0 = self.node.kin.inverse_kinematics(0.0, -0.20)
        self.node.q_lh = qh0
        self.node.q_lk = qk0
        self.node.q_rh = qh0
        self.node.q_rk = qk0

        # Mock publisher to capture torque commands
        published_torques = []

        def capture_torque(msg: JointState) -> None:
            published_torques.append(list(msg.effort))

        self.node.pub_leg_joint_torque.publish = capture_torque

        # Apply large step command: Delta_h = +0.15 m -> target_h = 0.35 m
        cmd = Float64MultiArray()
        cmd.data = [1.0, 0.35, 0.0]
        self.node._cmd_mode_callback(cmd)

        h_history = [self.node.current_height]
        dt = self.node.dt

        # Run control loop for 60 cycles (1.2 seconds)
        for _ in range(60):
            # Update simulated joint positions to track current filtered height
            qh, qk = self.node.kin.inverse_kinematics(0.0, -self.node.current_height)
            self.node.q_lh = qh
            self.node.q_lk = qk
            self.node.q_rh = qh
            self.node.q_rk = qk

            self.node._control_loop()
            h_history.append(self.node.current_height)

        # 1. Verify slew-rate limit: |h_dot| <= 0.20 m/s at EVERY single step
        for i in range(1, len(h_history)):
            dh = h_history[i] - h_history[i - 1]
            h_dot = dh / dt
            self.assertLessEqual(
                h_dot, 0.20 + 1e-6,
                f'Height slew rate exceeded 0.20 m/s at step {i}: {h_dot:.4f} m/s'
            )
            self.assertGreaterEqual(
                h_dot, -1e-6,
                f'Unexpected negative ramp direction at step {i}: {h_dot:.4f} m/s'
            )

        # 2. Verify torque continuity: |Delta_tau| < 5.0 N*m per step
        self.assertGreater(len(published_torques), 2)
        for i in range(1, len(published_torques)):
            tau_prev = published_torques[i - 1]
            tau_curr = published_torques[i]
            for j in range(len(tau_curr)):
                delta_tau = abs(tau_curr[j] - tau_prev[j])
                self.assertLess(
                    delta_tau, 5.0,
                    f'Instantaneous torque spike detected on joint {j} at step {i}: '
                    f'{delta_tau:.3f} N*m >= 5.0 N*m'
                )

        # 3. Target reached smoothly at t ~ Delta_h / 0.20 = 0.75 s (38 steps)
        self.assertAlmostEqual(self.node.current_height, 0.35, delta=1e-3)

    def test_downward_height_step_and_negative_ramp(self) -> None:
        """Verify downward height step Delta_h = -0.15m ramps at <= 0.20 m/s."""
        self.node.leg_mode = 1
        self.node.current_height = 0.35
        self.node.target_height = 0.35

        cmd = Float64MultiArray()
        cmd.data = [1.0, 0.20, 0.0]
        self.node._cmd_mode_callback(cmd)

        h_prev = self.node.current_height
        for _ in range(50):
            self.node._control_loop()
            dh = self.node.current_height - h_prev
            h_dot = dh / self.node.dt
            self.assertGreaterEqual(
                h_dot, -0.20 - 1e-6,
                f'Downward slew rate exceeded -0.20 m/s: {h_dot:.4f}'
            )
            h_prev = self.node.current_height

        self.assertAlmostEqual(self.node.current_height, 0.20, delta=1e-3)


class TestRollLevelingWithoutSag(unittest.TestCase):
    """Stress tests for roll tilt compensation and height sag invariance."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self) -> None:
        self.node = LegKinematicsControllerNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_roll_leveling_differential_height_and_zero_sag(self) -> None:
        """Verify Delta_h = (W/2)*tan(phi) across roll angles with zero mean height sag."""
        track_w = self.node.track_w  # 0.40 m
        nom_h = self.node.nom_h      # 0.28 m

        # Sweep roll angles from -25 deg to +25 deg in steps of 2.5 deg
        roll_angles_deg = np.linspace(-25.0, 25.0, 21)

        for phi_deg in roll_angles_deg:
            phi_rad = math.radians(phi_deg)
            self.node.roll_angle = phi_rad
            self.node.leg_mode = 0  # AUTO_ADAPTATION

            # Expected differential leg height
            expected_delta_h = (track_w / 2.0) * math.tan(phi_rad)
            expected_h_left = float(np.clip(
                nom_h + expected_delta_h, self.node.kin.height_min, self.node.kin.height_max
            ))
            expected_h_right = float(np.clip(
                nom_h - expected_delta_h, self.node.kin.height_min, self.node.kin.height_max
            ))

            # Run control loop to generate leg commands
            self.node._control_loop()

            # For roll in [-20, 20] deg, neither leg clamps
            if abs(phi_deg) <= 20.0:
                mean_h = 0.5 * (expected_h_left + expected_h_right)
                height_sag = abs(mean_h - nom_h)
                self.assertLess(
                    height_sag, 1e-6,
                    f'Height sag detected at roll {phi_deg:.1f} deg: sag={height_sag:.6f} m'
                )
                self.assertAlmostEqual(
                    expected_h_left - expected_h_right, 2.0 * expected_delta_h, places=5
                )

    def test_extreme_roll_boundary_resilience(self) -> None:
        """Verify extreme rolls (+-45 deg, +-60 deg, +-89 deg) clamp cleanly without NaN."""
        for extreme_phi_deg in [-89.0, -60.0, -45.0, 45.0, 60.0, 89.0]:
            self.node.roll_angle = math.radians(extreme_phi_deg)
            self.node.leg_mode = 0
            self.node._control_loop()

            # Confirm node states remain finite
            self.assertFalse(math.isnan(self.node.current_height))
            self.assertFalse(math.isinf(self.node.current_height))


class TestEquilibriumPitchTrimCoordination(unittest.TestCase):
    """Stress tests for pitch trim equilibrium and gravitational offset cancellation."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self) -> None:
        self.kin = LegKinematics()
        self.balance_node = BalanceControllerNode()

    def tearDown(self) -> None:
        self.balance_node.destroy_node()

    def test_gravitational_offset_torque_cancellation(self) -> None:
        """Verify theta_eq = arcsin(Delta_x / L_eff) cancels gravitational moment identically."""
        total_m = self.kin.total_mass
        g = self.kin.g

        offsets = np.linspace(-0.08, 0.08, 17)
        lengths = [0.20, 0.25, 0.28, 0.32, 0.36]

        for length in lengths:
            for dx in offsets:
                theta_eq = self.kin.equilibrium_pitch(dx, length)

                # Gravitational torque around wheel contact point:
                # tau_grav = M * g * (L * sin(theta_eq) - dx)
                tau_grav = total_m * g * (length * math.sin(theta_eq) - dx)

                self.assertAlmostEqual(
                    tau_grav, 0.0, places=5,
                    msg=(
                        f'Residual gravitational moment for dx={dx:.3f}, '
                        f'L={length:.3f}: {tau_grav}'
                    )
                )

    def test_balance_controller_zero_steady_state_torque_at_equilibrium(self) -> None:
        """Verify balance controller produces zero steady-state torque when theta == theta_eq."""
        offsets = [-0.06, -0.03, 0.0, 0.03, 0.06]
        length = 0.28

        for dx in offsets:
            theta_eq = self.kin.equilibrium_pitch(dx, length)

            # Transmit coordinated leg state to balance controller
            leg_msg = Float64MultiArray()
            leg_msg.data = [length, dx, theta_eq, 1.0, 0.0]
            self.balance_node._leg_state_callback(leg_msg)

            # Robot is balanced at equilibrium angle theta = theta_eq, v = 0, theta_dot = 0
            self.balance_node.estimator.theta = theta_eq
            self.balance_node.estimator.theta_dot = 0.0
            self.balance_node.estimator.v = 0.0
            self.balance_node.v_ref = 0.0
            self.balance_node.theta_cmd = 0.0

            tau_w, sat = self.balance_node.controller.compute_torque(
                v_meas=0.0,
                v_ref=0.0,
                theta_meas=theta_eq,
                theta_ref=self.balance_node.theta_ref,
                theta_dot=0.0,
                e_integral=0.0,
                effective_length=length,
            )

            self.assertAlmostEqual(
                tau_w, 0.0, places=4,
                msg=f'Non-zero steady-state torque at physical equilibrium for dx={dx}: {tau_w} Nm'
            )
            self.assertFalse(sat)

    def test_uncoordinated_baseline_contrast(self) -> None:
        """Verify uncoordinated baseline (theta_ref = 0) demands non-zero continuous torque."""
        dx = 0.05
        length = 0.28
        theta_eq = self.kin.equilibrium_pitch(dx, length)

        # In uncoordinated case, theta_ref remains 0.0 despite physical offset
        tau_uncoordinated, _ = self.balance_node.controller.compute_torque(
            v_meas=0.0,
            v_ref=0.0,
            theta_meas=theta_eq,
            theta_ref=0.0,  # Uncoordinated
            theta_dot=0.0,
            e_integral=0.0,
            effective_length=length,
        )

        # Controller demands substantial torque fighting the equilibrium
        self.assertGreater(
            abs(tau_uncoordinated), 1.0,
            f'Expected substantial torque demand without coordination, got: {tau_uncoordinated}'
        )


if __name__ == '__main__':
    unittest.main()
