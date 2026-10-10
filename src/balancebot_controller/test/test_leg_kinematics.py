#!/usr/bin/env python3
"""Comprehensive unit tests for LegKinematics and LegKinematicsControllerNode."""

import math
import unittest
from unittest.mock import MagicMock

from balancebot_controller.balance_controller import BalanceControllerNode
from balancebot_controller.leg_kinematics import LegKinematics, LegKinematicsControllerNode
import numpy as np
import rclpy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray


class TestLegKinematics(unittest.TestCase):
    """Test suite for pure mathematical LegKinematics class."""

    def setUp(self) -> None:
        self.kin = LegKinematics(
            l1=0.20,
            l2=0.20,
            height_min=0.18,
            height_max=0.38,
            offset_min=-0.08,
            offset_max=0.08,
            total_mass=12.4,
            g=9.81,
            tau_max=50.0,
        )

    def test_forward_kinematics_straight_and_symmetric(self) -> None:
        """Verify FK at straight downward configuration and symmetric crouch."""
        # Fully straight downward: q_h = 0, q_k = 0
        x0, z0 = self.kin.forward_kinematics(0.0, 0.0)
        self.assertAlmostEqual(x0, 0.0, places=5)
        self.assertAlmostEqual(z0, -0.40, places=5)

        # Symmetrical crouch: q_h = -30 deg, q_k = +60 deg
        qh = -math.radians(30.0)
        qk = math.radians(60.0)
        x_sym, z_sym = self.kin.forward_kinematics(qh, qk)
        self.assertAlmostEqual(x_sym, 0.0, places=4)
        expected_z = -2.0 * 0.20 * math.cos(math.radians(30.0))
        self.assertAlmostEqual(z_sym, expected_z, places=4)

    def test_inverse_kinematics_roundtrip(self) -> None:
        """Verify FK(IK(x, -h)) recovers (x, -h) across operational workspace."""
        for h in [0.20, 0.25, 0.28, 0.32, 0.36]:
            for x in [-0.06, -0.03, 0.0, 0.03, 0.06]:
                qh, qk = self.kin.inverse_kinematics(x, -h)
                x_calc, z_calc = self.kin.forward_kinematics(qh, qk)
                self.assertAlmostEqual(x_calc, x, delta=1e-3)
                self.assertAlmostEqual(-z_calc, h, delta=1e-3)

    def test_workspace_boundary_clamping(self) -> None:
        """Verify extreme height and offset targets are safely clamped."""
        # Height beyond reach (0.50 m)
        qh_high, qk_high = self.kin.inverse_kinematics(0.0, -0.50)
        _, z_high = self.kin.forward_kinematics(qh_high, qk_high)
        self.assertLessEqual(-z_high, 0.38 + 1e-3)

        # Height below minimum (0.10 m)
        qh_low, qk_low = self.kin.inverse_kinematics(0.0, -0.10)
        _, z_low = self.kin.forward_kinematics(qh_low, qk_low)
        self.assertGreaterEqual(-z_low, 0.18 - 1e-3)

        # Excessive forward and rearward offsets (+0.25 m, -0.25 m)
        qh_fwd, qk_fwd = self.kin.inverse_kinematics(0.25, -0.28)
        x_fwd, _ = self.kin.forward_kinematics(qh_fwd, qk_fwd)
        self.assertLessEqual(abs(x_fwd), 0.08 + 1e-3)

        qh_rear, qk_rear = self.kin.inverse_kinematics(-0.25, -0.28)
        x_rear, _ = self.kin.forward_kinematics(qh_rear, qk_rear)
        self.assertLessEqual(abs(x_rear), 0.08 + 1e-3)

    def test_jacobian_analytical_and_determinant(self) -> None:
        """Verify leg Jacobian shape, determinant, and numerical accuracy."""
        qh, qk = self.kin.inverse_kinematics(0.02, -0.28)
        j = self.kin.jacobian(qh, qk)
        self.assertEqual(j.shape, (2, 2))

        # Determinant det(J) = L1 * L2 * sin(qk) > 0.02
        det_j = float(np.linalg.det(j))
        expected_det = 0.20 * 0.20 * math.sin(qk)
        self.assertAlmostEqual(det_j, expected_det, places=4)
        self.assertGreater(det_j, 0.02)

        # Numerical differentiation check for J_11, J_21
        eps = 1e-6
        x_plus, z_plus = self.kin.forward_kinematics(qh + eps, qk)
        x_minus, z_minus = self.kin.forward_kinematics(qh - eps, qk)
        num_j11 = (x_plus - x_minus) / (2.0 * eps)
        num_j21 = (z_plus - z_minus) / (2.0 * eps)
        self.assertAlmostEqual(j[0, 0], num_j11, delta=1e-4)
        self.assertAlmostEqual(j[1, 0], num_j21, delta=1e-4)

    def test_virtual_model_control_spring_damping_and_symmetry(self) -> None:
        """Verify VMC Cartesian spring-damper wrench mapping and symmetry."""
        qh, qk = self.kin.inverse_kinematics(0.0, -0.28)

        # 1. Downward compression produces greater restorative torque
        tau_h_nom, tau_k_nom = self.kin.virtual_model_torque(
            qh, qk, target_z=0.28, current_z=0.28, z_vel=0.0
        )
        tau_h_comp, tau_k_comp = self.kin.virtual_model_torque(
            qh, qk, target_z=0.28, current_z=0.24, z_vel=0.0
        )
        self.assertGreater(abs(tau_k_comp), abs(tau_k_nom))

        # 2. Symmetrical stance (x=0, Fx=0) produces zero hip torque
        self.assertAlmostEqual(tau_h_nom, 0.0, delta=1e-4)
        self.assertAlmostEqual(tau_h_comp, 0.0, delta=1e-4)

        # 3. Damping opposes downward motion (stronger upward push)
        tau_h_up, tau_k_up = self.kin.virtual_model_torque(
            qh, qk, target_z=0.28, current_z=0.28, z_vel=0.5, d_z=100.0
        )
        tau_h_down, tau_k_down = self.kin.virtual_model_torque(
            qh, qk, target_z=0.28, current_z=0.28, z_vel=-0.5, d_z=100.0
        )
        self.assertGreater(abs(tau_k_down), abs(tau_k_up))

        # 4. Torque clamping to tau_max (50 Nm)
        tau_h_huge, tau_k_huge = self.kin.virtual_model_torque(
            qh, qk, target_z=0.50, current_z=0.10, k_z=10000.0
        )
        self.assertLessEqual(abs(tau_h_huge), 50.0)
        self.assertLessEqual(abs(tau_k_huge), 50.0)

    def test_dynamic_equilibrium_pitch(self) -> None:
        """Verify dynamic equilibrium pitch angle theta_eq = arcsin(delta_x / L_eff)."""
        length = 0.28

        # Positive offset -> forward lean (theta_eq > 0)
        theta_pos = self.kin.equilibrium_pitch(0.04, length)
        self.assertGreater(theta_pos, 0.0)
        self.assertAlmostEqual(theta_pos, math.asin(0.04 / length), places=4)

        # Negative offset -> backward lean (theta_eq < 0)
        theta_neg = self.kin.equilibrium_pitch(-0.04, length)
        self.assertLess(theta_neg, 0.0)
        self.assertAlmostEqual(theta_neg, math.asin(-0.04 / length), places=4)

        # Zero offset -> vertical stance (theta_eq = 0)
        theta_zero = self.kin.equilibrium_pitch(0.0, length)
        self.assertAlmostEqual(theta_zero, 0.0, places=6)

        # Small angle approximation error < 0.01 rad
        self.assertLess(abs(theta_pos - (0.04 / length)), 0.01)


class TestLegKinematicsControllerNode(unittest.TestCase):
    """Test suite for ROS 2 LegKinematicsControllerNode."""

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

    def test_node_initialization_defaults(self) -> None:
        """Verify node parameter declarations and initial states."""
        self.assertEqual(self.node.leg_mode, 0)  # AUTO_ADAPTATION
        self.assertAlmostEqual(self.node.current_height, 0.28, places=3)
        self.assertAlmostEqual(self.node.current_offset, 0.0, places=3)
        self.assertAlmostEqual(self.node.dt, 0.02, places=4)

    def test_joint_state_and_imu_callbacks(self) -> None:
        """Verify joint positions and IMU roll extraction."""
        # 1. JointState callback
        js = JointState()
        js.name = ['left_hip_joint', 'left_knee_joint', 'right_hip_joint', 'right_knee_joint']
        js.position = [-0.3, 0.6, -0.3, 0.6]
        js.velocity = [0.1, -0.2, 0.1, -0.2]
        self.node._joint_callback(js)

        self.assertAlmostEqual(self.node.q_lh, -0.3)
        self.assertAlmostEqual(self.node.q_lk, 0.6)
        self.assertAlmostEqual(self.node.dq_lh, 0.1)
        self.assertAlmostEqual(self.node.dq_lk, -0.2)

        # 2. IMU roll angle extraction: 5 deg roll
        roll_target = math.radians(5.0)
        imu_msg = Imu()
        imu_msg.orientation.w = math.cos(roll_target / 2.0)
        imu_msg.orientation.x = math.sin(roll_target / 2.0)
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = 0.0
        self.node._imu_callback(imu_msg)

        self.assertAlmostEqual(self.node.roll_angle, roll_target, places=3)

    def test_slew_rate_limiting_in_manual_mode(self) -> None:
        """Verify height and offset changes respect velocity slew rates."""
        # Command manual mode with large step change
        cmd = Float64MultiArray()
        cmd.data = [1.0, 0.36, 0.06]  # target_h = 0.36 (+0.08m), target_x = +0.06m
        self.node._cmd_mode_callback(cmd)

        self.assertEqual(self.node.leg_mode, 1)

        # Single step at dt = 0.02s: dh <= 0.20 * 0.02 = 0.004 m, dx <= 0.15 * 0.02 = 0.003 m
        h_prev = self.node.current_height
        x_prev = self.node.current_offset
        self.node._control_loop()

        self.assertLessEqual(abs(self.node.current_height - h_prev), 0.004 + 1e-5)
        self.assertLessEqual(abs(self.node.current_offset - x_prev), 0.003 + 1e-5)

        # Over 50 steps (1.0 s), targets are reached within tolerance
        for _ in range(50):
            self.node._control_loop()

        self.assertAlmostEqual(self.node.current_height, 0.36, delta=0.01)
        self.assertAlmostEqual(self.node.current_offset, 0.06, delta=0.01)

    def test_latched_mode_switching(self) -> None:
        """Verify manual mode remains latched until explicitly returned to auto."""
        # Switch to manual
        cmd_manual = Float64MultiArray()
        cmd_manual.data = [1.0, 0.24, -0.03]
        self.node._cmd_mode_callback(cmd_manual)
        self.assertEqual(self.node.leg_mode, 1)

        # Run control loop for multiple cycles without new commands
        for _ in range(20):
            self.node._control_loop()
        self.assertEqual(self.node.leg_mode, 1)

        # Switch back to auto
        cmd_auto = Float64MultiArray()
        cmd_auto.data = [0.0]
        self.node._cmd_mode_callback(cmd_auto)
        self.assertEqual(self.node.leg_mode, 0)

    def test_auto_adaptation_roll_leveling(self) -> None:
        """Verify auto mode adjusts left and right target heights for roll leveling."""
        # Set roll angle = +6 deg
        self.node.roll_angle = math.radians(6.0)
        self.node.leg_mode = 0

        # Mock publishers to capture published messages
        pub_state_mock = MagicMock()
        self.node.pub_leg_state.publish = pub_state_mock

        self.node._control_loop()
        pub_state_mock.assert_called_once()
        published_data = pub_state_mock.call_args[0][0].data

        # Array format: [eff_l, offset_x, theta_eq, mode, roll]
        self.assertEqual(len(published_data), 5)
        self.assertEqual(published_data[3], 0.0)  # auto mode code
        self.assertAlmostEqual(published_data[4], math.radians(6.0), places=3)

    def test_balance_controller_pitch_equilibrium_coordination(self) -> None:
        """Verify BalanceControllerNode dynamically shifts theta_ref when theta_eq is received."""
        balance_node = BalanceControllerNode()
        try:
            self.assertAlmostEqual(balance_node.theta_ref, 0.0)

            # Publish leg_state with theta_eq = 0.12 rad (~6.88 deg)
            leg_msg = Float64MultiArray()
            leg_msg.data = [0.28, 0.03, 0.12, 1.0, 0.0]
            balance_node._leg_state_callback(leg_msg)

            self.assertAlmostEqual(balance_node.theta_eq, 0.12, places=4)
            self.assertAlmostEqual(balance_node.theta_ref, 0.12, places=4)

            # In control loop, theta_ref stays coordinated
            balance_node._control_loop()
            self.assertAlmostEqual(balance_node.theta_ref, 0.12, places=4)
        finally:
            balance_node.destroy_node()


if __name__ == '__main__':
    unittest.main()
